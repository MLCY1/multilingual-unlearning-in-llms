import math
import os, random, argparse, json, shutil
from collections import defaultdict
from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import DPOTrainer, DPOConfig
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
from transformers.trainer_callback import TrainerCallback
from llm_config import load_llm_config
# =========================================
# =========== GLOBAL CONFIG ===============
# =========================================
SEED: int = 42
KL_ON_ANSWER_ONLY : bool = True
MAXIMUM_LEN: int = 512
KL_WEIGHT: float = 1.0
KL_RATIO_M: int = 10
ENABLE_KL_MINIBATCH = False
KL_MINIBATCH_SPLITS = 10
FOLDER_TO_TAG = {
    "English": "En",
    "Chinese": "Ch",
    "German": "De",
    "Russian": "Ru",
    "Turkish": "Tu"
}

TAG_CANON_ORDER = {
    "En": 0,
    "Ch": 1,
    "De": 2,
    "Ru": 3,
    "Tu": 4
}
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO Unlearning launcher")
    
    p.add_argument("--model_path", type=str, required=True, help="Path to finetuned model")
    p.add_argument("--output_path", type=str, required=True, help="Path to save unlearned model")
    p.add_argument("--forget_paths", nargs="+", required=True, help="Paths to forget data files")
    p.add_argument("--retain_paths", nargs="+", required=True, help="Paths to retain data files")
    p.add_argument("--idk_paths", nargs="+", required=True, help="Paths to IDK data files")

    p.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    p.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file")
    p.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed config file (alternative)")
    p.add_argument(
        "--llm-family",
        type=str.lower,
        choices=["qwen", "gemma"],
        default=None,
        help="Select LLM family (defaults to env LLM_FAMILY or inferred).",
    )

    args, _ = p.parse_known_args()
    return args

_ARGS = _parse_args()
MODEL_ID: str = _ARGS.model_path
OUTPUT_DIR: str = _ARGS.output_path
FORGET_PATHS: List[str] = _ARGS.forget_paths
RETAIN_PATHS: List[str] = _ARGS.retain_paths
IDK_PATHS: List[str] = _ARGS.idk_paths
_LLM_FAMILY_ARG: Optional[str] = _ARGS.llm_family

def _infer_model_size(value: str) -> float | None:
    base = os.path.basename(os.path.normpath(value))
    for token in base.split("-"):
        token = token.strip()
        if token.lower().endswith("b"):
            num = token[:-1]
            try:
                return float(num)
            except ValueError:
                continue
    return None

_MODEL_SIZE_VALUE = _infer_model_size(MODEL_ID)

def _assert_data_files_exist():
    missing = [p for p in (FORGET_PATHS + IDK_PATHS + RETAIN_PATHS) if not os.path.isfile(p)]
    if missing:
        msg = "Missing required data files:\n  " + "\n  ".join(missing)
        raise FileNotFoundError(msg)

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def make_prompt_builder(tokenizer, llm_family: str) -> Callable[[str], str]:
    def _render(question: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return _render


def make_ds_config(zero_stage: int) -> Dict:
    cfg = {
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }
    cfg["bf16"] = {"enabled": True}
    return cfg


def _infer_llm_family(model_path: str, arg_family: Optional[str]) -> str:
    if arg_family:
        return arg_family.lower()
    env_family = os.environ.get("LLM_FAMILY")
    if env_family:
        return env_family.lower()
    lower_path = model_path.lower()
    if "gemma" in lower_path:
        return "gemma"
    return "qwen"

# =========================================
# =========== Load Data ===============
# =========================================
    
def _tag_from_data_path(p: str) -> str:
    parts = p.split(os.sep)
    try:
        data_idx = parts.index("Data")
        if data_idx + 1 < len(parts):
            folder = parts[data_idx + 1]
            tag = FOLDER_TO_TAG.get(folder)
            if tag:
                return tag
    except (ValueError, IndexError):
        pass
    raise ValueError(
        f"Could not extract language tag from path: {p}. "
        f"Expected path like ./Data/English/... or ./Data/Chinese/..."
    )

def _read_idk_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    
    if not lines:
        raise ValueError(f"IDK file is empty: {path}")
    
    return lines

def _read_json_dict(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object mapping question -> answer")
    return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip()}

def load_data_strict() -> Tuple[
    List[Tuple[str, str, str]],   
    Dict[str, List[str]],           
    List[Tuple[str, str, str]]    
]:
    idk_by_lang: Dict[str, List[str]] = {}
    for p in IDK_PATHS:
        tag = _tag_from_data_path(p)
        lines = _read_idk_lines(p)
        idk_by_lang[tag] = idk_by_lang.get(tag, []) + lines

    forget_items: List[Tuple[str, str, str]] = []
    for p in FORGET_PATHS:
        tag = _tag_from_data_path(p)
        j = _read_json_dict(p)
        for q, neg in j.items():
            forget_items.append((q, neg, tag))   
    retain_items: List[Tuple[str, str, str]] = []
    for p in RETAIN_PATHS:
        tag = _tag_from_data_path(p)
        j = _read_json_dict(p)
        for q, ans in j.items():
            retain_items.append((q, ans, tag))
    if not forget_items:
        raise ValueError("No forget examples loaded from any UN language files.")
    needed = {tag for _, _, tag in forget_items}
    missing_or_empty = [t for t in needed if (t not in idk_by_lang or len(idk_by_lang[t]) == 0)]
    if missing_or_empty:
        raise ValueError(f"IDK pool missing/empty for languages: {', '.join(missing_or_empty)}")

    return forget_items, idk_by_lang, retain_items

def build_pref_dataset(tokenizer, prompt_builder: Callable[[str], str]) -> Dataset:
    rows = []
    forget_items, idk_by_lang, _ = load_data_strict()

    idk_pools: Dict[str, List[str]] = {}
    idk_offsets: Dict[str, int] = {}
    for tag, pool in idk_by_lang.items():
        if not pool:
            raise ValueError(f"IDK pool for language '{tag}' is empty.")
        shuffled = pool[:]
        random.shuffle(shuffled)
        idk_pools[tag] = shuffled
        idk_offsets[tag] = 0

    for q, neg, tag in forget_items:
        prompt = prompt_builder(q)
        pool = idk_pools[tag]
        idx = idk_offsets[tag]
        pos = pool[idx]
        idx += 1
        if idx >= len(pool):
            random.shuffle(pool)
            idx = 0
        idk_offsets[tag] = idx
        rows.append({"prompt": prompt, "chosen": pos, "rejected": neg})

    return Dataset.from_list(rows).shuffle(seed=SEED)

def build_retain_dataset(tokenizer, prompt_builder: Callable[[str], str]) -> Dataset:
    rows = []
    _, _, retain_items = load_data_strict()

    for q, ans, tag in retain_items:
        prompt = prompt_builder(q)
        ans = ans + tokenizer.eos_token
        rows.append({"prompt": prompt, "retain_target": ans, "lang_tag": tag})
    if not retain_items:
        raise ValueError("No retain examples loaded from any UN language files.")
    return Dataset.from_list(rows).shuffle(seed=SEED)

def build_training_datasets(tokenizer, prompt_builder: Callable[[str], str]) -> Tuple[Dataset, Dataset]:
    pref_ds  = build_pref_dataset(tokenizer, prompt_builder)
    retain_ds = build_retain_dataset(tokenizer, prompt_builder)
    return pref_ds, retain_ds

def make_retain_collate_fn(tokenizer):
    def _len_ids(txt: str) -> int:
        return len(tokenizer(txt, add_special_tokens=False).input_ids)

    def collate(examples):
        prompts = [ex["prompt"] for ex in examples]
        answers = [ex["retain_target"] for ex in examples]

        full_texts = [p + a for p, a in zip(prompts, answers)]
        full = tokenizer(
            full_texts, add_special_tokens=False,
            return_tensors="pt", padding=True, truncation=False
        )
        prompt_lens = torch.tensor([_len_ids(p) for p in prompts], dtype=torch.long)
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "prompt_len": prompt_lens,
        }
    return collate

def _identity_collate(batch):
    return batch

class DPOWithRetainKL(DPOTrainer):
    def __init__(self, *args, retain_dataset: Dataset, tokenizer, prompt_builder: Callable[[str], str], forget_items, idk_by_lang, **kwargs):
        super().__init__(*args, **kwargs)
        self.retain_dataset = retain_dataset
        self._tokenizer = tokenizer
        self._prompt_builder = prompt_builder
        self._retain_collate_fn = make_retain_collate_fn(self._tokenizer)
        self.kl_weight = KL_WEIGHT
        self.kl_ratio = KL_RATIO_M
        self.kl_on_answer_only = KL_ON_ANSWER_ONLY
        self._last_dpo_loss = None
        self._last_kl_loss = None
        self._last_total_loss = None
        
        self.forget_items = forget_items
        self.idk_by_lang = idk_by_lang

        per_dev = max(1, int(self.args.per_device_train_batch_size * self.kl_ratio))
        self._build_balanced_retain_loaders(per_dev)
        self._retain_iter = self._make_retain_iter()

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs.pop("loss", None) 
        super().log(logs, start_time=start_time)

    def _build_balanced_retain_loaders(self, per_dev: int) -> None:
        if "lang_tag" not in self.retain_dataset.column_names:
            raise ValueError("Retain dataset must include a 'lang_tag' column for balanced sampling.")
        lang_to_indices = defaultdict(list)
        for idx, tag in enumerate(self.retain_dataset["lang_tag"]):
            lang_to_indices[tag].append(idx)

        order_key = lambda t: TAG_CANON_ORDER.get(t, len(TAG_CANON_ORDER))
        self._retain_langs = sorted(lang_to_indices.keys(), key=order_key)
        if not self._retain_langs:
            raise ValueError("No retain languages were found in the dataset.")
        
        if len(self._retain_langs) == 1:
            if self.accelerator.is_main_process:
                print(f"[Retain] Only 1 language found: {self._retain_langs[0]}. Using simple sampling.")
            
            lang = self._retain_langs[0]
            subset = self.retain_dataset.select(lang_to_indices[lang])
            sampler = DistributedSampler(
                subset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.args.seed,
            )
            loader = DataLoader(
                subset,
                batch_size=per_dev,
                sampler=sampler,
                drop_last=True,
                collate_fn=_identity_collate,
                pin_memory=True,
            )
            self._retain_lang_samplers = {lang: sampler}
            self._retain_lang_loaders = {lang: loader}
            self._retain_lang_order = [lang]
            return
        
        if per_dev < len(self._retain_langs):
            raise ValueError(
                f"Retain KL batch size ({per_dev}) must be >= number of languages ({len(self._retain_langs)}). "
                "Increase KL_RATIO_M or per_device_train_batch_size."
            )

        per_lang_sizes = self._balanced_batch_sizes(per_dev, len(self._retain_langs))
        self._retain_lang_samplers: Dict[str, DistributedSampler] = {}
        self._retain_lang_loaders: Dict[str, DataLoader] = {}
        for lang, batch_size in zip(self._retain_langs, per_lang_sizes):
            if batch_size <= 0:
                raise ValueError("Retain batch size allocation per language must be at least 1 sample.")
            subset = self.retain_dataset.select(lang_to_indices[lang])
            sampler = DistributedSampler(
                subset,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
                seed=self.args.seed,
            )
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                collate_fn=_identity_collate,
                pin_memory=True,
            )
            self._retain_lang_samplers[lang] = sampler
            self._retain_lang_loaders[lang] = loader

        self._retain_lang_order = list(self._retain_lang_loaders.keys())
        if not self._retain_lang_order:
            raise ValueError("Failed to build retain dataloaders for any language.")

    @staticmethod
    def _balanced_batch_sizes(total: int, parts: int) -> List[int]:
        if parts <= 0:
            return []
        base = total // parts
        remainder = total - base * parts
        sizes = [base] * parts
        for i in range(remainder):
            sizes[i] += 1
        return sizes

    def _make_retain_iter(self):
        lang_iters = {lang: iter(loader) for lang, loader in self._retain_lang_loaders.items()}
        lang_order = list(self._retain_lang_order)
        collate_fn = self._retain_collate_fn

        while True:
            merged_examples = []
            for lang in lang_order:
                loader = self._retain_lang_loaders[lang]
                iterator = lang_iters[lang]
                try:
                    lang_examples = next(iterator)
                except StopIteration:
                    lang_iters[lang] = iter(loader)
                    lang_examples = next(lang_iters[lang])
                merged_examples.extend(lang_examples)

            if len(lang_order) > 1:
                random.shuffle(merged_examples)
            yield collate_fn(merged_examples)

    def _set_retain_epoch(self, epoch: int) -> None:
        for sampler in self._retain_lang_samplers.values():
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        self._retain_iter = self._make_retain_iter()

    def _rebuild_dpo_dataset(self, epoch: int) -> Dataset:
        """Rebuild the DPO preference dataset with fresh IDK shuffling for this epoch."""
        rows = []
        
        rng = random.Random(SEED + epoch)
        
        idk_pools: Dict[str, List[str]] = {}
        idk_offsets: Dict[str, int] = {}
        for tag, pool in self.idk_by_lang.items():
            shuffled = pool[:]
            rng.shuffle(shuffled)
            idk_pools[tag] = shuffled
            idk_offsets[tag] = 0

        for q, neg, tag in self.forget_items:
            prompt = self._prompt_builder(q)
            pool = idk_pools[tag]
            idx = idk_offsets[tag]
            pos = pool[idx]
            idx += 1
            if idx >= len(pool):
                rng.shuffle(pool)
                idx = 0
            idk_offsets[tag] = idx
            rows.append({"prompt": prompt, "chosen": pos, "rejected": neg})

        ds = Dataset.from_list(rows)
        ds = ds.shuffle(seed=SEED + epoch)
        return ds

    def _refresh_dpo_dataset_for_epoch(self, epoch: int) -> None:
        new_ds = self._rebuild_dpo_dataset(epoch)
        self.train_dataset = new_ds
        if hasattr(self, '_train_dataloader'):
            del self._train_dataloader

    def forward_kl_on_batch(self, model, teacher, batch, answer_only: bool = True) -> torch.Tensor:
        dev = self.accelerator.device

        ids  = batch["input_ids"].to(dev)
        attn = batch["attention_mask"].to(dev)
        pl   = batch["prompt_len"].to(dev)

        def _compute_chunk(chunk_ids, chunk_attn, chunk_pl):
            with torch.no_grad():
                t_logits = teacher(input_ids=chunk_ids, attention_mask=chunk_attn).logits
            s_logits = model(input_ids=chunk_ids, attention_mask=chunk_attn).logits

            t_logp = F.log_softmax(t_logits[:, :-1, :].float(), dim=-1)
            s_logp = F.log_softmax(s_logits[:, :-1, :].float(), dim=-1)

            kl_tok = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
            valid = chunk_attn.sum(dim=1) - 1
            if answer_only:
                start_idx = (chunk_pl - 1).clamp_min(0)
            else:
                start_idx = torch.zeros_like(valid)

            mask = torch.zeros_like(kl_tok, dtype=kl_tok.dtype)
            B, _ = kl_tok.shape
            for b in range(B):
                s = int(start_idx[b].item()); e = int(valid[b].item())
                if e > s:
                    mask[b, s:e] = 1.0

            kl_sum_per_ex  = (kl_tok * mask).sum(dim=1)
            len_per_ex     = mask.sum(dim=1).clamp_min(1.0)
            kl_mean_per_ex = kl_sum_per_ex / len_per_ex
            return kl_mean_per_ex.sum(), chunk_ids.size(0)

        if not ENABLE_KL_MINIBATCH or KL_MINIBATCH_SPLITS <= 1:
            kl_sum, count = _compute_chunk(ids, attn, pl)
            return kl_sum / count

        chunk_size = max(1, math.ceil(ids.size(0) / KL_MINIBATCH_SPLITS))
        total_sum = torch.zeros(1, device=dev)
        total_count = 0
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            chunk_ids = ids[start:end]
            chunk_attn = attn[start:end]
            chunk_pl = pl[start:end]
            kl_sum, count = _compute_chunk(chunk_ids, chunk_attn, chunk_pl)
            total_sum += kl_sum
            total_count += count
        total_count = max(1, total_count)
        return total_sum / total_count

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        base = super().compute_loss(model, inputs,
                                    return_outputs=return_outputs,
                                    num_items_in_batch=num_items_in_batch)
        if return_outputs:
            dpo_loss, dpo_outputs = base
        else:
            dpo_loss = base

        retain_batch = next(self._retain_iter) 
        kl_loss = self.forward_kl_on_batch(model, self.ref_model, retain_batch, answer_only=self.kl_on_answer_only)
        total = dpo_loss + self.kl_weight * kl_loss
        self._last_dpo_loss = float(dpo_loss.detach().mean().item())
        self._last_kl_loss = float(kl_loss.detach().mean().item())
        self._last_total_loss = float(total.detach().mean().item())

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({"DPO": self._last_dpo_loss,
                    "KL": self._last_kl_loss,
                    "Total": self._last_total_loss})
        return (total, dpo_outputs) if return_outputs else total

    def training_step(self, *args, **kwargs):
        return super().training_step(*args, **kwargs)

    def _inner_training_loop(self, *args, **kwargs):
        if getattr(self, "_retain_lang_samplers", None):
            epoch = int(self.state.epoch) if getattr(self.state, "epoch", None) is not None else 0
            self._set_retain_epoch(epoch)
        return super()._inner_training_loop(*args, **kwargs)

class RetainShuffleCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        
        if hasattr(self.trainer, '_refresh_dpo_dataset_for_epoch'):
            if self.trainer.is_world_process_zero:
                self.trainer.accelerator.print(f"[Epoch {epoch}] Rebuilding DPO dataset with fresh IDK shuffling...")
            self.trainer._refresh_dpo_dataset_for_epoch(epoch)
        
        if not hasattr(self.trainer, "_retain_lang_samplers") or not self.trainer._retain_lang_samplers:
            return control
        self.trainer._set_retain_epoch(epoch)
        return control

class LossReportCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self._last_logged_step = -1

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.trainer is None or logs is None:
            return control
        
        dpo = getattr(self.trainer, "_last_dpo_loss", None)
        kl = getattr(self.trainer, "_last_kl_loss", None)
        tot = getattr(self.trainer, "_last_total_loss", None)
        
        if dpo is not None and kl is not None and tot is not None:
            if hasattr(self.trainer, 'state') and hasattr(self.trainer.state, 'log_history'):
                custom_logs = {
                    'DPO': f'{dpo:.4f}',
                    'KL': f'{kl:.4f}',
                    'Total': f'{tot:.4f}'
                }
                logs.update(custom_logs)
        
        return control
    

_ZERO_STAGE = 3 if (_MODEL_SIZE_VALUE is not None and _MODEL_SIZE_VALUE > 1.5) else 2

def main():
    set_seed()
    llm_family = _infer_llm_family(MODEL_ID, _LLM_FAMILY_ARG)
    os.environ["LLM_FAMILY"] = llm_family
    _llm_cfg = load_llm_config(llm_family)

    if not os.path.isdir(MODEL_ID):
        raise FileNotFoundError(f"Model not found: {MODEL_ID}")
    _assert_data_files_exist()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_builder = make_prompt_builder(tokenizer, llm_family)

    forget_items, idk_by_lang, _ = load_data_strict()
    
    pref_ds, retain_ds = build_training_datasets(tokenizer, prompt_builder)

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    config.use_cache = False
    dtype = torch.bfloat16
    attn_impl = "flash_attention_2"
    if llm_family == "gemma":
        attn_impl = "eager"
        print("⚠️ Gemma family detected: using 'eager' attention implementation (recommended).")
    common_kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config, **common_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    ref_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    ref_config.use_cache = False
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=ref_config, **common_kwargs)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    ds_config = make_ds_config(_ZERO_STAGE)

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.25,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=MAXIMUM_LEN,
        max_completion_length=MAXIMUM_LEN,
        beta=0.1,
        bf16=True,
        fp16=False,
        loss_type="sigmoid",
        gradient_checkpointing=False,
        remove_unused_columns=False,
        logging_steps=1,
        deepspeed=ds_config,
        report_to=[],
        save_strategy="no",
        save_steps=0,
    )

    trainer = DPOWithRetainKL(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        tokenizer=tokenizer,
        train_dataset=pref_ds,
        retain_dataset=retain_ds,
        forget_items=forget_items,
        idk_by_lang=idk_by_lang,
        prompt_builder=prompt_builder,
    )
    trainer.add_callback(RetainShuffleCallback(trainer))
    trainer.add_callback(LossReportCallback(trainer))

    if os.environ.get("RANK", "0") == "0":
        per_dev_dpo = dpo_args.per_device_train_batch_size
        per_dev_kl  = max(1, int(per_dev_dpo * KL_RATIO_M))
        print(f"[Sanity] DPO per-device BS={per_dev_dpo}  KL per-device BS={per_dev_kl}")
        print(f"[Sanity] Pref size={len(pref_ds)}  Retain size={len(retain_ds)}")

    trainer.train()

    if trainer.is_world_process_zero:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        trainer.accelerator.print(f"Saved to: {OUTPUT_DIR}")
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isdir(path) and name.startswith("global_step"):
                trainer.accelerator.print(f"[Cleanup] Removing checkpoint folder: {path}")
                shutil.rmtree(path, ignore_errors=True)
if __name__ == "__main__":
    main()
