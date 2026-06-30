import argparse
from typing import List, Tuple, Dict, Callable
from datasets import Dataset
import json
import torch
import random
import numpy as np
import os
import torch.distributed as dist
import shutil
from sentence_transformers import CrossEncoder
import benchmarkTest
from benchmarkTest import compute_nil_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from llm_config import load_llm_config

# ----------------- config -----------------
SEED = 42
MAX_LEN = 512
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LANG_MAP = {
    "english": "En",
    "chinese": "Ch", 
    "german": "De",
    "russian": "Ru",
    "turkish": "Tu"
}

# ----------------- utils -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_abbr(langs: List[str]) -> List[str]:
    abbr = []
    for l in langs:
        if l not in LANG_MAP:
            raise ValueError(f"Unsupported language: '{l}'. Supported: {list(LANG_MAP.keys())}")
        abbr.append(LANG_MAP[l])
    return abbr

def load_pairs(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        d: Dict[str, str] = json.load(f)
    # expects a dict {question: answer}
    return [(q.strip(), a.strip()) for q, a in d.items()]

def make_prompt_builder(tokenizer, _llm_family: str) -> Callable[[str], str]:
    """
    Returns a function that renders a user prompt using the tokenizer's chat template.
    """
    def _render(question: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return _render


def build_training_dataset(tokenizer, paths: List[str], prompt_builder: Callable[[str], str], seed: int = 42) -> Dataset:
    def to_rows(pairs: List[Tuple[str, str]]) -> List[Dict[str, str]]:
        rows = []
        for q, a in pairs:
            prompt = prompt_builder(q)
            comp = a
            if tokenizer.eos_token and not comp.endswith(tokenizer.eos_token):
                comp = comp + tokenizer.eos_token
            rows.append({"prompt": prompt, "completion": comp})
        return rows

    all_rows: List[Dict[str, str]] = []
    for path in paths:
        pairs = load_pairs(path)
        all_rows.extend(to_rows(pairs))

    ds = Dataset.from_list(all_rows).shuffle(seed=seed)
    return ds

class PromptCompletionCollator:
    """
    Labels only completion tokens; prompt and PAD get -100
    """
    def __init__(self, tokenizer, max_length=2048):
        self.tok = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, str]]):
        prompts     = [f["prompt"]     for f in features]
        completions = [f["completion"] for f in features]

        # Get prompt lengths (no padding/truncation)
        prompt_tok = self.tok(prompts, add_special_tokens=False, padding=False, truncation=False, return_attention_mask=False)
        prompt_lens = [len(ids) for ids in prompt_tok["input_ids"]]

        # Tokenize concatenated sequences
        enc = self.tok(
            [p + c for p, c in zip(prompts, completions)],
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        # Build labels: ignore prompt and PAD
        labels = input_ids.clone()
        for i, pl in enumerate(prompt_lens):
            cut = min(pl, labels.size(1))    # if truncation clipped completion
            labels[i, :cut] = -100
        if getattr(self.tok, "pad_token_id", None) is not None:
            labels[labels == self.tok.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class EarlyStopNLICallback(TrainerCallback):
    def __init__(self, tokenizer, prompt_builder: Callable[[str], str], train_pairs: List[Tuple[str, str]], threshold=0.95, sample_ratio=0.1):
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.train_pairs = train_pairs
        self.threshold = threshold
        self.sample_ratio = sample_ratio


    def on_epoch_end(self, args, state, control, model, **kwargs):
        if state.epoch < 5:
            return control
        is_distributed = dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0
        is_rank0 = rank == 0
        device = next(model.parameters()).device

        stop_tensor = torch.zeros(1, device=device, dtype=torch.int32)

        sampled = None
        if is_rank0:
            print(f"\n🔍 Checking NLI score at epoch {state.epoch}...")
            sample_size = max(1, int(len(self.train_pairs) * self.sample_ratio))
            sample_size = min(sample_size, len(self.train_pairs))
            sampled = random.sample(self.train_pairs, sample_size)

        if is_distributed:
            payload = [sampled]
            dist.broadcast_object_list(payload, src=0)
            sampled = payload[0]

        if not sampled:
            if is_rank0:
                print("⚠️ No pairs available for early stopping check.")
            return control

        references = [ans for _, ans in sampled]

        shard_indices = list(range(rank, len(sampled), world_size))
        shard_questions = [sampled[i][0] for i in shard_indices]
        shard_predictions: List[Tuple[int, str]] = []

        model_was_training = model.training
        
        # Use context manager for eval mode
        with torch.inference_mode():
            model.eval()
            
            original_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"

            original_use_cache = getattr(model.config, "use_cache", None)
            if original_use_cache is not None:
                model.config.use_cache = True

            batch_size = 32
            for start in range(0, len(shard_questions), batch_size):
                batch_indices = shard_indices[start:start + batch_size]
                batch_q = shard_questions[start:start + batch_size]

                batch_prompts = [self.prompt_builder(q) for q in batch_q]

                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )

                prompt_len = inputs["input_ids"].shape[-1]
                continuations = outputs[:, prompt_len:]
                decoded = self.tokenizer.batch_decode(
                    continuations,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                for idx, text in zip(batch_indices, decoded):
                    shard_predictions.append((idx, text.strip()))
                
                # Clear CUDA cache after each batch
                torch.cuda.empty_cache()

            self.tokenizer.padding_side = original_padding_side
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache

        # Restore training state outside context manager
        if model_was_training:
            model.train()

        if is_distributed:
            gathered: List[List[Tuple[int, str]]] = [None] * world_size  # type: ignore
            dist.all_gather_object(gathered, shard_predictions)
            if is_rank0:
                flat = [item for bucket in gathered for item in bucket]
                flat.sort(key=lambda pair: pair[0])
                predictions = [pred for _, pred in flat]
            else:
                predictions = None
        else:
            shard_predictions.sort(key=lambda pair: pair[0])
            predictions = [pred for _, pred in shard_predictions]

        if is_rank0 and predictions is not None:
            # Validate prediction count
            if len(predictions) != len(references):
                print(f"⚠️ Prediction/reference mismatch: {len(predictions)} vs {len(references)}")
                return control
                
            nli_score, _ = compute_nil_score(predictions, references)
            print(f"📊 Average NLI Score: {nli_score:.4f} (threshold: {self.threshold})")
            if nli_score >= self.threshold:
                print(f"✅ Early stopping triggered! NLI score {nli_score:.4f} >= {self.threshold}")
                stop_tensor.fill_(1)

        if is_distributed:
            dist.broadcast(stop_tensor, src=0)

        if stop_tensor.item():
            control.should_training_stop = True

        return control


def make_ds_config(zero_stage: int = 3) -> Dict:
    if zero_stage not in (2, 3):
        zero_stage = 3
    cfg = {
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_accumulation_steps": "auto",
    }
    if zero_stage == 3:
        cfg["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
    cfg["bf16"] = {"enabled": True}
    return cfg

# ----------------- main -----------------
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("ft_language", type=str, nargs="+", help="Languages to finetune on")
    # parser.add_argument("--shuffle", type=str, required=True, help="Shuffle name (e.g., 'Original', 'Shuffle1')")
    parser.add_argument("--data-paths", type=str, nargs="+", required=True, help="Paths to forget/retain JSON files")
    # parser.add_argument("--val-path", type=str, required=True, help="Path to validation JSON file (English_holdout10.json)")
    parser.add_argument(
        "--llm-family",
        type=str,
        choices=["qwen", "gemma"],
        default=None,
        help="Select which LLM config to use (defaults to env LLM_FAMILY or qwen).",
    )

    args, _unknown = parser.parse_known_args()
    
    FT_LANGUAGES = [lang.strip().lower() for lang in args.ft_language]
    FT_LANG_ABBR = get_abbr(FT_LANGUAGES)
    # SHUFFLE = args.shuffle
    training_data_path = args.data_paths
    # validation_path = args.val_path
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]
    llm_cfg = load_llm_config(args.llm_family)

    if is_main_process:
        print(f"\n{'='*80}")
        print(f"🔥 FINETUNING CONFIGURATION")
        print(f"{'='*80}")
        print(f"📚 Languages: {', '.join(FT_LANGUAGES)}")
        # print(f"🔀 Shuffle: {SHUFFLE}")
        print(f"📁 Training data files ({len(training_data_path)}):")
        for i, path in enumerate(training_data_path, 1):
            print(f"   {i}. {path}")
        print(f"🤖 Base model: {llm_cfg.base_model_name} ({llm_cfg.family})")
        print(f"{'='*80}\n")

    OUTPUT_DIR = os.path.join(llm_cfg.models_local_root, f"{llm_cfg.finetuned_prefix}{''.join(FT_LANG_ABBR)}")
    set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(llm_cfg.base_model_local_path, use_fast=False)
    prompt_builder = make_prompt_builder(tok, llm_cfg.family)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    hf_cfg = AutoConfig.from_pretrained(llm_cfg.base_model_local_path, trust_remote_code=True)
    hf_cfg.use_cache = False

    dtype = torch.bfloat16
    attn_impl = "flash_attention_2"
    if llm_cfg.family == "gemma":
        attn_impl = "eager"
        if is_main_process:
            print("⚠️ Gemma family detected: using 'eager' attention implementation (recommended).")
    model_kwargs = dict(
        config=hf_cfg,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(
        llm_cfg.base_model_local_path,
        **model_kwargs,
    )

    # Optional: helps VRAM
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    training_data = build_training_dataset(tok, training_data_path, prompt_builder, seed=SEED)

    all_train_pairs = []
    for path in training_data_path:
        all_train_pairs.extend(load_pairs(path))

    # Initialize global NLI model for benchmarkTest functions
    NLI_PATH = "./Models/xlm-roberta-large-xnli"
    if is_main_process:
        print(f"🚀 Loading NLI model for early stopping: {NLI_PATH}")
    
    if torch.cuda.device_count() > 1:
        if is_main_process:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs for NLI model")
        benchmarkTest.nli_model = CrossEncoder(NLI_PATH, device="cuda", max_length=512)
        original_config = benchmarkTest.nli_model.model.config
        original_device = benchmarkTest.nli_model.model.device
        benchmarkTest.nli_model.model = torch.nn.DataParallel(benchmarkTest.nli_model.model)
        benchmarkTest.nli_model.model.device = original_device
        benchmarkTest.nli_model.model.config = original_config
        benchmarkTest.batch_size = 512 * torch.cuda.device_count()
        if is_main_process:
            print(f"📊 NLI batch size scaled to: {benchmarkTest.batch_size}")
    else:
        if is_main_process:
            print(f"🚀 Using single GPU for NLI model")
        benchmarkTest.nli_model = CrossEncoder(NLI_PATH, device="cuda", max_length=512)
        benchmarkTest.batch_size = 512
        if is_main_process:
            print(f"📊 NLI batch size: {benchmarkTest.batch_size}")


    collator = PromptCompletionCollator(tok, max_length=MAX_LEN)
    zero_stage = int(os.environ.get("ZERO_STAGE", "3"))
    ds_cfg = make_ds_config(zero_stage)

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_ratio=0.25,
        dataloader_num_workers=8,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=["tensorboard"],
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        deepspeed=ds_cfg,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tok,
        args=train_args,
        train_dataset=training_data,
        data_collator=collator,
        callbacks=[EarlyStopNLICallback(tok, prompt_builder, all_train_pairs, threshold=0.89, sample_ratio=0.1)]
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    
    if is_main_process:
        # Clean up any checkpoint directories created by DeepSpeed
        checkpoint_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            print(f"\n🧹 Cleaning up {len(checkpoint_dirs)} checkpoint(s)...")
            for ckpt_dir in checkpoint_dirs:
                ckpt_path = os.path.join(OUTPUT_DIR, ckpt_dir)
                try:
                    shutil.rmtree(ckpt_path)
                    print(f"   ✓ Removed: {ckpt_dir}")
                except Exception as e:
                    print(f"   ✗ Failed to remove {ckpt_dir}: {e}")
        # Remove ZeRO stage3 shard folders if present
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isdir(path) and name.startswith("global_step"):
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    print(f"   ✓ Removed ZeRO shard folder: {name}")
                except Exception as e:
                    print(f"   ✗ Failed to remove {name}: {e}")
        
        print(f"\n✅ Model saved to: {OUTPUT_DIR}\n")

if __name__ == "__main__":
    main()
