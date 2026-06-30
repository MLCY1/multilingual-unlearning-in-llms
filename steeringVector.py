import argparse
import json
import os
import subprocess
import benchmarkTest
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmarkTest import compute_nil_score
from transfer import LANG_MAP
from utils.utils import run_benchmark
from llm_config import load_llm_config
from typing import Optional, Dict, Sequence, Union


LANGUAGE_ORDER = ["English", "Chinese", "German", "Russian", "Turkish"]
MODEL_SIZE_ENV_KEYS = {
    "qwen": "QWEN_MODEL_SIZE",
    "gemma": "GEMMA_MODEL_SIZE",
}


def _normalise_family(llm_family: Optional[str]) -> str:
    return (llm_family or os.environ.get("LLM_FAMILY", "qwen")).lower()


def _apply_llm_env(env: Dict[str, str], llm_family: Optional[str], llm_model_size: Optional[str]) -> str:
    family = _normalise_family(llm_family)
    env["LLM_FAMILY"] = family
    size_env_key = MODEL_SIZE_ENV_KEYS.get(family)
    if llm_model_size and size_env_key:
        env[size_env_key] = llm_model_size
    return family


def _resolve_language(language: str, available: Dict[str, Dict]) -> str:
    language_map = {name.lower(): name for name in available}
    key = language.lower()
    if key not in language_map:
        raise ValueError(f"Unsupported language '{language}'. Options: {list(available)}")
    return language_map[key]


def _filter_language_paths(paths: Dict[str, Dict], languages: Optional[Sequence[str]]) -> Dict[str, Dict]:
    if not languages:
        return paths
    selected = {}
    for language in languages:
        resolved = _resolve_language(language, paths)
        selected[resolved] = paths[resolved]
    return selected


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        device_map = model.hf_device_map
        embed_keys = [k for k in device_map.keys() if "embed" in k or "wte" in k]
        key = embed_keys[0] if embed_keys else next(iter(device_map.keys()))
        target = device_map[key]
        if isinstance(target, int):
            return torch.device(f"cuda:{target}")
        return torch.device(target)
    dev = getattr(model, "device", None)
    if dev is not None:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_transformer_layers(model: torch.nn.Module):
    candidates = [
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "backbone", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "layers"):
            return candidate.layers
        if hasattr(candidate, "h"):
            return candidate.h
    raise ValueError("Could not locate transformer layers on the provided model.")

def _load_llm_cfg(llm_family: Optional[str], llm_model_size: Optional[str]):
    env_key = None
    prev_val = None
    family = _normalise_family(llm_family)
    if llm_model_size and family in MODEL_SIZE_ENV_KEYS:
        env_key = MODEL_SIZE_ENV_KEYS[family]
        prev_val = os.environ.get(env_key)
        os.environ[env_key] = llm_model_size
    try:
        return load_llm_config(family)
    finally:
        if env_key:
            if prev_val is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = prev_val

def train_new_unlearning_config(llm_family: Optional[str] = None, llm_model_size: Optional[str] = None):
    llm_cfg = _load_llm_cfg(llm_family, llm_model_size)
    prefix = llm_cfg.finetuned_prefix
    root = llm_cfg.models_local_root
    configs = {
        "Chinese": {
            "ft_model_path": os.path.join(root, f"{prefix}Ch"),
            "forget_data_path": "./Data/Chinese/Shuffle1/forget01.json",
            "retain_data_path": "./Data/Chinese/Shuffle1/retain99.json",
            "idk_data_path": "./Data/Chinese/Original/idk.txt",
            "output_model_path": os.path.join(root, f"{prefix}Ch-UNCh-Shuffle1"),
        },
        "English": {
            "ft_model_path": os.path.join(root, f"{prefix}En"),
            "forget_data_path": "./Data/English/Shuffle1/forget01.json",
            "retain_data_path": "./Data/English/Shuffle1/retain99.json",
            "idk_data_path": "./Data/English/Original/idk.txt",
            "output_model_path": os.path.join(root, f"{prefix}En-UNEn-Shuffle1"),
        },
        "German": {
            "ft_model_path": os.path.join(root, f"{prefix}De"),
            "forget_data_path": "./Data/German/Shuffle1/forget01.json",
            "retain_data_path": "./Data/German/Shuffle1/retain99.json",
            "idk_data_path": "./Data/German/Original/idk.txt",
            "output_model_path": os.path.join(root, f"{prefix}De-UNDe-Shuffle1"),
        },
        "Russian": {
            "ft_model_path": os.path.join(root, f"{prefix}Ru"),
            "forget_data_path": "./Data/Russian/Shuffle1/forget01.json",
            "retain_data_path": "./Data/Russian/Shuffle1/retain99.json",
            "idk_data_path": "./Data/Russian/Original/idk.txt",
            "output_model_path": os.path.join(root, f"{prefix}Ru-UNRu-Shuffle1"),
        },
        "Turkish": {
            "ft_model_path": os.path.join(root, f"{prefix}Tu"),
            "forget_data_path": "./Data/Turkish/Shuffle1/forget01.json",
            "retain_data_path": "./Data/Turkish/Shuffle1/retain99.json",
            "idk_data_path": "./Data/Turkish/Original/idk.txt",
            "output_model_path": os.path.join(root, f"{prefix}Tu-UNTu-Shuffle1"),
        },
    }
    return configs

def run_unlearning(
    config: Optional[Dict[str, Dict]] = None,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    num_gpus: int = 4,
):
    config = config or train_new_unlearning_config(llm_family, llm_model_size)
    for language, paths in config.items():
        cmd = [
            "deepspeed",
            f"--num_gpus={num_gpus}",
            "dpoUnlearning.py",
            "--model_path", paths['ft_model_path'],
            "--output_path", paths['output_model_path'],
            "--forget_paths", paths['forget_data_path'],
            "--retain_paths", paths['retain_data_path'],
            "--idk_paths", paths['idk_data_path'],
        ]
        if llm_family:
            cmd.extend(["--llm-family", llm_family])
        env = os.environ.copy()
        _apply_llm_env(env, llm_family, llm_model_size)
        result = subprocess.run(cmd, check=False, env=env)
        
        if result.returncode != 0:
            print("\n" + "❌" + "="*78 + "❌")
            print("║" + " "*78 + "║")
            print("║" + "⚠️  UNLEARNING FAILED!".center(77) + "║")
            print("║" + " "*78 + "║")
            print("❌" + "="*78 + "❌" + "\n")
            return None
        
        print("\n" + "✅" + "="*78 + "✅")
        print("║" + " "*78 + "║")
        print("║" + "🎉 UNLEARNING COMPLETED SUCCESSFULLY!".center(77) + "║")
        print("║" + " "*78 + "║")
        print("✅" + "="*78 + "✅")
        print(f"\n💾 Model Saved To: {paths['output_model_path']}\n")
    
    return paths['output_model_path']
    
    
def run_evaluation(llm_family: Optional[str] = None, llm_model_size: Optional[str] = None):
    config = train_new_unlearning_config(llm_family, llm_model_size)
    for language, paths in config.items():
        model_path = paths['output_model_path']
        data_path = {language:{"forget": paths['forget_data_path'], 
                            "retain": paths['retain_data_path']}}
        languages = [language]
        unlearn_lang_list = [language]
        shuffle_num = 1
        unlearn_key = f"{language}-UN{''.join([LANG_MAP.get(l.lower(), l[:2]) for l in unlearn_lang_list])}-{shuffle_num}"
        env_family = llm_family or os.environ.get("LLM_FAMILY")
        _apply_llm_env(os.environ, env_family, llm_model_size)
        run_benchmark(model_path, data_path, languages, LANG_MAP, shuffle_num, unlearning_flag = unlearn_key, llm_family=llm_family)
        
        
def get_steering_vector(
    ft_model_path: str = "./Models/Qwen2.5-7B-finetuned-FTEn",
    unlearn_model_path: str = "./Models/Qwen2.5-7B-finetuned-FTEn-UNEn-Shuffle1",
    data_path: str = "./Data/English/Shuffle1/forget01.json",
    batch_size: int = 40,
    max_length: int = 512,
    save_path: str = "./Data/SteeringVector/SV",
):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = list(data.keys())
    tokenizer = AutoTokenizer.from_pretrained(ft_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def _render_chat(s: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": s}],
            tokenize=False,
            add_generation_prompt=True,
        )

    prompts = [_render_chat(s) for s in samples]

    ft_model = AutoModelForCausalLM.from_pretrained(
        ft_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    unlearn_model = AutoModelForCausalLM.from_pretrained(
        unlearn_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    ft_device = _infer_model_device(ft_model)
    un_device = _infer_model_device(unlearn_model)
    steering_sum = None 
    total_count = 0 

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        attn = encoded["attention_mask"]
        last_idx = attn.sum(dim=1) - 1

        ft_inputs = {k: v.to(ft_device) for k, v in encoded.items()}
        un_inputs = {k: v.to(un_device) for k, v in encoded.items()}

        with torch.no_grad():
            ft_out = ft_model(**ft_inputs, output_hidden_states=True, return_dict=True)
            un_out = unlearn_model(**un_inputs, output_hidden_states=True, return_dict=True)
            
        def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
            norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
            return x / norm

        ft_hs = [h.to(torch.float32) for h in ft_out.hidden_states]
        un_hs = [h.to(torch.float32) for h in un_out.hidden_states]

        batch_sums = []
        for u, f in zip(un_hs, ft_hs):
            idx = last_idx.to(u.device)
            idx_exp = idx.view(-1, 1, 1).expand(-1, 1, u.size(-1))
            u_last = torch.gather(u, 1, idx_exp).squeeze(1)
            f_last = torch.gather(f, 1, idx_exp).squeeze(1)
            u_last = _l2_normalize_rows(u_last)
            f_last = _l2_normalize_rows(f_last)
            diff_sum = (u_last - f_last).sum(dim=0).cpu()
            batch_sums.append(diff_sum)

        if steering_sum is None:
            steering_sum = batch_sums
        else:
            steering_sum = [acc + diff for acc, diff in zip(steering_sum, batch_sums)]

        total_count += attn.size(0)

    if steering_sum is None or total_count == 0:
        raise ValueError("No activations accumulated for steering vector.")

    steering_vector = []
    for acc in steering_sum:
        g = acc / total_count
        g = g / (g.norm(p=2) + 1e-9)
        steering_vector.append(g)
    steering_vector = steering_vector[1:]

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save({"steering_vector": [v.cpu() for v in steering_vector]}, save_path)

    return steering_vector

def load_NIL():
    NLI_PATH = "./Models/xlm-roberta-large-xnli"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in [-1, 0]
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
            
def load_data(data_path: str):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def get_paths(
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
):
    llm_cfg = _load_llm_cfg(llm_family, llm_model_size)
    prefix = llm_cfg.finetuned_prefix
    root = llm_cfg.models_local_root
    paths = {
        "English": {
            "forget": "./Data/English/Original/forget01.json",
            "model": os.path.join(root, f"{prefix}En-UNEn-Original"),
        },
        "Chinese": {
            "forget": "./Data/Chinese/Original/forget01.json",
            "model": os.path.join(root, f"{prefix}Ch-UNCh-Original"),
        },
        "German": {
            "forget": "./Data/German/Original/forget01.json",
            "model": os.path.join(root, f"{prefix}De-UNDe-Original"),
        },
        "Russian": {
            "forget": "./Data/Russian/Original/forget01.json",
            "model": os.path.join(root, f"{prefix}Ru-UNRu-Original"),
        },
        "Turkish": {
            "forget": "./Data/Turkish/Original/forget01.json",
            "model": os.path.join(root, f"{prefix}Tu-UNTu-Original"),
        }
    }
    return _filter_language_paths(paths, languages)


def injecting_steering_vector(
    temperature: float = 0.0,
    steering_path: str = "./Data/SteeringVector/SV",
    alpha: float = 1.0,
    alpha_list: list = None,
    max_new_tokens: int = 128,
    save_path: str = "./Data/SteeringVector/alpha_sweep.json",
    random: bool = False,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
):
    load_NIL()

    paths = get_paths(llm_family, llm_model_size, languages=languages)
    sv_obj = torch.load(steering_path, map_location="cpu")
    base_steering = [v.to(torch.float32) for v in sv_obj["steering_vector"]]
    if random:
        steering_vector = []
        for v in base_steering:
            noise = torch.randn_like(v)
            norm = noise.norm(p=2) + 1e-9
            steering_vector.append(noise / norm)
    else:
        steering_vector = base_steering

    alpha_values = alpha_list if alpha_list is not None else [alpha]
    sweep_results = {}

    for alpha_val in tqdm(alpha_values, desc="Alpha sweep", total=len(alpha_values)):
        all_results = {}
        for lang, p in tqdm(paths.items(), desc="Models", total=len(paths)):
            unlearn_model_path = p["model"]
            forget_data_path = p["forget"]
            unlearn_model = AutoModelForCausalLM.from_pretrained(
                unlearn_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(unlearn_model_path, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            layers = _get_transformer_layers(unlearn_model)
            if len(steering_vector) != len(layers):
                raise ValueError(
                    f"Steering vector count ({len(steering_vector)}) does not match transformer layers ({len(layers)})."
                )

            forget_data = load_data(forget_data_path)
            questions = list(forget_data.keys())
            references = list(forget_data.values())
            batch_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in questions
            ]
            encoded_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            model_inputs = {k: v.to(_infer_model_device(unlearn_model)) for k, v in encoded_inputs.items()}
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": temperature > 0,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            def _run_generation():
                with torch.no_grad():
                    outputs = unlearn_model.generate(**model_inputs, **gen_kwargs)
                prompt_len = model_inputs["input_ids"].shape[1]
                return tokenizer.batch_decode(
                    outputs[:, prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            def _make_hook(vec: torch.Tensor, layer_num: int):
                vec = vec.detach()

                def _apply(hidden_states: torch.Tensor) -> torch.Tensor:
                    steer_vec = vec.to(device=hidden_states.device, dtype=hidden_states.dtype)

                    last = hidden_states[:, -1, :]

                    norms = last.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-9)

                    step = alpha_val * norms
                    delta = step * steer_vec

                    hidden_states = hidden_states.clone()
                    
                    hidden_states[:, -1, :] -= delta

                    return hidden_states

                def _hook(_module, _inputs, output):
                    if isinstance(output, torch.Tensor):
                        return _apply(output)
                    if isinstance(output, (tuple, list)) and output:
                        steered = _apply(output[0])
                        rest = output[1:]
                        if isinstance(output, tuple):
                            return (steered, *rest)
                        return [steered, *rest]
                    return output

                return _hook

            active_family = (llm_family or os.environ.get("LLM_FAMILY", "")).lower()
            if "qwen" in active_family:
                offsets = [0, 1, 2]
            else:
                offsets = [0, 1, 2, 3, 4, 5]
            max_offset = max(offsets)
            total_layers = len(steering_vector)
            if total_layers <= max_offset:
                raise ValueError(
                    f"Not enough layers ({total_layers}) to apply steering with offsets {offsets}."
                )
            loop_limit = total_layers - max_offset

            layer_nil_scores = {}
            for layer_idx in tqdm(range(loop_limit), desc=f"Steering layers ({lang}, alpha={alpha_val})", total=loop_limit):
                handles = []
                steered_layers = []
                for offset in offsets:
                    target_idx = layer_idx + offset
                    vec = steering_vector[target_idx]
                    steered_layers.append(target_idx + 1)
                    handles.append(
                        layers[target_idx].register_forward_hook(_make_hook(vec, target_idx + 1))
                    )
                try:
                    generations = _run_generation()
                finally:
                    for h in handles:
                        h.remove()
                nli_score, _ = compute_nil_score(generations, references)
                layer_nil_scores[layer_idx + 1] = nli_score
                print(
                    f"[{lang} | alpha={alpha_val}] Layers {steered_layers}/{total_layers} NIL score: {nli_score:.4f}"
                )

            all_results[lang] = layer_nil_scores

            try:
                del (
                    unlearn_model,
                    tokenizer,
                    model_inputs,
                    encoded_inputs,
                    batch_prompts,
                    generations,
                    layers,
                    forget_data,
                    questions,
                    references,
                )
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sweep_results[str(alpha_val)] = all_results

        for lang, scores in all_results.items():
            top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 NIL scores for {lang} at alpha={alpha_val}:")
            for layer, score in top5:
                print(f"  Layer {layer}: {score:.4f}")

    if save_path:
        def _compute_summary(data_dict: Dict[str, Dict]) -> Dict:
            alpha_avg = {}
            best_alpha = None
            best_avg = float("-inf")
            lang_best = {}
            for a_str, lang_scores in data_dict.items():
                per_lang_best = []
                for lang, scores in lang_scores.items():
                    if scores:
                        per_lang_best.append(max(scores.values()))
                        for layer, score in scores.items():
                            if (lang not in lang_best) or (score > lang_best[lang]["score"]):
                                lang_best[lang] = {"alpha": a_str, "layer": layer, "score": score}
                avg_score = sum(per_lang_best) / len(per_lang_best) if per_lang_best else 0.0
                alpha_avg[a_str] = avg_score
                if avg_score > best_avg:
                    best_avg = avg_score
                    best_alpha = a_str
            return {
                "alpha_avg_best_layer_per_lang": alpha_avg,
                "best_alpha": best_alpha,
                "best_alpha_avg": best_avg,
                "best_per_language": lang_best,
            }

        merged = {}
        if os.path.exists(save_path):
            try:
                with open(save_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                for k, v in existing.items():
                    if k != "_summary":
                        merged[k] = v
            except Exception as e:
                print(f"⚠️  Could not read existing results at {save_path}: {e}")

        for k, v in sweep_results.items():
            if k == "_summary":
                continue
            if k not in merged or not isinstance(merged.get(k), dict):
                merged[k] = v
                continue
            for lang, scores in v.items():
                merged[k][lang] = scores

        summary = _compute_summary(merged) if merged else {
            "alpha_avg_best_layer_per_lang": {},
            "best_alpha": None,
            "best_alpha_avg": float("-inf"),
            "best_per_language": {},
        }
        merged["_summary"] = summary

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"\nAlpha sweep results saved/merged to {save_path}")
        if summary["best_alpha"] is not None:
            print(f"Best alpha by avg best-layer NIL: {summary['best_alpha']} (avg={summary['best_alpha_avg']:.4f})")
            for lang, info in summary["best_per_language"].items():
                print(f"  {lang}: alpha={info['alpha']} layer={info['layer']} NIL={info['score']:.4f}")

def plot_alpha_heatmap(
    title: Union[str, Sequence[str]],
    alpha_file: Union[str, Sequence[str]] = "./Data/SteeringVector/alpha_sweep.json",
    save_path: str = "./Data/SteeringVector/alpha100_heatmap.pdf",
    vmax: Union[int, Sequence[Optional[int]], None] = None,
):
    def _as_list(val):
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]

    def _prepare_heatmap_data(data_dict: Dict):
        summary = data_dict.get("_summary", {})
        best_per_language = summary.get("best_per_language", {})
        if not best_per_language:
            raise ValueError("No per-language alpha selection found in summary.")

        desired_order = ["English", "Chinese", "German", "Russian", "Turkish"]
        languages = [lang for lang in desired_order if lang in best_per_language]
        for lang in best_per_language:
            if lang not in languages:
                languages.append(lang)

        max_layer = 0
        for lang in languages:
            lang_alpha = str(best_per_language[lang]["alpha"])
            lang_scores = data_dict.get(lang_alpha, {}).get(lang, {})
            if lang_scores:
                max_layer = max(max_layer, max(int(k) for k in lang_scores.keys()))
        if max_layer == 0:
            raise ValueError("No layer scores found for per-language best alphas.")

        mat = np.full((len(languages), max_layer), np.nan, dtype=float)
        for i, lang in enumerate(languages):
            lang_alpha = str(best_per_language[lang]["alpha"])
            lang_scores = data_dict.get(lang_alpha, {}).get(lang, {})
            for layer_str, score in lang_scores.items():
                layer_idx = int(layer_str) - 1
                if 0 <= layer_idx < max_layer:
                    mat[i, layer_idx] = score

        return languages, max_layer, mat

    alpha_files = _as_list(alpha_file)
    for path in alpha_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Alpha sweep file not found: {path}")

    titles_in = _as_list(title)
    if len(alpha_files) == 1:
        subplot_titles = [titles_in[0]]
        suptitle = None
    else:
        if len(titles_in) == len(alpha_files):
            subplot_titles = titles_in
            suptitle = None
        else:
            subplot_titles = [os.path.splitext(os.path.basename(p))[0] for p in alpha_files]
            suptitle = titles_in[0] if titles_in else None

    vmax_list = _as_list(vmax)
    if len(vmax_list) == 1:
        vmax_list = vmax_list * len(alpha_files)
    elif len(vmax_list) != len(alpha_files):
        raise ValueError("vmax must be a single value or match the number of alpha files.")

    datasets = []
    max_layers = []
    max_langs = 0
    for path in alpha_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        languages, max_layer, mat = _prepare_heatmap_data(data)
        datasets.append((languages, max_layer, mat))
        max_layers.append(max_layer)
        max_langs = max(max_langs, len(languages))

    max_layer_overall = max(max_layers) if max_layers else 0
    width_per_layer = 0.2
    per_panel_width = max(4.5, max_layer_overall * width_per_layer)
    fig_width = per_panel_width * len(alpha_files)
    fig_height = max(3.5, max_langs * 0.7)
    fig, axes = plt.subplots(
        1,
        len(alpha_files),
        figsize=(fig_width, fig_height),
        facecolor="white",
        constrained_layout=True,
    )
    if len(alpha_files) == 1:
        axes = [axes]

    ims = []
    for idx, (ax, (languages, max_layer, mat), sub_title, vmax_val) in enumerate(
        zip(axes, datasets, subplot_titles, vmax_list)
    ):
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap="Blues",
            vmin=None,
            vmax=vmax_val,
        )
        ims.append(im)
        ax.set_yticks(ticks=range(len(languages)), labels=languages, fontsize=8)
        ax.set_xticks(
            ticks=list(range(max_layer)),
            labels=[str(i + 1) for i in range(max_layer)],
            rotation=90,
        )
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Language", fontsize=11)
        ax.set_title(sub_title, fontsize=12)
        if idx > 0:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    cbar = fig.colorbar(ims[-1], ax=axes, pad=0.02)
    cbar.set_label("NLI score", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Heatmap saved to {save_path}")


def _parse_alpha_list(values: Sequence[str]) -> list:
    alphas = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                alphas.append(float(item))
    if not alphas:
        raise ValueError("At least one alpha value is required.")
    return alphas


def _default_artifact_paths(
    llm_family: Optional[str],
    llm_model_size: Optional[str],
    aux_language: str,
    random_baseline: bool,
) -> Dict[str, str]:
    llm_cfg = _load_llm_cfg(llm_family, llm_model_size)
    model_name = llm_cfg.base_model_name.replace(os.sep, "_")
    baseline_tag = "random" if random_baseline else "steering"
    file_prefix = f"{model_name}_{aux_language}_{baseline_tag}"
    return {
        "steering_path": f"./Data/SteeringVector/{model_name}_{aux_language}_SV.pt",
        "alpha_save_path": f"./Data/SteeringVector/{file_prefix}_alpha_sweep.json",
        "heatmap_save_path": f"./Data/SteeringVector/{file_prefix}_best_heatmap.pdf",
    }


def run_full_steering_process(
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    aux_language: str = "English",
    target_languages: Optional[Sequence[str]] = None,
    alpha_list: Optional[Sequence[float]] = None,
    steering_path: Optional[str] = None,
    alpha_save_path: Optional[str] = None,
    heatmap_save_path: Optional[str] = None,
    plot_title: Optional[str] = None,
    random_baseline: bool = False,
    num_gpus: int = 4,
    batch_size: int = 40,
    max_length: int = 512,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    vmax: Optional[float] = None,
    skip_unlearning: bool = False,
    skip_steering_vector: bool = False,
    skip_alpha_sweep: bool = False,
    skip_plot: bool = False,
) -> Dict[str, str]:
    family = _normalise_family(llm_family)
    _apply_llm_env(os.environ, family, llm_model_size)
    config = train_new_unlearning_config(family, llm_model_size)
    aux_language = _resolve_language(aux_language, config)
    defaults = _default_artifact_paths(family, llm_model_size, aux_language, random_baseline)
    steering_path = steering_path or defaults["steering_path"]
    alpha_save_path = alpha_save_path or defaults["alpha_save_path"]
    heatmap_save_path = heatmap_save_path or defaults["heatmap_save_path"]
    alpha_values = list(alpha_list) if alpha_list is not None else [1.0]
    aux_config = {aux_language: config[aux_language]}
    aux_paths = aux_config[aux_language]

    if not skip_unlearning:
        print(f"\n[1/4] Running auxiliary unlearning for {aux_language} ({family})")
        output_model_path = run_unlearning(
            config=aux_config,
            llm_family=family,
            llm_model_size=llm_model_size,
            num_gpus=num_gpus,
        )
        if output_model_path is None:
            raise RuntimeError("Auxiliary unlearning failed; stopping full steering process.")
    else:
        print(f"\n[1/4] Skipping auxiliary unlearning for {aux_language}")

    if not skip_steering_vector:
        print(f"\n[2/4] Extracting steering vector to {steering_path}")
        get_steering_vector(
            ft_model_path=aux_paths["ft_model_path"],
            unlearn_model_path=aux_paths["output_model_path"],
            data_path=aux_paths["forget_data_path"],
            batch_size=batch_size,
            max_length=max_length,
            save_path=steering_path,
        )
    else:
        print(f"\n[2/4] Skipping steering vector extraction; using {steering_path}")

    if not skip_alpha_sweep:
        baseline_label = "random baseline" if random_baseline else "steering vector"
        print(f"\n[3/4] Sweeping {baseline_label} alphas {alpha_values}")
        injecting_steering_vector(
            steering_path=steering_path,
            alpha_list=alpha_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            save_path=alpha_save_path,
            random=random_baseline,
            llm_family=family,
            llm_model_size=llm_model_size,
            languages=target_languages,
        )
    else:
        print(f"\n[3/4] Skipping alpha sweep; using {alpha_save_path}")

    if not skip_plot:
        if plot_title is None:
            mode = "Random baseline" if random_baseline else "Steering vector"
            size = f" {llm_model_size}" if llm_model_size else ""
            plot_title = f"{family}{size} {mode} best-alpha heatmap"
        print(f"\n[4/4] Plotting best-result heatmap to {heatmap_save_path}")
        plot_alpha_heatmap(
            title=plot_title,
            alpha_file=alpha_save_path,
            save_path=heatmap_save_path,
            vmax=vmax,
        )
    else:
        print("\n[4/4] Skipping heatmap plot")

    return {
        "steering_path": steering_path,
        "alpha_save_path": alpha_save_path,
        "heatmap_save_path": heatmap_save_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run auxiliary unlearning, extract a steering vector, sweep alphas, and plot the best-result heatmap."
    )
    parser.add_argument("--llm-family", default=os.environ.get("LLM_FAMILY", "qwen").lower(), type=str.lower, choices=sorted(MODEL_SIZE_ENV_KEYS))
    parser.add_argument("--llm-model-size", default=None, help="Model size override, e.g. 7B, 8B, 9B.")
    parser.add_argument("--aux-language", default="English", help="Language used for auxiliary unlearning and steering-vector extraction.")
    parser.add_argument("--target-languages", nargs="+", default=None, help="Languages to evaluate during steering injection. Defaults to all configured languages.")
    parser.add_argument("--alpha-list", nargs="+", default=["1.0"], help="Alpha values, space-separated or comma-separated.")
    parser.add_argument("--steering-path", default=None, help="Where to save/load the extracted steering vector.")
    parser.add_argument("--alpha-save-path", default=None, help="Where to save/load alpha sweep JSON results.")
    parser.add_argument("--heatmap-save-path", default=None, help="Where to save the best-result heatmap.")
    parser.add_argument("--plot-title", default=None, help="Title for the generated heatmap.")
    parser.add_argument("--random-baseline", action="store_true", help="Use random unit vectors with the steering-vector shapes.")
    parser.add_argument("--num-gpus", type=int, default=4, help="Number of GPUs for deepspeed unlearning.")
    parser.add_argument("--batch-size", type=int, default=40, help="Batch size for steering-vector extraction.")
    parser.add_argument("--max-length", type=int, default=512, help="Prompt max length for extraction/injection tokenization.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation length during alpha sweep.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature during alpha sweep.")
    parser.add_argument("--vmax", type=float, default=None, help="Optional heatmap colorbar maximum.")
    parser.add_argument("--skip-unlearning", action="store_true", help="Reuse an existing auxiliary unlearned model.")
    parser.add_argument("--skip-steering-vector", action="store_true", help="Reuse an existing steering vector file.")
    parser.add_argument("--skip-alpha-sweep", action="store_true", help="Reuse an existing alpha sweep JSON file.")
    parser.add_argument("--skip-plot", action="store_true", help="Do not generate a heatmap.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, str]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    alpha_values = _parse_alpha_list(args.alpha_list)
    return run_full_steering_process(
        llm_family=args.llm_family,
        llm_model_size=args.llm_model_size,
        aux_language=args.aux_language,
        target_languages=args.target_languages,
        alpha_list=alpha_values,
        steering_path=args.steering_path,
        alpha_save_path=args.alpha_save_path,
        heatmap_save_path=args.heatmap_save_path,
        plot_title=args.plot_title,
        random_baseline=args.random_baseline,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        vmax=args.vmax,
        skip_unlearning=args.skip_unlearning,
        skip_steering_vector=args.skip_steering_vector,
        skip_alpha_sweep=args.skip_alpha_sweep,
        skip_plot=args.skip_plot,
    )


if __name__ == "__main__":
    main()

