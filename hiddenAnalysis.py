import argparse
import os
import gc
import random
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Literal, Sequence
import json
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from pathlib import Path
from transfer import LANG_MAP
LLM_FAMILY = os.environ.get("LLM_FAMILY")
if LLM_FAMILY:
    os.environ["LLM_FAMILY"] = LLM_FAMILY
from utils.utils import get_model_path, run_task2_benchmark
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_config import REMOTE_MODEL_ROOT, load_llm_config
from matplotlib.legend_handler import HandlerTuple
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, VPacker
from matplotlib.text import Text


LANGUAGES = ["English", "Chinese", "German", "Russian", "Turkish"]
MODEL_SIZE_ENV_KEYS = {
    "qwen": "QWEN_MODEL_SIZE",
    "gemma": "GEMMA_MODEL_SIZE",
}


def _normalise_family(llm_family: Optional[str] = None) -> str:
    return (llm_family or os.environ.get("LLM_FAMILY", "qwen")).lower()


def _apply_model_size_env(llm_family: Optional[str], llm_model_size: Optional[str]) -> str:
    family = _normalise_family(llm_family)
    os.environ["LLM_FAMILY"] = family
    env_key = MODEL_SIZE_ENV_KEYS.get(family)
    if llm_model_size and env_key:
        os.environ[env_key] = llm_model_size
    return family


def _resolve_language(language: str) -> str:
    language_map = {name.lower(): name for name in LANGUAGES}
    key = language.lower()
    if key not in language_map:
        raise ValueError(f"Unsupported language '{language}'. Options: {LANGUAGES}")
    return language_map[key]


def _parse_layer_list(values: Sequence[str]) -> List[int]:
    layers: List[int] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                layers.append(int(item))
    if not layers:
        raise ValueError("At least one PCA layer is required.")
    return layers

def prompting_test_in_another_language(llm_family: Optional[str] = None, llm_model_size: Optional[str] = None):
    return benchmark_answer_in_finetuned_language(
        llm_family=llm_family,
        llm_model_size=llm_model_size,
    )



def benchmark_answer_in_finetuned_language(
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    model_languages: Optional[Sequence[str]] = None,
    input_languages: Optional[Sequence[str]] = None,
    benchmark_result_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    For each single-language finetuned model, prompt it with foreign-language
    questions and ask it to answer in its finetuned language.

    run_task2_benchmark uses SHUFFLE-NUM=-2, so benchmarkTest.py evaluates both:
      - TargetLanguage: normal prompt, target-language references.
      - FinetunedLanguage: prompt includes "answer in <FT language>" and references
        are replaced with the aligned answers from the FT language dataset.
    """
    family = _apply_model_size_env(llm_family or LLM_FAMILY, llm_model_size)
    model_languages = [_resolve_language(lang) for lang in (model_languages or LANGUAGES)]
    requested_inputs = [_resolve_language(lang) for lang in input_languages] if input_languages else None
    results: Dict[str, Dict[str, Any]] = {}

    for ft_lang in model_languages:
        model_path = get_model_path(ft_lang, LANG_MAP, llm_family=family)
        foreign_languages = [
            lang for lang in (requested_inputs or LANGUAGES)
            if lang != ft_lang
        ]
        if not foreign_languages:
            print(f"[!] No foreign input languages for {ft_lang}; skipping.")
            continue

        test_data = {
            lang: {
                "forget": f"./Data/{lang}/Original/forget01.json",
                "retain": f"./Data/{lang}/Original/retain99.json",
            }
            for lang in foreign_languages
        }
        print(
            f"\n=== FT-language response benchmark: model={ft_lang}, "
            f"inputs={', '.join(foreign_languages)} ==="
        )
        run_task2_benchmark(
            model_path,
            test_data,
            [ft_lang],
            LANG_MAP,
            llm_family=family,
            benchmark_result_dir=benchmark_result_dir,
        )
        results[ft_lang] = {
            "model_path": model_path,
            "input_languages": foreign_languages,
            "benchmark_result_dir": benchmark_result_dir or "./Data/BenchmarkResult",
        }

    return results


def setup_env(seed: int = 42):
    """Light determinism and fast math; assert GPU environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_available(), "CUDA is required."
    assert torch.cuda.device_count() >= 4, "This script assumes 4 GPUs are visible."

def load_causal_lm_fp16(model_id: str, device_map: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    return tok, model

# ===================== 1) Data loading (simple dict) =====================

def load_questions_from_qa_json(path: str) -> List[str]:
    # Loading only the questions
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data.keys())


# ===================== 2) Prompt building (question-only) =====================

def render_user_only(tok, content: str) -> str:
    return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False
        )

def build_prompts_and_question_spans(
    tok,
    questions: List[str],
    add_instruction_to_user: bool = False,
    instruction_language: str = "En",
) -> Tuple[List[str], List[Tuple[int, int]]]:
    prompts, spans = [], []
    if instruction_language == "En":
        instruction_text = "You must only answer this question in English."
    elif instruction_language == "Ch":
        instruction_text = "你必须仅用中文回答这个问题"
    elif instruction_language == "De":
        instruction_text = "Bitte beantworte diese Frage ausschließlich auf Deutsch."
    elif instruction_language == "Ru":
        instruction_text = "Пожалуйста, отвечай на этот вопрос только на русском языке."
    elif instruction_language == "Tu":
        instruction_text = "Lütfen bu soruyu yalnızca Türkçe olarak cevapla."
    else:
        raise Exception("No Language Matched")
    instr = instruction_text.strip()
    for q in questions:
        content = f"{q}\n{instr}" if add_instruction_to_user else q
        rendered = render_user_only(tok, content)
        # Find the last occurrence of 'q' in the rendered string; robust if 'q' appears earlier anywhere.
        start = rendered.rfind(q)
        assert start != -1, "Question text not found in rendered prompt."
        end = start + len(q) + len("\n" + instr) if add_instruction_to_user else start + len(q)
        prompts.append(rendered)
        spans.append((start, end))
    return prompts, spans

# ===================== 3) Pairwise cosine (aligned items) =====================
T_CRIT_95 = 2.02  # 95% t-critical for n~40.

@torch.no_grad()
def pairwise_cosine_mean(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    sims = F.cosine_similarity(X.to(torch.float32), Y.to(torch.float32), dim=1, eps=eps)
    return float(sims.mean().cpu())

@torch.no_grad()
def pairwise_cosine_mean_ci(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-12,
    t_crit: float = T_CRIT_95,
) -> Tuple[float, float, float]:
    sims = F.cosine_similarity(X.to(torch.float32), Y.to(torch.float32), dim=1, eps=eps)
    mean = float(sims.mean().cpu())
    n = int(sims.numel())
    if n < 2:
        return mean, mean, mean
    std = sims.std(unbiased=True)
    if torch.isnan(std):
        return mean, mean, mean
    sem = std / (n ** 0.5)
    margin = float((t_crit * sem).cpu())
    return mean, mean - margin, mean + margin

# ===================== 4) Representation extraction (question-only) =====================
@torch.no_grad()
def layerwise_reps(model, tok, full_prompts: List[str], batch_size: int = 40, max_length: int = 512,
    debug: bool = False,
) -> List[torch.Tensor]:
    assert len(full_prompts) > 0, "Provide at least one prompt."
    tok.padding_side = "right"
    if hasattr(tok, "truncation_side"):
        tok.truncation_side = "right"

    probe = tok(
        full_prompts[0],
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_dev = model.get_input_embeddings().weight.device

    # Just a forward pass to get model dimensions
    out0 = model(
        **{k: v.to(input_dev) for k, v in probe.items()},
        output_hidden_states=True,
        use_cache=False,
    )
    Lp1 = len(out0.hidden_states)  # 1 (embeddings) + Number of layers
    H = out0.hidden_states[-1].shape[-1]  # Hidden size
    del out0, probe

    per_layer_chunks: List[List[torch.Tensor]] = [[] for _ in range(Lp1)]
    N_total = len(full_prompts)
    for s in range(0, N_total, batch_size):
        e = min(s + batch_size, N_total)
        prompts_b = full_prompts[s:e]
        enc = tok(
            prompts_b,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors=None,
        )
        B = len(prompts_b)
        T = len(enc["attention_mask"][0])
        attn_cpu = torch.tensor(enc["attention_mask"], dtype=torch.bool)
        # The last index of tok before padding, including special tokens
        prefill_last_idx_cpu = attn_cpu.long().sum(dim=1) - 1 

        if debug:
            for i, _ in enumerate(range(B)):
                bi_dbg = i
                enc0 = enc.encodings[bi_dbg]
                debug_window = 100
                li = int(prefill_last_idx_cpu[bi_dbg].item())
                wL = max(li - debug_window, 0)
                wR = min(li + debug_window, T - 1)
                tid = enc0.ids[li]
                dec_piece = tok.decode([tid], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                cs = enc.token_to_chars(bi_dbg, li)
                raw_piece = "" if cs is None else prompts_b[bi_dbg][cs.start:cs.end]
                print("\n" + "=" * 96)
                print(f"[PREFILL DEBUG] batch_start={s}, sample={s+bi_dbg}")
                print(f"Prefill token idx: {li}  id={tid}  decoded={repr(dec_piece)}  raw={repr(raw_piece)}")
                print("-" * 96)
                print("Idx Spc Pad  ID       RawToken               DecodedPiece           (start,end)  Substring")
                spmask = enc["special_tokens_mask"][bi_dbg]
                attdbg = enc["attention_mask"][bi_dbg]
                for t in range(wL, wR + 1):
                    is_sp = "Y" if bool(spmask[t]) else "N"
                    is_pad = "Y" if not bool(attdbg[t]) else "N"
                    tid_t = enc0.ids[t]
                    dec = tok.decode([tid_t], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    cst = enc.token_to_chars(bi_dbg, t)
                    if cst is None:
                        l, r, sub = None, None, ""
                    else:
                        l, r = cst.start, cst.end
                        sub = prompts_b[bi_dbg][l:r]
                    mark = "<-- prefill_last" if t == li else ""
                    print(f"{t:>3}  {is_sp}  {is_pad} {tid_t:>6}  {enc0.tokens[t]!r:<22} {dec!r:<20} ({str(l):>4},{str(r):>4})  {sub!r} {mark}")
                print("=" * 96 + "\n")

        enc_pt = enc.convert_to_tensors("pt")

        if "offset_mapping" in enc_pt:
            enc_pt.pop("offset_mapping")

        enc_pt = {k: v.to(input_dev) for k, v in enc_pt.items()}

        # Forward once to get hidden states from all layers
        out = model(**enc_pt, output_hidden_states=True, use_cache=False)

        for l, h in enumerate(out.hidden_states):
            if h.dtype != torch.float32:
                h = h.to(torch.float32) 
            
            # The current batch size (may be smaller on last batch)
            B_ = h.size(0)
            idx = prefill_last_idx_cpu.to(h.device)

            # The specific vector for the last token
            pooled = h[torch.arange(B_, device=h.device), idx, :]

            per_layer_chunks[l].append(pooled.cpu())

        del out, enc, enc_pt, attn_cpu
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    reps = [torch.cat(chunks, dim=0) for chunks in per_layer_chunks]
    N = len(full_prompts)
    assert reps[-1].shape == (N, H), f"Expected final-layer reps {(N, H)}, got {reps[-1].shape}."
    assert len(reps) == Lp1, f"Expected {Lp1} layers (incl. embeddings), got {len(reps)}."
    # Drop the embedding layer representations
    return reps[1:]

def get_hidden_reps_base_model(
    model_path,
    main_language,
    add_instruction: bool = False,
    save_plot: bool = True,
    base_model_path: Optional[str] = None,
    ft_model_paths: Optional[Dict[str, str]] = None,
    return_ci: bool = False,
):
    languages = LANGUAGES
    other_languages = [lang for lang in languages if lang != main_language]
    main_abbr = LANG_MAP.get(main_language.lower())
    if main_abbr is None:
        raise ValueError(f"Unsupported main_language='{main_language}'. Expected one of {list(LANG_MAP.keys())}.")
    def _display_tag(tag: str) -> str:
        return "BS" if tag == "Base" else tag

    main_tag = _model_tag(model_path)
    main_tag_display = _display_tag(main_tag)
    if main_tag == "UN":
        base_model_path = None
    load_base_model = bool(
        base_model_path
        and Path(base_model_path) != Path(model_path)
        and main_tag != "UN"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto", 
        attn_implementation="flash_attention_2",
    ).eval()
    base_tokenizer = None
    base_model = None
    base_reps_result: Optional[Dict[str, List[torch.Tensor]]] = None
    base_tag_display: Optional[str] = None
    if load_base_model:
        base_tag_display = _display_tag(_model_tag(base_model_path))
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        base_reps_result = {}
    ft_reps_result: Optional[Dict[str, Dict[str, Any]]] = None
    ft_model_cache: Dict[str, Any] = {}
    if ft_model_paths:
        ft_reps_result = {}
    test_data_path = {lang: f"./Data/{lang}/Original/forget01.json" for lang in languages}
    retain_data_path = {lang: f"./Data/{lang}/Original/retain99.json" for lang in languages}
    reps_result: Dict[str, List[torch.Tensor]] = {}
    retain_reps_result: Optional[Dict[str, List[torch.Tensor]]] = {} if main_tag == "Base" else None

    # 1) Collect layerwise representations per language (last-token pooling)
    for lang in languages:
        data_path = test_data_path[lang]
        questions = load_questions_from_qa_json(data_path)
        prompts, spans = build_prompts_and_question_spans(
            tokenizer,
            questions,
            add_instruction_to_user=add_instruction,
            instruction_language=main_abbr,
        )
        reps = layerwise_reps(model, tokenizer, prompts)
        reps_result[lang] = reps
        if retain_reps_result is not None:
            rt_questions = load_questions_from_qa_json(retain_data_path[lang])
            if len(rt_questions) > 40:
                rt_questions = random.sample(rt_questions, 40)
            rt_prompts, _ = build_prompts_and_question_spans(
                tokenizer,
                rt_questions,
                add_instruction_to_user=add_instruction,
                instruction_language=main_abbr,
            )
            retain_reps_result[lang] = layerwise_reps(model, tokenizer, rt_prompts)
        if base_reps_result is not None and base_model is not None and base_tokenizer is not None:
            base_prompts, _ = build_prompts_and_question_spans(
                base_tokenizer,
                questions,
                add_instruction_to_user=add_instruction,
                instruction_language=main_abbr,
            )
            base_reps_result[lang] = layerwise_reps(base_model, base_tokenizer, base_prompts)
        if ft_reps_result is not None and ft_model_paths is not None:
            if lang not in ft_model_paths:
                raise ValueError(f"No finetuned model path provided for language '{lang}'.")
            ft_path = ft_model_paths[lang]
            if ft_path not in ft_model_cache:
                ft_tag_display = _display_tag(_model_tag(ft_path))
                ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
                ft_model = AutoModelForCausalLM.from_pretrained(
                    ft_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                ).eval()
                ft_model_cache[ft_path] = {
                    "tokenizer": ft_tokenizer,
                    "model": ft_model,
                    "tag": ft_tag_display,
                }
            ft_tokenizer = ft_model_cache[ft_path]["tokenizer"]
            ft_model = ft_model_cache[ft_path]["model"]
            ft_tag_display = ft_model_cache[ft_path]["tag"]
            ft_prompts, _ = build_prompts_and_question_spans(
                ft_tokenizer,
                questions,
                add_instruction_to_user=add_instruction,
                instruction_language=main_abbr,
            )
            ft_reps = layerwise_reps(ft_model, ft_tokenizer, ft_prompts)
            ft_reps_result[lang] = {"reps": ft_reps, "tag": ft_tag_display}
    # Cleanup any cached FT models after use
    for entry in ft_model_cache.values():
        del entry["model"], entry["tokenizer"]
    if ft_model_cache:
        torch.cuda.empty_cache()
        gc.collect()

    assert main_language in reps_result, f"main_language '{main_language}' not found."
    main_reps = reps_result[main_language]
    num_layers = len(main_reps)
    layers_x = list(range(1, num_layers + 1))

    # 2) Compute layerwise cosine similarities: main vs each other language
    cos_curves: Dict[str, List[float]] = {}
    ci_curves: Optional[Dict[str, Tuple[List[float], List[float]]]] = {} if return_ci else None
    for lang in other_languages:
        sims = []
        lows: List[float] = []
        highs: List[float] = []
        peer_reps = reps_result[lang]
        for l in range(num_layers):
            a = main_reps[l]
            b = peer_reps[l]
            min_n = min(a.size(0), b.size(0))
            if min_n == 0:
                raise ValueError(f"No samples to compare for layer {l} between {main_language} and {lang}.")
            if a.size(0) != b.size(0):
                # align by truncating to the smaller set to avoid shape errors
                a = a[:min_n]
                b = b[:min_n]
            if ci_curves is not None:
                mean, low, high = pairwise_cosine_mean_ci(a, b)
                sims.append(mean)
                lows.append(low)
                highs.append(high)
            else:
                sims.append(pairwise_cosine_mean(a, b))
        cos_curves[f"{main_language} ({main_tag_display}) vs {lang} ({main_tag_display})"] = sims
        if ci_curves is not None:
            ci_curves[f"{main_language} ({main_tag_display}) vs {lang} ({main_tag_display})"] = (lows, highs)
    
    if retain_reps_result is not None:
        retain_main_reps = retain_reps_result.get(main_language)
        if retain_main_reps is None:
            raise ValueError(f"No retain representations found for main language '{main_language}'.")
        # Compare main-language forget vs retain sets directly
        sims_main_rt = []
        lows_main_rt: List[float] = []
        highs_main_rt: List[float] = []
        for l in range(num_layers):
            a = main_reps[l]
            b = retain_main_reps[l]
            min_n = min(a.size(0), b.size(0))
            if min_n == 0:
                raise ValueError(f"No retain samples to compare for layer {l} between forget and retain sets of {main_language}.")
            if ci_curves is not None:
                mean, low, high = pairwise_cosine_mean_ci(a[:min_n], b[:min_n])
                sims_main_rt.append(mean)
                lows_main_rt.append(low)
                highs_main_rt.append(high)
            else:
                sims_main_rt.append(pairwise_cosine_mean(a[:min_n], b[:min_n]))
        cos_curves[f"{main_language} ({main_tag_display}) vs {main_language} (RT)"] = sims_main_rt
        if ci_curves is not None:
            ci_curves[f"{main_language} ({main_tag_display}) vs {main_language} (RT)"] = (lows_main_rt, highs_main_rt)
        for lang in other_languages:
            sims = []
            lows: List[float] = []
            highs: List[float] = []
            peer_reps = retain_reps_result[lang]
            for l in range(num_layers):
                a = retain_main_reps[l]
                b = peer_reps[l]
                min_n = min(a.size(0), b.size(0))
                if min_n == 0:
                    raise ValueError(f"No retain samples to compare for layer {l} between {main_language} and {lang}.")
                if a.size(0) != b.size(0):
                    a = a[:min_n]
                    b = b[:min_n]
                if ci_curves is not None:
                    mean, low, high = pairwise_cosine_mean_ci(a, b)
                    sims.append(mean)
                    lows.append(low)
                    highs.append(high)
                else:
                    sims.append(pairwise_cosine_mean(a, b))
            cos_curves[f"{main_language} (RT) vs {lang} (RT)"] = sims
            if ci_curves is not None:
                ci_curves[f"{main_language} (RT) vs {lang} (RT)"] = (lows, highs)
    
    if ft_reps_result:
        for lang in languages:
            peer_entry = ft_reps_result.get(lang)
            if peer_entry is None:
                continue
            sims = []
            lows: List[float] = []
            highs: List[float] = []
            peer_reps = peer_entry["reps"]
            peer_tag = peer_entry["tag"]
            for l in range(num_layers):
                a = main_reps[l]
                b = peer_reps[l]
                min_n = min(a.size(0), b.size(0))
                if min_n == 0:
                    raise ValueError(f"No samples to compare for layer {l} between {main_language} ({main_tag_display}) and {lang} ({peer_tag}).")
                if a.size(0) != b.size(0):
                    a = a[:min_n]
                    b = b[:min_n]
                if ci_curves is not None:
                    mean, low, high = pairwise_cosine_mean_ci(a, b)
                    sims.append(mean)
                    lows.append(low)
                    highs.append(high)
                else:
                    sims.append(pairwise_cosine_mean(a, b))
            cos_curves[f"{main_language} ({main_tag_display}) vs {lang} ({peer_tag})"] = sims
            if ci_curves is not None:
                ci_curves[f"{main_language} ({main_tag_display}) vs {lang} ({peer_tag})"] = (lows, highs)

    if base_reps_result is not None and base_tag_display:
        for lang in languages:
            sims = []
            lows: List[float] = []
            highs: List[float] = []
            peer_reps = base_reps_result[lang]
            for l in range(num_layers):
                a = main_reps[l]
                b = peer_reps[l]
                min_n = min(a.size(0), b.size(0))
                if min_n == 0:
                    raise ValueError(f"No samples to compare for layer {l} between {main_language} (FT) and {lang} (BS).")
                if a.size(0) != b.size(0):
                    a = a[:min_n]
                b = b[:min_n]
                if ci_curves is not None:
                    mean, low, high = pairwise_cosine_mean_ci(a, b)
                    sims.append(mean)
                    lows.append(low)
                    highs.append(high)
                else:
                    sims.append(pairwise_cosine_mean(a, b))
            cos_curves[f"{main_language} ({main_tag_display}) vs {lang} ({base_tag_display})"] = sims
            if ci_curves is not None:
                ci_curves[f"{main_language} ({main_tag_display}) vs {lang} ({base_tag_display})"] = (lows, highs)
    
    if main_tag == "UN":
        cos_curves = {k: v for k, v in cos_curves.items() if "(BS)" not in k}
        if ci_curves is not None:
            ci_curves = {k: v for k, v in ci_curves.items() if "(BS)" not in k}

    tag = _model_tag(model_path)
    if save_plot:
        diagram_name = f"{main_abbr}_{tag}_{'Instr' if add_instruction else 'NoInstr'}"
        if add_instruction:
            title_text = f"{tag}-{main_abbr} • instruction to answer in {main_language}"
        else:
            title_text = f"{tag}-{main_abbr} • no instruction"
        out_pdf = plot_layerwise_cosines(
            main_language,
            layers_x,
            cos_curves,
            ci_curves=ci_curves,
            out_dir="./Data/CosineSimilarityDiagram",
            diagram_name=diagram_name,
            title_text=title_text,
        )
        print(f"[✓] Saved plot to {out_pdf}")

    if return_ci:
        return cos_curves, ci_curves
    return cos_curves


def plot_layerwise_cosines(
    main_language: str,
    layers_x: List[int],
    cos_curves: Dict[str, List[float]],
    ci_curves: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
    out_dir: str = "./Data/CosineSimilarityDiagram",
    diagram_name: Optional[str] = None,
    title_text: Optional[str] = None,
) -> str:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    palette = plt.get_cmap("tab20")
    def _style_from_label(lbl: str) -> Dict[str, Any]:
        if "(RT)" in lbl:
            return {"linestyle": "--", "marker": "s", "linewidth": 2.3}
        if "(BS)" in lbl:
            return {"linestyle": "-", "marker": "o", "linewidth": 2.4}
        if "(FT)" in lbl:
            return {"linestyle": "-.", "marker": "^", "linewidth": 2.3}
        if "(UN)" in lbl:
            return {"linestyle": "-", "marker": "o", "linewidth": 2.2}
        return {"linestyle": "-", "marker": "o", "linewidth": 2.2}
    for idx, curve_label in enumerate(sorted(cos_curves.keys())):
        color = palette(idx % palette.N)
        style = _style_from_label(curve_label)
        display_label = curve_label.replace("(RT)", "(RND)")
        if ci_curves is not None and curve_label in ci_curves:
            low, high = ci_curves[curve_label]
            max_len = min(len(layers_x), len(low), len(high))
            if max_len > 0:
                ax.fill_between(
                    layers_x[:max_len],
                    low[:max_len],
                    high[:max_len],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
        ax.plot(
            layers_x,
            cos_curves[curve_label],
            label=display_label,
            color=color,
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.8 if style["marker"] == "s" else 4.5,
        )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine similarity", fontsize=12)
    title_final = title_text if title_text else f"Layerwise similarity to {main_language}"
    ax.set_title(title_final, fontsize=13, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    base_name = diagram_name if diagram_name else f"{main_language}_layerwise_cosine"
    out_pdf = str(out_dir_path / f"{base_name}.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def _model_tag(model_path: str) -> str:
    """Return a short tag for the model path: 'UN' if unlearned, 'FT' if finetuned, else 'Base'."""
    name = Path(model_path).name.lower()
    if "unlearn" in name or re.search(r"(?:^|[-_])un", name):
        return "UN"
    if "finetuned" in name or re.search(r"(?:^|[-_])ft", name):
        return "FT"
    return "Base"


def plot_language_panels(
    main_language: str,
    entries: List[Dict[str, Any]],
    out_dir: str = "./Data/CosineSimilarityDiagram",
    family_tag: Optional[str] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Combine runs into a single panel PDF with a wider legend that fully encloses the lines.
    family_tag (e.g., qwen/gemma) is used to namespace outputs so different families do not overwrite.
    """
    if not entries:
        raise ValueError(f"No entries provided for {main_language}.")
    out_dir_path = Path(out_dir)
    family_slug = None
    if family_tag:
        family_slug = re.sub(r"[^A-Za-z0-9_-]+", "", family_tag.strip()).lower()
        default_out_dir = Path("./Data/CosineSimilarityDiagram")
        if family_slug and out_dir_path == default_out_dir and output_file is None:
            out_dir_path = out_dir_path / family_slug
    if output_file:
        output_path = Path(output_file)
        out_dir_path = output_path.parent if str(output_path.parent) != "." else Path(".")
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Determine layer axis using the first non-empty curve across all entries
    def _infer_layers_x(all_entries: List[Dict[str, Any]]) -> List[int]:
        for ent in all_entries:
            for vals in ent["cos_curves"].values():
                if len(vals) > 0:
                    return list(range(1, len(vals) + 1))
        raise ValueError(f"No non-empty cosine curves found for {main_language}.")

    layers_x = _infer_layers_x(entries)

    # Setup Subplots
    fig, axes = plt.subplots(
        2,
        len(entries),
        figsize=(5.6 * len(entries), 7.6),
        sharex=False,
        sharey=False,
    )
    if len(entries) == 1:
        axes = np.array(axes).reshape(2, 1)
    # Distinct, paper-friendly palette
    pos_color_grid = [
        ["#1f77b4", "#e15759", "#2ca02c", "#9467bd", "#ff7f0e"],
        ["#17becf", "#bcbd22", "#7f7f7f", "#e377c2", "#8c564b"],
        ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#eeca3b"],
        ["#72b7b2", "#e45756", "#59a14f", "#9d755d", "#edc948"],
    ]
    palette = mcolors.ListedColormap([c for row in pos_color_grid for c in row])
    def _color_for_pos(row_idx: int, col_idx: int):
        if row_idx < len(pos_color_grid) and col_idx < len(pos_color_grid[0]):
            return pos_color_grid[row_idx][col_idx]
        # Fallback if we ever exceed the grid
        colors = palette.colors if hasattr(palette, "colors") else []
        if colors:
            return colors[(row_idx * len(pos_color_grid[0]) + col_idx) % len(colors)]
        return palette((row_idx * len(pos_color_grid[0]) + col_idx) / max(1, palette.N))

    lang_order = [
        LANG_MAP[key]
        for key in ["english", "chinese", "german", "russian", "turkish"]
        if key in LANG_MAP
    ]
    lang_color_map = {abbr: _color_for_pos(0, idx) for idx, abbr in enumerate(lang_order)}

    # Font scaling
    title_fs = 16
    axis_fs = 15
    tick_fs = 12
    legend_fs = 12
    header_marker_size = 18

    # Sort entries
    def _panel_sort_key(entry: Dict[str, Any]) -> int:
        tag = entry["tag"]
        add = entry["add_instruction"]
        if tag == "Base": return 0
        if tag == "FT" and not add: return 1
        if tag == "FT" and add: return 2
        if tag == "UN": return 3
        return 99

    panel_order = sorted(entries, key=_panel_sort_key)

    # --- Helpers ---
    def _abbr_side(side: str) -> str:
        m = re.match(r"([A-Za-z]+)", side.strip())
        lang = m.group(1) if m else side.strip()
        return LANG_MAP.get(lang.lower(), lang[:2])

    def _get_peer_abbr(label_str: str) -> str:
        if " vs " in label_str:
            parts = label_str.split(" vs ")
            return _abbr_side(parts[1])
        return _abbr_side(label_str)
    
    def _extract_peer_tag(label_str: str) -> str:
        if " vs " in label_str:
            right = label_str.split(" vs ", 1)[1]
        else:
            right = label_str
        m = re.search(r"\(([^)]+)\)", right)
        if m:
            return m.group(1).strip()
        return "FT"

    def _display_tag(tag: str) -> str:
        if tag in {"RT", "BS"}:
            return ""
        return tag

    def _line_style(panel_tag: str, peer_tag: str) -> Dict[str, Any]:
        """
        Explicit per-panel/per-peer styling to avoid accidental fallthroughs.
        Key cases the user asked for:
          - FT panel: BS column should use square markers and long dash.
          - UN panel: FT column should use square markers and long dash.
        """
        style = {"linestyle": "-", "marker": "o", "linewidth": 2.1, "dashes": None}
        if panel_tag == "FT":
            if peer_tag == "BS":
                style.update({"linestyle": "--", "marker": "s", "linewidth": 2.4, "dashes": (16, 6)})
            elif peer_tag == "RT":
                style.update({"linestyle": "--", "marker": "s", "linewidth": 2.3, "dashes": (12, 4)})
            elif peer_tag == "FT":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.2})
            elif peer_tag == "UN":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.1})
        elif panel_tag == "UN":
            if peer_tag == "FT":
                style.update({"linestyle": "--", "marker": "s", "linewidth": 2.4, "dashes": (16, 6)})
            elif peer_tag == "RT":
                style.update({"linestyle": "--", "marker": "s", "linewidth": 2.3, "dashes": (12, 4)})
            elif peer_tag == "UN":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.1})
            elif peer_tag == "BS":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.2})
        else:  # Base or other tags
            if peer_tag == "RT":
                style.update({"linestyle": "--", "marker": "s", "linewidth": 2.3, "dashes": (12, 4)})
            elif peer_tag == "BS":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.3})
            elif peer_tag == "FT":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.2})
            elif peer_tag == "UN":
                style.update({"linestyle": "-", "marker": "o", "linewidth": 2.1})
        return style
    
    def _apply_dash_pattern(
        line_obj,
        style: Dict[str, Any],
        *,
        for_legend: bool = False,
        handle_len: Optional[float] = None,
    ) -> None:
        """
        Ensure dash patterns carry through; compress dashes for legend handles so short
        handle lengths still show a visible gap.
        """
        dash_seq = style.get("dashes")
        if not dash_seq:
            return
        use_seq = dash_seq
        if for_legend:
            target = handle_len if handle_len else 4.0
            total = sum(dash_seq[:2]) if len(dash_seq) >= 2 else dash_seq[0]
            if total > 0 and target > 0:
                scale = min(1.0, (target * 0.8) / total)
                use_seq = tuple(max(1.0, d * scale) for d in dash_seq)
        line_obj.set_linestyle((0, use_seq))
        line_obj.set_dashes(use_seq)
        line_obj.set_dash_capstyle("butt")

    def _bottom_tag_for_panel(panel_tag: str) -> Optional[str]:
        if panel_tag == "Base":
            return "RT"
        if panel_tag == "FT":
            return "BS"
        if panel_tag == "UN":
            return "FT"
        return None

    def _instruction_label(add_instruction: bool, language: str) -> str:
        return "Cross-lingual prompting" if add_instruction else ""

    def _split_curves_by_tag(
        curves_dict: Dict[str, List[float]],
        bottom_tag: Optional[str],
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        top_curves: Dict[str, List[float]] = {}
        bottom_curves: Dict[str, List[float]] = {}
        for curve_label, vals in curves_dict.items():
            peer_tag = _extract_peer_tag(curve_label)
            if bottom_tag and peer_tag == bottom_tag:
                bottom_curves[curve_label] = vals
            else:
                top_curves[curve_label] = vals
        return top_curves, bottom_curves

    def _split_ci_by_tag(
        ci_dict: Optional[Dict[str, Tuple[List[float], List[float]]]],
        bottom_tag: Optional[str],
    ) -> Tuple[Optional[Dict[str, Tuple[List[float], List[float]]]], Optional[Dict[str, Tuple[List[float], List[float]]]]]:
        if ci_dict is None:
            return None, None
        top_ci: Dict[str, Tuple[List[float], List[float]]] = {}
        bottom_ci: Dict[str, Tuple[List[float], List[float]]] = {}
        for curve_label, bands in ci_dict.items():
            peer_tag = _extract_peer_tag(curve_label)
            if bottom_tag and peer_tag == bottom_tag:
                bottom_ci[curve_label] = bands
            else:
                top_ci[curve_label] = bands
        return top_ci, bottom_ci

    def _plot_panel_axis(
        ax,
        curves_subset: Dict[str, List[float]],
        panel_tag: str,
        color_by_peer: Dict[str, str],
        ci_curves_subset: Optional[Dict[str, Tuple[List[float], List[float]]]] = None,
        force_solid: bool = False,
        show_legend: bool = True,
    ) -> Optional[Tuple[float, float]]:
        if not curves_subset:
            ax.axis("off")
            return None

        line_store: Dict[str, Dict[str, Any]] = {}
        min_y: Optional[float] = None
        max_y: Optional[float] = None

        # Plot lines
        for idx, curve_label in enumerate(sorted(curves_subset.keys())):
            peer_tag = _extract_peer_tag(curve_label)
            peer = _get_peer_abbr(curve_label)
            color = color_by_peer.get(peer)
            if not color:
                palette_colors = palette.colors if hasattr(palette, "colors") else []
                color = palette_colors[idx % len(palette_colors)] if palette_colors else palette(idx / max(1, palette.N))
            y_vals_full = curves_subset[curve_label]
            max_len = min(len(layers_x), len(y_vals_full))
            if max_len == 0:
                print(f"[!] Empty curve '{curve_label}' in panel '{panel_tag}'. Skipping.")
                continue
            if len(y_vals_full) != len(layers_x):
                print(f"[!] Length mismatch for curve '{curve_label}' in panel '{panel_tag}': "
                    f"x={len(layers_x)} vs y={len(y_vals_full)}. Truncating to {max_len}.")
            x_vals = layers_x[:max_len]
            y_vals = y_vals_full[:max_len]
            style = _line_style(panel_tag, peer_tag)
            if force_solid:
                style["linestyle"] = "-"
                style["dashes"] = None

            y_min = float(np.nanmin(y_vals)) if len(y_vals) else None
            y_max = float(np.nanmax(y_vals)) if len(y_vals) else None
            if y_min is not None:
                min_y = y_min if min_y is None else min(min_y, y_min)
            if y_max is not None:
                max_y = y_max if max_y is None else max(max_y, y_max)

            if ci_curves_subset is not None and curve_label in ci_curves_subset:
                low, high = ci_curves_subset[curve_label]
                ci_len = min(len(x_vals), len(low), len(high))
                if ci_len > 0:
                    low_ci = np.array(low[:ci_len], dtype=float)
                    high_ci = np.array(high[:ci_len], dtype=float)
                    ci_min = float(np.nanmin(low_ci))
                    ci_max = float(np.nanmax(high_ci))
                    min_y = ci_min if min_y is None else min(min_y, ci_min)
                    max_y = ci_max if max_y is None else max(max_y, ci_max)
                    ax.fill_between(
                        x_vals[:ci_len],
                        low_ci,
                        high_ci,
                        color=color,
                        alpha=0.18,
                        linewidth=0,
                        zorder=1,
                    )

            line, = ax.plot(
                x_vals,
                y_vals[:len(x_vals)],
                label=curve_label,
                color=color,
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                marker=None,
                markersize=0,
            )
            _apply_dash_pattern(line, style)

            if peer not in line_store:
                line_store[peer] = {}
            line_store[peer][peer_tag] = line

        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
        ax.tick_params(labelsize=tick_fs)
        if not show_legend:
            return (min_y, max_y) if min_y is not None and max_y is not None else None

        # -------------------------------------------------------
        # Matrix Legend Logic
        # -------------------------------------------------------
        unique_tags = sorted({k for entry in line_store.values() for k in entry.keys()})
        if unique_tags:
            sorted_peers = sorted(line_store.keys())

            # Constants
            handle_len = 4.2
            text_pad   = -2.0
            column_spacing = 3.0
            border_pad = 0.8

            if len(unique_tags) == 1:
                single_tag = unique_tags[0]
                row_sep = legend_fs * 0.35
                col_sep = legend_fs * 0.7
                text_col_w = legend_fs * 2.6
                line_len = legend_fs * 2.6
                line_height = legend_fs * 0.9

                def _text_cell(text: str) -> DrawingArea:
                    da = DrawingArea(text_col_w, line_height, 0, 0)
                    txt = Text(
                        text_col_w * 0.5,
                        line_height * 0.5,
                        text,
                        ha="center",
                        va="center",
                        fontsize=legend_fs,
                    )
                    da.add_artist(txt)
                    return da

                def _line_cell(peer_name: str) -> DrawingArea:
                    base_line = line_store[peer_name][single_tag]
                    style_for_cell = _line_style(panel_tag, single_tag)
                    if force_solid:
                        style_for_cell["linestyle"] = "-"
                        style_for_cell["dashes"] = None
                    color = color_by_peer.get(peer_name, base_line.get_color())
                    da = DrawingArea(line_len, line_height, 0, 0)
                    y = line_height * 0.5
                    line = plt.Line2D(
                        [0, line_len * 0.5, line_len],
                        [y, y, y],
                        color=color,
                        linewidth=style_for_cell["linewidth"],
                        linestyle=style_for_cell["linestyle"],
                        marker=style_for_cell["marker"],
                        markersize=4.0,
                        markevery=[1],
                    )
                    _apply_dash_pattern(line, style_for_cell, for_legend=True, handle_len=2.6)
                    da.add_artist(line)
                    return da

                rows = [
                    HPacker(
                        children=[_text_cell("Lang"), _text_cell(_display_tag(single_tag))],
                        align="center",
                        pad=0,
                        sep=col_sep,
                    )
                ]
                for peer_name in sorted_peers:
                    rows.append(
                        HPacker(
                            children=[_text_cell(peer_name), _line_cell(peer_name)],
                            align="center",
                            pad=0,
                            sep=col_sep,
                        )
                    )
                legend_box = VPacker(children=rows, align="center", pad=0, sep=row_sep)
                anchored = AnchoredOffsetbox(
                    loc="lower left",
                    child=legend_box,
                    frameon=True,
                    borderpad=0.4,
                    pad=0.2,
                )
                ax.add_artist(anchored)
                return (min_y, max_y) if min_y is not None and max_y is not None else None

            # Handles
            inv_handle = plt.Line2D([], [], color="none", marker="", linestyle="")

            # Column 1: Language Names
            final_handles = [inv_handle] + [inv_handle for _ in sorted_peers]
            final_labels  = ["Lang"] + sorted_peers

            if panel_tag == "UN":
                preferred_order = ["RT", "BS", "FT", "UN"]
            else:
                preferred_order = ["RT", "BS", "UN", "FT"]
            tag_order = [t for t in preferred_order if t in unique_tags] + [t for t in unique_tags if t not in preferred_order]

            # Apply language-based colors to plotted lines for consistency across subplots
            for peer_name, tag_map in line_store.items():
                for tag_name, line_obj in tag_map.items():
                    color = color_by_peer.get(peer_name)
                    if color:
                        line_obj.set_color(color)
                        line_obj.set_markerfacecolor(color)
                        line_obj.set_markeredgecolor(color)

            for idx_tag, tag in enumerate(tag_order):
                col_handles = [inv_handle]
                for col_idx, p in enumerate(sorted_peers):
                    if tag in line_store[p]:
                        base_line = line_store[p][tag]
                        style_for_cell = _line_style(panel_tag, tag)
                        if force_solid:
                            style_for_cell["linestyle"] = "-"
                            style_for_cell["dashes"] = None
                        color = color_by_peer.get(p, base_line.get_color())
                        proxy = plt.Line2D(
                            [],
                            [],
                            color=color,
                            linewidth=style_for_cell["linewidth"],
                            linestyle=style_for_cell["linestyle"],
                            marker=style_for_cell["marker"],
                            markersize=4.0,
                        )
                        _apply_dash_pattern(proxy, style_for_cell, for_legend=True, handle_len=handle_len)
                        col_handles.append(proxy)
                    else:
                        col_handles.append(inv_handle)
                col_labels = [_display_tag(tag)] + [""] * len(sorted_peers)
                final_handles.extend(col_handles)
                final_labels.extend(col_labels)

            ax.legend(
                final_handles,
                final_labels,
                loc="lower left",
                ncol=len(tag_order) + 1,
                fontsize=legend_fs,
                alignment="center",

                # Layout
                handlelength=handle_len,
                handletextpad=text_pad, 
                columnspacing=column_spacing,
                borderpad=border_pad,

                framealpha=0.95
            )

        elif len(curves_subset) == 4:
            # Simple Legend
            h, l = ax.get_legend_handles_labels()
            clean_labels = []
            for lbl in l:
                if " vs " in lbl:
                    left, right = lbl.split(" vs ", 1)
                    clean_labels.append(f"{_abbr_side(left)} vs {_abbr_side(right)}")
                else:
                    clean_labels.append(_abbr_side(lbl))

            ax.legend(
                h,
                clean_labels,
                loc="lower left",
                fontsize=legend_fs,
                ncol=2 if len(h) > 3 else 1,
                columnspacing=1.0,
                handlelength=2.5
            )
        return (min_y, max_y) if min_y is not None and max_y is not None else None

    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    for col_idx, entry in enumerate(panel_order):
        tag = entry["tag"]
        add_instr = entry["add_instruction"]
        curves = entry["cos_curves"]
        ci_curves = entry.get("ci_curves")
        all_peers = sorted({_get_peer_abbr(lbl) for lbl in curves.keys()})
        color_by_peer: Dict[str, str] = {}
        fallback_peers: List[str] = []
        for peer in all_peers:
            if peer in lang_color_map:
                color_by_peer[peer] = lang_color_map[peer]
            else:
                fallback_peers.append(peer)
        for idx, peer in enumerate(fallback_peers):
            color_by_peer[peer] = _color_for_pos(1, idx)

        bottom_tag = _bottom_tag_for_panel(tag)
        top_curves, bottom_curves = _split_curves_by_tag(curves, bottom_tag)
        top_ci, bottom_ci = _split_ci_by_tag(ci_curves, bottom_tag)

        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        show_legend = col_idx == 0
        top_ylim = _plot_panel_axis(
            ax_top,
            top_curves,
            tag,
            color_by_peer,
            ci_curves_subset=top_ci,
            show_legend=show_legend,
        )
        bottom_ylim = _plot_panel_axis(
            ax_bottom,
            bottom_curves,
            tag,
            color_by_peer,
            ci_curves_subset=bottom_ci,
            force_solid=True,
            show_legend=show_legend,
        )
        if top_ylim or bottom_ylim:
            if top_ylim and bottom_ylim:
                shared_min = min(top_ylim[0], bottom_ylim[0])
                shared_max = max(top_ylim[1], bottom_ylim[1])
            else:
                shared_min, shared_max = top_ylim if top_ylim else bottom_ylim
            pad = 0.02 * (shared_max - shared_min) if shared_max > shared_min else 0.01
            ax_top.set_ylim(shared_min - pad, shared_max + pad)
            ax_bottom.set_ylim(shared_min - pad, shared_max + pad)

        # Titles (top row only)
        instr_label = _instruction_label(add_instr, main_language)
        instr_suffix = f" ({instr_label})" if instr_label else ""
        if tag == "Base": subtitle = f"Base Model{instr_suffix}"
        elif tag == "FT": subtitle = f"FT {main_language}{instr_suffix}"
        elif tag == "UN": subtitle = f"UN English{instr_suffix}"
        else: subtitle = f"{tag}"
        ax_top.set_title(subtitle, fontsize=title_fs, pad=2)

        if tag == "Base":
            bottom_title = f"vs Random{instr_suffix}"
        elif tag == "FT":
            bottom_title = f"vs Base Model{instr_suffix}"
        elif tag == "UN":
            un_instr = _instruction_label(False, main_language)
            un_suffix = f" ({un_instr})" if un_instr else ""
            bottom_title = f"vs FT English{un_suffix}"
        else:
            bottom_title = ""
        if bottom_title:
            ax_bottom.set_title(bottom_title, fontsize=title_fs, pad=2)

    # Global Axis Formatting
    nrows, ncols = axes.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            if not ax.get_visible():
                continue
            if c == 0:
                ax.set_ylabel("Cosine similarity", fontsize=axis_fs)
            else:
                ax.tick_params(labelleft=False)
            if r == nrows - 1:
                ax.set_xlabel("Layer", fontsize=axis_fs)
            else:
                ax.set_xlabel("")
            ax.tick_params(labelsize=tick_fs)

    fig.subplots_adjust(left=0.04, right=0.99, wspace=0.08, hspace=0.14, bottom=0.07, top=0.96)

    if output_file:
        out_pdf = str(Path(output_file))
    else:
        name_prefix = f"{family_slug}_" if family_slug else ""
        out_pdf = str(out_dir_path / f"{name_prefix}{LANG_MAP[main_language.lower()]}_combined.pdf")
    fig.savefig(out_pdf)
    plt.close(fig)
    return out_pdf


def _load_llm_cfg(llm_family: Optional[str] = None, llm_model_size: Optional[str] = None):
    """Load LLM config with optional Qwen/Gemma size override."""
    family = _normalise_family(llm_family)
    env_key = None
    prev_val = None
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



def run_single_anchor_language_similarity_plot(
    anchor_language: str = "English",
    anchor_model_type: str = "FT",
    add_instruction: bool = False,
    out_dir: str = "./Data/CosineSimilarityDiagram",
    output_file: Optional[str] = None,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
) -> str:
    """
    Plot layerwise hidden-state cosine similarity across five languages using one
    anchor language. The anchor model is Base/FT/UN for the anchor language; the
    four non-anchor languages are compared to that anchor representation per layer.
    """
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    family_slug = re.sub(r"[^A-Za-z0-9_-]+", "", cfg.family).lower()
    anchor_language = _resolve_language(anchor_language)
    model_type = anchor_model_type.upper()
    if model_type in {"BASE", "BS"}:
        model_path = cfg.base_model_local_path
        model_type = "Base"
    elif model_type in {"FT", "UN"}:
        model_path = get_model_path_by_pattern(anchor_language, model_type, llm_family, llm_model_size)
    else:
        raise ValueError("anchor_model_type must be one of: base, ft, un")

    print(
        f"\n=== Anchor cosine similarity: anchor={anchor_language}, "
        f"model={model_type}, family={cfg.family} ==="
    )
    cos_curves, ci_curves = get_hidden_reps_base_model(
        model_path,
        anchor_language,
        add_instruction=add_instruction,
        save_plot=False,
        return_ci=True,
    )
    if not cos_curves:
        raise ValueError("No cosine curves were produced for anchor-language similarity.")

    first_curve = next(iter(cos_curves.values()))
    layers_x = list(range(1, len(first_curve) + 1))
    anchor_abbr = LANG_MAP[anchor_language.lower()]
    instr_tag = "Instr" if add_instruction else "NoInstr"
    diagram_name = f"{family_slug}_{anchor_abbr}_{model_type}_{instr_tag}_anchor_cosine"
    title_text = f"{cfg.family} {model_type}: {anchor_language} anchor vs other languages"
    default_out_dir = Path("./Data/CosineSimilarityDiagram")
    out_dir_path = Path(out_dir)
    output_dir = str(out_dir_path / family_slug) if family_slug and out_dir_path == default_out_dir else str(out_dir_path)
    output_diagram_name = diagram_name
    if output_file:
        output_path = Path(output_file)
        output_dir = str(output_path.parent if str(output_path.parent) != "." else Path("."))
        output_diagram_name = output_path.stem

    out_path = plot_layerwise_cosines(
        anchor_language,
        layers_x,
        cos_curves,
        ci_curves=ci_curves,
        out_dir=output_dir,
        diagram_name=output_diagram_name,
        title_text=title_text,
    )
    print(f"[✓] Anchor-language cosine plot saved to: {out_path}")
    return out_path


def run_anchor_language_similarity_plot(
    anchor_language: str = "English",
    out_dir: str = "./Data/CosineSimilarityDiagram",
    output_file: Optional[str] = None,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
) -> str:
    """Generate the original 2-row x 4-column cosine-similarity panel.

    Columns are Base, FT, FT with cross-lingual prompting, and UN for the
    selected anchor language. Rows split the main cross-language comparisons
    from the panel-specific baseline comparisons, matching the original
    task2.py plotting logic.
    """
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    anchor_language = _resolve_language(anchor_language)
    family_tag = cfg.family
    base_model_path = cfg.base_model_local_path
    ft_path = get_model_path_by_pattern(anchor_language, "FT", llm_family, llm_model_size)
    un_path = get_model_path_by_pattern(anchor_language, "UN", llm_family, llm_model_size)
    tasks = [
        (anchor_language, base_model_path, False),
        (anchor_language, ft_path, False),
        (anchor_language, ft_path, True),
        (anchor_language, un_path, False),
    ]

    entries: List[Dict[str, Any]] = []
    for lang, model_path, add_instr in tasks:
        label = f"{Path(model_path).name}_{lang}_{'Instr' if add_instr else 'NoInstr'}"
        print(f"\n[+] Running panel cosine: model={model_path}, anchor={lang}, instruction={add_instr}")
        model_tag = _model_tag(model_path)
        base_compare_path = base_model_path if model_tag == "FT" else None
        ft_compare_paths = None
        if model_tag == "UN":
            ft_single_path = get_model_path_by_pattern(lang, "FT", llm_family, llm_model_size)
            ft_compare_paths = {candidate_lang: ft_single_path for candidate_lang in LANGUAGES}
        cos_curves, ci_curves = get_hidden_reps_base_model(
            model_path,
            lang,
            add_instruction=add_instr,
            save_plot=False,
            base_model_path=base_compare_path,
            ft_model_paths=ft_compare_paths,
            return_ci=True,
        )
        print(f"[✓] Done: {label}")
        entries.append({
            "tag": model_tag,
            "add_instruction": add_instr,
            "cos_curves": cos_curves,
            "ci_curves": ci_curves,
        })

    out_pdf = plot_language_panels(
        anchor_language,
        entries,
        out_dir=out_dir,
        family_tag=family_tag,
        output_file=output_file,
    )
    print(f"[✓] Anchor-language 2x4 cosine panel saved to: {out_pdf}")
    return out_pdf


def run_all_language_similarity_plots(llm_family: Optional[str] = None, llm_model_size: Optional[str] = None):
    """Generate cosine plots for EN/CH, base + finetuned, with/without instruction."""
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    family_tag = cfg.family
    root = cfg.models_local_root
    prefix = cfg.finetuned_prefix
    base_model_path = os.path.join(root, cfg.base_model_name)
    en_ft_path = os.path.join(root, f"{prefix}En")
    ch_ft_path = os.path.join(root, f"{prefix}Ch")
    en_un_path = os.path.join(root, f"{prefix}En-UNEn-Original")
    ch_un_path = os.path.join(root, f"{prefix}Ch-UNCh-Original")
    tasks = [
        # Base model (without instruction)
        ("English", base_model_path, False),
        ("Chinese", base_model_path, False),
        # Finetuned models (with/without instruction)
        ("English", en_ft_path, False),
        ("English", en_ft_path, True),
        ("Chinese", ch_ft_path, False),
        ("Chinese", ch_ft_path, True),
        # Unlearned models (without instruction)
        ("English", en_un_path, False),
        ("Chinese", ch_un_path, False),
    ]

    combined_results = {}
    collected: Dict[str, List[Dict[str, Any]]] = {"English": [], "Chinese": []}
    for lang, path, add_instr in tasks:
        label = f"{Path(path).name}_{lang}_{'Instr' if add_instr else 'Base'}"
        print(f"\n[+] Running similarity plot: model={path}, main_language={lang}, instruction={add_instr}")
        model_tag = _model_tag(path)
        base_compare_path = base_model_path if model_tag == "FT" else None
        ft_compare_paths = None
        if model_tag == "UN":
            # For UN runs, compare against a single FT model (main-language FT) across all languages
            ft_single_path = get_model_path_by_pattern(lang, "FT", llm_family, llm_model_size)
            ft_compare_paths = {
                l: ft_single_path
                for l in LANGUAGES
            }
        cos_curves, ci_curves = get_hidden_reps_base_model(
            path,
            lang,
            add_instruction=add_instr,
            save_plot=False,
            base_model_path=base_compare_path,
            ft_model_paths=ft_compare_paths,
            return_ci=True,
        )
        print(f"[✓] Done: {label}")
        collected[lang].append({
            "tag": _model_tag(path),
            "add_instruction": add_instr,
            "cos_curves": cos_curves,
            "ci_curves": ci_curves,
        })

    # Combine into per-language panel PDFs
    for lang, entries in collected.items():
        combined_pdf = plot_language_panels(lang, entries, family_tag=family_tag)
        combined_results[f"{lang}_combined"] = combined_pdf
        print(f"[✓] Combined panel for {lang}: {combined_pdf}")
    return combined_results


def get_model_path_by_pattern(lang: str, model_type: str, llm_family: Optional[str] = None, llm_model_size: Optional[str] = None) -> str:
    """
    Constructs the file path based on your directory structure.
    Adjust base_root if your models are elsewhere.
    """
    abbr = LANG_MAP.get(lang.lower(), lang[:2]) # En, Ch, De...
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    base_root = cfg.models_local_root
    prefix = cfg.finetuned_prefix

    if model_type == "FT":
        return os.path.join(base_root, f"{prefix}{abbr}")
    if model_type == "UN":
        return os.path.join(base_root, f"{prefix}{abbr}-UN{abbr}-Original")
    raise ValueError("model_type must be 'FT' or 'UN'")

def get_vectors_for_model(
    model_path: str, 
    data_path: str, 
    lang_abbr: str,
    batch_size: int = 40
) -> List[torch.Tensor]:
    """
    Helper: Loads one model, processes one dataset, returns vectors (CPU), cleans up.
    """
    print(f"    [Loader] Loading: {Path(model_path).name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
    except OSError:
        print(f"    [!] Error: Could not load {model_path}. Skipping.")
        return None

    # Load Questions
    questions = load_questions_from_qa_json(data_path)
    
    # Build Prompts (No specific instruction to keep representation pure)
    prompts, _ = build_prompts_and_question_spans(
        tokenizer, 
        questions, 
        add_instruction_to_user=False,
        instruction_language=lang_abbr
    )

    # Extract Vectors
    # Returns list of tensors [Layer0, Layer1... Layer32]
    reps = layerwise_reps(model, tokenizer, prompts, batch_size=batch_size)

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return reps

def plot_ft_vs_un_similarity(
    out_dir: str = "./Data/CosineSimilarityDiagram",
    output_file: Optional[str] = None,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
) -> Optional[str]:
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    family_slug = re.sub(r"[^A-Za-z0-9_-]+", "", cfg.family).lower()
    default_out_dir = Path("./Data/CosineSimilarityDiagram")
    base_out_dir = Path(out_dir)
    out_dir_path = base_out_dir / family_slug if family_slug and base_out_dir == default_out_dir else base_out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_prefix = f"{family_slug}_" if family_slug else ""

    languages = [_resolve_language(lang) for lang in (languages or LANGUAGES)]
    final_curves = {} # Stores the similarity list for each language

    # Loop through each language independently
    for lang in languages:
        print(f"\n=== Processing Language: {lang} ===")
        abbr = LANG_MAP.get(lang.lower(), "En")
        
        # 1. Define Paths
        ft_path = get_model_path_by_pattern(lang, "FT", llm_family, llm_model_size)
        un_path = get_model_path_by_pattern(lang, "UN", llm_family, llm_model_size)
        data_path = f"./Data/{lang}/Original/forget01.json"
        
        # 2. Get FT Vectors
        print("  -> Step 1: Getting Finetuned (FT) Representations")
        ft_reps = get_vectors_for_model(ft_path, data_path, abbr)
        if ft_reps is None: raise ValueError(f"Failed to get FT representations for {lang} from {ft_path}")

        # 3. Get UN Vectors
        print("  -> Step 2: Getting Unlearned (UN) Representations")
        un_reps = get_vectors_for_model(un_path, data_path, abbr)
        if un_reps is None: raise ValueError(f"Failed to get UN representations for {lang} from {un_path}")

        # 4. Calculate Similarity (FT vs UN)
        print("  -> Step 3: Calculating Similarity")
        sims_per_layer = []
        num_layers = len(ft_reps)
        
        for l in range(num_layers):
            v_ft = ft_reps[l]    # Finetuned Vector
            v_un = un_reps[l]    # Unlearned Vector
            assert v_ft.size() == v_un.size(), f"Size mismatch at layer {l} for {lang}: FT {v_ft.size()} vs UN {v_un.size()}"
            sim = pairwise_cosine_mean(v_ft, v_un)
            sims_per_layer.append(sim)
            
        final_curves[lang] = sims_per_layer
        print(f"  [✓] Finished {lang}")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    if not final_curves:
        print("No data collected.")
        return None

    print("\n[Plotting] Generating final diagram...")
    # Determine X-axis based on layer count of first valid result
    first_lang = next(iter(final_curves))
    layers_x = list(range(1, len(final_curves[first_lang]) + 1))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    
    # Standard Colors
    colors = {
        "English": "#1f77b4", # Blue
        "Chinese": "#d62728", # Red
        "German": "#2ca02c",  # Green
        "Russian": "#9467bd", # Purple
        "Turkish": "#ff7f0e"  # Orange
    }

    for lang in languages:
        if lang in final_curves:
            ax.plot(
                layers_x,
                final_curves[lang],
                label=lang,
                color=colors.get(lang, "black"),
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                alpha=0.9
            )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity (FT vs UN)", fontsize=12)
    ax.set_title("Impact of Unlearning on Internal Representations for Trained Languages", fontsize=13, pad=10)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)

    # Legend inside bottom-left
    ax.legend(
        loc="lower left",
        frameon=True,
        fontsize=10,
        ncol=1,
        framealpha=0.9
    )
    
    plt.tight_layout()
    
    if output_file:
        output_path = Path(output_file)
        out_dir_path = output_path.parent if str(output_path.parent) != "." else Path(".")
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_path = str(output_path)
    else:
        out_filename = f"{out_prefix}Impact_of_Unlearning_FT_vs_UN.pdf"
        out_path = str(out_dir_path / out_filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Plot saved to: {out_path}")
    return out_path

def get_vectors_specific_questions(
    model_path: str,
    questions: List[str],
    lang_abbr: str,
    batch_size: int = 40
) -> List[torch.Tensor]:
    """
    Loads model, extracts vectors for the EXACT list of questions provided.
    """
    print(f"    [Loader] Loading: {Path(model_path).name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
    except OSError:
        print(f"    [!] Error: Could not load {model_path}.")
        return None

    # No instruction added to test raw understanding
    prompts, _ = build_prompts_and_question_spans(
        tokenizer, 
        questions, 
        add_instruction_to_user=False, 
        instruction_language=lang_abbr
    )

    reps = layerwise_reps(model, tokenizer, prompts, batch_size=batch_size)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return reps

def plot_pca_for_language(
    lang: str,
    reps_bs: List[torch.Tensor],
    reps_ft: List[torch.Tensor],
    reps_un: List[torch.Tensor],
    target_layers: List[int],
    out_dir: str
):
    num_plots = len(target_layers)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4.5))
    if num_plots == 1: axes = [axes]
    
    total_layers_avail = len(reps_ft)

    for i, layer_idx in enumerate(target_layers):
        ax = axes[i]

        actual_idx = _resolve_pca_layer_index(layer_idx, total_layers_avail)
        _plot_pca_panel(
            ax,
            reps_bs,
            reps_ft,
            reps_un,
            actual_idx,
            show_legend=(i == 0),
        )

        # Styling
        layer_name = "Last Layer" if actual_idx == total_layers_avail - 1 else f"Layer {actual_idx}"
        ax.set_title(layer_name, fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    out_path = Path(out_dir) / f"{lang}_PCA_Separation.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Saved PCA plot to: {out_path}")


def _resolve_pca_layer_index(layer_idx: int, total_layers_avail: int) -> int:
    if layer_idx < 0:
        actual_idx = total_layers_avail + layer_idx
    else:
        actual_idx = layer_idx
    if actual_idx < 0:
        raise ValueError(f"Warning: Layer request {layer_idx} is invalid")
    return min(actual_idx, total_layers_avail - 1)


def _plot_pca_panel(
    ax,
    reps_bs: List[torch.Tensor],
    reps_ft: List[torch.Tensor],
    reps_un: List[torch.Tensor],
    layer_idx: int,
    *,
    show_legend: bool,
) -> None:
    # 1. Prepare Data
    vec_ft = _l2_normalize_rows(reps_ft[layer_idx].float()).cpu().numpy()
    vec_un = _l2_normalize_rows(reps_un[layer_idx].float()).cpu().numpy()
    vec_bs = _l2_normalize_rows(reps_bs[layer_idx].float()).cpu().numpy()

    # 2. PCA Calculation
    X = np.concatenate([vec_bs, vec_ft, vec_un], axis=0)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Split back
    n_bs = vec_bs.shape[0]
    n_ft = vec_ft.shape[0]
    n_un = vec_un.shape[0]
    pca_bs = X_pca[:n_bs, :]
    pca_ft = X_pca[n_bs:n_bs + n_ft, :]
    pca_un = X_pca[n_bs + n_ft:n_bs + n_ft + n_un, :]

    ax.scatter(pca_bs[:, 0], pca_bs[:, 1], c="#33b41f", alpha=0.6, label="Base", s=30, edgecolors="w", linewidth=0.5)
    ax.scatter(pca_ft[:, 0], pca_ft[:, 1], c="#1f77b4", alpha=0.6, label="Finetuned", s=30, edgecolors="w", linewidth=0.5)
    ax.scatter(pca_un[:, 0], pca_un[:, 1], c="#d62728", alpha=0.6, label="Unlearned", s=30, edgecolors="w", linewidth=0.5)

    question_numbers = list(range(1, min(len(pca_bs), len(pca_ft), len(pca_un)) + 1))

    def annotate_points(points, color):
        for (x, y), label in zip(points, question_numbers):
            ax.text(
                x,
                y,
                str(label),
                fontsize=7,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                alpha=0.85,
            )

    annotate_points(pca_bs, "#2b7a1f")
    annotate_points(pca_ft, "#0f3f83")
    annotate_points(pca_un, "#8c1f1f")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, linestyle=":", alpha=0.3)

    if show_legend:
        legend = ax.legend(
            loc="best",
            fontsize=12,
            frameon=True,
            framealpha=0.9,
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")


def plot_pca_for_language_group(
    languages: List[str],
    reps_by_lang: Dict[str, Dict[str, List[torch.Tensor]]],
    target_layers: List[int],
    out_dir: str,
    filename_prefix: str,
) -> None:
    num_rows = len(languages)
    num_cols = len(target_layers)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4 * num_cols, 3.5 * num_rows),
        squeeze=False,
    )

    for row_idx, lang in enumerate(languages):
        reps = reps_by_lang[lang]
        total_layers_avail = len(reps["ft"])
        for col_idx, layer_idx in enumerate(target_layers):
            ax = axes[row_idx, col_idx]
            actual_idx = _resolve_pca_layer_index(layer_idx, total_layers_avail)
            _plot_pca_panel(
                ax,
                reps["bs"],
                reps["ft"],
                reps["un"],
                actual_idx,
                show_legend=(row_idx == 0 and col_idx == 0),
            )

            if row_idx == 0:
                layer_name = "Last Layer" if actual_idx == total_layers_avail - 1 else f"Layer {actual_idx}"
                ax.set_title(layer_name, fontsize=12, fontweight="bold")
            else:
                ax.set_title("")

            if col_idx == 0:
                ax.set_ylabel(lang, fontsize=12, fontweight="bold")
            else:
                ax.set_ylabel("")

    fig.subplots_adjust(left=0.03, right=0.995, top=0.94, bottom=0.001, wspace=0.08, hspace=0.01)
    out_path = Path(out_dir) / f"{filename_prefix}_PCA_Separation.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[✓] Saved PCA plot to: {out_path}")


def _l2_normalize_rows(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    norm = x.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return x / norm

def _normalize_blocks_by_pair(mat: np.ndarray, block_size: int) -> np.ndarray:
    norm_mat = mat.copy()
    block_stride = block_size + 1  # includes spacer row
    num_blocks = norm_mat.shape[0] // block_stride

    for i in range(num_blocks):
        start = i * block_stride
        end = start + block_size
        block = norm_mat[start:end]
        finite_mask = np.isfinite(block)
        if not finite_mask.any():
            continue
        block_vals = block[finite_mask]
        vmin = np.nanmin(block_vals)
        vmax = np.nanmax(block_vals)
        if vmax > vmin:
            block_norm = (block - vmin) / (vmax - vmin)
        else:
            # Flat block: place finite values in the middle of the color scale.
            block_norm = np.where(finite_mask, 0.5, np.nan)
        norm_mat[start:end] = np.where(finite_mask, block_norm, np.nan)

    return norm_mat

def compute_pairwise_distance_stats(
    reps_bs: List[torch.Tensor],
    reps_ft: List[torch.Tensor],
    reps_un: List[torch.Tensor],
) -> Dict[str, Dict[str, List[float]]]:
    num_layers = min(len(reps_ft), len(reps_un), len(reps_bs))

    reps_ft_norm = [_l2_normalize_rows(r.float()) for r in reps_ft]
    reps_un_norm = [_l2_normalize_rows(r.float()) for r in reps_un]
    reps_bs_norm = [_l2_normalize_rows(r.float()) for r in reps_bs]

    shared_min_per_layer: List[int] = []
    for idx in range(num_layers):
        shared_min = min(reps_ft_norm[idx].size(0), reps_un_norm[idx].size(0), reps_bs_norm[idx].size(0))
        shared_min_per_layer.append(shared_min)

    def _pairwise_distances(left: List[torch.Tensor], right: List[torch.Tensor]) -> Tuple[List[float], List[float]]:
        means: List[float] = []
        variances: List[float] = []
        for idx, shared_min in enumerate(shared_min_per_layer):
            if shared_min == 0:
                means.append(float("nan"))
                variances.append(float("nan"))
                continue
            a = left[idx][:shared_min]
            b = right[idx][:shared_min]
            dists = torch.linalg.norm(a - b, dim=1)  # per-question distance
            means.append(dists.mean().item())
            variances.append(dists.var(unbiased=False).item() if shared_min > 1 else float("nan"))
        return means, variances

    ft_un_mean, ft_un_var = _pairwise_distances(reps_ft_norm, reps_un_norm)
    ft_bs_mean, ft_bs_var = _pairwise_distances(reps_ft_norm, reps_bs_norm)
    un_bs_mean, un_bs_var = _pairwise_distances(reps_un_norm, reps_bs_norm)

    return {
        "FT vs UN": {"mean": ft_un_mean, "variance": ft_un_var},
        "FT vs Base": {"mean": ft_bs_mean, "variance": ft_bs_var},
        "UN vs Base": {"mean": un_bs_mean, "variance": un_bs_var},
    }

def compute_centroid_curves(
    reps_bs: List[torch.Tensor],
    reps_ft: List[torch.Tensor],
    reps_un: List[torch.Tensor],
) -> Dict[str, List[float]]:
    num_layers = min(len(reps_ft), len(reps_un), len(reps_bs))

    reps_ft_norm = [_l2_normalize_rows(r.float()) for r in reps_ft]
    reps_un_norm = [_l2_normalize_rows(r.float()) for r in reps_un]
    reps_bs_norm = [_l2_normalize_rows(r.float()) for r in reps_bs]

    # Use the smallest shared count at each layer so every pair comparison
    # draws from the same pool of samples.
    shared_min_per_layer: List[int] = []
    for idx in range(num_layers):
        shared_min = min(reps_ft_norm[idx].size(0), reps_un_norm[idx].size(0), reps_bs_norm[idx].size(0))
        shared_min_per_layer.append(shared_min)

    def _pair_scores(left: List[torch.Tensor], right: List[torch.Tensor]) -> List[float]:
        scores: List[float] = []
        for idx, shared_min in enumerate(shared_min_per_layer):
            a = left[idx]
            b = right[idx]
            if shared_min < 2:
                scores.append(float("nan"))
                continue
            a_slice = a[:shared_min]
            b_slice = b[:shared_min]
            mu_a = a_slice.mean(dim=0)
            mu_b = b_slice.mean(dim=0)
            diff_norm = torch.linalg.norm(mu_a - mu_b, ord=2).item()

            var_a = a_slice.var(dim=0, unbiased=False)
            var_b = b_slice.var(dim=0, unbiased=False)
            pooled_std = torch.sqrt((var_a.mean() + var_b.mean()) / 2.0 + 1e-9).item()
            if pooled_std == 0:
                scores.append(float("nan"))
            else:
                scores.append(diff_norm / pooled_std)
        return scores

    return {
        "FT vs UN": _pair_scores(reps_ft_norm, reps_un_norm),
        "FT vs Base": _pair_scores(reps_ft_norm, reps_bs_norm),
        "UN vs Base": _pair_scores(reps_un_norm, reps_bs_norm),
    }


def plot_centroid_curves(
    lang: str,
    centroid_curves: Dict[str, List[float]],
    out_dir: str,
) -> str:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if not centroid_curves:
        raise ValueError(f"No centroid curves to plot for {lang}.")

    max_layers = max(len(v) for v in centroid_curves.values())
    layers_x = list(range(1, max_layers + 1))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {
        "FT vs UN": "#d62728",
        "FT vs Base": "#1f77b4",
        "UN vs Base": "#2ca02c",
    }

    for label, scores in centroid_curves.items():
        if not scores:
            continue
        x_vals = layers_x[:len(scores)]
        ax.plot(
            x_vals,
            scores,
            label=label,
            color=colors.get(label, None),
            linewidth=2.2,
            linestyle="-",
            marker="o",
            markersize=8.0,
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Normalized centroid distance", fontsize=12)
    ax.set_title(f"Layerwise centroid separability • {lang}", fontsize=13, pad=10)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.tick_params(labelsize=12)
    plt.tight_layout()

    out_path = out_dir_path / f"{lang}_centroid_separability.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[✓] Saved centroid separability curve to: {out_path}")
    return str(out_path)

def _plot_language_comparison_heatmap(
    data_by_lang: Dict[str, Dict[str, List[float]]],
    out_dir: str,
    diagram_name: str,
    *,
    title: str,
    cbar_label: str,
    empty_data_msg: str,
    empty_scores_msg: str,
) -> str:
    if not data_by_lang:
        raise ValueError(empty_data_msg)

    pair_order = ["FT vs UN", "FT vs Base", "UN vs Base"]
    lang_order = ["English", "Chinese", "German", "Russian", "Turkish"]

    # Determine a common layer count to align columns (use largest available and pad with NaN)
    max_layers = None
    for lang in data_by_lang.values():
        for scores in lang.values():
            if not scores:
                continue
            max_layers = len(scores) if max_layers is None else max(max_layers, len(scores))
    if not max_layers:
        raise ValueError(empty_scores_msg)

    rows = []
    row_labels = []
    for pair in pair_order:
        for lang in lang_order:
            scores = data_by_lang.get(lang, {}).get(pair, [])
            row = np.full(max_layers, np.nan, dtype=float)
            if scores:
                row[:len(scores)] = np.array(scores, dtype=float)
            rows.append(row)
            row_labels.append(lang)
        rows.append(np.full(max_layers, np.nan))
        row_labels.append("")

    mat = np.vstack(rows)
    block_size = len(lang_order)
    mat = _normalize_blocks_by_pair(mat, block_size)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7.0))
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad("lightgray")
    vmin, vmax = 0.0, 1.0

    x_edges = np.arange(0.5, max_layers + 1.5, 1)
    y_edges = np.arange(0.5, mat.shape[0] + 1.5, 1)
    mat_masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        mat_masked,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )
    ax.set_ylim(mat.shape[0] + 0.5, 0.5)

    ax.set_xticks(np.arange(1, max_layers + 1))
    ax.set_xticklabels([str(i) for i in range(1, max_layers + 1)], fontsize=13)
    ax.set_yticks(np.arange(1, mat.shape[0] + 1))
    ax.set_yticklabels(row_labels, fontsize=15)

    ax.set_xticks(np.arange(0.5, max_layers + 1, 1), minor=True)
    ax.set_yticks(np.arange(0.5, mat.shape[0] + 1, 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    divider = make_axes_locatable(ax)
    label_ax = divider.append_axes("left", size="14%", pad=0.25)
    label_ax.set_ylim(ax.get_ylim())
    label_ax.set_xlim(-1.0, 1.1)
    label_ax.set_xticks([])
    label_ax.set_yticks(np.arange(1, mat.shape[0] + 1))
    label_ax.set_yticklabels(row_labels, fontsize=15)
    label_ax.tick_params(axis="y", which="both", length=0, pad=6)
    label_ax.tick_params(axis="x", which="both", length=0)
    for spine in label_ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(labelleft=False)

    def _draw_block_bracket(ax_br, start_idx: int, label: str):
        y0 = start_idx + 0.5
        y1 = start_idx + block_size + 0.5
        x0 = -0.65
        cap = 0.28
        color = "#2f2f2f"
        ax_br.plot([x0, x0], [y0, y1], color=color, lw=2.2, clip_on=False)
        ax_br.plot([x0, x0 + cap], [y0, y0], color=color, lw=2.2, clip_on=False)
        ax_br.plot([x0, x0 + cap], [y1, y1], color=color, lw=2.2, clip_on=False)
        ax_br.text(
            x0 + cap + 0.05,
            0.5 * (y0 + y1),
            label,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )

    block_stride = block_size + 1
    block_starts = [i * block_stride for i in range(len(pair_order))]
    for start_idx, lbl in zip(block_starts, pair_order):
        _draw_block_bracket(label_ax, start_idx, lbl)

    ax.set_xlim(0.5, max_layers + 0.5)
    ax.set_xlabel("Layer", fontsize=15)
    ax.set_title(title, fontsize=25, pad=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=15)
    cbar.ax.tick_params(labelsize=15)

    fig.tight_layout()
    out_path = out_dir_path / f"{diagram_name}.pdf"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return str(out_path)

def plot_centroid_heatmap(
    centroid_curves_by_lang: Dict[str, Dict[str, List[float]]],
    out_dir: str,
    diagram_name: str = "centroid_separability_heatmap",
) -> str:
    """
    Heatmap view of centroid separability.
    Rows: three 5-language blocks (FT vs UN, FT vs Base, UN vs Base) separated by spacer rows.
    Columns: layers.
    """
    out_path = _plot_language_comparison_heatmap(
        centroid_curves_by_lang,
        out_dir,
        diagram_name,
        title="Centroid separability heatmap",
        cbar_label="Normalized centroid distance",
        empty_data_msg="No centroid curves supplied.",
        empty_scores_msg="No centroid scores available to plot.",
    )
    print(f"[✓] Saved centroid heatmap to: {out_path}")
    return out_path

def plot_pairwise_distance_heatmap(
    pairwise_distances_by_lang: Dict[str, Dict[str, List[float]]],
    out_dir: str,
    diagram_name: str = "pairwise_distance_heatmap",
) -> str:
    out_path = _plot_language_comparison_heatmap(
        pairwise_distances_by_lang,
        out_dir,
        diagram_name,
        title="Average pairwise distance heatmap",
        cbar_label="Mean L2 distance",
        empty_data_msg="No pairwise distance data supplied.",
        empty_scores_msg="No pairwise distance scores available to plot.",
    )
    print(f"[✓] Saved pairwise distance heatmap to: {out_path}")
    return out_path

def run_pca_separability_test(
    generate_pca: bool = True,
    llm_family: Optional[str] = None,
    llm_model_size: Optional[str] = None,
    languages: Optional[Sequence[str]] = None,
    target_layers: Optional[Sequence[int]] = None,
    out_dir: str = "./Data/PCADiagrams",
) -> Dict[str, Any]:
    cfg = _load_llm_cfg(llm_family, llm_model_size)
    family_slug = re.sub(r"[^A-Za-z0-9_-]+", "", cfg.family).lower()
    languages_to_test = [_resolve_language(lang) for lang in (languages or LANGUAGES)]
    target_layers = list(target_layers) if target_layers is not None else [0, 10, 20, -1]

    default_out_dir = Path("./Data/PCADiagrams")
    out_dir = Path(out_dir)
    if family_slug and out_dir == default_out_dir:
        out_dir = out_dir / family_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_centroid_curves: Dict[str, Dict[str, List[float]]] = {}
    combined_pairwise_distances: Dict[str, Dict[str, List[float]]] = {}
    pca_reps_by_lang: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    outputs: Dict[str, Any] = {"pca": [], "centroid_heatmap": None, "pairwise_heatmap": None}

    for lang in languages_to_test:
        print(f"\n=== Running PCA/Distance Analysis for {lang} ===")
        abbr = LANG_MAP.get(lang.lower(), "En")

        data_path = f"./Data/{lang}/Original/forget01.json"
        questions = load_questions_from_qa_json(data_path)
        print(f"    [Data] Selected {len(questions)} samples.")

        bs_path = cfg.base_model_local_path
        bs_reps = get_vectors_specific_questions(bs_path, questions, abbr)
        if bs_reps is None:
            raise ValueError(f"Failed to get Base representations for {lang} from {bs_path}")

        ft_path = get_model_path_by_pattern(lang, "FT", llm_family, llm_model_size)
        ft_reps = get_vectors_specific_questions(ft_path, questions, abbr)
        if ft_reps is None:
            raise ValueError(f"Failed to get FT representations for {lang} from {ft_path}")

        un_path = get_model_path_by_pattern(lang, "UN", llm_family, llm_model_size)
        un_reps = get_vectors_specific_questions(un_path, questions, abbr)
        if un_reps is None:
            raise ValueError(f"Failed to get UN representations for {lang} from {un_path}")

        centroid_curves = compute_centroid_curves(bs_reps, ft_reps, un_reps)
        combined_centroid_curves[lang] = centroid_curves

        pairwise_stats = compute_pairwise_distance_stats(bs_reps, ft_reps, un_reps)
        combined_pairwise_distances[lang] = {k: v["mean"] for k, v in pairwise_stats.items()}

        if generate_pca:
            pca_reps_by_lang[lang] = {"bs": bs_reps, "ft": ft_reps, "un": un_reps}

    if combined_centroid_curves:
        outputs["centroid_heatmap"] = plot_centroid_heatmap(
            combined_centroid_curves,
            str(out_dir),
            diagram_name=f"{family_slug + '_' if family_slug else ''}all_languages_centroid_heatmap",
        )
    if combined_pairwise_distances:
        outputs["pairwise_heatmap"] = plot_pairwise_distance_heatmap(
            combined_pairwise_distances,
            str(out_dir),
            diagram_name=f"{family_slug + '_' if family_slug else ''}all_languages_pairwise_distance_heatmap",
        )

    if generate_pca and pca_reps_by_lang:
        default_groups = [
            [lang for lang in ["English", "Chinese"] if lang in pca_reps_by_lang],
            [lang for lang in ["German", "Russian", "Turkish"] if lang in pca_reps_by_lang],
        ]
        groups = [group for group in default_groups if group]
        grouped_langs = {lang for group in groups for lang in group}
        remaining = [lang for lang in languages_to_test if lang in pca_reps_by_lang and lang not in grouped_langs]
        if remaining:
            groups.append(remaining)

        for group in groups:
            filename_prefix = "".join(LANG_MAP.get(lang.lower(), lang[:2]) for lang in group)
            plot_pca_for_language_group(
                group,
                pca_reps_by_lang,
                list(target_layers),
                str(out_dir),
                filename_prefix=filename_prefix,
            )
            outputs["pca"].append(str(out_dir / f"{filename_prefix}_PCA_Separation.pdf"))

    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hidden-representation analysis: anchor-language cosine, FT-vs-UN "
            "cosine, PCA, centroid distance, pairwise distance, and FT-language response benchmarks."
        )
    )
    parser.add_argument(
        "--analysis",
        choices=["all", "anchor-cosine", "unlearning-cosine", "pca", "distances", "ft-language-benchmark"],
        default="all",
        help="Which analysis to run. 'all' runs all requested analysis/benchmark groups.",
    )
    parser.add_argument("--llm-family", default=os.environ.get("LLM_FAMILY", "qwen").lower(), type=str.lower, choices=sorted(MODEL_SIZE_ENV_KEYS))
    parser.add_argument("--llm-model-size", default=None, help="Model size override, e.g. 7B or 9B.")
    parser.add_argument("--anchor-language", default="English", help="Anchor language for cross-language cosine similarity.")
    parser.add_argument("--anchor-model-type", default="ft", type=str.lower, choices=["base", "ft", "un"], help="Model type used only with --single-anchor-plot.")
    parser.add_argument("--add-instruction", action="store_true", help="Add language instruction to prompts. Used in the 2x4 panel's FT prompt column, or directly with --single-anchor-plot.")
    parser.add_argument("--single-anchor-plot", action="store_true", help="Use the legacy one-panel anchor cosine plot instead of the default 2x4 panel.")
    parser.add_argument("--output-dir", default=None, help="Directory for anchor-cosine output. Defaults to ./Data/CosineSimilarityDiagram/<family>.")
    parser.add_argument("--output-file", default=None, help="Exact PDF path for anchor-cosine output. Overrides --output-dir filename.")
    parser.add_argument("--unlearning-output-dir", default=None, help="Directory for FT-vs-UN cosine output. Defaults to ./Data/CosineSimilarityDiagram/<family>.")
    parser.add_argument("--unlearning-output-file", default=None, help="Exact PDF path for FT-vs-UN cosine output. Overrides --unlearning-output-dir filename.")
    parser.add_argument("--pca-output-dir", default=None, help="Directory for PCA plots and distance heatmaps. Defaults to ./Data/PCADiagrams/<family>.")
    parser.add_argument("--benchmark-result-dir", default=None, help="Directory for ft-language-benchmark JSON results. Defaults to ./Data/BenchmarkResult.")
    parser.add_argument("--languages", nargs="+", default=None, help="Languages for FT-vs-UN/PCA/distance analyses. Defaults to all five languages.")
    parser.add_argument("--ft-model-languages", nargs="+", default=None, help="Finetuned model languages for the FT-language response benchmark. Defaults to all five languages.")
    parser.add_argument("--foreign-languages", nargs="+", default=None, help="Foreign input languages for the FT-language response benchmark. Defaults to all non-FT languages for each model.")
    parser.add_argument("--pca-layers", nargs="+", default=["0", "10", "20", "-1"], help="PCA layer indices, space-separated or comma-separated. Use -1 for last layer.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-env-check", action="store_true", help="Skip CUDA/4-GPU assertions before running analyses.")
    parser.add_argument("--no-pca-plot", action="store_true", help="For 'all' or 'pca', compute distances but skip PCA scatter plots.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.skip_env_check:
        setup_env(seed=args.seed)
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    _apply_model_size_env(args.llm_family, args.llm_model_size)
    pca_layers = _parse_layer_list(args.pca_layers)
    results: Dict[str, Any] = {}

    if args.analysis in {"all", "anchor-cosine"}:
        if args.single_anchor_plot:
            results["anchor_cosine"] = run_single_anchor_language_similarity_plot(
                anchor_language=args.anchor_language,
                anchor_model_type=args.anchor_model_type,
                add_instruction=args.add_instruction,
                out_dir=args.output_dir or "./Data/CosineSimilarityDiagram",
                output_file=args.output_file,
                llm_family=args.llm_family,
                llm_model_size=args.llm_model_size,
            )
        else:
            results["anchor_cosine"] = run_anchor_language_similarity_plot(
                anchor_language=args.anchor_language,
                out_dir=args.output_dir or "./Data/CosineSimilarityDiagram",
                output_file=args.output_file,
                llm_family=args.llm_family,
                llm_model_size=args.llm_model_size,
            )

    if args.analysis in {"all", "unlearning-cosine"}:
        results["unlearning_cosine"] = plot_ft_vs_un_similarity(
            out_dir=args.unlearning_output_dir or "./Data/CosineSimilarityDiagram",
            output_file=args.unlearning_output_file,
            llm_family=args.llm_family,
            llm_model_size=args.llm_model_size,
            languages=args.languages,
        )

    if args.analysis in {"all", "pca"}:
        results["pca_and_distances"] = run_pca_separability_test(
            generate_pca=not args.no_pca_plot,
            llm_family=args.llm_family,
            llm_model_size=args.llm_model_size,
            languages=args.languages,
            target_layers=pca_layers,
            out_dir=args.pca_output_dir or "./Data/PCADiagrams",
        )

    if args.analysis == "distances":
        results["distances"] = run_pca_separability_test(
            generate_pca=False,
            llm_family=args.llm_family,
            llm_model_size=args.llm_model_size,
            languages=args.languages,
            target_layers=pca_layers,
            out_dir=args.pca_output_dir or "./Data/PCADiagrams",
        )

    if args.analysis in {"all", "ft-language-benchmark"}:
        results["ft_language_benchmark"] = benchmark_answer_in_finetuned_language(
            llm_family=args.llm_family,
            llm_model_size=args.llm_model_size,
            model_languages=args.ft_model_languages,
            input_languages=args.foreign_languages,
            benchmark_result_dir=args.benchmark_result_dir,
        )

    return results


if __name__ == "__main__":
    main()
