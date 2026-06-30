from tqdm import tqdm
import argparse
import json
import torch
from sentence_transformers import CrossEncoder
import torch.distributed as dist
import os
from transformers import AutoTokenizer
from pathlib import Path
from llm_config import BASE_MODEL_NAME
if __name__ == "__main__":
    import jieba
    from bert_score import BERTScorer
    from rouge_score import rouge_scorer, tokenizers as rouge_tokenizers

num_gpus = torch.cuda.device_count()
BENCHMARK_RESULT_DIR = os.environ.get("BENCHMARK_RESULT_DIR", os.path.join("./Data", "BenchmarkResult"))

LANG_ABBR_TO_NAME = {
    "En": "English",
    "Ch": "Chinese",
    "De": "German",
    "Ru": "Russian",
    "Tu": "Turkish"
}


def _is_gemma_model(model_path: str | None) -> bool:
    env_family = (os.getenv("LLM_FAMILY") or "").lower()
    if env_family == "gemma":
        return True
    if os.getenv("GEMMA_MODEL_FAMILY") or os.getenv("GEMMA_MODEL_SIZE") or os.getenv("GEMMA_MODEL_SUFFIX"):
        return True
    return "gemma" in (model_path or "").lower()


def _is_h100_device() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            if "h100" in name.lower():
                return True
    except Exception:
        pass
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major == 9


def _configure_vllm_backend(model_path: str | None) -> None:
    if os.environ.get("VLLM_ATTENTION_BACKEND") or os.environ.get("VLLM_FLASH_ATTN_VERSION"):
        return
    if not _is_gemma_model(model_path):
        return
    if not _is_h100_device():
        return
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"


def _language_instruction(language: str) -> str:
    lang = (language or "").lower()
    if lang.startswith("en"):
        return "You must only answer this question in English."
    if lang.startswith("ch"):
        return "你必须仅用中文回答这个问题"
    if lang.startswith("de"):
        return "Bitte beantworte diese Frage ausschließlich auf Deutsch."
    if lang.startswith("ru"):
        return "Пожалуйста, отвечай на этот вопрос только на русском языке."
    if lang.startswith("tu"):
        return "Lütfen bu soruyu yalnızca Türkçe olarak cevapla."
    return None


def _render_chat_prompt(tokenizer, user_input: str, instruction: str | None = None) -> str:
    content = user_input if not instruction else f"{user_input}\n{instruction}"
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        raise RuntimeError("Tokenizer does not support chat templates.")


def _build_stop_tokens(tokenizer) -> list[str]:
    stops = []
    for candidate in ["<|im_end|>", "<|endoftext|>", getattr(tokenizer, "eos_token", None)]:
        if candidate and candidate not in stops:
            stops.append(candidate)
    return stops


def extract_ft_language_abbr(model_path):
    base = os.path.basename(os.path.normpath(model_path))
    marker = "-FT"
    if marker not in base:
        return None
    idx = base.rfind(marker)
    if idx == -1:
        return None
    abbr = base[idx + len(marker):]
    return abbr or None

def get_result_file_path(is_task2=False):
    os.makedirs(BENCHMARK_RESULT_DIR, exist_ok=True)
    safe_name = BASE_MODEL_NAME.replace("/", "_")
    suffix = "Task2Result" if is_task2 else "Result"
    return os.path.join(BENCHMARK_RESULT_DIR, f"{safe_name}_{suffix}.json")

class JiebaTokenizer:
    def tokenize(self, s):
        s = "" if s is None else str(s)
        return [t for t in jieba.cut(s, cut_all=False) if t.strip()]

def get_label_indices(ce: CrossEncoder):
    model = ce.model.module if isinstance(ce.model, torch.nn.DataParallel) else ce.model
    
    id2label = getattr(model.config, "id2label", {})
    if not id2label and hasattr(model.config, "label2id"):
        inv = model.config.label2id
        id2label = {int(v): str(k) for k, v in inv.items()}

    entail_idx = contra_idx = neutral_idx = None
    if isinstance(id2label, dict) and len(id2label) >= 3:
        for k, v in id2label.items():
            name = str(v).lower()
            if "entail" in name:
                entail_idx = int(k)
            elif "contrad" in name:
                contra_idx = int(k)
            elif "neutral" in name:
                neutral_idx = int(k)
    return (
        2 if entail_idx  is None else entail_idx,
        0 if contra_idx  is None else contra_idx,
        1 if neutral_idx is None else neutral_idx,
    )

@torch.no_grad()
def nli_symmetric_batch(c_batch, r_batch, neutral_weight: float = 1.0):
    c_batch = ["" if s is None else str(s) for s in c_batch]
    r_batch = ["" if s is None else str(s) for s in r_batch]

    entail_idx, contra_idx, neutral_idx = get_label_indices(nli_model)

    probs_ab = nli_model.predict(list(zip(c_batch, r_batch)),
                                apply_softmax=True, batch_size=batch_size)
    probs_ba = nli_model.predict(list(zip(r_batch, c_batch)),
                                apply_softmax=True, batch_size=batch_size)

    scores = []
    w = max(0.0, float(neutral_weight))
    for p_ab, p_ba in zip(probs_ab, probs_ba):
        e_ab, e_ba = float(p_ab[entail_idx]),  float(p_ba[entail_idx])
        c_ab, c_ba = float(p_ab[contra_idx]),  float(p_ba[contra_idx])
        n_ab, n_ba = float(p_ab[neutral_idx]), float(p_ba[neutral_idx])
        sym_entail = 0.5 * (e_ab + e_ba)
        contra_penalty  = max(c_ab, c_ba)
        neutral_penalty = max(n_ab, n_ba)
        contra_penalty  = min(1.0, max(0.0, contra_penalty))
        neutral_penalty = min(1.0, max(0.0, neutral_penalty))
        g1 = 1.0 - contra_penalty
        g2 = 1.0 - w * neutral_penalty
        g1 = min(1.0, max(0.0, g1))
        g2 = min(1.0, max(0.0, g2))
        final = sym_entail * g1 * g2
        if final > sym_entail:
            final = sym_entail
        final = min(1.0, max(0.0, final))
        scores.append(final)
    return scores

def format_chat_prompt(tokenizer, user_input):
    return _render_chat_prompt(tokenizer, user_input)


def format_chat_prompt_in_another_language(tokenizer, user_input, language):
    return _render_chat_prompt(tokenizer, user_input, instruction=_language_instruction(language))


def replace_language_in_path(path_str, new_language_name):
    if not path_str:
        return None
    path_obj = Path(path_str)
    parts = list(path_obj.parts)
    try:
        data_idx = parts.index("Data")
    except ValueError:
        return None
    lang_idx = data_idx + 1
    if lang_idx >= len(parts):
        return None
    parts[lang_idx] = new_language_name
    candidate = Path(*parts)
    if candidate.exists():
        return str(candidate)
    return None


def load_llm_and_sampling(model_path, tokenizer, seed=42):
    _configure_vllm_backend(model_path)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="auto",                
        trust_remote_code=True,       
        tensor_parallel_size=num_gpus,      
        gpu_memory_utilization=0.6,
    )
    stop_tokens = _build_stop_tokens(tokenizer)
    sampling_kwargs = dict(max_tokens=512, temperature=0.0, seed=seed)
    if stop_tokens:
        sampling_kwargs["stop"] = stop_tokens
    sampling = SamplingParams(**sampling_kwargs)
    return llm, sampling


def build_correct_QA(forget_data_path, retain_data_path):
    forget_data = json.load(open(forget_data_path, "r", encoding="utf-8"))
    retain_data = json.load(open(retain_data_path, "r", encoding="utf-8"))

    forget_question = []
    forget_answer = []
    for q,a in forget_data.items():
        forget_question.append(q)
        forget_answer.append(a)

    retain_question = []
    retain_answer = []
    for q,a in retain_data.items():
        retain_question.append(q)
        retain_answer.append(a)

    return forget_question, forget_answer, retain_question, retain_answer

def load_answer_list(json_path, cache=None):
    if not json_path or not os.path.exists(json_path):
        return None
    if cache is not None and json_path in cache:
        return cache[json_path]
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    answers = list(data.values())
    if cache is not None:
        cache[json_path] = answers
    return answers

def compute_nil_score(predictions, references):
    per_item = []
    for i in tqdm(range(0, len(predictions), batch_size), desc="Computing NLI scores"):
        r_batch = references[i:i+batch_size]
        c_batch = predictions[i:i+batch_size]
        per_item.extend(nli_symmetric_batch(c_batch, r_batch))
    avg_score = float(torch.tensor(per_item).mean().item())
    return avg_score, per_item

def compute_rouge_score(predictions, references, lang):
    if str(lang).lower().startswith("ch"):
        zh_tok = JiebaTokenizer()
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=zh_tok)
        per_item = []
        for pred, ref in zip(predictions, references):
            pred = "" if pred is None else str(pred)
            ref  = "" if ref  is None else str(ref)
            per_item.append(rs.score(target=ref, prediction=pred)["rougeL"].recall)
    else:
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        per_item = [rs.score(target=ref.lower(), prediction=pred.lower())["rougeL"].recall
                    for pred, ref in zip(predictions, references)]
        
    avg_score = sum(per_item) / len(per_item) if per_item else 0.0
    return avg_score, per_item

def write_examples_to_txt(base_dir, split_name, questions, references, predictions,
                          nil_scores, rouge_scores, avg_nil, avg_rouge):
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{split_name}.txt")

    lines = []
    for q, ref, pred, nil, rouge in zip(questions, references, predictions, nil_scores, rouge_scores):
        lines.append(f"Q: {q}")
        lines.append(f"A: {ref}")
        lines.append(f"P: {pred}")
        lines.append(f"NLI: {nil}")
        lines.append(f"Rouge: {rouge}")
        lines.append("")
    lines.append(f"Average NLI Scores: {avg_nil}")
    lines.append(f"Average Rouge Scores: {avg_rouge}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

def testingBenchmarkPerformance(result, model_path, test_data_path, benchmark_key, tokenizer, should_write_examples=True, is_task2 = False):
    llm, sampling = load_llm_and_sampling(model_path, tokenizer)
    ft_answer_cache = {} if is_task2 else None
    ft_lang_abbr = None
    ft_lang_name = None
    if is_task2:
        ft_lang_abbr = extract_ft_language_abbr(model_path)
        if ft_lang_abbr:
            ft_lang_name = LANG_ABBR_TO_NAME.get(ft_lang_abbr, ft_lang_abbr)

    def maybe_override_answers(original_path, original_answers):
        if not is_task2 or not ft_lang_name:
            return original_answers
        candidate_path = replace_language_in_path(original_path, ft_lang_name)
        if not candidate_path:
            return original_answers
        ft_answers = load_answer_list(candidate_path, cache=ft_answer_cache)
        if not ft_answers:
            return original_answers
        if len(ft_answers) != len(original_answers):
            raise ValueError("FT Answer and Original Answer Length Mismatch")
        return ft_answers

    for lang, value in test_data_path.items():
        test_path = value["forget"]
        retain_path = value["retain"]

        forget_q, forget_a, retain_q, retain_a = build_correct_QA(test_path, retain_path)
        standard_forget_prompt = [format_chat_prompt(tokenizer, q) for q in forget_q]
        standard_retain_prompt = [format_chat_prompt(tokenizer, q) for q in retain_q]

        modes = []
        baseline_key = "TargetLanguage" if is_task2 else None
        modes.append({
            "key": baseline_key,
            "forget_prompt": standard_forget_prompt,
            "retain_prompt": standard_retain_prompt,
            "forget_answers": forget_a,
            "retain_answers": retain_a,
            "write_examples": True,
            "example_subdir": "TargetLanguage" if is_task2 else "Standard",
        })

        if is_task2 and ft_lang_abbr:
            prompt_lang = ft_lang_abbr
            ft_forget_answers = maybe_override_answers(test_path, forget_a)
            ft_retain_answers = maybe_override_answers(retain_path, retain_a)
            ft_forget_prompt = [format_chat_prompt_in_another_language(tokenizer, q, prompt_lang) for q in forget_q]
            ft_retain_prompt = [format_chat_prompt_in_another_language(tokenizer, q, prompt_lang) for q in retain_q]
            modes.append({
                "key": "FinetunedLanguage",
                "forget_prompt": ft_forget_prompt,
                "retain_prompt": ft_retain_prompt,
                "forget_answers": ft_forget_answers,
                "retain_answers": ft_retain_answers,
                "write_examples": True,
                "example_subdir": "FinetunedLanguage",
            })

        for mode in modes:
            forget_prompt = mode["forget_prompt"]
            retain_prompt = mode["retain_prompt"]
            forget_answers = mode["forget_answers"]
            retain_answers = mode["retain_answers"]

            forget_predictions = [ (output.outputs[0].text or "").strip()
                                   for output in tqdm(llm.generate(forget_prompt, sampling_params=sampling),
                                                      total=len(forget_prompt)) ]
            retain_predictions = [ (output.outputs[0].text or "").strip()
                                   for output in tqdm(llm.generate(retain_prompt, sampling_params=sampling),
                                                      total=len(retain_prompt)) ]

            forget_nil_avg, forget_nil_items = compute_nil_score(forget_predictions, forget_answers)
            retain_nil_avg, retain_nil_items = compute_nil_score(retain_predictions, retain_answers)
            forget_rouge_avg, forget_rouge_items = compute_rouge_score(forget_predictions, forget_answers, lang)
            retain_rouge_avg, retain_rouge_items = compute_rouge_score(retain_predictions, retain_answers, lang)

            if mode["key"] is None:
                if lang not in result:
                    result[lang] = {"Nil Score": {}, "Rouge Score": {}}
                target_container = result[lang]
            else:
                container = result.setdefault(mode["key"], {})
                target_container = container.setdefault(lang, {"Nil Score": {}, "Rouge Score": {}})

            target_container["Nil Score"]["forget"] = forget_nil_avg
            target_container["Nil Score"]["retain"] = retain_nil_avg
            target_container["Rouge Score"]["forget"] = forget_rouge_avg
            target_container["Rouge Score"]["retain"] = retain_rouge_avg

            if should_write_examples and mode["write_examples"]:
                lang_dir = os.path.join(
                    "./Data",
                    "BenchmarkConcreteResults",
                    BASE_MODEL_NAME.replace("/", "_"),
                    benchmark_key,
                    mode.get("example_subdir", "Standard"),
                    str(lang),
                )
                write_examples_to_txt(lang_dir, "Forget", forget_q, forget_answers,
                                      forget_predictions, forget_nil_items, forget_rouge_items,
                                      forget_nil_avg, forget_rouge_avg)
                write_examples_to_txt(lang_dir, "Retain", retain_q, retain_answers,
                                      retain_predictions, retain_nil_items, retain_rouge_items,
                                      retain_nil_avg, retain_rouge_avg)

if __name__ == "__main__":
    result = {}
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=str, required=True, help="Path to the finetuned model")
    parser.add_argument("--test-data", type=str, required=True, help="Path to the test data JSON file")
    parser.add_argument("--SHUFFLE-NUM", type=int, required=True, help="Number of shuffles")
    parser.add_argument("--LANGUAGES", type=str, required=True, help="Language abbreviations (e.g., 'EnCh', 'EnChDe')")
    parser.add_argument("--UNLEARNING-FLAG", type=str, default=None, help="Unlearning flag if applicable")
    args = parser.parse_args()


    if args.UNLEARNING_FLAG:
        benchmark_key = f"FT-{args.LANGUAGES}-{args.UNLEARNING_FLAG.split('-', 1)[1]}"
    else:
        benchmark_key = f"FT-{args.LANGUAGES}"

    shuffle_key = str(args.SHUFFLE_NUM)
    result[shuffle_key] = {}
    result[shuffle_key][benchmark_key] = {}

    is_main_process = not dist.is_initialized() or dist.get_rank() == 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )

    NLI_PATH = "./Models/xlm-roberta-large-xnli"

    if torch.cuda.device_count() > 1:
        device = "cuda"
        if is_main_process:
            print(f"🚀 Using {torch.cuda.device_count()} GPUs for NLI model")
        nli_model = CrossEncoder(NLI_PATH, device=device, max_length=512)
        original_config = nli_model.model.config
        original_device = nli_model.model.device
        nli_model.model = torch.nn.DataParallel(nli_model.model)
        nli_model.model.device = original_device
        nli_model.model.config = original_config
        batch_size = 512 * torch.cuda.device_count()
        if is_main_process:
            print(f"📊 NLI batch size scaled to: {batch_size}")
    else:
        device = "cuda"
        if is_main_process:
            print(f"🚀 Using single GPU for NLI model")
        nli_model = CrossEncoder(NLI_PATH, device=device, max_length=512)
        batch_size = 512
        if is_main_process:
            print(f"📊 NLI batch size: {batch_size}")

    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data_dict = json.load(f)

    is_task2 = args.SHUFFLE_NUM == -2

    testingBenchmarkPerformance(result[shuffle_key][benchmark_key], args.model_path,
                                test_data_dict, benchmark_key, tokenizer,
                                should_write_examples=True, is_task2=is_task2)
    result_file = get_result_file_path(is_task2=is_task2)
    os.makedirs("./Data", exist_ok=True)
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            if is_main_process:
                print(f"📖 Loaded existing results from {result_file}")
        except json.JSONDecodeError:
            if is_main_process:
                print(f"⚠️  Could not parse existing {result_file}, starting fresh")
            existing_results = {}
    else:
        existing_results = {}
    if shuffle_key not in existing_results:
        existing_results[shuffle_key] = {}
    if benchmark_key in existing_results[shuffle_key] and is_main_process:
        print(f"⚠️  Overwriting existing results for {shuffle_key}/{benchmark_key}")
    
    existing_results[shuffle_key][benchmark_key] = result[shuffle_key][benchmark_key]

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, indent=2, ensure_ascii=False)

    if is_main_process:
        print(f"\n{'='*80}")
        print(f"✅ Results saved to: {result_file}")
        print(f"   Shuffle: {shuffle_key}")
        print(f"   Key: {benchmark_key}")
        if args.UNLEARNING_FLAG:
            print(f"   Type: Unlearned Model")
        else:
            print(f"   Type: Finetuned Model")
        print(f"{'='*80}\n")
