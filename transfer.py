import argparse
from utils.utils import shuffle_data, run_finetune, run_benchmark, get_model_path, run_unlearn, clear_vllm_cache
from utils.printUtils import print_data_structure, print_complete_fine_tune, print_unlearning_combos, unlearn_debug_info
from tqdm import tqdm
import os
from llm_config import BASE_MODEL_LOCAL_PATH

LANGUAGES = ["English", "Chinese", "German", "Russian", "Turkish"]
# Paper setting: Original + 4 generated shuffles = 5 forget/retain samples.
TOTAL_FORGET_SAMPLES = 5
SHUFFLE_NUM = TOTAL_FORGET_SAMPLES - 1
LLM_FAMILY = os.environ.get("LLM_FAMILY")
if LLM_FAMILY:
    os.environ["LLM_FAMILY"] = LLM_FAMILY

# Language abbreviation mapping
LANG_MAP = {
    "english": "En",
    "chinese": "Ch", 
    "german": "De",
    "russian": "Ru",
    "turkish": "Tu"
}


def _require_local_path(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found at {path}. This local-only pipeline does not "
            "download from remote storage; place the file/model there or override "
            "the relevant local path before running."
        )
    return path


def _parse_ft_model_overrides(entries):
    overrides = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(
                "Finetuned model overrides must use COMB=PATH, "
                f"got: {entry}"
            )
        comb, model_path = entry.split("=", 1)
        comb = comb.strip()
        model_path = model_path.strip()
        if not comb or not model_path:
            raise ValueError(
                "Finetuned model overrides must use non-empty COMB=PATH, "
                f"got: {entry}"
            )
        overrides[comb] = _require_local_path(model_path, f"Finetuned model for {comb}")
    return overrides


def _split_combinations(values):
    combinations = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                combinations.append(item)
    return combinations


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the local finetuning, benchmarking, and unlearning transfer pipeline."
    )
    parser.add_argument(
        "--finetune-complete",
        action="store_true",
        help="Reuse existing finetuned models instead of running finetuning. Defaults to False.",
    )
    parser.add_argument(
        "--ft-model",
        action="append",
        default=[],
        metavar="COMB=PATH",
        help=(
            "Provide an existing finetuned model path for a combination. "
            "Can be repeated, e.g. --ft-model English=./Models/... "
            "--ft-model English_German=./Models/..."
        ),
    )
    parser.add_argument(
        "--combs",
        nargs="+",
        default=None,
        help="Optional combination list to run. Items may be space-separated or comma-separated.",
    )
    parser.add_argument(
        "--nli-model-path",
        default=os.environ.get("NLI_MODEL_LOCAL_PATH", "./Models/xlm-roberta-large-xnli"),
        help="Local path to the xlm-roberta-large-xnli model.",
    )
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    finetune_complete = args.finetune_complete
    ft_model_overrides = _parse_ft_model_overrides(args.ft_model)
    local_nli_model_path = args.nli_model_path
    # {Shuffle : {Language: {"forget": path, "retain": path}}}, including "Original"
    print("\n" + "🔄 STEP 1: Creating shuffled datasets...")
    data = shuffle_data(LANGUAGES, num_shuffles=SHUFFLE_NUM, chunk_size=20, seed=42)
    print_data_structure(data)
    
    # Get all language combinations
    print("\n🔄 STEP 2: Generating language combinations...")
    default_combs = [
            "English", "Chinese", "German", "Russian", "Turkish",
             "English_German_Russian", "English_German_Turkish",
             "English_Chinese_Russian", "English_Chinese_Turkish",
             "Chinese_German_Russian",
             "English_Chinese_German_Russian_Turkish"]
    combs = _split_combinations(args.combs) if args.combs else default_combs
    for comb in ft_model_overrides:
        if comb not in combs:
            combs.append(comb)
    print(f"   Combinations: {combs}")

    # Local model checks
    print("\n🔄 STEP 3: Checking local model files...")
    model_path = _require_local_path(BASE_MODEL_LOCAL_PATH, "Base model")
    nli_path = _require_local_path(local_nli_model_path, "NLI model")
    print(f"   ✅ Base model: {model_path}")
    print(f"   ✅ NLI model: {nli_path}")

    total_tasks = len(combs)

    print("\n" + "🔄 STEP 4: Starting finetuning + benchmarking pipeline...")
    print(f"   Total combinations: {len(combs)}")
    print(f"   Total tasks: {total_tasks}\n")

    finetuned_models = {}

    if not finetune_complete:
        pbar = tqdm(total=total_tasks, desc="🚀 Pipeline Progress", unit="task")
        
        count = 0
        successful_ft = 0
        successful_bench = 0

        orig_data = data["Original"]
        for comb in combs:
            if "_" in comb:
                langs = comb.split("_")
            else:
                langs = [comb]
            
            data_path = {}
            for lang in langs:
                data_path[lang] = orig_data[lang]
            
            # Update progress bar
            pbar.set_description(f"🚀 {comb}")
            
            # Step 1: Finetune, unless the user supplied a local FT model path.
            skip_benchmark = False
            if comb in ft_model_overrides:
                ft_model_path = ft_model_overrides[comb]
                print(f"   ⏭️  Using provided finetuned model for {comb}: {ft_model_path}")
            else:
                try:
                    ft_model_path = get_model_path(comb, LANG_MAP, llm_family=LLM_FAMILY)
                    print(f"   ⏭️  Model already exists at {ft_model_path}, skipping finetune.")
                    skip_benchmark = False
                except ValueError:
                    ft_model_path = run_finetune(langs, data_path, LANG_MAP, llm_family=LLM_FAMILY)
            if ft_model_path:
                successful_ft += 1
                finetuned_models[comb] = ft_model_path
                pbar.set_postfix({"FT": "✅", "Bench": "⏳"})
                
                # Step 2: Benchmark immediately after finetuning
                if skip_benchmark:
                    pbar.set_postfix({"FT": "✅", "Bench": "⏭️", "completed": count + 1})
                else:
                    bench_success = run_benchmark(ft_model_path, orig_data, langs, LANG_MAP, shuffle_num=0, unlearning_flag=None, llm_family=LLM_FAMILY)
                    
                    if bench_success:
                        successful_bench += 1
                        pbar.set_postfix({"FT": "✅", "Bench": "✅", "completed": count + 1})
                    else:
                        pbar.set_postfix({"FT": "✅", "Bench": "❌", "completed": count + 1})
            else:
                pbar.set_postfix({"FT": "❌", "Bench": "⏭️", "completed": count + 1})
            
            count += 1
            pbar.update(1)

        print("   ✅ Finetuning outputs are stored locally; no remote upload is performed.")
        pbar.close()
    else:
        for comb in combs:
            if comb in ft_model_overrides:
                finetuned_models[comb] = ft_model_overrides[comb]
                print(f"   ✅ Using provided finetuned model for {comb}: {finetuned_models[comb]}")
            else:
                finetuned_models[comb] = get_model_path(comb, LANG_MAP, llm_family=LLM_FAMILY)
        # Set counts for summary
        count = len(combs)
        successful_ft = len(finetuned_models)
        successful_bench = 0  # Not tracked when loading existing models
        print("="*80)
        print("🚀 FINETUNING SUMMARY")
        print(f"finetuned_models: {finetuned_models}")
        print("="*80)

    print_complete_fine_tune(finetune_complete, count, successful_ft, successful_bench, finetuned_models, combs)

    # ==================== STAGE 4: UNLEARNING WITH SHUFFLES ====================
    total_unlearn_tasks, unlearn_plan = print_unlearning_combos(data, SHUFFLE_NUM, LANGUAGES, finetuned_models, combs, LANG_MAP)

    pbar_unlearn = tqdm(total=total_unlearn_tasks, desc="🧠 Unlearning Progress", unit="task")
    
    unlearn_count = 0
    unlearn_successful = 0
    bench_after_unlearn = 0
    unlearned_models = {}  # Store unlearned model paths
    debug = False
    if debug:
        unlearn_debug_info(data, finetuned_models, unlearn_plan)
    vllm_clear_counter = 0

    for shuffle_idx, (shuffle_name, languages_paths) in enumerate(data.items()):
        for ft_lang, model_path in finetuned_models.items():
            if "_" in ft_lang:
                ft_langs = ft_lang.split("_")
            else:
                ft_langs = [ft_lang]
            unlearn_langs = unlearn_plan[ft_lang]

            for unlearn_lang in unlearn_langs:
                print(f"FT:{ft_lang}")
                print(f"unlearn_lang: {unlearn_lang}")

                if isinstance(unlearn_lang, str):
                    unlearn_lang_list = [unlearn_lang]
                else:
                    unlearn_lang_list = list(unlearn_lang)
                # Run all unlearning tasks for this ft_lang on this shuffle
                output_path = run_unlearn(model_path, ft_langs, unlearn_lang_list, languages_paths, LANG_MAP, shuffle_name, pbar_unlearn, ft_lang)

                # Track this unlearning task
                unlearn_count += 1
                
                if output_path:
                    unlearn_successful += 1
                    unlearn_key = f"{ft_lang}-UN{''.join([LANG_MAP.get(l.lower(), l[:2]) for l in unlearn_lang_list])}-{shuffle_name}"
                    unlearned_models[unlearn_key] = output_path
                    
                    # Benchmark with ALL languages from current shuffle
                    benchmark_data_path = languages_paths

                    # Run benchmark immediately after successful unlearning
                    bench_success = run_benchmark(
                        output_path, 
                        benchmark_data_path,
                        ft_langs,
                        LANG_MAP, 
                        shuffle_num=shuffle_idx,
                        unlearning_flag=unlearn_key
                    )
                    
                    if bench_success:
                        bench_after_unlearn += 1

                    if vllm_clear_counter >= 10:
                        clear_vllm_cache()
                        vllm_clear_counter = 0
                    vllm_clear_counter += 1

                    print(f"   💾 Keeping unlearned model locally at: {output_path}")
                # Update progress bar after each unlearn+benchmark completes
                pbar_unlearn.set_postfix({"✅": unlearn_successful, "❌": unlearn_count - unlearn_successful, "📊": bench_after_unlearn})
                pbar_unlearn.update(1)
    
    pbar_unlearn.close()

    print("\n" + "="*80)
    print("📊 UNLEARNING SUMMARY")
    print("="*80)
    print(f"   Total tasks: {unlearn_count}")
    print(f"   Successful: {unlearn_successful}")
    print(f"   Failed: {unlearn_count - unlearn_successful}")
    if unlearn_count > 0:
        print(f"   Success rate: {unlearn_successful/unlearn_count*100:.1f}%")
    print("\n   Unlearned models:")
    for key, path in unlearned_models.items():
        print(f"      • {key}: {path}")
    print("="*80 + "\n")
