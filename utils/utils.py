import json
from collections import defaultdict
import os, random
from itertools import combinations as iter_combinations
import os, sys, json, subprocess, argparse
from pathlib import Path
import json
import math
from collections import defaultdict
from statistics import mean, stdev
import re
from pathlib import Path
import shutil
try:
    from ..llm_config import load_llm_config
except ImportError:
    try:
        from llm_config import load_llm_config
    except ImportError:
        _UTILS_DIR = Path(__file__).resolve().parent
        _PROJECT_ROOT = _UTILS_DIR.parent
        if str(_PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(_PROJECT_ROOT))
        from llm_config import load_llm_config


def clear_vllm_cache(
    project_root: str = "/data/gpfs/projects/punim2348",
    dry_run: bool = False,
):
    cache_dir = Path(project_root) / ".cache" / "vllm"

    if not cache_dir.exists():
        print(f"No vLLM cache found at {cache_dir}")
        return

    if dry_run:
        n_files = sum(1 for _ in cache_dir.rglob("*"))
        print(f"[DRY RUN] Would remove {n_files} files from {cache_dir}")
        return

    shutil.rmtree(cache_dir)
    print(f"Cleared vLLM cache at {cache_dir}")

def find_duplicates(file_path):
    """
    Read a JSONL file and find duplicate items.
    
    Args:
        file_path: Path to the JSONL file
    """
    seen = {}  # Store {json_string: line_number}
    duplicates = {}  # Store {json_string: [line_numbers]}
    
    # Also track by a specific key if the JSON has one
    seen_by_key = {}  # Store {key_value: line_number}
    key_duplicates = {}
    
    total_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            try:
                # Parse the JSON object
                item = json.loads(line)
                
                # Convert to string for full content comparison
                item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
                
                # Check if we've seen this exact content before
                if item_str in seen:
                    if item_str not in duplicates:
                        duplicates[item_str] = [seen[item_str], line_num]
                    else:
                        duplicates[item_str].append(line_num)
                else:
                    seen[item_str] = line_num
                
                # If there's an 'id' or similar key, check for key duplicates
                if isinstance(item, dict):
                    for key_name in ['id', 'question', 'prompt', 'text']:  # common key names
                        if key_name in item:
                            key_value = str(item[key_name])
                            if key_value in seen_by_key:
                                if key_value not in key_duplicates:
                                    key_duplicates[key_value] = [seen_by_key[key_value], line_num]
                                else:
                                    key_duplicates[key_value].append(line_num)
                            else:
                                seen_by_key[key_value] = line_num
                            break  # Only check first matching key
                    
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                continue
    
    # Print results
    print(f"\nTotal lines read: {total_lines}")
    print(f"Unique items: {len(seen)}")
    print(f"Expected duplicates: {total_lines - len(seen)}\n")
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate item(s) by FULL CONTENT:\n")
        for item_str, line_numbers in duplicates.items():
            print(f"Duplicate found at lines: {line_numbers}")
            print(f"Content: {item_str[:500]}...")
            print("-" * 80)
    else:
        print("No duplicates found by full content!")
    
    if key_duplicates:
        print(f"\nFound {len(key_duplicates)} duplicate(s) by KEY:\n")
        for key_val, line_numbers in key_duplicates.items():
            print(f"Duplicate key at lines: {line_numbers}")
            print(f"Key value: {key_val[:200]}...")
            print("-" * 80)
    
    return duplicates, key_duplicates

def shuffle_data(Languages, num_shuffles=3, chunk_size=20, seed=42):
    """
    Shuffle data by treating consecutive items as chunks.
    
    Args:
        num_shuffles: Number of different shuffles to create
        chunk_size: Number of consecutive items to keep together (default: 20)
        seed: Random seed for reproducibility
    
    Returns:
        dict: Mapping of shuffle name to data paths per language
    """
    language_paths = {}
    
    for lang in Languages:
        forget_path = f"./Data/{lang}/Original/forget01.json"
        retain_path = f"./Data/{lang}/Original/retain99.json"
        language_paths[lang] = (forget_path, retain_path)
    
    shuffled_results = {"Original": {}}
    
    for lang, (forget_path, retain_path) in language_paths.items():
        with open(forget_path, 'r', encoding='utf-8') as f:
            forget_data = json.load(f)  # This is a dict
        with open(retain_path, 'r', encoding='utf-8') as f:
            retain_data = json.load(f)  # This is a dict
        
        # Convert dict to list of (key, value) tuples (each pair is one item)
        forget_items = list(forget_data.items())
        retain_items = list(retain_data.items())
        
        # Combine: retain first, then forget (so forget appears after retain)
        combined = retain_items + forget_items
        
        # Create chunks of consecutive items
        chunks = []
        for i in range(0, len(combined), chunk_size):
            chunks.append(combined[i:i + chunk_size])

        shuffled_results["Original"][lang] = {
            'forget': forget_path,
            'retain': retain_path
        }
        
        # Create multiple shuffles
        for shuffle_idx in range(num_shuffles):
            # Reset seed for reproducibility: each shuffle gets a deterministic seed
            rng = random.Random(seed + shuffle_idx)
            
            # Shuffle the chunks (not the items within chunks)
            shuffled_chunks = chunks.copy()
            rng.shuffle(shuffled_chunks)

            # Flatten back to single list
            shuffled_items = [item for chunk in shuffled_chunks for item in chunk]
            
            # Split back into retain (first 99%) and forget (last 1%)
            split_point = len(retain_items)
            shuffled_retain_items = shuffled_items[:split_point]
            shuffled_forget_items = shuffled_items[split_point:]
            
            # Convert back to dict format
            shuffled_forget = dict(shuffled_forget_items)
            shuffled_retain = dict(shuffled_retain_items)
            
            # Save to new directory
            shuffle_name = f"Shuffle{shuffle_idx + 1}"
            output_dir = f"./Data/{lang}/{shuffle_name}"
            os.makedirs(output_dir, exist_ok=True)
            
            with open(f"{output_dir}/forget01.json", 'w', encoding='utf-8') as f:
                json.dump(shuffled_forget, f, indent=2, ensure_ascii=False)
            
            with open(f"{output_dir}/retain99.json", 'w', encoding='utf-8') as f:
                json.dump(shuffled_retain, f, indent=2, ensure_ascii=False)
            
            # Store results
            if shuffle_name not in shuffled_results:
                shuffled_results[shuffle_name] = {}
            shuffled_results[shuffle_name][lang] = {
                'forget': f"{output_dir}/forget01.json",
                'retain': f"{output_dir}/retain99.json"
            }
            
            print(f"Created {shuffle_name} for {lang}: {len(shuffled_forget)} forget + {len(shuffled_retain)} retain")
    
    return shuffled_results

def convert_jsonl_to_json(jsonl_path, output_path=None, lang="english"):
    """
    Read a JSONL file and convert it to a JSON dictionary.
    
    Args:
        jsonl_path: Path to the JSONL file (e.g., "./Data/holdout10.json")
        output_path: Optional output path. If None, uses "{lang}_holdout10.json"
        lang: Language name for default output filename
    
    Returns:
        dict: The converted data as a dictionary
    """
    data_dict = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                
                # Assuming each line has a question-answer pair
                # Adjust key names based on your actual JSONL structure
                if isinstance(item, dict):
                    # If item has 'question' and 'answer' keys
                    if 'question' in item and 'answer' in item:
                        data_dict[item['question']] = item['answer']
                    # If item has 'prompt' and 'completion' keys
                    elif 'prompt' in item and 'completion' in item:
                        data_dict[item['prompt']] = item['completion']
                    # If item is already a dict with single key-value
                    elif len(item) == 1:
                        key, value = list(item.items())[0]
                        data_dict[key] = value
                    else:
                        print(f"[WARNING] Line {line_num}: Unexpected format - {item}")
                        
            except json.JSONDecodeError as e:
                print(f"[ERROR] Line {line_num}: Invalid JSON - {e}")
                continue
    
    # Set default output path if not provided
    if output_path is None:
        output_path = f"./{lang.capitalize()}_holdout10.json"
    
    # Save as JSON dictionary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Converted {len(data_dict)} items from {jsonl_path} to {output_path}")
    return data_dict

def get_abbr(langs, LANG_MAP):
    """Convert language names to abbreviations.
    
    Args:
        langs: Can be a string (single language), tuple, or list of languages
        LANG_MAP: Dictionary mapping language names to abbreviations
    
    Returns:
        List of abbreviations
    """
    # Handle single string input - convert to list first
    if isinstance(langs, str):
        langs = [langs]
    
    return [LANG_MAP.get(lang.lower(), lang[:2].upper()) for lang in langs]

def get_combinations(LANGUAGES):
    # Single languages
    combinations = LANGUAGES.copy()
    
    # Pairs
    combinations.extend(
        f"{LANGUAGES[i]}_{LANGUAGES[j]}" 
        for i in range(len(LANGUAGES) - 1) 
        for j in range(i + 1, len(LANGUAGES))
    )
    
    # All languages combined
    if len(LANGUAGES) > 2:
        combinations.append("_".join(LANGUAGES))
    
    return combinations

def run_finetune(languages, data_path, LANG_MAP, llm_family=None):
    """
    Run finetuning for given language combination.
    
    Args:
        languages: List of language names
        data_path: Dict mapping language to forget/retain paths
        shuffle_name: Name of the shuffle (e.g., "Original", "Shuffle1")
        llm_family: Optional LLM family override (e.g., "qwen", "gemma")
    
    Returns:
        str: Path to finetuned model if successful, None otherwise
    """
    llm_cfg = load_llm_config(llm_family)
    abbr = get_abbr(languages, LANG_MAP)
    lang_abbr_str = ''.join(abbr)
    model_dir = f"{llm_cfg.finetuned_prefix}{lang_abbr_str}"
    model_path = os.path.join(llm_cfg.models_local_root, model_dir)
    all_paths = []
    for lang in languages:
        print(f"\n  {lang}:")
        print(f"    • Forget (1%):  {data_path[lang]['forget']}")
        print(f"    • Retain (99%): {data_path[lang]['retain']}")
        all_paths.append(data_path[lang]['forget'])
        all_paths.append(data_path[lang]['retain'])
    
    print("\n" + "🔥" + "="*78 + "🔥")
    print("║" + " "*78 + "║")
    print("║" + "🚀 FINETUNING STAGE".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🔥" + "="*78 + "🔥")
    print(f"\n📚 Training Languages: {', '.join(languages)} → [FT-{lang_abbr_str}]")
    print(f"\n💾 Output Model Path:")
    print(f"   └─ {model_path}")
    print(f"\n📁 Training Data Configuration:")
    for lang in languages:
        print(f"   • {lang.capitalize()}:")
        print(f"      ├─ Forget (1%):  {data_path[lang]['forget']}")
        print(f"      └─ Retain (99%): {data_path[lang]['retain']}")
    
    cmd = [
        "deepspeed",
        "--num_gpus=4",
        "finetune.py",
        *languages,
        "--data-paths", *all_paths,
    ]
    if llm_family:
        cmd.extend(["--llm-family", llm_family])
    
    print(f"[CMD] {' '.join(cmd)}\n")
    env = os.environ.copy()
    if llm_family:
        env["LLM_FAMILY"] = llm_family

    result = subprocess.run(cmd, check=False, env=env)
    
    if result.returncode != 0:
        print("\n" + "❌" + "="*78 + "❌")
        print("║" + " "*78 + "║")
        print("║" + "⚠️  FINETUNING FAILED!".center(77) + "║")
        print("║" + " "*78 + "║")
        print("❌" + "="*78 + "❌" + "\n")
        return None
    
    print("\n" + "✅" + "="*78 + "✅")
    print("║" + " "*78 + "║")
    print("║" + "🎉 FINETUNING COMPLETED SUCCESSFULLY!".center(77) + "║")
    print("║" + " "*78 + "║")
    print("✅" + "="*78 + "✅")
    print(f"\n💾 Model Saved To: {model_path}\n")
    return model_path

def run_benchmark(model_path, data_path, languages, LANG_MAP, shuffle_num = 0, unlearning_flag = None, llm_family=None):
    """
    Run benchmarking with vLLM in a different conda environment.
    
    Args:
        model_path: Path to the finetuned model
        data_path: Dict mapping language to forget/retain paths
        languages: List of language names that were finetuned
        shuffle_num: Shuffle number (0 for Original, 1+ for shuffles)
        llm_family: Optional LLM family override to pass through to subprocesses.
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Save data_path dict to temporary JSON file
    lang_abbr = get_abbr(languages, LANG_MAP)
    lang_abbr_str = ''.join(lang_abbr)
    
    # Save data_path dict to temporary JSON file
    os.makedirs("./Data/temp", exist_ok=True)
    temp_config_path = f"./Data/temp/test_config_{shuffle_num}_{lang_abbr_str}.json"
    
    with open(temp_config_path, 'w') as f:
        json.dump(data_path, f, indent=2)
    if unlearning_flag:
        cmd = [
            "conda", "run", "-n", "inference", "--no-capture-output",
            "python", "benchmarkTest.py",
            "--model-path", model_path,
            "--test-data", temp_config_path,
            "--SHUFFLE-NUM", str(shuffle_num),
            "--LANGUAGES", lang_abbr_str,
            "--UNLEARNING-FLAG", unlearning_flag
        ]
    else:
        cmd = [
            "conda", "run", "-n", "inference", "--no-capture-output",
            "python", "benchmarkTest.py",
            "--model-path", model_path,
            "--test-data", temp_config_path,
            "--SHUFFLE-NUM", str(shuffle_num),
            "--LANGUAGES", lang_abbr_str
        ]    

    print("\n" + "🎯" + "="*78 + "🎯")
    print("║" + " "*78 + "║")
    print("║" + "🧪 BENCHMARK TESTING STAGE".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🎯" + "="*78 + "🎯")
    print(f"\n📊 Model Path:")
    print(f"   └─ {model_path}")
    # print(f"\n🌍 Testing Languages: {', '.join(languages)} → [{lang_abbr_str}]")
    print(f"\n🔀 Shuffle Configuration: #{shuffle_num}")
    print(f"\n📁 Test Data Configuration:")
    for lang, paths in data_path.items():
        print(f"   • {lang.capitalize()}:")
        print(f"      ├─ Forget: {paths['forget']}")
        print(f"      └─ Retain: {paths['retain']}")
    print(f"\n🐍 Conda Environment: inference")
    print(f"\n⚙️  Command: {' '.join(cmd[:4])} ...")
    print("\n" + "▶" + "="*78 + "▶")
    print("   🚀 STARTING BENCHMARK EVALUATION...")
    print("▶" + "="*78 + "▶" + "\n")

    env = os.environ.copy()
    if llm_family:
        env["LLM_FAMILY"] = llm_family

    result = subprocess.run(cmd, check=False, env=env)

    if result.returncode != 0:
        print("\n" + "❌" + "="*78 + "❌")
        print("║" + " "*78 + "║")
        print("║" + f"⚠️  BENCHMARK FAILED FOR: {model_path}".center(77) + "║")
        print("║" + " "*78 + "║")
        print("❌" + "="*78 + "❌" + "\n")
        return False

    print("\n" + "✅" + "="*78 + "✅")
    print("║" + " "*78 + "║")
    print("║" + "🎉 BENCHMARK COMPLETED SUCCESSFULLY!".center(77) + "║")
    print("║" + " "*78 + "║")
    print("✅" + "="*78 + "✅" + "\n")
    return True

def get_model_path(combination, LANG_MAP, llm_family=None):
    """
    Given a combination string (e.g., 'english_chinese'), return the model path for that combination.
    
    Args:
        combination: Language combination name.
        LANG_MAP: Mapping of language names to abbreviations.
        llm_family: Optional LLM family override (e.g., "qwen", "gemma").
    """
    llm_cfg = load_llm_config(llm_family)
    models_dir = llm_cfg.models_local_root
    if "_" in combination:
        langs = combination.split("_")
    else:
        langs = [combination]
    abbr = ''.join(get_abbr(langs, LANG_MAP))
    model_dir = f"{llm_cfg.finetuned_prefix}{abbr}"
    model_path = os.path.join(models_dir, model_dir)
    if os.path.exists(model_path):
        return model_path
    else:
        raise ValueError(f"Model path not found for combination: {combination} (expected: {model_path})")

def get_unlearn_combinations(finetuned_langs, all_available_langs):
    """
    Generate all possible unlearn combinations for given finetuned languages.
    
    For a model finetuned on certain languages, we can unlearn:
    1. Each individual language from all available languages (as strings)
    2. All possible combinations of the finetuned languages (as tuples)
    
    Args:
        finetuned_langs: List of languages the model was finetuned on
        all_available_langs: List of all available languages
    
    Returns:
        List containing strings (for individual languages) and tuples (for combinations)
    
    Examples:
        # Model finetuned on English only
        finetuned_langs = ["English"]
        all_available_langs = ["English", "Chinese", "German"]
        Returns: ["English", "Chinese", "German"]
        
        # Model finetuned on English + Chinese
        finetuned_langs = ["English", "Chinese"]
        all_available_langs = ["English", "Chinese", "German"]
        Returns: ["English", "Chinese", "German", ("English", "Chinese")]
        
        # Model finetuned on all three languages
        finetuned_langs = ["English", "Chinese", "German"]
        all_available_langs = ["English", "Chinese", "German"]
        Returns: ["English", "Chinese", "German", 
                  ("English", "Chinese"), ("English", "German"), ("Chinese", "German"),
                  ("English", "Chinese", "German")]
    """
    # Fixed combinations for 5 languages to reduce computational cost
    if len(finetuned_langs) > 3:
        return [
            # Individual languages
            "English", "Chinese", "German", "Russian", "Turkish",
            # Selected 3-language combinations
            ("English", "German", "Russian"), ("English", "German", "Turkish"), ("English", "Chinese", "Russian"),
            ("German", "Chinese", "Russian"),
            # All 5 languages together
            ("English", "Chinese", "German", "Russian", "Turkish")
        ]
    
    unlearn_combinations = []
    
    # Add each individual language as a string (not tuple)
    for lang in all_available_langs:
        unlearn_combinations.append(lang)
    
    # Add all possible combinations of finetuned languages (size 2 to n)
    n = len(finetuned_langs)
    if n > 1:
        for r in range(2, n + 1):
            for combo in iter_combinations(finetuned_langs, r):
                unlearn_combinations.append(combo)
    
    return unlearn_combinations


def _run_unlearn_helper(model_path, unlearn_lang_list, data_path, LANG_MAP, shuffle_name):
    """
    Perform unlearning on a finetuned model.
    
    Args:
        model_path: Path to the finetuned model
        unlearn_lang_list: List of languages to unlearn
        data_path: Dictionary with 'Forget', 'Retain', and 'IDK' data paths
        LANG_MAP: Language abbreviation mapping
        shuffle_name: Name of the shuffle being used
    """
    print("\n" + "🧠" + "="*78 + "🧠")
    print("║" + " "*78 + "║")
    print("║" + "🔓 UNLEARNING STAGE".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🧠" + "="*78 + "🧠")
    
    # Model Information
    model_name = model_path.split('/')[-1]
    print(f"\n🤖 Base Model: {model_name}")
    print(f"   └─ {model_path}")
    
    # Unlearning Configuration
    unlearn_abbr = ''.join([LANG_MAP.get(lang.lower(), lang[:2]) for lang in unlearn_lang_list])
    print(f"\n🎯 Unlearning Languages: {', '.join(unlearn_lang_list)} → [UN-{unlearn_abbr}]")
    print(f"   └─ Shuffle: {shuffle_name}")
    
    # Output Model Path
    output_model_name = f"{model_name}-UN{unlearn_abbr}-{shuffle_name}"
    output_path = f"./Models/{output_model_name}"
    print(f"\n💾 Output Model Path:")
    print(f"   └─ {output_path}")
    
    # Data Information
    print(f"\n📁 Unlearning Data Configuration:")
    
    # Group forget data by language
    print(f"   Forget Data (to remove):")
    for idx, path in enumerate(data_path['Forget']):
        if idx < len(unlearn_lang_list):
            lang = unlearn_lang_list[idx]
            print(f"      • {lang}:")
            print(f"         └─ {path}")
        else:
            print(f"      • {path}")
    
    # Group retain data by language
    print(f"   Retain Data (to preserve):")
    for idx, path in enumerate(data_path['Retain']):
        lang_found = None
        for lang in LANG_MAP.keys():
            if lang.capitalize() in path:
                lang_found = lang.capitalize()
                break
        
        if lang_found:
            print(f"      • {lang_found}:")
            print(f"         └─ {path}")
        else:
            print(f"      • {path}")
    
    # IDK data
    print(f"   IDK Data (I Don't Know responses):")
    for idx, path in enumerate(data_path['IDK']):
        if idx < len(unlearn_lang_list):
            lang = unlearn_lang_list[idx]
            print(f"      • {lang}:")
            print(f"         └─ {path}")
        else:
            print(f"      • {path}")

    audit_dir = "./Data/UnlearnDataPathCheck"
    os.makedirs(audit_dir, exist_ok=True)
    file_stem = output_model_name
    ft_idx = file_stem.find("FT")
    if ft_idx != -1:
        file_stem = file_stem[ft_idx:]
    audit_file = os.path.join(audit_dir, f"{file_stem}.txt")

    with open(audit_file, "w", encoding="utf-8") as f:
        f.write("Forget Data:\n")
        for path in data_path["Forget"]:
            f.write(f"{path}\n")
        f.write("\nRetain Data:\n")
        for path in data_path["Retain"]:
            f.write(f"{path}\n")
        f.write("\nIDK Data:\n")
        for path in data_path["IDK"]:
            f.write(f"{path}\n")

    print(f"\n📝 Logged forget/retain/IDK paths to: {audit_file}")
    
    print("\n" + "─"*80)
    print("⏳ Starting unlearning process...")
    print("─"*80)
     
    cmd = [
        "deepspeed",
        "--num_gpus=4",
        "dpoUnlearning.py",
        "--model_path", model_path,
        "--output_path", output_path,
        "--forget_paths", *data_path['Forget'],
        "--retain_paths", *data_path['Retain'],
        "--idk_paths", *data_path['IDK'],
    ]
    
    result = subprocess.run(cmd, check=False)
    
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
    print(f"\n💾 Model Saved To: {output_path}\n")
    return output_path
    
def run_unlearn(model_path, ft_lang_list, unlearn_lang_list, languages_paths, LANG_MAP, shuffle_name, pbar_unlearn, ft_lang):        
    data_path = {'Forget': [], "Retain": [], "IDK": []}
    
    # Determine forget data (always the unlearned languages)
    for lang in unlearn_lang_list:
        data_path['Forget'].append(languages_paths[lang]['forget'])
    
    # Determine retain data based on rules
    if len(ft_lang_list) == 1:
        # Rule 1: Model finetuned on single language
        # Retain data is always the FT language
        retain_langs = ft_lang_list
    elif len(ft_lang_list) == 2 and len(unlearn_lang_list) == 1:
        # Rule 2: Model finetuned on two languages, unlearning one
        if unlearn_lang_list[0] not in ft_lang_list:
            # Unlearning a language not in FT languages
            # Retain both FT languages
            retain_langs = ft_lang_list
        else:
            # Unlearning one of the FT languages
            # Retain data should be the language being unlearned
            retain_langs = unlearn_lang_list
    else:
        # Default: Retain the unlearned languages
        retain_langs = unlearn_lang_list
    
    for lang in retain_langs:
        data_path['Retain'].append(languages_paths[lang]['retain'])
    
    # Load IDK files (always from Original folder, same across all shuffles)
    for lang in unlearn_lang_list:
        idk_path = f"./Data/{lang}/Original/idk.txt"
        data_path['IDK'].append(idk_path)
    
    # Update progress bar description
    unlearn_desc = '_'.join(unlearn_lang_list)
    pbar_unlearn.set_description(f"🧠 {ft_lang} → UN{unlearn_desc} [{shuffle_name}]")
    
    # Run unlearning
    output_path = _run_unlearn_helper(
        model_path, 
        unlearn_lang_list, 
        data_path, 
        LANG_MAP, 
        shuffle_name
    )
    
    return output_path


def run_task2_benchmark(model_path, data_path, languages, LANG_MAP, llm_family=None, benchmark_result_dir=None):
    lang_abbr = ''.join(get_abbr(languages, LANG_MAP))
    os.makedirs("./Data/temp", exist_ok=True)
    temp_config_path = f"./Data/temp/task2_{lang_abbr}.json"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(data_path, f, indent=2, ensure_ascii=False)

    cmd = [
        "conda", "run", "-n", "inference", "--no-capture-output",
        "python", "benchmarkTest.py",
        "--model-path", model_path,
        "--test-data", temp_config_path,
        "--SHUFFLE-NUM", "-2",
        "--LANGUAGES", lang_abbr
    ]

    print("\n" + "🎯" + "="*78 + "🎯")
    print("║" + " "*78 + "║")
    print("║" + f"🧪 TASK2 BENCHMARK :: {languages[0]} MODEL".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🎯" + "="*78 + "🎯")
    print(f"\n📊 Model Path: {model_path}")
    print(f"📁 Temp Config: {temp_config_path}")
    print(f"🌍 Target Languages: {', '.join(data_path.keys())}")
    print(f"\n⚙️  Command: {' '.join(cmd[:4])} ...\n")

    env = os.environ.copy()
    if llm_family:
        env["LLM_FAMILY"] = llm_family
    if benchmark_result_dir:
        env["BENCHMARK_RESULT_DIR"] = benchmark_result_dir

    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed for {model_path} with exit code {result.returncode}")
    print(f"✅ Completed benchmark for {model_path}")

if __name__ == "__main__":
    print(get_unlearn_combinations(["English", "Chinese", "Russian"], ["English", "Chinese", "German", "Russian"]))
