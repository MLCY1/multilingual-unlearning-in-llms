from pathlib import Path
import sys
try:
    from utils.utils import get_abbr, get_unlearn_combinations
except ImportError:
    _UTILS_DIR = Path(__file__).resolve().parent
    if str(_UTILS_DIR) not in sys.path:
        sys.path.insert(0, str(_UTILS_DIR))
    from utils import get_abbr, get_unlearn_combinations


def print_data_structure(data):
    print(f"📖 Data Structure (shuffle_data output):")
    print(f"   Type: {type(data)}")
    print(f"   Keys (shuffles): {list(data.keys())}")
    print(f"\n   Structure:")
    for shuffle_name, languages_paths in data.items():
        print(f"   '{shuffle_name}': {{")
        for lang, paths in languages_paths.items():
            print(f"      '{lang}': {{")
            print(f"         'forget': '{paths['forget']}',")
            print(f"         'retain': '{paths['retain']}'")
            print(f"      }},")
        print(f"   }}")
    print()

def print_complete_fine_tune(finetune_complete, count, successful_ft, successful_bench, finetuned_models, combs):
    print("\n" + "🎊" + "="*78 + "🎊")
    print("║" + " "*78 + "║")
    print("║" + "✅ Step 4 COMPLETE: FINETUNING".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🎊" + "="*78 + "🎊")
    print(f"\n📊 Finetuning Summary:")
    if not finetune_complete:
        print(f"   ├─ Total combinations: {count}")
        print(f"   ├─ Successful finetunings: {successful_ft}/{count}")
        print(f"   └─ Successful benchmarks: {successful_bench}/{count}")
    else:
        print(f"   ├─ Total combinations: {count}")
        print(f"   ├─ Loaded from existing models: {successful_ft}/{count}")
        print(f"   └─ Status: ⏭️  Skipped finetuning (using existing models)")

    print(f"\n💾 Finetuned Models:")
    if finetuned_models:
        for comb in combs:
            if comb in finetuned_models:
                print(f"   ├─ [{comb}] → {finetuned_models[comb]}")
            else:
                print(f"   ├─ [{comb}] → ❌ NOT FOUND")
    else:
        print(f"   └─ ⚠️  No models found!")
    print("\n" + "="*80 + "\n")

def print_unlearning_combos(data, SHUFFLE_NUM, LANGUAGES, finetuned_models, combs, LANG_MAP):
    print("\n" + "🔄" + "="*78 + "🔄")
    print("║" + " "*78 + "║")
    print("║" + "🧠 Step 5: UNLEARNING (WITH SHUFFLED DATA)".center(77) + "║")
    print("║" + " "*78 + "║")
    print("🔄" + "="*78 + "🔄")
    
    print(f"\n🎲 Using {len(data)} dataset variations (Original + {SHUFFLE_NUM} shuffles)...")
    print(f"   Languages: {', '.join(LANGUAGES)}")
    print(f"   Random seed: 42\n")
        
    # Calculate total unlearning tasks
    total_unlearn_tasks = 0
    unlearn_plan = {}  # Dictionary to store unlearn combinations per model
    
    print(f"📋 Unlearning Plan:")
    print(f"   ├─ Finetuned models: {len(finetuned_models)}")
    print(f"   ├─ Dataset variations: {len(data)} (Original + {len(data)-1} shuffles)")
    print(f"   ├─ Unlearn combinations per model:")
    
    for comb in combs:
        if comb not in finetuned_models:
            continue
        if "_" in comb:
            langs = comb.split("_")
        else:
            langs = [comb]
        
        unlearn_combos = get_unlearn_combinations(langs, LANGUAGES)
        unlearn_plan[comb] = unlearn_combos
        total_unlearn_tasks += len(unlearn_combos) * len(data)
        
        # Show details with proper formatting
        print(f"   │   └─ [{comb}]:")
        print(f"   │       ├─ Finetuned on: {', '.join(langs)}")
        print(f"   │       ├─ Unlearn options: {len(unlearn_combos)}")
        for idx, unlearn_combo in enumerate(unlearn_combos):
            unlearn_abbr = ''.join(get_abbr(unlearn_combo, LANG_MAP))
            lang_display = unlearn_combo
            # Don't add comma if it's the last item
            if idx == len(unlearn_combos) - 1:
                print(f"   │       │   └─ UN{unlearn_abbr} ({lang_display})")
            else:
                print(f"   │       │   ├─ UN{unlearn_abbr} ({lang_display})")
        print(f"   │       └─ Total tasks: {len(unlearn_combos)} × {len(data)} shuffles = {len(unlearn_combos) * len(data)}")
    
    print(f"   └─ Total unlearning tasks: {total_unlearn_tasks}\n")

    print(f"📖 Unlearning Dictionary (unlearn_plan):")
    for ft_comb, unlearn_combos in unlearn_plan.items():
        print(f"   '{ft_comb}': [")
        for idx, combo in enumerate(unlearn_combos):
            # Don't add comma if it's the last item
            if idx == len(unlearn_combos) - 1:
                print(f"      {combo}")
            else:
                print(f"      {combo},")
        print(f"   ]")
    print()
    return total_unlearn_tasks, unlearn_plan


def unlearn_debug_info(data, finetuned_models, unlearn_plan):
    task_num = 0
    print("\n" + "="*80)
    print("🔍 DEBUG: Unlearning Task Details")
    print("="*80)
    for shuffle_idx, (shuffle_name, languages_paths) in enumerate(data.items()):
        print(f"\n📁 Shuffle: {shuffle_name}")
        print("─" * 80)
        
        for ft_lang, model_path in finetuned_models.items():
            unlearn_langs = unlearn_plan[ft_lang]
            
            # Determine FT languages
            if "_" in ft_lang:
                ft_lang_list = ft_lang.split("_")
            else:
                ft_lang_list = [ft_lang]
            
            print(f"\n   🤖 FT-Model: {ft_lang}")
            print(f"      Path: {model_path}")
            print(f"      FT Languages: {ft_lang_list}")
            print(f"      Unlearn options: {unlearn_langs}")
            
            for idx, unlearn_lang in enumerate(unlearn_langs, 1):
                task_num += 1
                if isinstance(unlearn_lang, str):
                    unlearn_lang_list = [unlearn_lang]
                else:
                    # It's a tuple or list
                    unlearn_lang_list = list(unlearn_lang)
                
                data_path = {'Forget':[], "Retain":[]}
                
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
                
                print(f"\n      ├─ Task #{task_num}: Unlearn {unlearn_lang}")
                print(f"      │  Unlearn Languages: {unlearn_lang_list}")
                print(f"      │  Retain Languages: {retain_langs}")
                print(f"      │  Forget data ({len(data_path['Forget'])} files):")
                for f_path in data_path['Forget']:
                    print(f"      │     • {f_path}")
                
                print(f"      │  Retain data ({len(data_path['Retain'])} files):")
                for r_path in data_path['Retain']:
                    print(f"      │     • {r_path}")
                
                if idx < len(unlearn_langs):
                    print(f"      │")
    
    print("\n" + "="*80)
    print(f"📊 Total unlearning tasks to execute: {task_num}")
    print("="*80 + "\n")
