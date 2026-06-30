# Multilingual Unlearning in LLMs: Reproduction Guide

This repository accompanies the paper **Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility**. The paper studies whether unlearning targeted facts in one language transfers to other languages, how that transfer appears in hidden representations, and how much suppressed knowledge can be recovered with inference-time steering.

The experiments extend TOFU to five aligned languages:

- English
- Chinese
- German
- Russian
- Turkish

The pipeline supports two model families:

- Qwen, used in the paper as Qwen2.5-7B-Instruct
- Gemma, used in the paper as Gemma2-9B-Instruct

Multilingual answer correctness is evaluated with `xlm-roberta-large-xnli`.

## Repository Layout

```text
.
+-- Data/
|   +-- English/
|   +-- Chinese/
|   +-- German/
|   +-- Russian/
|   +-- Turkish/
|   +-- BenchmarkResult/
|   +-- CosineSimilarityDiagram/
|   +-- PCADiagrams/
|   `-- SteeringVector/
+-- Models/
+-- transfer.py
+-- finetune.py
+-- dpoUnlearning.py
+-- benchmarkTest.py
+-- hiddenAnalysis.py
+-- steeringVector.py
+-- llm_config.py
`-- utils/
```

Important scripts:

- `transfer.py`: full fine-tune, benchmark, unlearn, benchmark pipeline.
- `finetune.py`: full-parameter knowledge injection on aligned TOFU facts.
- `dpoUnlearning.py`: DPO unlearning with retain-set KL regularization.
- `benchmarkTest.py`: vLLM generation plus multilingual NLI scoring.
- `hiddenAnalysis.py`: cosine similarity, PCA, centroid distance, pairwise distance, and cross-lingual prompting tests.
- `steeringVector.py`: Testing how much suppressed knowledge can be recovered with inference-time steering.

## Environment

The curated dependency file is `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you use `transfer.py` or the helper benchmark launchers, create an environment named `inference`, because benchmark commands are launched as:

```bash
conda run -n inference --no-capture-output python benchmarkTest.py ...
```

All commands below assume you are inside this directory:

```bash
cd "Multilingual Unlearning"
```

## Local Model Setup
For Qwen2.5-7B:

```bash
export LLM_FAMILY=qwen
export QWEN_MODEL_FAMILY=Qwen2.5
export QWEN_MODEL_SIZE=7B
export QWEN_MODEL_SUFFIX=Instruct
export LLM_MODELS_LOCAL_ROOT=./Models
```

Expected base checkpoint:

```text
./Models/Qwen2.5-7B-Instruct
```

For Gemma2-9B:

```bash
export LLM_FAMILY=gemma
export GEMMA_MODEL_FAMILY=Gemma2
export GEMMA_MODEL_SIZE=9B
export GEMMA_MODEL_SUFFIX=Instruct
export LLM_MODELS_LOCAL_ROOT=./Models
```

Expected base checkpoint:

```text
./Models/Gemma2-9B-Instruct
```

Expected NLI evaluator:

```text
./Models/xlm-roberta-large-xnli
```

## Data Format

Each language directory should contain the original forget/retain split and IDK templates:

```text
Data/<Language>/Original/forget01.json
Data/<Language>/Original/retain99.json
Data/<Language>/Original/idk.txt
```

## Full Transfer Experiment

Run the full pipeline:

```bash
python transfer.py
```

This performs:

1. Build shuffled forget/retain splits.
2. Fine-tune on each selected language combination.
3. Benchmark each fine-tuned model.
4. Run DPO unlearning on shuffled forget sets.
5. Benchmark each unlearned model across all languages.

## Fine-Tuning Only

Fine-tune Qwen on English:

```bash
deepspeed --num_gpus=4 finetune.py English \
  --data-paths \
  ./Data/English/Original/forget01.json \
  ./Data/English/Original/retain99.json \
  --llm-family qwen
```

Fine-tune on multiple languages:

```bash
deepspeed --num_gpus=4 finetune.py English German Russian \
  --data-paths \
  ./Data/English/Original/forget01.json ./Data/English/Original/retain99.json \
  ./Data/German/Original/forget01.json ./Data/German/Original/retain99.json \
  ./Data/Russian/Original/forget01.json ./Data/Russian/Original/retain99.json \
  --llm-family qwen
```

## DPO Unlearning Only

The main paper method uses DPO unlearning. For each forget prompt, the ground-truth answer is the negative response and a language-specific IDK/refusal answer is the preferred response. Retain examples are used for KL regularization.

Example: unlearn English from an English fine-tuned Qwen model using `Shuffle2`.

```bash
deepspeed --num_gpus=4 dpoUnlearning.py \
  --model_path ./Models/Qwen2.5-7B-finetuned-FTEn \
  --output_path ./Models/Qwen2.5-7B-finetuned-FTEn-UNEn-Shuffle2 \
  --forget_paths ./Data/English/Shuffle2/forget01.json \
  --retain_paths ./Data/English/Shuffle2/retain99.json \
  --idk_paths ./Data/English/Original/idk.txt \
  --llm-family qwen
```

## Hidden Representation Analysis

`hiddenAnalysis.py` reproduces the paper's representation experiments:

1. Cross-lingual prompting: ask the model to answer in the fine-tuned language when the input is in a foreign language.
2. Cross-language cosine similarity using one anchor language.
3. Before-vs-after unlearning cosine similarity within each language.
4. PCA separation for base, fine-tuned, and unlearned representations.
5. Centroid-distance and pairwise-distance heatmaps.

Run all analyses:

```bash
python hiddenAnalysis.py \
  --analysis all \
  --llm-family qwen \
  --llm-model-size 7B \
  --anchor-language English
```

Run only anchor-language cosine similarity. This produces the original 2-row x 4-column panel: Base, FT, FT with cross-lingual prompting, and UN.

```bash
python hiddenAnalysis.py \
  --analysis anchor-cosine \
  --llm-family qwen \
  --llm-model-size 7B \
  --anchor-language English \
  --output-dir ./Data/CosineSimilarityDiagram/custom
```

Run FT-vs-UN cosine similarity:

```bash
python hiddenAnalysis.py \
  --analysis unlearning-cosine \
  --llm-family qwen \
  --llm-model-size 7B \
  --unlearning-output-dir ./Data/CosineSimilarityDiagram/custom
```

Run PCA and distance heatmaps:

```bash
python hiddenAnalysis.py \
  --analysis pca \
  --llm-family qwen \
  --llm-model-size 7B \
  --pca-layers 0 10 20 -1 \
  --pca-output-dir ./Data/PCADiagrams/custom
```

Run only the cross-lingual prompting benchmark:

```bash
python hiddenAnalysis.py \
  --analysis ft-language-benchmark \
  --llm-family qwen \
  --llm-model-size 7B \
  --benchmark-result-dir ./Data/BenchmarkResult/custom
```

## Steering-Vector Reversibility

The steering experiment has four stages:

1. Run auxiliary unlearning in one language.
2. Extract layer-wise steering vectors from fine-tuned vs. unlearned hidden states.
3. Sweep alpha values while injecting the vector during generation.
4. Plot the best-alpha heatmap.

Full Qwen run:

```bash
python steeringVector.py \
  --llm-family qwen \
  --llm-model-size 7B \
  --aux-language English \
  --target-languages English Chinese German Russian Turkish \
  --alpha-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

Random-vector baseline:

```bash
python steeringVector.py \
  --llm-family qwen \
  --llm-model-size 7B \
  --aux-language English \
  --target-languages English Chinese German Russian Turkish \
  --alpha-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
  --random-baseline
```

## Citation

```bibtex
@inproceedings{xiang2026multilingualunlearning,
  title = {Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility},
  author = {Xiang, Chaoyi and Ohrimenko, Olga and Rubinstein, Benjamin I. P. and Frermann, Lea},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026}
}
```
