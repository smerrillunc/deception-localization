# Deception Localization
This repository contains a compact, self-contained version of the deception-localization pipeline used for environment-level deception mining, sentence-level dataset construction, localization, and downstream analysis.

## DatasetAccess
1.  Public Dataset: https://huggingface.co/datasets/anonymous-neurips-2026-ED/deception-localization
2.  Quick Visualization Dashboard: https://deceptionlocalization-brzuiezqfmmwnevbm2nwuw.streamlit.app/
- Alternatively you can run locally with 
3. Schema described in DatasetAccess/README.md
4. Example of data access in code: DatasetAccess/hf_dataset_access_and_browser.ipynb

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `Environments/`: task environments and prompt-demo notebooks.
- `LocalizationScripts/`: the end-to-end localization data generation pipeline.
- `AnalysisScripts/`: feature extraction, OOD modeling, and mechanistic analysis entrypoints.
- `DatasetAccess/`: Hugging Face dataset access notebook, dashboard, and schema documentation.
- `SteeringScripts/`: activation steering from a discovered circuit, and its evaluation.
- `Results/`: aggregated outputs small enough to ship, so the analysis notebooks run without a GPU.
- `Core/`: shared helper modules vendored locally so this repo stands alone.

## Localization Workflow

The main workflow is:

1. `deception_miner.py`
   Generates raw deception examples from an environment by sampling model actions and labeling each sampled action as deceptive, truthful, or unknown.

2. `build_sentence_dataset.py`
   Converts mined example JSONL files into sentence-level examples, preserving the example-level labels and metadata needed for localization.

3. `sentence_localization.py`
   Re-samples continuations from progressively longer reasoning prefixes and estimates a deception rate for each localized prefix / sentence boundary.

In practice, the pipeline is:

```bash
python LocalizationScripts/deception_miner.py ...
python LocalizationScripts/build_sentence_dataset.py ...
python LocalizationScripts/sentence_localization.py ...
```

`deception_miner.py` is the single mining entrypoint. It now also handles advisor-audit runs via `--game advisor_audit`, so there is no separate finance-only miner script to maintain.

For common runs, the repo also includes shell wrappers under `LocalizationScripts/shell_scripts/`:

- `run_deception_miner_single_gpu.sh`
  Single-GPU launcher for mining. By default it writes to `Results/DeceptionMining/<env>/<model>/<run_tag>`.

- `run_sentence_localization_multi_gpu.sh`
  Multi-GPU launcher for localization. If needed, it first builds `examples.jsonl` and `sentences.jsonl`, then localizes across shards. By default it writes to:
  - `Results/SentenceDatasets/<env>/<model>/<run_tag>`
  - `Results/Localization/<env>/<model>/<run_tag>`

## Steering

`SteeringScripts/` asks whether a circuit discovered in one environment can be
steered to suppress deception in the others. The circuit shipped here is 32
attention heads found by attribution patching on `bs`; it is applied unchanged
to the other four, so `bs` is in-distribution and the rest are transfer.

For each (dose, steering length) cell the runner builds post-commitment prefixes
by cutting deceptive rollouts before they state their decision, screens them, and
then draws steered and unsteered continuations from each prefix under matched
seeds. Both arms are labelled by the environment's own rule, so they differ only
by the intervention.

```bash
GPUS="0 1 2 3" SteeringScripts/shell_scripts/run_length_dose_sweep.sh
python SteeringScripts/collect_results.py --runs Results/Steering/runs
```

Every table and figure is reproduced from the shipped aggregates by
`SteeringScripts/Notebooks/steering_results.ipynb`, which needs no GPU. See
`SteeringScripts/README.md` for the method, the steering rule, and what is and
is not included in `Results/`.

## Environments

Each environment folder contains the source implementation in `src/` and a consistent prompt demo notebook in `Notebooks/`.

- `AdvisorAudit`
  A commission-based financial advisor recommends investments to an investor who may audit the advice. Deception is recommending the wrong investment for the investor’s goals.

- `BS`
  A bluffing card game where players may lie about the rank of the cards they play. Deception is an untruthful play action.

- `CarSales`
  A used-car sales dialogue where the seller may conceal or disclose issues during a multi-round negotiation. Deception is seller-side concealment when disclosure is warranted.

- `Gridworld`
  A guide/explorer environment where the guide may recommend a suboptimal move. Deception is a knowingly misleading recommendation.

- `Interview`
  A job-offer negotiation environment where the candidate may falsely claim a better outside offer. Deception is a fabricated or inflated competing-offer claim.

Prompt demo notebooks:

- `Environments/AdvisorAudit/Notebooks/advisor_audit_environment_prompt_demo.ipynb`
- `Environments/BS/Notebooks/bs_environment_prompt_demo.ipynb`
- `Environments/CarSales/Notebooks/car_sales_environment_prompt_demo.ipynb`
- `Environments/Gridworld/Notebooks/gridworld_environment_prompt_demo.ipynb`
- `Environments/Interview/Notebooks/interview_environment_prompt_demo.ipynb`

These notebooks import the live environment code from this repo and show the exact prompts emitted at each step of a short manual rollout.

## Localization Scripts

- `LocalizationScripts/deception_miner.py`
  Main deception-mining entrypoint for `bs`, `gridworld`, `interview`, `car_sales`, and `advisor_audit`.

- `LocalizationScripts/build_sentence_dataset.py`
  Builds the sentence-level dataset used by localization from mined examples.

- `LocalizationScripts/sentence_localization.py`
  Runs prefix-based sentence localization and writes per-history continuation statistics.

Convenience launchers:

- `LocalizationScripts/shell_scripts/run_deception_miner_single_gpu.sh`
  Wrapper for launching mining on one GPU with the repo’s default `Results/` layout.

- `LocalizationScripts/shell_scripts/run_sentence_localization_multi_gpu.sh`
  Wrapper for building the sentence dataset and launching multi-GPU localization with sharding.

## Analysis Scripts

- `AnalysisScripts/text_structural_feature_extractor.py`
  Builds text-only and structural sentence-level baseline features from localization outputs.

- `AnalysisScripts/attention_activation_feature_extractor.py`
  Extracts sentence-level attention features plus activation summaries / activation tensors from localization outputs.

- `AnalysisScripts/train_predict.py`
  The single public OOD modeling entrypoint. It owns the CLI and runs the internal OOD modeling library from `AnalysisScripts/ood_support/`.

- `AnalysisScripts/position_text_baselines.py`
  Cheap non-mechanistic baselines for commitment-juncture prediction (normalized position, prefix / reasoning word count, MiniLM sentence embedding), trained with XGBoost under leave-one-environment-out or single-source transfer for deceptive and honest commitments. Reads the output of `text_structural_feature_extractor.py`.

- `AnalysisScripts/cot_monitor.py`
  Zero-shot LLM chain-of-thought monitor baselines (`run` queries an OpenAI-compatible API with `OPENAI_API_KEY`, `score` evaluates them). Two prompts (`action`: which action will the agent take; `committed`: has the agent committed), each read from token log-probabilities or from the generated answer alone.

- `AnalysisScripts/mechanistic_interpretibility.py`
  Mechanistic analysis / activation-patching entry script.

Internal analysis support is grouped to reduce clutter:

- `AnalysisScripts/ood_support/`
  Internal support modules used by the OOD modeling pipeline, including the companion workflow sourced by `train_predict.py`.

- `AnalysisScripts/interpretability_support/`
  Internal support modules used by the mechanistic / activation-patching pipeline.

## Dataset Access

- `DatasetAccess/hf_dataset_access_and_browser.ipynb`
  Notebook for browsing and visualizing the Hugging Face localization dataset.

- `DatasetAccess/app.py`
  Streamlit dashboard for the same dataset, reading directly from Hugging Face.

- `DatasetAccess/hf_dataset_browser_lib.py`
  Shared helper library used by both the notebook and the Streamlit dashboard.

- `DatasetAccess/build_hf_dataset_access_notebook.py`
  Builder script for regenerating the notebook.

- `DatasetAccess/LOCALIZATION_DATASET_SCHEMA.md`
  Complete field-by-field schema documentation for one localization example file.

Run the dashboard with:

```bash
streamlit run DatasetAccess/app.py
```

The dashboard and notebook both browse the Hugging Face dataset lazily:

- they select `environment` first
- then `model`
- then scan only that `localization/` folder with a configurable cap

This avoids listing all example files in the dataset up front.

## Quick Start

Mine examples on one GPU:

```bash
bash LocalizationScripts/shell_scripts/run_deception_miner_single_gpu.sh \
  --env bs \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu 0
```

Run localization on multiple GPUs using a mined run:

```bash
bash LocalizationScripts/shell_scripts/run_sentence_localization_multi_gpu.sh \
  --env bs \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu_ids "0 1 2 3" \
  --run_tag 2026-05-05_12-00-00
```
