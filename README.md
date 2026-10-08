# Deception Localization
This repository contains a compact, self-contained version of the deception-localization pipeline used for environment-level deception mining, sentence-level dataset construction, localization, and downstream analysis.

## Dataset

https://huggingface.co/datasets/anonymous-neurips-2026-ED/deception-localization

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Layout

- `Environments/`: task environments and prompt-demo notebooks.
- `LocalizationScripts/`: the end-to-end localization data generation pipeline.
- `AnalysisScripts/`: feature extraction, OOD modeling, and mechanistic analysis entrypoints.
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

Two steering evaluations run off the circuit `AnalysisScripts/AttributionPatching/attribution_patching.py`
discovers, both under `AnalysisScripts/`:

- **In distribution** — `attribution_patching.py steering` generates from held-out
  prompts in the environment the circuit was found in.
- **Across environments** — `prefix_steering.py` applies the same circuit, unchanged,
  to the four environments it was *not* found in, from post-commitment prefixes.

```bash
python AnalysisScripts/Steering/prefix_steering.py --env car_sales \
  --vector-path AnalysisScripts/Steering/steering_support/bs_circuit.pt \
  --alpha 1.0 --window fixed:250 --delta-mode relative \
  --out Results/Steering/runs/car_sales_a1.0_L250.jsonl

python AnalysisScripts/Steering/collect_steering_results.py --runs Results/Steering/runs
```

Scripts write under `Results/`, which is not tracked.

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

Three experiments, one folder each. See `AnalysisScripts/README.md` for the
method behind each.

- `AnalysisScripts/OODModeling/`
  Can a commitment juncture be predicted, and does the predictor transfer to an
  environment it never saw? Feature extraction (`text_structural_feature_extractor.py`,
  `attention_activation_feature_extractor.py`), leave-one-environment-out modeling
  (`train_predict.py`), and the cheap baselines it has to beat
  (`position_text_baselines.py`, `cot_monitor.py`).

- `AnalysisScripts/AttributionPatching/`
  Which attention heads carry the commitment? `attribution_patching.py` ranks heads
  by `target_grad * (source - target)` over the commitment sentence, selects a
  circuit, and saves its steering vectors. `interpretability_support/activation_patching.py`
  is the causal counterpart that swaps activations outright instead of approximating.

- `AnalysisScripts/Steering/`
  Does editing those heads suppress deception? `prefix_steering.py` applies the
  circuit to the four environments it was *not* discovered in, from
  post-commitment prefixes, with the aggregation, lenient rescoring and
  coherence-judge passes alongside it.

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
