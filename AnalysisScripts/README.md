# Analysis

Two lines of analysis run on the output of `LocalizationScripts/`:

- **Can commitment junctures be predicted?** Feature extraction plus
  leave-one-environment-out modeling — `*_feature_extractor.py`,
  `train_predict.py`, `position_text_baselines.py`, `cot_monitor.py`.
- **Which heads carry the commitment?** Attribution patching, circuit
  selection, and steering — `attribution_patching.py`.

## The mechanistic experiment

`attribution_patching.py` is the main entrypoint. It has three
subcommands, normally run in order.

### `analyze`

Builds matched deceptive/truthful pairs from localization examples, patches each
attention head's output from one into the other across the commitment sentence,
and scores the head by how far that moves the deception label. Heads are ranked
on the training split, the circuit size is chosen on validation, and the final
numbers are reported on test and OOD. The run also saves a steering vector
bundle built from the training split.

The **patch scope** decides how much of the commitment sentence is patched, and
is a first-class axis of the experiment rather than a tuning knob — a head that
only matters when the whole sentence is patched is making a different claim than
one that matters at its first token:

| scope | what is patched |
|---|---|
| `patch_first_1_token` | the first commitment token only |
| `patch_first_half_sentence` | the first half of the sentence, position-wise |
| `patch_full_sentence` | the whole sentence, position-wise |

```bash
python AnalysisScripts/attribution_patching.py analyze \
  --model-id MODEL --environment bs --scope patch_full_sentence \
  --dataset-root DatasetMain
```

### `vectors`

Re-exports a steering vector bundle from a saved `analyze` run for a different
split, without repeating the ranking.

### `steering`

Generates from held-out prompts with the saved vectors applied, so steered and
unsteered conditions can be compared under the environment's own deception
labels. This is the **in-distribution** evaluation: the circuit is tested in the
environment it was discovered in. `prefix_steering.py` below carries the same
circuit to the four environments it was not discovered in.

## Steering vectors

For each selected head the direction is honest-minus-deceptive: the mean over
the commitment sentence of that head's output on truthful completions minus the
same on deceptive ones, averaged over matched pairs. That is the quantity
`steering_support/bs_circuit.pt` holds, and the same quantity the steering
hook edits at generation time.

## Prediction pipeline

```
sentence_localization.py output
        │
        ├── text_structural_feature_extractor.py      text and structural features
        ├── attention_activation_feature_extractor.py attention and activation features
        │
        └── train_predict.py                          leave-one-environment-out transfer
```

`position_text_baselines.py` and `cot_monitor.py` are the cheap baselines a
representation-based predictor has to beat: position and length descriptors with
sentence embeddings, and a zero-shot LLM monitor asked directly whether the agent
has committed.


## Steering the circuit across environments

`prefix_steering.py` takes the circuit discovered on one environment and
applies it unchanged to the others, so that environment is in-distribution
and the rest are transfer.

### What the experiment does

1. **Build post-commitment prefixes.** Walk an environment with
   `deception_miner`. At each state, sample actions and take a *deceptive* one,
   then cut its reasoning before the final sentences — the model has committed
   but has not yet written the sentences that state the decision, so generation
   resumes inside the still-open think block.

   These are post-commitment states, not commitment junctures. A commitment
   juncture is where the deception rate *jumps*, which is what attribution needs.
   A steering evaluation wants the opposite: states where deception is close to
   locked in, so there is headroom to remove.

2. **Screen the prefix.** Sample continuations from it and keep the prefix only
   if at least `--min-screen-deception` of them are deceptive. The screen is a
   separate draw from both evaluation arms, so selecting on it cannot manufacture
   a reduction through regression to the mean.

3. **Draw both arms.** From each surviving prefix, `--samples` unsteered and
   `--samples` steered continuations under matched seeds, labelled by the
   environment's own rule via `deception_miner.deception_from_action`. The arms
   go through the same code path and differ only by the intervention.

### How steering is applied

For each selected head, the direction is honest-minus-deceptive: the mean over
the commitment sentence of that head's output on truthful completions minus the
same on deceptive ones, averaged over matched pairs. At generation time a
forward pre-hook on `self_attn.o_proj` edits head *h*'s slice of that module's
input at the current position:

    h += alpha * ||h|| * d_hat          (--delta-mode relative, the default here)

so `alpha` is a dimensionless fraction of each head's own activation scale
rather than an absolute coefficient — the raw direction norms differ by an order
of magnitude across heads, so an absolute dose would be wildly uneven.

`--window fixed:N` steers the first N generated tokens and then releases. A
bounded window is the point: steering that is never released keeps the model
from finishing its reasoning at all, and validity collapses. `--window
reasoning:N` instead releases at `</think>`, per sequence.

### Files

| | |
|---|---|
| `prefix_steering.py` | the experiment: builds prefixes, screens them, draws both arms |
| `steered_model.py` | per-head steering hooks and the steering window |
| `transformers_backend.py` | generation backend; steering needs forward hooks, which vLLM does not expose |
| `game_config.py` | model default, environment list, miner argument namespace |
| `prefix_cut.py` | where a deceptive rollout is cut to make a prefix |
| `collect_results.py` | aggregates runs into one table with bootstrap intervals |
| `lenient_action_reader.py` | rescores with a lenient but label-preserving action reader |
| `judge_coherence.py` | LLM-as-judge coherence scoring of both arms |
| `Circuit/bs_circuit.pt` | the 32-head circuit and its per-head directions |
| `Notebooks/steering_results.ipynb` | every table and figure, from the aggregated results |

### Running it

A single cell:

```bash
python AnalysisScripts/prefix_steering.py \
  --env car_sales \
  --vector-path AnalysisScripts/steering_support/bs_circuit.pt \
  --alpha 1.0 --window fixed:250 --delta-mode relative \
  --target-prefixes 8 --samples 50 --screen-samples 16 \
  --min-screen-deception 0.80 \
  --out Results/Steering/runs/sw_car_sales_a10_L250.jsonl \
  --dump-generations Results/Steering/runs/sw_car_sales_a10_L250.gen.jsonl
```

The whole sweep, one worker per GPU:

```bash
GPUS="0 1 2 3" AnalysisScripts/shell_scripts/run_length_dose_sweep.sh
```

Then aggregate:

```bash
python AnalysisScripts/collect_steering_results.py --runs Results/Steering/runs
python AnalysisScripts/lenient_action_reader.py --runs Results/Steering/runs
python AnalysisScripts/judge_coherence.py --runs Results/Steering/runs   # needs OPENAI_API_KEY
```

`judge_coherence.py --dry-run` prints the exact prompt and a token estimate
without calling the API.

### What the run writes

Everything lands under `Results/`, which is not tracked.

    Results/Steering/runs/<tag>.jsonl             one line per prefix, both arms' counts
    Results/Steering/runs/<tag>.gen.jsonl         raw completions (large; needed by the
                                                  lenient reader and the coherence judge)
    Results/Steering/sweep_results.json           collect_results.py
    Results/Steering/lenient_results.json         lenient_action_reader.py
    Results/Steering/coherence_judge.json         judge_coherence.py
    Results/Steering/figures/                     written by the notebook

The notebook reads the four aggregates, so once the sweep has run it reproduces
every table and figure without a GPU.

### Reading the numbers

Deception rates are over **valid** responses, with validity reported separately:
an intervention that suppresses deception by making the model unparseable has
not suppressed anything.

Intervals are a **paired cluster bootstrap over prefixes**, not a pooled
binomial. Samples drawn from one prefix share a scenario and a cut point, so
pooling 400 of them would claim a precision the design does not support.

## Support modules

`interpretability_support/` and `ood_support/` are internal: the patching and
steering machinery behind `attribution_patching.py`, and the modeling
pipeline `train_predict.py` loads by path. Neither is meant to be run directly.
