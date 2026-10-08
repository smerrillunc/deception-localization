# Analysis

Three experiments run on the output of `LocalizationScripts/`, one folder each.

| folder | question |
|---|---|
| `OODModeling/` | Can a commitment juncture be predicted, and does the predictor transfer to an environment it never saw? |
| `AttributionPatching/` | Which attention heads carry the commitment? |
| `Steering/` | Does editing those heads suppress deception, in distribution and out of it? |

Each folder holds its own entrypoints and a `*_support/` module that is not
meant to be run directly.

---

# OODModeling

```
sentence_localization.py output
        │
        ├── text_structural_feature_extractor.py      text and structural features
        ├── attention_activation_feature_extractor.py attention and activation features
        │
        └── train_predict.py                          leave-one-environment-out transfer
```

`train_predict.py` is the entrypoint; it loads the pipeline in `ood_support/` by
path. `position_text_baselines.py` and `cot_monitor.py` are the cheap baselines a
representation-based predictor has to beat: position and length descriptors with
sentence embeddings, and a zero-shot LLM monitor asked directly whether the agent
has committed.

Outputs land under `Results/OODModeling/`.

---

# AttributionPatching

`attribution_patching.py` is the main entrypoint, with three subcommands normally
run in order.

## `analyze`

Ranks attention heads by attribution patching on the training split: for each
head it computes `target_grad * (source - target)` summed over the commitment
sentence, the gradient-based approximation to the effect of patching a truthful
activation into a deceptive trace. Circuit size is chosen on validation, final
numbers are reported on test and OOD, and a steering vector bundle is saved from
the training split.

The **patch scope** decides how much of the commitment sentence is involved, and
is a first-class axis of the experiment rather than a tuning knob — a head that
only matters when the whole sentence is patched is making a different claim than
one that matters at its first token:

| scope | what is patched |
|---|---|
| `patch_first_1_token` | the first commitment token only |
| `patch_first_half_sentence` | the first half of the sentence, position-wise |
| `patch_full_sentence` | the whole sentence, position-wise |

```bash
python AnalysisScripts/AttributionPatching/attribution_patching.py analyze \
  --model-id MODEL --environment bs --scope patch_full_sentence \
  --dataset-root DatasetMain
```

## `vectors`

Re-exports a steering vector bundle from a saved `analyze` run for a different
split, without repeating the ranking.

## `steering`

Generates from held-out prompts with the saved vectors applied, so steered and
unsteered conditions can be compared under the environment's own deception
labels. This is the **in-distribution** evaluation: the circuit is tested in the
environment it was discovered in.

## Attribution against activation patching

`interpretability_support/activation_patching.py` is the causal counterpart.
Where attribution patching approximates a head's effect from gradients, it swaps
a donor's activations into the target outright and re-runs generation, measuring
the effect rather than approximating it. It also owns the matched
deceptive/truthful pair construction and cache that the attribution pass
consumes, and runs standalone as its own experiment.

`interpretability_support/activation_steering.py` backs the `steering`
subcommand.

## Steering vectors

For each selected head the direction is honest-minus-deceptive: the mean over
the commitment sentence of that head's output on truthful completions minus the
same on deceptive ones, averaged over matched pairs. That is the quantity
`Steering/steering_support/bs_circuit.pt` holds, and the same quantity the
steering hook edits at generation time.

---

# Steering

`prefix_steering.py` takes the circuit discovered on one environment and applies
it unchanged to the others, so that environment is in-distribution and the rest
are transfer. The in-distribution evaluation is
`AttributionPatching/attribution_patching.py steering`.

## What the experiment does

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

## How steering is applied

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

## Files

| | |
|---|---|
| `prefix_steering.py` | the experiment: builds prefixes, screens them, draws both arms |
| `steering_support/steered_model.py` | per-head steering hooks and the steering window |
| `steering_support/transformers_backend.py` | generation backend; steering needs forward hooks, which vLLM does not expose |
| `steering_support/game_config.py` | model default, environment list, miner argument namespace |
| `steering_support/prefix_cut.py` | where a deceptive rollout is cut to make a prefix |
| `collect_steering_results.py` | aggregates runs into one table with bootstrap intervals |
| `lenient_action_reader.py` | rescores with a lenient but label-preserving action reader |
| `judge_coherence.py` | LLM-as-judge coherence scoring of both arms |
| `steering_support/bs_circuit.pt` | the 32-head circuit and its per-head directions |
| `Notebooks/steering_results.ipynb` | every table and figure, from the aggregated results |

## Running it

A single cell:

```bash
python AnalysisScripts/Steering/prefix_steering.py \
  --env car_sales \
  --vector-path AnalysisScripts/Steering/steering_support/bs_circuit.pt \
  --alpha 1.0 --window fixed:250 --delta-mode relative \
  --target-prefixes 8 --samples 50 --screen-samples 16 \
  --min-screen-deception 0.80 \
  --out Results/Steering/runs/sw_car_sales_a10_L250.jsonl \
  --dump-generations Results/Steering/runs/sw_car_sales_a10_L250.gen.jsonl
```

The whole sweep, one worker per GPU:

```bash
GPUS="0 1 2 3" AnalysisScripts/Steering/shell_scripts/run_length_dose_sweep.sh
```

Then aggregate:

```bash
python AnalysisScripts/Steering/collect_steering_results.py --runs Results/Steering/runs
python AnalysisScripts/Steering/lenient_action_reader.py --runs Results/Steering/runs
python AnalysisScripts/Steering/judge_coherence.py --runs Results/Steering/runs   # needs OPENAI_API_KEY
```

`judge_coherence.py --dry-run` prints the exact prompt and a token estimate
without calling the API.

## What the run writes

Everything lands under `Results/`, which is not tracked.

    Results/Steering/runs/<tag>.jsonl             one line per prefix, both arms' counts
    Results/Steering/runs/<tag>.gen.jsonl         raw completions (large; needed by the
                                                  lenient reader and the coherence judge)
    Results/Steering/sweep_results.json           collect_steering_results.py
    Results/Steering/lenient_results.json         lenient_action_reader.py
    Results/Steering/coherence_judge.json         judge_coherence.py
    Results/Steering/figures/                     written by the notebook

The notebook reads the four aggregates, so once the sweep has run it reproduces
every table and figure without a GPU.

## Reading the numbers

Deception rates are over **valid** responses, with validity reported separately:
an intervention that suppresses deception by making the model unparseable has
not suppressed anything.

Intervals are a **paired cluster bootstrap over prefixes**, not a pooled
binomial. Samples drawn from one prefix share a scenario and a cut point, so
pooling 400 of them would claim a precision the design does not support.
