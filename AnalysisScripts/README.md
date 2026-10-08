# Analysis

Two lines of analysis run on the output of `LocalizationScripts/`:

- **Can commitment junctures be predicted?** Feature extraction plus
  leave-one-environment-out modeling — `*_feature_extractor.py`,
  `train_predict.py`, `position_text_baselines.py`, `cot_monitor.py`.
- **Which heads carry the commitment?** Attribution patching, circuit
  selection, and steering — `mechanistic_interpretability.py`.

## The mechanistic experiment

`mechanistic_interpretability.py` is the main entrypoint. It has three
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
python AnalysisScripts/mechanistic_interpretability.py analyze \
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
environment it was discovered in. `SteeringScripts/` carries the same circuit to
the four environments it was not discovered in.

## Steering vectors

For each selected head the direction is honest-minus-deceptive: the mean over
the commitment sentence of that head's output on truthful completions minus the
same on deceptive ones, averaged over matched pairs. That is the quantity
`SteeringScripts/Circuit/bs_circuit.pt` holds, and the same quantity the steering
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

## Support modules

`interpretability_support/` and `ood_support/` are internal: the patching and
steering machinery behind `mechanistic_interpretability.py`, and the modeling
pipeline `train_predict.py` loads by path. Neither is meant to be run directly.
