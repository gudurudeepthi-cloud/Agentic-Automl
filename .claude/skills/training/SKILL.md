---
name: training
description: Train the planned candidate models on preprocessed train data with cross-validation, select one, and persist it with a training_summary. Use after preprocessing is complete; the default modeling engine.
---

# training

## What it does
Fits each `modeling_plan.candidate_models` entry with the planned validation
strategy, compares them on the primary metric, and saves the selected model.

## Inputs
- `preprocessed_data/train.parquet`, `state/workflow_plan.json` (`modeling_plan`),
  `prediction_contract.json` (metric, weights).

## Output
- `training_summary` → `outputs/model/training_summary.json`
- Flips gate `training_complete` (unblocks evaluation, explainability,
  auto-research and parity-ml-export).
- Files: `outputs/model/model.joblib` (or the library's native format), plus
  per-candidate artifacts if useful.

## Key notes
- **Never touch the test split here.** Selection uses CV on train only.
- Fix and record `random_seed`. Use `weight_column` as `sample_weight` if set.
- Respect `validation_strategy`: stratified K-fold for classification,
  time-series split for temporal contracts, grouped folds when
  `entity_id_column` repeats.
- **Confirm before heavy compute.** Estimate the runtime from a single fold
  first. For anything over a few minutes, tell the user and run it in the
  background (`run_in_background: true`).
- Early stopping should use a validation fold carved from train, not test.

## Before you start
```
agentic-automl artifacts begin-skill training
```

## How to persist
Write `work/training_summary.json`, then:
```
agentic-automl artifacts save-artifact training_summary --payload work/training_summary.json --source-skill training
```

## Steps
1. Write `work/train_<model>.py` (one per candidate, or one parameterised
   script). It loads train, runs CV, logs fold scores, refits on full train,
   saves the model, and appends its result to the summary draft.
2. Run a quick single-fold smoke run, report the timing estimate, then run the
   full CV.
3. **Feature selection** (on CV, never on test), as **its own script**,
   `work/feature_selection.py`, that the training script reads, so it can be
   found, re-run and audited separately. Put shared CV/model code in a module
   (e.g. `work/modeling.py`) so both scripts use the same folds. Drop near-duplicates first
   (|Spearman| > 0.95: keep the more interpretable one), then compute
   permutation importance of each raw feature on every validation fold.
   Remove features whose mean importance is ≤ 0, re-score, and keep the
   reduced set only if it is within 1 std of the full set. Save the evidence
   to `outputs/model/feature_selection.json`.
4. Select the model by mean CV primary metric. Prefer the simpler model when
   scores overlap within a standard deviation, and say so in `selection_rationale`.
5. Summary fields: `models[]` (`name`, `library`, `hyperparameters`,
   `cv_score`, plus optional `cv_score_std`, `train_time_seconds`,
   `artifact_path`), `selected_model`, `selection_rationale`,
   `primary_metric`, `cv_strategy`, `training_script`,
   `model_artifact_path`, `random_seed`.
6. Persist, then follow `next-step` (it may route to `auto-research` or
   `parity-ml-export` before `evaluation`).
