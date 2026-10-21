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
1. Start from the templates:
   `cp templates/aaml_common.py templates/feature_selection.py templates/train.py work/`.
   Shared CV/model code lives in `aaml_common.py`, so both scripts use the same
   folds (time-series for temporal contracts, group folds when
   `entity_id_column` repeats). Adapt them to the plan: families, search
   spaces, imbalance handling.
2. **Smoke run first:** `work/feature_selection.py --smoke` fits one fold per
   family and prints a runtime estimate. Report it, and confirm before heavy
   compute. Run the full job in the background if it takes more than a few minutes.
3. **Feature selection** (on CV, never on test) runs as **its own script**,
   `work/feature_selection.py`, which `train.py` reads, so it can be found,
   re-run and audited separately. It records the plan's drops, removes
   near-duplicates (|Spearman| > 0.95: keep the original over an engineered
   transform), and tunes list-valued `search_space` entries. It then computes
   permutation importance of each raw feature on every validation fold, drops
   features with mean importance ≤ 0 (`--rule stable` is stricter), re-scores,
   and keeps the reduced set only if it is within 1 std of the full set. The
   evidence goes to `outputs/model/feature_selection.json`.
4. `work/train.py` selects the family with a paired per-fold comparison,
   simplest first: a more complex family must beat the incumbent by more than
   the std of the per-fold gap. It refits every candidate on full train and
   saves the winner's CV predictions (for threshold choice in evaluation).
   Use the same `--folds` for both scripts. Put the paired gaps in
   `selection_rationale`.
5. Summary fields: `models[]` (`name`, `library`, `hyperparameters`,
   `cv_score`, plus optional `cv_score_std`, `train_time_seconds`,
   `artifact_path`), `selected_model`, `selection_rationale`,
   `primary_metric`, `cv_strategy`, `training_script`,
   `model_artifact_path`, `random_seed`.
6. Persist, then follow `next-step` (it may route to `auto-research` or
   `parity-ml-export` before `evaluation`).
