---
name: auto-research
description: Iteratively improve on the trained baseline with a budgeted hypothesis -> change -> CV-measure loop (features, model families, hyperparameters), keeping only changes that beat the incumbent. Use when prediction_contract.model_preference is auto_research; runs between training and evaluation.
---

# auto-research

## What it does
A disciplined experiment loop on top of the training baseline. Each iteration
states one hypothesis, makes one change, measures it with the **same CV** as
training, and keeps the change only if it beats the incumbent.

## Inputs
- `training_summary.json` (baseline score, CV strategy, seed),
  `preprocessed_data/train.parquet`, the plan, and the profiles.
- Budget: `prediction_contract.research_max_experiments` (default **15**).
  `model_space` entries are hints, not limits.

## Output
- `auto_research_summary` → `outputs/model/auto_research_summary.json`
- Flips gate `auto_research_complete`. `evaluation` then uses `best_model_artifact_path`.
- Files: `outputs/model/model_auto_research.joblib` (the best model) and one
  script per experiment under `work/research/`.

## Key notes
- **Test data stays untouched.** Every comparison is CV on train with the
  training folds and seed.
- One change per iteration, so the effect can be attributed. Changes that
  touch preprocessing must stay inside a pipeline fit per fold.
- "Beats" means the mean CV improves by more than the fold standard deviation
  (or a threshold you state up front). Otherwise keep the simpler incumbent.
- Stop early when 3 consecutive iterations fail to improve, or when the
  budget or compute limit is hit. Record `stop_reason`.
- Confirm the total compute estimate with the user before starting the loop.

## Before you start
```
agentic-automl artifacts begin-skill auto-research
```

## How to persist
Write `work/auto_research_summary.json`, then:
```
agentic-automl artifacts save-artifact auto_research_summary --payload work/auto_research_summary.json --source-skill auto-research
```

## Steps
1. Read the baseline CV score. Write a ranked list of hypotheses from the
   profiles (e.g. "target-encode high-cardinality `merchant_id`", "class
   weights for 2% positives", "monotone constraints", "try CatBoost").
2. For each iteration: write `work/research/iter_<n>.py`, run it, and record
   `{iteration, hypothesis, change, metric_value, kept, script}`.
3. After the loop, refit the best configuration on full train and save it.
4. Summary fields: `primary_metric`, `baseline_metric`, `best_metric`,
   `iterations[]`, `best_model_artifact_path`, `stop_reason`, `notes` (what
   was learned, including the negative results).
5. Persist, then follow `next-step` (→ `evaluation`).
