---
name: workflow-planning
description: Propose and get approval for the modeling plan (candidate models, validation strategy, preprocessing plan and engine, parity/distributed requirements, risks) from the contract and profiles. Use after data-profiling, and whenever the model family, features or preprocessing engine change.
---

# workflow-planning

## What it does
Turns the contract and the profiles into a concrete plan the user signs off on
before any heavy compute runs.

## Inputs
- `prediction_contract.json`, `dataset_profile.json`, `column_profile.json`.
- The domain overlay, if `ml_domain` is set.

## Output
- `workflow_plan` → `state/workflow_plan.json`
- Flips gate `workflow_plan_approved` (unblocks preprocessing).

## Key notes
- **Preprocessing engine** (`preprocessing_plan.engine`):
  - `parity-ml` (default, used when the field is omitted): ONNX-exportable
    preprocessing, required for `deployment_intent: production_parity`.
  - `open_source`: sklearn/pandas pipeline. Choose it when parity-ml isn't
    installed in the runtime (`import parity_ml` fails), or when the intent is
    `exploratory_baseline` and the user opts out.
  Check availability with a one-line script before you propose `parity-ml`.
  If it's missing and the intent is `production_parity`, tell the user. Don't
  silently downgrade.
- Fields that overlap the contract (e.g. `ml_domain`): **the plan wins**.
- `model_preference` shapes `modeling_plan.candidate_models`:
  `single_baseline` → one model (gradient boosting is the default for
  tabular data); `user_specified` → exactly the contract's `model_space`;
  `auto_research` → the baseline here, with iteration in `auto-research`.
- `scale_assumption: large_data_distributed_tbd` → add a
  `distributed_training_plan` (`backend`, `cluster`, `notes`). Runtime wiring
  (Dataproc/Spark, Ray) is environment-specific, so document the intent and
  confirm it with the user.

## Before you start
```
agentic-automl artifacts begin-skill workflow-planning
```

## How to persist
1. Write `work/workflow_plan.json` and copy it to
   `work/decision_feedback/workflow_plan.proposal.json` before presenting it.
2. After approval:
```
agentic-automl artifacts save-artifact workflow_plan --payload work/workflow_plan.json --source-skill workflow-planning
```

## Data-driven model-family recommendation
Unless the contract fixes the family (`user_specified`), recommend it from
evidence, not habit. Start from the template:
`cp .claude/scripts/aaml_common.py .claude/scripts/model_family_screen.py work/`.
It uses **train only**, with the same encoder and CV folds training will use
(time-ordered folds for temporal contracts, group folds when the entity repeats),
so the screen and training agree. Pass the planned drops with `--drop`:
1. **Characterise the data**: rows, features, share categorical, max
   cardinality, missingness, positive rate, and how many features carry
   univariate signal.
2. **Screen 3–4 families** with quick, sensible defaults, each with the
   preprocessing it needs: a regularised linear model (scaled numerics, log
   transforms, one-hot), a bagged-tree ensemble (random forest / extra trees),
   and gradient boosting (LightGBM/HistGB). Add others only if the data
   suggests them (e.g. text → linear on TF-IDF; very small n → linear/kNN).
3. **Rank by mean CV primary metric**, and read the gaps **paired per fold**
   (same folds for every family: compare the per-fold differences and their
   std, not overlapping error bars):
   - boosting ≫ linear → non-linear effects/interactions → recommend GBDT;
   - paired gap to linear ≤ its std → prefer linear (simpler, interpretable,
     calibrated), and keep the best family as a contender for training to confirm;
   - bagging ≈ boosting with lower variance → bagging is a robust alternative.
   Factor in the contract (`production_parity` → ONNX-exportable families;
   interpretability needs from the domain overlay).
4. Save the evidence to `work/model_family_advice.json`. Put the recommended
   family first in `candidate_models`, plus the linear model as a reference
   baseline (useful even under `single_baseline`). State the gaps in the rationale.

## Feature engineering
Propose features from the profile and the domain, then screen them before
committing them to the plan. Write them in `work/features.py`, starting from
`cp .claude/scripts/features_template.py work/features.py`. Its point-in-time
helpers cover prior counts, time since previous, ratio to history, rolling counts,
novelty flags and cyclical encodings. `aaml_common.split()` applies it before the
split in every later step, and `ENGINEERED` must equal the plan's `engineered_features`:
- **Point-in-time only.** Per-entity history features (time since previous event,
  amount vs the entity's past mean, first-time merchant/country) must use
  strictly earlier rows (sorted by time, `shift`/expanding windows). Never use
  future rows, and never use statistics computed on the full table.
- Prefer ratios, deltas, novelty flags and cyclical/time encodings over raw
  identifiers. A high-cardinality ID with no signal can still be useful as a
  **novelty** feature ("first time this customer used this merchant").
- Screen each engineered feature with the out-of-fold univariate probe and
  include it in the family screen. Keep the decision to retain features for
  training-time selection.

## Steps
1. Read the contract and both profiles. List the decisions the data forces:
   leakage suspects to drop, high-cardinality categoricals, imbalance
   handling, time-awareness. Run the feature-engineering screen and the
   model-family screen above.
2. Draft the plan:
   - `objective`: one sentence.
   - `modeling_plan`: `candidate_models[]` (`name`, `library`, `rationale`,
     optional `search_space`), `validation_strategy` (consistent with the
     contract split, e.g. stratified 5-fold CV on train, or time-series CV
     for temporal splits), `primary_metric` (= contract), plus optional
     `selection_criteria` and `class_imbalance_strategy`.
   - `preprocessing_plan`: `engine`, `drop_columns` (ids, leakage, constants),
     and optionally `categorical_encoding`, `numeric_scaling`, `imputation`,
     `engineered_features`.
   - `parity_requirements` (parity-ml only): `max_abs_diff` (e.g. 1e-5),
     `n_samples`, `opset`.
   - `steps`: the ordered skill chain you will run, including auto-research or
     parity-ml-export if applicable.
   - `risks`, `compute_notes`: expected runtime and memory, and anything needing approval.
3. Present the plan as a short table plus the rationale for each
   non-obvious choice. Call out compute cost. **Wait for explicit approval.**
4. Persist, then follow `next-step`.
