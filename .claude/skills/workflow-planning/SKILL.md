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

## Steps
1. Read the contract and both profiles. List the decisions the data forces:
   leakage suspects to drop, high-cardinality categoricals, imbalance
   handling, time-awareness.
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
