---
name: data-preprocessing
description: Build, fit and persist the open-source (sklearn/pandas) preprocessing pipeline from the approved workflow plan, and write train/test parquet. Use when workflow_plan.preprocessing_plan.engine is open_source.
---

# data-preprocessing (open_source engine)

## What it does
Implements `preprocessing_plan` as a reproducible sklearn pipeline, fit on
train only, and materialises model-ready train/test data.

## Inputs
- `state/workflow_plan.json` (`preprocessing_plan`), `prediction_contract.json` (split),
  `column_profile.json` (roles, types).

## Output
- `preprocessing_summary` → `outputs/pipeline/preprocessing_summary.json`
  with **`engine: "open_source"`** (the validator rejects anything else from this skill).
- Flips gate `preprocessing_complete` (unblocks `training`).
- Files: `outputs/pipeline/preprocessor.joblib`, `preprocessed_data/{train,test}.parquet`.

## Key notes
- **Fit on train only.** Test passes through `transform` only. Anything that
  learns from data (imputers, encoders, scalers, target encoding) goes inside
  the pipeline.
- The split must match `dataset_profile.split_summary` row counts exactly.
- Keep the target, id and weight columns out of the feature matrix, but carry
  id and weight alongside in the parquet files for evaluation.
- Prefer `ColumnTransformer` + `Pipeline`. Use target encoding only with
  out-of-fold fitting.

## Before you start
```
agentic-automl artifacts begin-skill data-preprocessing
```

## How to persist
Write `work/preprocessing_summary.json`, then:
```
agentic-automl artifacts save-artifact preprocessing_summary --payload work/preprocessing_summary.json --source-skill data-preprocessing
```

## Steps
1. Start from the template:
   `cp templates/aaml_common.py templates/preprocess.py work/`
   (plus `work/features.py` from planning, if the plan has engineered features).
   It already does the steps below. It fails if a leakage suspect isn't dropped,
   if a high-cardinality categorical would be one-hot encoded, or if row counts
   differ from the profile. It also records the raw-feature → encoded-column map
   that feature selection needs. Adapt it to the plan's encoding/imputation
   choices. `work/preprocess.py` should:
   1. load the segments and apply the contract split;
   2. drop `preprocessing_plan.drop_columns`;
   3. build the `ColumnTransformer` from the plan (imputation, encoding, scaling);
   4. fit on train, then transform train and test;
   5. save `outputs/pipeline/preprocessor.joblib` and
      `preprocessed_data/train.parquet` / `test.parquet` (features, target,
      plus id/weight when present);
   6. write the summary draft.
2. Run it and check for: no NaNs left in the outputs (unless the model
   handles them), matching column sets between train and test, and row counts
   equal to the profile.
3. Summary fields: `engine` (`open_source`), `pipeline_script`,
   `pipeline_artifact`, `input_features`, `output_feature_count`,
   `transformations[]` (`{step, columns, description}`), `dropped_columns`,
   `train_rows`, `test_rows`, plus optional `validation_rows` and
   `leakage_checks` (what you verified).
4. Persist, and report feature counts before and after to the user.
