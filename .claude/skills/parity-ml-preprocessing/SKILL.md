---
name: parity-ml-preprocessing
description: Build the ONNX-exportable parity-ml preprocessing pipeline from the approved workflow plan (the default engine), fit on train, and write train/test parquet. Use when workflow_plan.preprocessing_plan.engine is parity-ml or omitted.
---

# parity-ml-preprocessing (default engine)

## What it does
Implements `preprocessing_plan` with the **parity-ml** library, so that
preprocessing can later be fused with the model into one ONNX graph that takes
raw inputs. This is what makes training-serving parity checkable.

## Inputs
- `state/workflow_plan.json` (`preprocessing_plan`, `parity_requirements`),
  `prediction_contract.json`, `column_profile.json`.

## Output
- `preprocessing_summary` → `outputs/pipeline/preprocessing_summary.json`
  with **`engine: "parity-ml"`**.
- Flips gate `preprocessing_complete`. After training, `next-step` routes to
  `parity-ml-export` before evaluation.
- Files: the fitted parity-ml pipeline under `outputs/pipeline/`, and
  `preprocessed_data/{train,test}.parquet`.

## Key notes
- **Environment-specific integration.** parity-ml is an internal library.
  Its exact API, install location and supported transforms depend on your
  runtime image. Before writing code:
  1. check that `import parity_ml` works in the runtime Python, and read its
     docstrings or README for the transform and fit API;
  2. if it isn't available, **stop** and tell the user. Offer to go back to
     `workflow-planning` and switch to `engine: open_source` (see
     `.claude/trial-backtrack.md`). Don't hand-roll an imitation and label it parity-ml.
- Use only transforms the library can export to ONNX. Keep any feature
  engineering it can't express out of the pipeline, and flag it in the plan.
- The same rules as open-source preprocessing apply: fit on train only, split
  counts match the profile, and target/id/weight stay out of the features.

## Before you start
```
agentic-automl artifacts begin-skill parity-ml-preprocessing
```

## How to persist
Write `work/preprocessing_summary.json`, then:
```
agentic-automl artifacts save-artifact preprocessing_summary --payload work/preprocessing_summary.json --source-skill parity-ml-preprocessing
```

## Steps
1. Verify the library is available (one small script: `work/check_parity_ml.py`).
2. Write `work/preprocess_parity_ml.py`. It loads and splits the data per the
   contract, builds the parity-ml pipeline from the plan, fits it on train,
   transforms both splits, saves the fitted pipeline and the parquet files,
   and writes the summary draft.
3. Spot-check that transformed train/test have identical schemas and no
   unexpected NaNs.
4. Summary: same fields as `data-preprocessing`, with `engine: "parity-ml"`
   and `pipeline_artifact` pointing to the fitted parity-ml pipeline.
5. Persist, then follow `next-step` (→ `training`).
