---
name: parity-ml-export
description: Fuse the parity-ml preprocessing pipeline and the trained model into one ONNX graph on raw inputs, verify numerical parity against the Python pipeline, and record onnx_deployment. Use after training when preprocessing used the parity-ml engine (next-step suggests it before evaluation).
---

# parity-ml-export

## What it does
Produces the deployable artifact: one ONNX model that takes **raw** feature
columns and returns predictions, with a measured parity check against the
Python preprocessing + model.

## Inputs
- The fitted parity-ml pipeline (`preprocessing_summary.pipeline_artifact`),
  the selected model (`training_summary.model_artifact_path`), and
  `workflow_plan.parity_requirements`.

## Output
- `onnx_deployment` → `outputs/model/onnx_deployment.json`
- Flips gate `onnx_deployment_complete`. Evaluation then runs through the ONNX graph.
- File: `outputs/model/model.onnx`.

## Key notes
- **Environment-specific integration.** Fusion and export use the parity-ml
  library's export API (and onnxruntime for the check). Confirm both import in
  the runtime Python and read the library docs for the fuse/export call. If
  either is missing, stop and tell the user.
- The parity check compares Python predictions with ONNX predictions on the
  same raw rows (at least `parity_requirements.n_samples`, default 1000,
  drawn from test). `passed` means `max_abs_diff <= tolerance` (default
  `parity_requirements.max_abs_diff`, else 1e-5).
- **Never loosen the tolerance to make it pass.** A failure is a finding. Report
  it and investigate (dtype casts, category handling, missing-value encoding).

## Before you start
```
agentic-automl artifacts begin-skill parity-ml-export
```

## How to persist
Write `work/onnx_deployment.json`, then:
```
agentic-automl artifacts save-artifact onnx_deployment --payload work/onnx_deployment.json --source-skill parity-ml-export
```

## Steps
1. Write `work/export_onnx.py`. It loads the pipeline and model, fuses and
   exports them to `outputs/model/model.onnx` at the planned `opset`, then
   runs the parity check on raw test rows with onnxruntime.
2. Record `input_signature[]` (`name`, `dtype`, `shape`) from the exported graph.
3. Summary: `onnx_path`, `opset`, `input_signature`, `parity_check`
   (`passed`, `max_abs_diff`, `tolerance`, `n_samples`), `export_script`,
   `fused_preprocessing: true`.
4. Persist even when the check fails (`passed: false`) so the result is on
   record, and tell the user before moving on to evaluation.
