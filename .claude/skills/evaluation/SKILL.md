---
name: evaluation
description: Evaluate the selected model once on the held-out test split - contract metrics, trust checks (leakage, too-good-to-be-true, split integrity), per-segment results, plots and a plain-language narrative. Use after training (and after auto-research / parity-ml-export when the chain includes them); completes the canonical workflow.
---

# evaluation

## What it does
The single, honest look at the held-out data. Computes the contract's
metrics, checks whether the result can be trusted, and explains it to the
business owner.

## Inputs
- `preprocessed_data/test.parquet` and the model chosen as follows: the
  auto-research best model if `auto_research_summary` exists, otherwise
  `training_summary.model_artifact_path`. On the parity-ml path, use
  **raw** test rows through `outputs/model/model.onnx`.
- `prediction_contract.json` (`primary_metric`, `report_metrics`, labels).

## Output
- `evaluation_summary` → `outputs/report/evaluation_summary.json`
- Flips gate `evaluation_complete`: the canonical workflow is complete.
  `next-step` then offers `explainability` and `model-development-documentation`.
- Plots in `outputs/report/`.

## Key notes
- **Evaluate once.** Don't tune thresholds or models on test. If you choose a
  decision threshold, choose it on CV/out-of-fold train predictions and only
  apply it here.
- **Trust checks are required.** Each is `{name, passed, detail}`. Include at least:
  - `leakage`: no feature in `column_profile` marked `leakage_suspect` was used,
    and the top-feature importances are plausible;
  - `too_good_to_be_true`: the metric isn't implausibly high for the domain.
    With rare positives (under ~1%), ROC-AUC near 0.99 is normal and tells you
    little. Judge by PR-AUC relative to the base rate, and by whether a single
    feature carries most of the importance or nearly separates the classes alone;
  - `cv_vs_test_gap`: the test metric is within a reasonable distance of the CV score;
  - `split_integrity`: test rows and target rate match `dataset_profile.split_summary`;
  - on the parity path, `onnx_parity` (from `onnx_deployment.parity_check`).
  A failed check doesn't block saving. It must be visible in the narrative.
- `baseline_comparison`: compare against a trivial baseline (prior / majority
  class / mean predictor), and against an earlier trial if the user names one.

## Before you start
```
agentic-automl artifacts begin-skill evaluation
```

## How to persist
Write `work/evaluation_summary.json`, then:
```
agentic-automl artifacts save-artifact evaluation_summary --payload work/evaluation_summary.json --source-skill evaluation
```

## Steps
1. Start from the template:
   `cp .claude/scripts/aaml_common.py .claude/scripts/evaluate.py work/`.
   It refuses to run once `evaluation_summary` exists, unless you pass
   `--rerun-reason`, which is recorded. It picks any threshold on the CV
   predictions (`--threshold-rule f1|top:<frac>`) and computes a bootstrap CI for
   the primary metric, which the `cv_vs_test_gap` tolerance uses. Never paste in
   noise or "leaked model" figures from elsewhere. Rewrite its draft `narrative`
   for the business owner. `work/evaluate.py` loads test and the model, predicts, computes
   `primary_metric` and every `report_metric`, the trivial baseline,
   per-segment metrics (by `data_segments` roles or a business segment column
   if the user named one), and plots (ROC/PR or residuals, calibration,
   confusion matrix) into `outputs/report/`.
2. Run the trust checks and record each one.
3. Write `narrative`: 3–6 sentences for a non-technical reader covering what
   the model does, how good it is relative to the baseline, whether it can
   be trusted, and the main caveat.
4. Summary fields: `model_evaluated`, `split_evaluated`, `primary_metric`,
   `primary_metric_value`, `metrics`, `baseline_comparison`, `trust_checks`,
   `narrative`, `plots`, `findings` (bullet-sized facts), `evaluation_script`,
   plus optional `threshold` and `per_segment`.
5. Persist and present: metric table, trust check results, narrative. Then
   offer the optional skills.
