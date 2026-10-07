---
name: explainability
description: Explain the trained model - global feature importance (SHAP or permutation), dependence plots and sanity findings - and record explainability_summary. Optional; offer after evaluation, or when the user asks why the model predicts what it does.
---

# explainability

## What it does
Shows which features drive the model and checks that they make domain sense.
It's often the fastest way to catch leakage that evaluation missed.

## Inputs
- The model used in evaluation, `preprocessed_data/test.parquet` (or a
  sample of it), `column_profile.json`, and the preprocessing pipeline for
  mapping encoded features back to raw names.

## Output
- `explainability_summary` → `outputs/report/explainability_summary.json`
- Flips gate `explainability_complete`.
- Plots in `outputs/report/` (summary/beeswarm, top dependence plots).

## Key notes
- Prefer **SHAP** (`TreeExplainer` for tree models). Fall back to
  **permutation importance** on the primary metric if shap isn't installed
  (it's in the optional `xai` extra) or the model isn't supported. Name the
  method you used.
- Aggregate one-hot or encoded columns back to their raw feature for global
  importance, so the result is readable.
- Use a sample (e.g. 2,000 rows, fixed seed) for SHAP on large data.
- Any top feature that is marked `leakage_suspect`, or that is
  implausible for the domain, is a **finding**. Raise it with the user.

## Before you start
```
agentic-automl artifacts begin-skill explainability
```

## How to persist
Write `work/explainability_summary.json`, then:
```
agentic-automl artifacts save-artifact explainability_summary --payload work/explainability_summary.json --source-skill explainability
```

## Steps
1. Write `work/explain.py`. It computes global importance (sorted
   `{feature, importance}`), saves plots, and writes the summary draft.
2. Review the top 10 features against the column profile and domain sense.
3. Summary fields: `method`, `global_importance[]`, `plots`, `findings`,
   `explainability_script`, plus optional `notes`.
4. Persist and present the top drivers in plain language.
