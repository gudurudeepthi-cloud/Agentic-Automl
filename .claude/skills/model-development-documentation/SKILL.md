---
name: model-development-documentation
description: Write the model development document (MDD) - problem, data, methodology, results, trust checks, limitations - assembled from the trial's validated artifacts. Optional; offer after evaluation, or when the user asks for a write-up, model card or review document.
---

# model-development-documentation

## What it does
Assembles a reviewer-ready Markdown document from the artifacts. Every number
in it comes from a saved artifact, not from memory.

## Inputs
- All saved artifacts (contract, profiles, plan, preprocessing, training,
  optional auto-research / ONNX, evaluation, optional explainability), the
  plots in `outputs/report/`, and the domain overlay if `ml_domain` is set
  (it may require extra sections).

## Output
- `model_development_documentation` → `outputs/report/model_development_documentation.json`
- Flips gate `model_development_documentation_complete`.
- The document: `outputs/report/model_development_documentation.md`.

## Key notes
- **Generate, don't transcribe.** Write `work/build_mdd.py`, which reads the
  artifact JSONs and renders the Markdown, so that re-running after a change
  keeps the document accurate.
- State limitations honestly: failed trust checks, sampling, untested
  segments, parity results, and the assumptions in `scale_assumption`.
- Keep it scannable: tables for metrics and data, short paragraphs for reasoning.

## Before you start
```
agentic-automl artifacts begin-skill model-development-documentation
```

## How to persist
Write `work/model_development_documentation.json`, then:
```
agentic-automl artifacts save-artifact model_development_documentation --payload work/model_development_documentation.json --source-skill model-development-documentation
```

## Steps
1. Write and run `work/build_mdd.py`. Default sections:
   1. Executive summary (from `evaluation_summary.narrative`)
   2. Business context and prediction contract
   3. Data: sources, segments, split, profile highlights, quality flags
   4. Methodology: preprocessing, candidates, validation, selection
      (+ auto-research log)
   5. Results: metrics against the baseline, per-segment results, plots
   6. Trust checks and explainability
   7. Deployment (ONNX parity), if applicable
   8. Limitations and recommended next steps
   9. Reproducibility: scripts, seeds, package version, trial id
2. Summary fields: `document_path`, `summary` (2–3 sentences), `sections`
   (the headings you wrote), `limitations`, plus optional `generated_at`.
3. Persist, and point the user to the document path.
