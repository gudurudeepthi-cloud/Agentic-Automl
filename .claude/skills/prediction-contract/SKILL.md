---
name: prediction-contract
description: Define and get user approval for the prediction contract (target, task, data segments, split, metrics, model preference, deployment intent). Use first for any request to build, train or evaluate a model, and whenever the target, split or metric changes.
---

# prediction-contract

## What it does
Turns the user's goal and dataset into a precise, approved problem definition:
what is predicted, from which files, how train and test are separated, how
success is measured, and what kind of modeling is wanted. Every later step
reads this contract.

## Inputs
- `goal` / `dataset_path` from `agentic-automl artifacts status` (or ask).
- The data itself. Inspect it before asking anything.

## Output
- Artifact `prediction_contract` → `prediction_contract.json`
- Flips gate `prediction_contract_approved` (unblocks `data-profiling`).

## Key notes
- **Ask, never infer, the target column.** Propose candidates if you like, but
  the user names it.
- Ask for business context (who uses the prediction, for what decision). It
  drives the metric choice.
- Everything else you **propose** from inspection, and the user confirms.
- Problem semantics (`task_type`, `primary_metric`, `report_metrics`,
  `model_space`) are free strings. Enums exist only where the platform branches.
- `dataset_location` is `gcs` iff `dataset_path` starts with `gs://`.
- Domain overlay: if the problem matches an entry in
  `.claude/skills/domains/registry.json` (e.g. `credit_risk`), propose `ml_domain`
  and read that overlay before finalising metrics and split.

## Before you start
```
agentic-automl artifacts begin-skill prediction-contract
```
If `rerun: true`, read `.claude/trial-backtrack.md` first.

## How to persist
1. Write the proposal to `work/prediction_contract.json`, **and copy it to
   `work/decision_feedback/prediction_contract.proposal.json`** before showing
   it to the user. The validator diffs proposal against approval for decision
   feedback.
2. After the user approves (with any edits applied to `work/prediction_contract.json`):
```
agentic-automl artifacts save-artifact prediction_contract --payload work/prediction_contract.json --source-skill prediction-contract
```

## Steps
1. **Inspect the data** with a script (`work/inspect_dataset.py`, run with the
   runtime Python): list the files under `dataset_path`, their formats and
   counts, then read a sample for columns, dtypes, row count and null rates.
   For GCS, use `gcsfs`/pandas.
2. **Determine `data_segments`**: one entry per physical file group
   `{path, format, role, file_count}`. Use either a single `full` segment or
   separate `train` + `test` segments. `other` covers extra files such as
   out-of-time samples.
3. **Ask the three questions** you can't answer yourself: confirm the dataset
   path if it's ambiguous, ask for the target column, and ask for the
   business context.
4. **Derive the target facts**: `target_kind` (binary / multiclass /
   continuous / other) and `target_labels` from the actual values: exactly
   2 for binary, 3 or more for multiclass, `null` for continuous.
5. **Choose the split**, recommending one from what you saw:
   - separate train/test files → `pre_split` (all split extras `null`)
   - a fold/split column → `predefined_column` + `split_column`, `split_train_value`, `split_test_value`
   - time-ordered data with leakage risk → `temporal` + `time_column` + `split_ratio`
   - otherwise → `random_stratified` (classification) or `random`, with `split_ratio` (e.g. 0.2)
   Unused split fields must be `null`. The schema enforces this.
6. **Metrics**: pick `primary_metric` to match the business decision (e.g.
   `roc_auc` for ranking, `pr_auc` for rare positives, `rmse`/`mae` for
   regression) plus 2–4 `report_metrics`.
7. **Model preference + deployment intent**:
   - `deployment_intent`: `exploratory_baseline` (default) or `production_parity`
     (needs a deployable ONNX artifact; implies the parity-ml path).
   - `model_preference`: `single_baseline` (one strong default model; with
     `exploratory_baseline`, `model_space` must be `null`), `user_specified`
     (`model_space` lists at least one family), or `auto_research` (iterate
     after the baseline; `model_space` = optional hints; optional
     `research_max_experiments`, default 15).
   - `scale_assumption`: `local_workstation` unless the data clearly needs a
     cluster (`large_data_distributed_tbd`).
8. Fill the remaining fields: `task_type` (short free text), `data_source`
   (internal / external), `schema_files` (data dictionaries, or `[]`),
   `entity_id_column`, `weight_column` (`null` if none).
9. Show the user a compact table of the proposal and the reasoning for the
   split and metric. **Wait for explicit approval.** Then persist.
10. If you haven't already, set the slug: `agentic-automl artifacts set-trial-slug <slug>`.
