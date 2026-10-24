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
- **Interview first, then draft.** Before any contract is written, run the intake
  interview below. Every answer is either the user's own words or a proposal
  from inspection that the user explicitly confirmed. Nothing is silently assumed.
- **Ask, never infer, the target column.** Propose candidates if you like, but
  the user names it.
- Ask for business context (who uses the prediction, for what decision). It
  drives the metric choice.
- Record the interview in `intake_answers`, plus `project_name`,
  `prediction_point`, `required_features` and `excluded_columns`. Profiling
  turns `excluded_columns` into role `drop`, and feature selection never
  removes `required_features`.
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
1. **Get the dataset path** (from `status`, or ask). Nothing else can start without it.
2. **Inspect the data**: `cp templates/inspect_dataset.py work/` (with
   `aaml_common.py`), then `<runtime python> work/inspect_dataset.py <path> [<path> ...]`.
   It lists files, formats, columns, dtypes and nulls, and writes
   `work/intake_candidates.json` with proposals: target, time, entity-id, row-id,
   split-column and "ask if post-outcome" columns. These are prompts for questions,
   not decisions.
3. **Determine `data_segments`**: one entry per physical file group
   `{path, format, role, file_count}`. Use either a single `full` segment or
   separate `train` + `test` segments. `other` covers extra files such as
   out-of-time samples.
4. **Run the intake interview.** Ask in at most two rounds, grouped, with your
   proposal pre-filled wherever inspection gives one, so the user mostly confirms.
   Use the `AskUserQuestion` tool for the choice questions (candidate columns as
   options) and plain numbered questions for free text. Ask every item below; skip
   one only when the user already answered it in this conversation, and say so.

   **Round 1: the problem**
   | # | Question | Pre-fill from | Contract field |
   |---|---|---|---|
   | 1 | Project / use-case name | the user's request | `project_name` |
   | 2 | Who uses the prediction, for what decision, how often | the user's request | `business_context` |
   | 3 | **Target column** and the positive class (or what a "good" value is) | candidates listed, never chosen | `target_column`, `target_labels` |
   | 4 | **When is the prediction made**, and what is known by then? | — | `prediction_point` |
   | 5 | **Date/time column**: which column dates each row, and should the test set be the latest period? | `candidates.time` | `time_column`, `split_strategy` |
   | 6 | **Entity**: does a customer/account/device repeat across rows? Which column identifies it? | `candidates.entity_id` | `entity_id_column` |

   **Round 2: features and success**
   | # | Question | Pre-fill from | Contract field |
   |---|---|---|---|
   | 7 | **Columns that must not be used**: recorded after the outcome, unavailable at prediction time, PII, or barred by policy | `candidates.ask_if_post_outcome`, row ids | `excluded_columns` (with a reason each) |
   | 8 | **Columns that must be kept** as features (business or regulatory reasons) | — | `required_features` |
   | 9 | Existing split column or separate train/test files? | `candidates.split_column`, inventory | `split_strategy` |
   | 10 | Cost of errors (false alarm vs miss, or over/under-prediction); any metric you already report on | — | `primary_metric`, `report_metrics` |
   | 11 | Exploration or production (ONNX parity)? One baseline, named families, or auto-research? | default `exploratory_baseline` + `single_baseline` | `deployment_intent`, `model_preference` |
   | 12 | Row weights, or a data dictionary to read? | — | `weight_column`, `schema_files` |

   Ask questions 4, 5 and 7 even when the data seems to settle them. They are where
   leakage comes from: a date column decides whether a random split leaks the
   future, and a column filled after the outcome looks like a perfect feature.
   If an answer contradicts the data (the named date column doesn't parse, or the
   target has 5 levels when the user said yes/no), show what you saw and ask again.
   Don't pick for them.
5. **Derive the target facts**: `target_kind` (binary / multiclass /
   continuous / other) and `target_labels` from the actual values: exactly
   2 for binary, 3 or more for multiclass, `null` for continuous.
6. **Choose the split** from the answers to questions 5 and 9:
   - separate train/test files → `pre_split` (all split extras `null`)
   - a fold/split column → `predefined_column` + `split_column`, `split_train_value`, `split_test_value`
   - a date column and the model will score future rows → `temporal` + `time_column` + `split_ratio`
   - otherwise → `random_stratified` (classification) or `random`, with `split_ratio` (e.g. 0.2)
   Unused split fields must be `null`. The schema enforces this. If the user named
   a date column but you recommend a random split, say why. The schema allows
   `time_column` only with a temporal split, so otherwise record the date column
   in `intake_answers` and `notes`, and exclude it or engineer features from it
   (as the user decided).
7. **Metrics**: pick `primary_metric` to match the cost of errors from question 10
   (e.g. `roc_auc` for ranking, `pr_auc` for rare positives, `rmse`/`mae` for
   regression) plus 2–4 `report_metrics`.
8. **Model preference + deployment intent**:
   - `deployment_intent`: `exploratory_baseline` (default) or `production_parity`
     (needs a deployable ONNX artifact; implies the parity-ml path).
   - `model_preference`: `single_baseline` (one strong default model; with
     `exploratory_baseline`, `model_space` must be `null`), `user_specified`
     (`model_space` lists at least one family), or `auto_research` (iterate
     after the baseline; `model_space` = optional hints; optional
     `research_max_experiments`, default 15).
   - `scale_assumption`: `local_workstation` unless the data clearly needs a
     cluster (`large_data_distributed_tbd`).
9. Fill the remaining fields: `task_type` (short free text), `data_source`
   (internal / external), `schema_files` (data dictionaries, or `[]`),
   `entity_id_column`, `weight_column` (`null` if none), and the intake fields:
   `project_name`, `prediction_point`, `excluded_columns` (`[{column, reason}]`),
   `required_features`, and `intake_answers`
   (`[{question, answer, source: user|confirmed_proposal|default}]`, one per question asked).
10. Show the user a compact table of the proposal: the intake answers first, then
    the split and metric with the reasoning for each. **Wait for explicit approval.**
    Then persist. The validator rejects contradictions (target excluded, a column
    both required and excluded, or an id/time column listed as a required feature).
11. If you haven't already, set the slug from the project name:
    `agentic-automl artifacts set-trial-slug <slug>`.
