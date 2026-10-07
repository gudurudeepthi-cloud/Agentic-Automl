# Agentic AutoML: runtime instructions

You are working inside one **trial**: a self-contained workspace where a human
describes a dataset and a goal, and you take it from problem definition to an
evaluated model. You are the ML reasoner: you profile, plan, write and run the
code, and make the calls. A Python validator (`agentic-automl artifacts`) owns
the trial's state and enforces step order. It never trains anything. You do
that, by writing and running Python.

## 1. Intake

1. Read `state/run_state.json` (via `agentic-automl artifacts status`) for a
   recorded `goal` and `dataset_path`. If either is missing, ask for it.
2. Route the request:
   - **(A) Build a model** → `agentic-automl artifacts begin-skill prediction-contract`.
   - **(B) Dataset-only questions** (profile, quality, "what's in here") →
     `data-profiling`. It still needs an approved prediction contract first,
     so draft a minimal one with the user.
   - **(C) Anything else** → confirm scope in one sentence before acting.
3. **Explore before asking.** Inspect file formats, columns, dtypes, row counts,
   and split layout yourself. The only things you ask for are:
   - the dataset path,
   - the **target column** (always asked, never inferred from values or names),
   - the business context (what the prediction is used for).

   Everything else you propose, and the user confirms or corrects.
4. Once dataset and goal are clear, name the trial:
   `agentic-automl artifacts set-trial-slug <short-kebab-slug>` (applied on next resume).

## 2. Who owns what

| Wrapper (validator) owns | You own |
|---|---|
| Directory layout, `state/run_state.json` | ML reasoning and decisions |
| JSON-schema validation of every artifact | All code under `work/` |
| Gates and step ordering | Artifact payloads (drafted in `work/`) |

**Never write an artifact path or `state/` directly.** A PreToolUse hook rejects
it. The registered paths are `prediction_contract.json`,
`dataset_profile.json`, `column_profile.json`, `state/workflow_plan.json`, and
the `*_summary.json` / `onnx_deployment.json` /
`model_development_documentation.json` files under `outputs/`. Always:

1. Write the draft to `work/<artifact_key>.json`.
2. `agentic-automl artifacts save-artifact <key> --payload work/<key>.json --source-skill <skill>`.
3. On rejection (non-zero exit, JSON `errors`), fix the payload and retry. If a
   gate is unmet, run the skill named in `redirect_to`. Never work around the
   validator.

## 3. Trial layout

```
work/                 lab notebook: your scripts, drafts, scratch outputs
outputs/pipeline/     fitted preprocessing pipeline + preprocessing_summary.json
outputs/model/        model files, training/auto-research/ONNX summaries
outputs/report/       evaluation, explainability, documentation, plots
preprocessed_data/    train/test parquet produced by preprocessing
state/                wrapper-owned (run_state.json, workflow_plan.json); read-only to you
```

- `work/` is a lab notebook. Give scripts descriptive names
  (`work/profile_columns.py`, `work/train_lightgbm.py`), not `test.py` or `tmp2.py`.
- Deliverables go in `outputs/`. Scratch never does.

## 4. Python runtime

- Use **`/opt/conda/bin/python`** for all data and ML work (pandas, sklearn,
  lightgbm, and so on). It has the modeling stack, and the launcher does not.
- **Write every script to a file under `work/`, then run it.** No heredocs
  (`python - <<EOF`) and no long `python -c` one-liners. They bypass
  auto-approval and leave nothing reproducible behind.
- Keep `agentic-automl ...` for validator and lifecycle commands only.
- Single commands only. No `&&`, `;` or pipes when one command will do.
  Chained commands always need manual approval.

## 5. Validator CLI contract

| When | Command |
|---|---|
| Before any work for a skill | `agentic-automl artifacts begin-skill <skill>`. If `allowed` is false, go to `redirect_to` and tell the user why. |
| To persist an artifact | `agentic-automl artifacts save-artifact <key> --payload work/<key>.json --source-skill <skill>` |
| Unsure what's next | `agentic-automl artifacts next-step` (derived from the artifacts on disk, not your memory) |
| Inspect state | `agentic-automl artifacts status` |
| Dataset + goal are clear | `agentic-automl artifacts set-trial-slug <slug>` |

`--trial` defaults to `$TRIAL_DIR`, which the launcher sets.

If `begin-skill` reports `rerun: true` with `downstream_artifacts_present`, you are
going back to an earlier step. See `.claude/trial-backtrack.md`.

## 6. Workflow

Canonical baseline chain (each skill's `SKILL.md` has the recipe):

```
prediction-contract → data-profiling → workflow-planning
  → data-preprocessing | parity-ml-preprocessing → training → evaluation
```

Branches:
- **Preprocessing engine** comes from `workflow_plan.preprocessing_plan.engine`:
  `parity-ml` (default) → `parity-ml-preprocessing`; `open_source` →
  `data-preprocessing`.
- **parity-ml path:** `parity-ml-preprocessing → training → parity-ml-export → evaluation`.
  Evaluation runs on raw inputs through the fused ONNX graph (preprocessing + model).
- **Auto-research:** when `prediction_contract.model_preference == "auto_research"`,
  `auto-research` runs between training and evaluation (budget
  `research_max_experiments`, default 15).
- **Optional, after evaluation:** `explainability`, `model-development-documentation`.

**Skill overrides:** after reading a platform `SKILL.md`, check
`~/.agentic-automl/skills/overrides/<skill>.md`. If it exists, follow it where
it conflicts (the override wins).

**Domain overlays:** if `ml_domain` is set (the workflow plan wins over the
contract), read the overlay listed in `.claude/skills/domains/registry.json`.

## 7. Confirmation UX

- Before any Bash command whose purpose isn't obvious, write a 1–2 sentence preamble.
- **Confirm with the user before** approving the prediction contract,
  approving the workflow plan, and starting any heavy compute (large training
  runs, hyperparameter searches, distributed jobs).
- Long jobs: run with `run_in_background: true` and read the output with the
  background-task output tool. Don't poll with `sleep` or `pgrep`.
- Report results plainly: metric values, what was checked, and what wasn't.

## 8. Data access (GCS)

- Credentials are inherited from the environment. Run `gsutil` and `gcsfs` bare.
- Fall back to an explicit credentials path only after an actual credentials error.
- Verify GCS access with Python (`gcsfs` / `pandas.read_*("gs://...")`), not `gsutil`.
- Mutating `gsutil` commands (`cp`, `mv`, `rm`, `rsync`) need user approval
  unless the trial runs in trust mode (`work/trust.md`).

## 9. Feedback

When something fails and you recover, or the user corrects a decision, record
it so the platform can improve:
`agentic-automl feedback-episodes insight|rationale ...` (see the
`aaml-feedback` skill). Personal workflow tweaks go through the `aaml-override`
skill. Never edit platform skills in place.
