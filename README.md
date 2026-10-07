# Agentic AutoML

A validator-gated Claude Code harness for supervised tabular ML. A thin Python
CLI (`agentic-automl`) launches Claude Code inside a per-trial workspace. The
agent does all ML reasoning and writes all the code, while the Python
validator owns state, schema validation and step ordering.

## Install

```bash
pip install -e ".[dev]"          # launcher + tests (only runtime dep: jsonschema)
pip install -e ".[phase1]"       # modeling stack, for the trial runtime only
```

The modeling stack is expected in the trial runtime (default
`/opt/conda/bin/python`). Point at another interpreter with
`agentic-automl config set-runtime-python <path>`.

## Use

```bash
agentic-automl new --dataset data/churn.csv --goal "predict churn for retention outreach"
agentic-automl resume                 # most recent trial; --fresh / --summarize for a new chat
agentic-automl fork <trial> --name try-xgb
agentic-automl status --trial <trial>
agentic-automl summarize-trials
agentic-automl                        # pick a recent trial interactively
```

Pass Claude Code flags after `--`, e.g. `agentic-automl resume -- --permission-mode plan`.
`--trust` (or `AGENTIC_AUTOML_TRUST=1`) gives a trial broader auto-approval for the
runtime Python and mutating `gsutil`. It is sticky per trial and is not skip-permissions.

Trials live in `trials/<UTC timestamp>[--slug]/` (gitignored). Override the
location with `AGENTIC_AUTOML_TRIALS_DIR`.

## Workflow

```
prediction-contract → data-profiling → workflow-planning
  → parity-ml-preprocessing | data-preprocessing → training
  → [auto-research] → [parity-ml-export] → evaluation
  → optional: explainability, model-development-documentation
```

See `.claude/skills/SKILLS.md` for every skill, the artifact it produces, and its gate.

## Validator (`agentic-automl artifacts`)

All subcommands accept `--trial <dir|id>` (default `$TRIAL_DIR`) and print JSON.

| Command | Purpose |
|---|---|
| `begin-skill <name>` | Gate check; returns `allowed`, `redirect_to`, state snapshot |
| `save-artifact <key> --payload <file> --source-skill <name>` | Schema + gate validation, then canonical write |
| `next-step` | Next skill in the chain, or `canonical_workflow_complete` |
| `status` | Trial snapshot |
| `set-trial-slug <slug>` | Rename applied on next resume |
| `validate-write` | PreToolUse guard (hook JSON on stdin) |

Other sub-tools: `session-analysis`, `feedback-episodes`, `bash-permissions`,
`library-skills`, `overrides list|reconcile`, `config show|set-*`.

## Tests

```bash
pytest
```
