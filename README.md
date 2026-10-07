# Agentic AutoML

A validator-gated Claude Code harness for supervised tabular ML. A thin Python
CLI (`agentic-automl`) launches Claude Code inside a per-trial workspace; the
agent does all ML reasoning and code, while the Python validator owns state,
schema validation and step ordering.

## Install

```bash
pip install -e ".[dev]"          # launcher + tests (only runtime dep: jsonschema)
pip install -e ".[phase1]"       # modeling stack for the trial runtime
```

## Validator (`agentic-automl artifacts`)

All subcommands accept `--trial <dir|id>` and print JSON.

| Command | Purpose |
|---|---|
| `begin-skill <name>` | Gate check; returns `allowed`, `redirect_to`, state snapshot |
| `save-artifact <key> --payload <file> --source-skill <name>` | Schema + gate validation, then canonical write |
| `next-step` | Next skill in the chain, or `canonical_workflow_complete` |
| `status` | Trial snapshot |
| `set-trial-slug <slug>` | Rename applied on next resume |
| `validate-write` | PreToolUse guard (hook JSON on stdin) |

Registries live in `src/agentic_automl/contracts.py`; schemas in `schemas/`.

## Tests

```bash
pytest
```
