# Contributor orientation

This repo is the *harness*, not a trial. The runtime instructions loaded inside
each trial workspace live in `.claude/CLAUDE.md`.

- `src/agentic_automl/contracts.py` — the registries (artifacts, gates, requirements, skill map, order). Change the contract here first.
- `src/agentic_automl/artifacts.py` — validator and sole writer of `state/run_state.json` and artifact files.
- `schemas/` — one draft-07 schema per artifact plus `run_state.schema.json`. Adding an artifact means: registry entry, gate, requirements, schema, `SKILL_PRODUCES` entry, and test fixtures in `tests/conftest.py`.
- The Python never trains models; ML logic belongs in `.claude/skills/*/SKILL.md`.
- Run `pytest` before committing.
