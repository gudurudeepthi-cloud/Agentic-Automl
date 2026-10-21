# Contributor orientation

This repo is the *harness*, not a trial. The runtime instructions loaded inside
each trial workspace live in `.claude/CLAUDE.md`.

- `src/agentic_automl/contracts.py`: the registries (artifacts, gates, requirements, skill map, order). Change the contract here first.
- `src/agentic_automl/artifacts.py`: validator, and the only writer of `state/run_state.json` and artifact files.
- `src/agentic_automl/cli.py`: launcher (new/resume/fork/...) and sub-tool dispatch. `state.py` has trial discovery, user config and transcript helpers.
- `schemas/`: one draft-07 schema per artifact, plus `run_state.schema.json`. Adding an artifact means: registry entry, gate, requirements, schema, `SKILL_PRODUCES` entry, a `SKILL.md`, and test fixtures in `tests/conftest.py`.
- `.claude/` is the runtime template copied into every trial. `.claude/hooks.json` holds the PreToolUse guards. Claude Code doesn't read that file itself; the launcher merges it into each trial's `settings.json`, which keeps the guards out of sessions in this repo.
- `ml_domain` enums in the contract/plan schemas must match `.claude/skills/domains/registry.json` (a test enforces this).
- The Python never trains models. ML logic belongs in `.claude/skills/*/SKILL.md`
  and in the step templates in `scripts/steps/`. The launcher copies them into every trial as
  `templates/`, and the agent copies them from there into `work/`. They run under the
  trial runtime Python, not the launcher; `tests/test_step_templates.py` runs them end to
  end when sklearn is available.
- The rest of `scripts/` is contributor tooling (evals, benchmarks, seeds, release; see
  `scripts/README.md`) and never reaches a trial: the synthetic-data generator reveals the
  answer key. Most are thin wrappers over `src/agentic_automl/` modules.
- Run `pytest` before committing.
