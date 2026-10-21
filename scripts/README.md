# scripts/

Contributor tooling: evals, benchmarks, seeds and releases. None of it reaches a trial.
The step templates the agent copies are in `steps/` and are the only part shipped into
trials (as `.claude/scripts/`). Most scripts are thin wrappers over `src/agentic_automl/` modules.

Run with a Python that has this repo's dependencies, e.g. `.venv\Scripts\python.exe scripts/<name>`.
Seed generation also needs the trial runtime Python with the ML stack, which it calls for you.

| Script | What it does | Backed by |
|---|---|---|
| **Evals** | | |
| `run_agent_evals.py` | Run scenarios by tier: `deterministic` (free), `auth` (preflight), `component` / `e2e` / `real` (paid, need `--trust-runtime`) | `eval_harness.main` |
| `recommend_evals.py` | Which tiers a diff needs (`--base origin/main --head HEAD`); recommend-only | `eval_harness.recommend_evals_cli` |
| **Benchmarks** | | |
| `audit_benchmark_datasets.py` | Readiness of every `benchmarks/datasets.json` entry (`--json`, `--strict`) | `benchmark_readiness.main` |
| `inspect_benchmark_dataset.py` | One dataset's entry, readiness, fixtures, scenarios, seeds (`--id`) | `benchmark_readiness` |
| `download_datasets.py` | Fetch/generate datasets into `data/benchmarks/<id>/` (gitignored) | registry `access` |
| `prepare_benchmark_samples.py` | Validate `samples.json` manifests (rows, sha256); `--write` rebuilds them | `benchmark_samples.schema.json` |
| **Seeds and fixtures** | | |
| `generate_synthetic_benchmarks.py` | Deterministic CSV from a JSON config (stdlib) | `synthetic_benchmarks` |
| `generate_dataset_seeds.py` | Smoke CSV + seed trials `after_contract` to `evaluated`, by running the step templates through the validator | `eval_seeds` |
| `generate_seed_profiles.py` | Same, stopped after profiling (cheap) | `eval_seeds` |
| `generate_training_seed.py` | Same, stopped after training | `eval_seeds` |
| `generate_answer_bank.py` | Scenario `answer_bank` from the registry + seeded contract (`--print` to preview) | `eval_seed_roles` |
| `generate_synthetic_fraud.py` | The synthetic fraud dataset (writes an answer key; never give it to a trial) | stdlib |
| **Release** | | |
| `prepare_release.py` | Bump the version in `pyproject.toml` and `__init__.py` (`patch`/`minor`/`major`/`final`, `--pre rc`, `--dry-run`) | |
| `inspect_tool_distribution.py` | Check a wheel: required members present, nothing release-blocking, final version | |
| `check_tool_distribution.sh` | Build wheel + inspect + sandboxed `uv tool install` smoke | |
| `publish_artifactory.sh` | Build + release check + `twine upload` to Artifactory (`ARTIFACTORY_URL/USER/TOKEN` or `--repository`; `--dry-run`) | |
| **Trials** | | |
| `summarize_trials.py` | One row per trial | `trial_summary.main` |

Typical loops:

```
# dataset onboarding
python scripts/download_datasets.py --id telco_churn
python scripts/generate_dataset_seeds.py --dataset telco_churn
python scripts/prepare_benchmark_samples.py --write --id telco_churn
python scripts/generate_answer_bank.py --dataset telco_churn
python scripts/audit_benchmark_datasets.py

# before a PR
python scripts/recommend_evals.py --base origin/main --head HEAD
python scripts/run_agent_evals.py --tier deterministic --report-md eval-report.md

# release
python scripts/prepare_release.py patch
scripts/check_tool_distribution.sh --release-ready
scripts/publish_artifactory.sh --dry-run
```
