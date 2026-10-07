# Skills index (modeler runtime)

Each skill lives in `<skill>/SKILL.md`. Always start a skill with
`agentic-automl artifacts begin-skill <skill>` and persist with `save-artifact`.
Contributor/maintainer skills are kept out of this index on purpose.

## Canonical chain

| # | Skill | Produces (artifact → gate) | Requires |
|---|---|---|---|
| 1 | [prediction-contract](prediction-contract/SKILL.md) | `prediction_contract` → `prediction_contract_approved` | none |
| 2 | [data-profiling](data-profiling/SKILL.md) | `dataset_profile` → `dataset_profile_complete`, `column_profile` → `column_profile_complete` | contract |
| 3 | [workflow-planning](workflow-planning/SKILL.md) | `workflow_plan` → `workflow_plan_approved` | both profiles |
| 4a | [parity-ml-preprocessing](parity-ml-preprocessing/SKILL.md) (default engine) | `preprocessing_summary` → `preprocessing_complete` | plan |
| 4b | [data-preprocessing](data-preprocessing/SKILL.md) (`open_source`) | `preprocessing_summary` → `preprocessing_complete` | plan |
| 5 | [training](training/SKILL.md) | `training_summary` → `training_complete` | preprocessing |
| 6 | [evaluation](evaluation/SKILL.md) | `evaluation_summary` → `evaluation_complete` | training |

## Branches (inserted by `next-step`)

| Skill | When | Produces | Requires |
|---|---|---|---|
| [auto-research](auto-research/SKILL.md) | contract `model_preference == "auto_research"`, between training and evaluation | `auto_research_summary` | training |
| [parity-ml-export](parity-ml-export/SKILL.md) | preprocessing `engine == "parity-ml"` and no ONNX yet, before evaluation | `onnx_deployment` | training |

## Optional, after evaluation

| Skill | Produces | Requires |
|---|---|---|
| [explainability](explainability/SKILL.md) | `explainability_summary` | training |
| [model-development-documentation](model-development-documentation/SKILL.md) | `model_development_documentation` | evaluation |

## Feedback

| Skill | Purpose |
|---|---|
| [aaml-override](aaml-override/SKILL.md) | Record a personal override for a platform skill (`~/.agentic-automl/skills/overrides/<skill>.md`) |
| [aaml-feedback](aaml-feedback/SKILL.md) | Reconcile overrides + feedback episodes and (optionally) file them as GitHub issues |

## Shared material

- `domains/registry.json`: optional `ml_domain` overlays (e.g. `credit_risk`).
- `references/`: how-tos shared across skills (GCS auth, long-running jobs).
- `planned/`: draft skills that are **not** wired into the validator. Don't use them in trials.
