"""Registries that define the artifact contract: artifacts, gates, ordering.

Everything the validator enforces is declared here. The agent never reads this
module directly; it interacts through `agentic-automl artifacts <sub>`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

PACKAGE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

_ARTIFACT_PATHS: dict[str, tuple[str, str]] = {
    "prediction_contract": ("prediction_contract.json", "Prediction contract"),
    "dataset_profile": ("dataset_profile.json", "Dataset profile"),
    "column_profile": ("column_profile.json", "Column profile"),
    "workflow_plan": ("state/workflow_plan.json", "Workflow plan"),
    "preprocessing_summary": ("outputs/pipeline/preprocessing_summary.json", "Preprocessing summary"),
    "training_summary": ("outputs/model/training_summary.json", "Training summary"),
    "onnx_deployment": ("outputs/model/onnx_deployment.json", "ONNX deployment"),
    "auto_research_summary": ("outputs/model/auto_research_summary.json", "Auto-research summary"),
    "evaluation_summary": ("outputs/report/evaluation_summary.json", "Evaluation summary"),
    "explainability_summary": ("outputs/report/explainability_summary.json", "Explainability summary"),
    "model_development_documentation": (
        "outputs/report/model_development_documentation.json",
        "Model development documentation",
    ),
}

ARTIFACT_REGISTRY: dict[str, dict[str, str]] = {
    key: {"path": path, "schema": f"{key}.schema.json", "label": label}
    for key, (path, label) in _ARTIFACT_PATHS.items()
}

RUN_STATE_PATH = "state/run_state.json"
RUN_STATE_SCHEMA = "run_state.schema.json"

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

Predicate = Callable[[dict[str, Any]], bool]


def _exists(_payload: dict[str, Any]) -> bool:
    # Gates are existence-based: a payload only reaches disk after passing
    # schema + upstream-gate validation, so presence is sufficient.
    return True


_GATE_NAMES: dict[str, str] = {
    "prediction_contract_approved": "prediction_contract",
    "dataset_profile_complete": "dataset_profile",
    "column_profile_complete": "column_profile",
    "workflow_plan_approved": "workflow_plan",
    "preprocessing_complete": "preprocessing_summary",
    "training_complete": "training_summary",
    "onnx_deployment_complete": "onnx_deployment",
    "auto_research_complete": "auto_research_summary",
    "evaluation_complete": "evaluation_summary",
    "explainability_complete": "explainability_summary",
    "model_development_documentation_complete": "model_development_documentation",
}

GATE_REGISTRY: dict[str, dict[str, Any]] = {
    gate: {
        "satisfied_by": key,
        "predicate": _exists,
        "description": f"{ARTIFACT_REGISTRY[key]['label']} saved via save-artifact",
    }
    for gate, key in _GATE_NAMES.items()
}

GATE_FOR_ARTIFACT: dict[str, str] = {key: gate for gate, key in _GATE_NAMES.items()}

ARTIFACT_GATE_REQUIREMENTS: dict[str, list[str]] = {
    "prediction_contract": [],
    "dataset_profile": ["prediction_contract_approved"],
    "column_profile": ["prediction_contract_approved"],
    "workflow_plan": ["dataset_profile_complete", "column_profile_complete"],
    "preprocessing_summary": ["workflow_plan_approved"],
    "training_summary": ["preprocessing_complete"],
    "onnx_deployment": ["training_complete"],
    "evaluation_summary": ["training_complete"],
    "explainability_summary": ["training_complete"],
    "auto_research_summary": ["training_complete"],
    "model_development_documentation": ["evaluation_complete"],
}

# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

SKILL_PRODUCES: dict[str, list[str]] = {
    "prediction-contract": ["prediction_contract"],
    "data-profiling": ["column_profile", "dataset_profile"],
    "workflow-planning": ["workflow_plan"],
    "data-preprocessing": ["preprocessing_summary"],  # open_source opt-out
    "parity-ml-preprocessing": ["preprocessing_summary"],  # default engine
    "training": ["training_summary"],
    "parity-ml-export": ["onnx_deployment"],
    "evaluation": ["evaluation_summary"],
    "explainability": ["explainability_summary"],
    "auto-research": ["auto_research_summary"],
    "model-development-documentation": ["model_development_documentation"],
}

WORKFLOW_ORDER: list[str] = [
    "prediction-contract",
    "data-profiling",
    "workflow-planning",
    "data-preprocessing",
    "training",
    "evaluation",
]

OPTIONAL_SKILLS_AFTER_CANONICAL: tuple[str, ...] = (
    "explainability",
    "model-development-documentation",
)

# workflow_plan.preprocessing_plan.engine -> preprocessing skill
PREPROCESSING_SKILL_FOR_ENGINE: dict[str, str] = {
    "parity-ml": "parity-ml-preprocessing",
    "open_source": "data-preprocessing",
}
DEFAULT_PREPROCESSING_ENGINE = "parity-ml"

TRIAL_SUBDIRS: tuple[str, ...] = (
    "state",
    "work",
    "outputs/pipeline",
    "outputs/model",
    "outputs/report",
    "preprocessed_data",
)


def required_gates_for_skill(skill: str) -> list[str]:
    gates: list[str] = []
    for key in SKILL_PRODUCES[skill]:
        for gate in ARTIFACT_GATE_REQUIREMENTS[key]:
            if gate not in gates:
                gates.append(gate)
    return gates


def producer_skill_for_artifact(key: str) -> str:
    """Canonical skill that produces `key` (WORKFLOW_ORDER wins on ties)."""
    candidates = [s for s, keys in SKILL_PRODUCES.items() if key in keys]
    for skill in WORKFLOW_ORDER:
        if skill in candidates:
            return skill
    return candidates[0]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def source_root() -> Path | None:
    """The repo root when running from a source checkout, else None (installed wheel)."""
    root = PACKAGE_DIR.parents[1]
    if (root / "pyproject.toml").is_file() and (root / "schemas").is_dir():
        return root
    return None



def schemas_dir() -> Path:
    bundled = PACKAGE_DIR / "_data" / "schemas"
    if bundled.is_dir():
        return bundled
    root = source_root()
    if root is not None:
        return root / "schemas"
    raise FileNotFoundError("Cannot locate schemas directory (source checkout or wheel _data)")


def claude_template_dir() -> Path:
    """The `.claude` tree copied into each trial workspace."""
    bundled = PACKAGE_DIR / "_data" / "claude"
    if bundled.is_dir():
        return bundled
    root = source_root()
    if root is not None:
        return root / ".claude"
    raise FileNotFoundError("Cannot locate .claude template directory")


def step_templates_dir() -> Path:
    """Step-template scripts (repo `scripts/`), copied into each trial as `.claude/scripts/`.

    `scripts/dev/` (contributor tools such as the synthetic-data generator) is never copied.
    """
    bundled = PACKAGE_DIR / "_data" / "scripts"
    if bundled.is_dir():
        return bundled
    root = source_root()
    if root is not None:
        return root / "scripts"
    raise FileNotFoundError("Cannot locate scripts directory")


def storage_dir() -> Path:
    """Root for trials/ and results/, kept outside the code repo.

    Precedence: $AGENTIC_AUTOML_STORAGE_DIR > config `storage_dir` > ~/agentic-automl.
    """
    from agentic_automl.state import load_config  # lazy: state imports this module

    env = os.environ.get("AGENTIC_AUTOML_STORAGE_DIR") or load_config().get("storage_dir")
    return Path(env).expanduser().resolve() if env else Path.home() / "agentic-automl"


def trials_dir() -> Path:
    env = os.environ.get("AGENTIC_AUTOML_TRIALS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return storage_dir() / "trials"


def results_dir() -> Path:
    """Durable results ledger; defaults to a sibling of the trials dir."""
    env = os.environ.get("AGENTIC_AUTOML_RESULTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return trials_dir().parent / "results"


def artifact_path(trial_dir: str | os.PathLike[str], key: str) -> Path:
    return Path(trial_dir) / ARTIFACT_REGISTRY[key]["path"]


def schema_path(key: str) -> Path:
    if key == "run_state":
        return schemas_dir() / RUN_STATE_SCHEMA
    return schemas_dir() / ARTIFACT_REGISTRY[key]["schema"]


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.realpath(path))


def artifact_key_for_path(
    trial_dir: str | os.PathLike[str], candidate: str | os.PathLike[str]
) -> str | None:
    """Reverse lookup used by the write guard. Relative candidates resolve against trial_dir."""
    trial = Path(trial_dir)
    cand = Path(candidate)
    if not cand.is_absolute():
        cand = trial / cand
    target = _norm(cand)
    for key in ARTIFACT_REGISTRY:
        if _norm(artifact_path(trial, key)) == target:
            return key
    return None


def is_run_state_path(trial_dir: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> bool:
    trial = Path(trial_dir)
    cand = Path(candidate)
    if not cand.is_absolute():
        cand = trial / cand
    return _norm(cand) == _norm(trial / RUN_STATE_PATH)
