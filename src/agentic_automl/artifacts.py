"""Validator + run_state owner, exposed as `agentic-automl artifacts <sub>`.

state/run_state.json is the single source of truth for a trial. Only this
module writes it or any registered artifact file; the agent goes through
`save-artifact`, and the PreToolUse guard (`validate-write`) blocks direct
edits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from agentic_automl import __version__
from agentic_automl.contracts import (
    ARTIFACT_GATE_REQUIREMENTS,
    ARTIFACT_REGISTRY,
    DEFAULT_PREPROCESSING_ENGINE,
    GATE_FOR_ARTIFACT,
    GATE_REGISTRY,
    OPTIONAL_SKILLS_AFTER_CANONICAL,
    PREPROCESSING_SKILL_FOR_ENGINE,
    RUN_STATE_PATH,
    SKILL_PRODUCES,
    TRIAL_SUBDIRS,
    artifact_key_for_path,
    artifact_path,
    is_run_state_path,
    producer_skill_for_artifact,
    required_gates_for_skill,
    schema_path,
    trials_dir,
)

DECISION_FEEDBACK_KEYS = ("prediction_contract", "workflow_plan")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
WRITE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}


class ArtifactError(Exception):
    """A rejected operation. `payload` is emitted as JSON by the CLI."""

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.payload = {"ok": False, "error": message, **extra}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


@lru_cache(maxsize=None)
def _validator(key: str) -> Draft7Validator:
    schema = _read_json(schema_path(key))
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def validate_payload(key: str, payload: Any) -> list[str]:
    """Schema errors for `payload`, formatted as '<json.path>: <message>'."""
    errors = sorted(_validator(key).iter_errors(payload), key=lambda e: list(e.absolute_path))
    out = []
    for err in errors:
        where = "$" + "".join(f"[{p!r}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path)
        out.append(f"{where}: {err.message}")
    if key == "prediction_contract" and not out:
        out += _contract_column_conflicts(payload)
    return out


def _contract_column_conflicts(c: dict) -> list[str]:
    """Intake answers that contradict each other (the schema can't express these)."""
    excluded = {e["column"] for e in c.get("excluded_columns") or []}
    required = set(c.get("required_features") or [])
    non_features = {c["target_column"], *(c.get(k) for k in ("time_column", "entity_id_column", "weight_column",
                                                            "split_column") if c.get(k))}
    out = []
    if c["target_column"] in excluded:
        out.append(f"$.excluded_columns: the target {c['target_column']!r} cannot be excluded")
    if required & excluded:
        out.append(f"$.required_features: {sorted(required & excluded)} are also in excluded_columns")
    if required & non_features:
        out.append(f"$.required_features: {sorted(required & non_features)} are target/time/id/weight/split "
                   "columns, not features")
    return out


# ---------------------------------------------------------------------------
# Trial + state
# ---------------------------------------------------------------------------


def resolve_trial(trial: str | os.PathLike[str] | None) -> Path:
    """Accept a trial directory, a trial id under trials_dir(), or fall back to env/cwd.

    The launcher exports TRIAL_DIR; AGENTIC_AUTOML_TRIAL is an explicit override.
    """
    if trial is None:
        trial = os.environ.get("AGENTIC_AUTOML_TRIAL") or os.environ.get("TRIAL_DIR")
    if trial is None:
        cwd = Path.cwd()
        if (cwd / RUN_STATE_PATH).is_file():
            return cwd.resolve()
        raise ArtifactError("No trial given: pass --trial or set TRIAL_DIR")
    candidate = Path(trial).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    by_id = trials_dir() / str(trial)
    if by_id.is_dir():
        return by_id.resolve()
    raise ArtifactError(f"Trial not found: {trial}")


def _state_file(trial_dir: Path) -> Path:
    return Path(trial_dir) / RUN_STATE_PATH


def ensure_state(
    trial_dir: str | os.PathLike[str],
    *,
    goal: str | None = None,
    dataset_path: str | None = None,
    forked_from: str | None = None,
) -> dict[str, Any]:
    """Create (or load) run_state.json and the standard trial subdirectories."""
    trial = Path(trial_dir)
    for sub in TRIAL_SUBDIRS:
        (trial / sub).mkdir(parents=True, exist_ok=True)
    path = _state_file(trial)
    if path.is_file():
        state = _read_json(path)
        changed = False
        for field, value in (("goal", goal), ("dataset_path", dataset_path)):
            if value is not None and state.get(field) != value:
                state[field] = value
                changed = True
        if changed:
            save_state(trial, state)
        return state
    ts = now_iso()
    state = {
        "trial_id": trial.name,
        "created_at": ts,
        "updated_at": ts,
        "artifacts": [],
        "gates": {gate: False for gate in GATE_REGISTRY},
        "sessions": [],
        "validation_log": [],
        "goal": goal,
        "dataset_path": dataset_path,
        "pending_slug": None,
        "forked_from": forked_from,
        "package_version": __version__,
        "last_claude_session_id": None,
        "previous_trial_dirs": [],
        "active_skill": None,
        "developer_focus": [],
    }
    save_state(trial, state, touch=False)
    return state


def load_state(trial_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = _state_file(Path(trial_dir))
    if not path.is_file():
        raise ArtifactError(f"Trial has no run_state.json: {trial_dir}")
    return _read_json(path)


def save_state(trial_dir: str | os.PathLike[str], state: dict[str, Any], *, touch: bool = True) -> None:
    if touch:
        state["updated_at"] = now_iso()
    errors = validate_payload("run_state", state)
    if errors:
        raise ArtifactError("run_state.json failed validation (internal error)", errors=errors)
    _write_json_atomic(_state_file(Path(trial_dir)), state)


def _log(state: dict[str, Any], action: str, ok: bool, *, key: str | None = None,
         skill: str | None = None, errors: list[str] | None = None) -> None:
    entry: dict[str, Any] = {"at": now_iso(), "action": action, "ok": ok, "key": key, "skill": skill}
    if errors:
        entry["errors"] = errors
    state["validation_log"].append(entry)


def recorded_artifacts(trial_dir: Path, state: dict[str, Any]) -> set[str]:
    """Keys that were saved through save-artifact and are still on disk."""
    return {
        entry["key"]
        for entry in state["artifacts"]
        if entry["key"] in ARTIFACT_REGISTRY and artifact_path(trial_dir, entry["key"]).is_file()
    }


def load_artifact(trial_dir: Path, key: str) -> dict[str, Any] | None:
    path = artifact_path(trial_dir, key)
    if not path.is_file():
        return None
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def evaluate_gates(trial_dir: str | os.PathLike[str], state: dict[str, Any] | None = None) -> dict[str, bool]:
    trial = Path(trial_dir)
    state = state if state is not None else load_state(trial)
    present = recorded_artifacts(trial, state)
    gates: dict[str, bool] = {}
    for gate, spec in GATE_REGISTRY.items():
        key = spec["satisfied_by"]
        payload = load_artifact(trial, key) if key in present else None
        gates[gate] = payload is not None and bool(spec["predicate"](payload))
    return gates


# ---------------------------------------------------------------------------
# Workflow navigation
# ---------------------------------------------------------------------------


def _preprocessing_skill(trial: Path, present: set[str]) -> str:
    plan = load_artifact(trial, "workflow_plan") if "workflow_plan" in present else None
    engine = ((plan or {}).get("preprocessing_plan") or {}).get("engine", DEFAULT_PREPROCESSING_ENGINE)
    return PREPROCESSING_SKILL_FOR_ENGINE[engine]


def _producer_for_gate(trial: Path, gate: str, present: set[str]) -> str:
    key = GATE_REGISTRY[gate]["satisfied_by"]
    if key == "preprocessing_summary":
        return _preprocessing_skill(trial, present)
    return producer_skill_for_artifact(key)


def _unmet_gates(skill: str, gates: dict[str, bool]) -> list[str]:
    return [g for g in required_gates_for_skill(skill) if not gates[g]]


def _earliest_runnable(trial: Path, skill: str, gates: dict[str, bool], present: set[str]) -> str:
    """Walk upstream from `skill` until reaching a skill whose gates all hold."""
    unmet = _unmet_gates(skill, gates)
    if not unmet:
        return skill
    return _earliest_runnable(trial, _producer_for_gate(trial, unmet[0], present), gates, present)


def _downstream_artifacts(keys: list[str]) -> set[str]:
    """Artifacts whose gate requirements transitively depend on any of `keys`."""
    produced_gates = {GATE_FOR_ARTIFACT[k] for k in keys}
    downstream: set[str] = set()
    changed = True
    while changed:
        changed = False
        for key, reqs in ARTIFACT_GATE_REQUIREMENTS.items():
            if key not in downstream and key not in keys and produced_gates.intersection(reqs):
                downstream.add(key)
                produced_gates.add(GATE_FOR_ARTIFACT[key])
                changed = True
    return downstream


def _chain(trial: Path, present: set[str]) -> list[tuple[str, str]]:
    """The effective baseline chain for this trial as (skill, reason) pairs."""
    chain = [
        ("prediction-contract", "Define and approve the prediction contract."),
        ("data-profiling", "Profile the dataset and its columns."),
        ("workflow-planning", "Propose and approve the modeling plan."),
        (_preprocessing_skill(trial, present), "Build the preprocessing pipeline from the approved plan."),
        ("training", "Train candidate models and select one."),
    ]
    contract = load_artifact(trial, "prediction_contract") if "prediction_contract" in present else None
    if contract and contract.get("model_preference") == "auto_research":
        chain.append(("auto-research", "Contract requests auto_research: iterate on the trained baseline."))
    pre = load_artifact(trial, "preprocessing_summary") if "preprocessing_summary" in present else None
    if pre and pre.get("engine") == "parity-ml":
        chain.append(("parity-ml-export", "parity-ml pipeline: export and parity-check ONNX before evaluation."))
    chain.append(("evaluation", "Evaluate the selected model on the held-out split."))
    return chain


def next_step(trial_dir: str | os.PathLike[str]) -> dict[str, Any]:
    trial = Path(trial_dir)
    state = load_state(trial)
    present = recorded_artifacts(trial, state)
    completed: list[str] = []
    for skill, reason in _chain(trial, present):
        if not set(SKILL_PRODUCES[skill]) <= present:
            return {
                "next_skill": skill,
                "reason": reason,
                "canonical_workflow_complete": False,
                "completed_skills": completed,
            }
        completed.append(skill)
    optional = [s for s in OPTIONAL_SKILLS_AFTER_CANONICAL if not set(SKILL_PRODUCES[s]) <= present]
    return {
        "next_skill": None,
        "reason": "Canonical workflow complete.",
        "canonical_workflow_complete": True,
        "completed_skills": completed,
        "optional_next_skills": optional,
    }


def snapshot(trial_dir: str | os.PathLike[str], state: dict[str, Any] | None = None) -> dict[str, Any]:
    trial = Path(trial_dir)
    state = state if state is not None else load_state(trial)
    present = recorded_artifacts(trial, state)
    return {
        "trial_id": state["trial_id"],
        "trial_dir": str(trial),
        "goal": state.get("goal"),
        "dataset_path": state.get("dataset_path"),
        "active_skill": state.get("active_skill"),
        "pending_slug": state.get("pending_slug"),
        "artifacts": {key: key in present for key in ARTIFACT_REGISTRY},
        "gates": evaluate_gates(trial, state),
    }


def begin_skill(trial_dir: str | os.PathLike[str], skill: str) -> dict[str, Any]:
    trial = Path(trial_dir)
    if skill not in SKILL_PRODUCES:
        raise ArtifactError(f"Unknown skill: {skill}", known_skills=sorted(SKILL_PRODUCES))
    state = load_state(trial)
    gates = evaluate_gates(trial, state)
    present = recorded_artifacts(trial, state)
    unmet = _unmet_gates(skill, gates)
    result: dict[str, Any] = {"skill": skill, "allowed": not unmet, "redirect_to": None, "unmet_gates": unmet}
    if unmet:
        result["redirect_to"] = _earliest_runnable(trial, skill, gates, present)
        result["message"] = (
            f"{skill} is blocked by {', '.join(unmet)}. Run {result['redirect_to']} first."
        )
    else:
        state["active_skill"] = skill
        rerun = set(SKILL_PRODUCES[skill]) & present
        if rerun:
            stale = sorted(_downstream_artifacts(SKILL_PRODUCES[skill]) & present)
            result["rerun"] = True
            result["downstream_artifacts_present"] = stale
            if stale:
                result["message"] = (
                    "Re-running an upstream skill: downstream artifacts "
                    f"{stale} were built from the previous version and should be revisited."
                )
    _log(state, "begin-skill", not unmet, skill=skill, errors=[result["message"]] if unmet else None)
    save_state(trial, state)
    result["state"] = snapshot(trial, state)
    return result


# ---------------------------------------------------------------------------
# Decision feedback
# ---------------------------------------------------------------------------


def _diff(proposed: Any, approved: Any, path: str = "$") -> list[dict[str, Any]]:
    if isinstance(proposed, dict) and isinstance(approved, dict):
        changes: list[dict[str, Any]] = []
        for k in list(proposed) + [k for k in approved if k not in proposed]:
            if k not in approved:
                changes.append({"path": f"{path}.{k}", "change": "removed", "proposed": proposed[k]})
            elif k not in proposed:
                changes.append({"path": f"{path}.{k}", "change": "added", "approved": approved[k]})
            else:
                changes.extend(_diff(proposed[k], approved[k], f"{path}.{k}"))
        return changes
    if proposed != approved:
        return [{"path": path, "change": "modified", "proposed": proposed, "approved": approved}]
    return []


def decision_proposal_path(trial: Path, key: str) -> Path:
    return trial / "work" / "decision_feedback" / f"{key}.proposal.json"


def _capture_decision(trial: Path, key: str, payload: dict[str, Any], skill: str) -> dict[str, Any] | None:
    proposal_file = decision_proposal_path(trial, key)
    if key not in DECISION_FEEDBACK_KEYS or not proposal_file.is_file():
        return None
    try:
        proposed = _read_json(proposal_file)
    except (OSError, json.JSONDecodeError):
        return None
    changes = _diff(proposed, payload)
    episode = {
        "id": uuid.uuid4().hex[:12],
        "type": "decision",
        "at": now_iso(),
        "skill": skill,
        "artifact": key,
        "accepted_as_proposed": not changes,
        "changes": changes,
        "status": "open",
    }
    _append_jsonl(trial / "work" / "feedback_episodes.jsonl", episode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.replace(proposal_file, proposal_file.with_name(f"{key}.proposal.{stamp}.applied.json"))
    return {"episode_id": episode["id"], "n_changes": len(changes)}


# ---------------------------------------------------------------------------
# save-artifact
# ---------------------------------------------------------------------------


def save_artifact(trial_dir: str | os.PathLike[str], key: str, payload: Any, source_skill: str) -> dict[str, Any]:
    trial = Path(trial_dir)
    if key not in ARTIFACT_REGISTRY:
        raise ArtifactError(f"Unknown artifact key: {key}", known_artifacts=sorted(ARTIFACT_REGISTRY))
    if source_skill not in SKILL_PRODUCES:
        raise ArtifactError(f"Unknown skill: {source_skill}", known_skills=sorted(SKILL_PRODUCES))
    if key not in SKILL_PRODUCES[source_skill]:
        allowed = [s for s, keys in SKILL_PRODUCES.items() if key in keys]
        raise ArtifactError(f"Skill {source_skill} may not write {key}", skills_that_may_write=allowed)

    state = load_state(trial)

    def reject(message: str, errors: list[str], **extra: Any) -> ArtifactError:
        _log(state, "save-artifact", False, key=key, skill=source_skill, errors=errors)
        save_state(trial, state)
        return ArtifactError(message, artifact=key, errors=errors, **extra)

    errors = validate_payload(key, payload)
    if errors:
        raise reject(f"{key} failed schema validation; fix the payload and retry.", errors)

    if key == "preprocessing_summary":
        expected_engine = {v: k for k, v in PREPROCESSING_SKILL_FOR_ENGINE.items()}[source_skill]
        if payload["engine"] != expected_engine:
            raise reject(
                f"{source_skill} must save engine={expected_engine!r}",
                [f"$.engine: expected {expected_engine!r}, got {payload['engine']!r}"],
            )

    gates = evaluate_gates(trial, state)
    unmet = [g for g in ARTIFACT_GATE_REQUIREMENTS[key] if not gates[g]]
    if unmet:
        present = recorded_artifacts(trial, state)
        run_first = _earliest_runnable(trial, source_skill, gates, present)
        raise reject(
            f"{key} requires gates {unmet}; run {run_first} first.",
            [f"gate not satisfied: {g}" for g in unmet],
            unmet_gates=unmet,
            redirect_to=run_first,
        )

    decision = _capture_decision(trial, key, payload, source_skill)

    rel = ARTIFACT_REGISTRY[key]["path"]
    _write_json_atomic(artifact_path(trial, key), payload)
    previous = [e for e in state["artifacts"] if e["key"] == key]
    version = max((e.get("version", 1) for e in previous), default=0) + 1
    state["artifacts"] = [e for e in state["artifacts"] if e["key"] != key]
    state["artifacts"].append(
        {"key": key, "path": rel, "source_skill": source_skill, "saved_at": now_iso(), "version": version}
    )
    state["gates"] = evaluate_gates(trial, state)
    _log(state, "save-artifact", True, key=key, skill=source_skill)
    save_state(trial, state)

    result: dict[str, Any] = {
        "ok": True,
        "artifact": key,
        "path": rel,
        "version": version,
        "gate_satisfied": GATE_FOR_ARTIFACT[key],
        "next_step": next_step(trial),
    }
    if decision:
        result["decision_feedback"] = decision
    return result


# ---------------------------------------------------------------------------
# Slug / focus
# ---------------------------------------------------------------------------


def set_trial_slug(trial_dir: str | os.PathLike[str], slug: str) -> dict[str, Any]:
    if not SLUG_RE.match(slug):
        raise ArtifactError("Slug must be lowercase letters, digits and dashes (max 48 chars)", slug=slug)
    trial = Path(trial_dir)
    state = load_state(trial)
    state["pending_slug"] = slug
    save_state(trial, state)
    return {"ok": True, "pending_slug": slug, "note": "Directory is renamed on the next resume."}


def apply_pending_slug(trial_dir: str | os.PathLike[str]) -> Path:
    """Rename the trial dir to <trial_id>--<slug> if a slug is pending. Returns the (new) dir."""
    trial = Path(trial_dir).resolve()
    state = load_state(trial)
    slug = state.get("pending_slug")
    if not slug:
        return trial
    base = trial.name.split("--", 1)[0]
    target = trial.with_name(f"{base}--{slug}")
    state["pending_slug"] = None
    if target == trial:
        save_state(trial, state)
        return trial
    if target.exists():
        raise ArtifactError(f"Cannot apply slug: {target} already exists")
    state["previous_trial_dirs"] = [*state.get("previous_trial_dirs", []), str(trial)]
    state["trial_id"] = target.name
    save_state(trial, state)
    os.replace(trial, target)
    return target


def record_developer_focus(trial_dir: str | os.PathLike[str], text: str) -> dict[str, Any]:
    trial = Path(trial_dir)
    state = load_state(trial)
    state.setdefault("developer_focus", []).append({"at": now_iso(), "text": text})
    save_state(trial, state)
    return {"ok": True, "developer_focus": state["developer_focus"]}


def record_session(trial_dir: str | os.PathLike[str], mode: str, claude_session_id: str | None = None) -> None:
    trial = Path(trial_dir)
    state = load_state(trial)
    state["sessions"].append({"started_at": now_iso(), "mode": mode, "claude_session_id": claude_session_id})
    if claude_session_id:
        state["last_claude_session_id"] = claude_session_id
    save_state(trial, state)


def attach_session_id(trial_dir: str | os.PathLike[str], claude_session_id: str) -> None:
    """Record the Claude session id discovered after the chat exits on the latest session."""
    trial = Path(trial_dir)
    state = load_state(trial)
    if state["sessions"]:
        state["sessions"][-1]["claude_session_id"] = claude_session_id
    state["last_claude_session_id"] = claude_session_id
    save_state(trial, state)


# ---------------------------------------------------------------------------
# Write guard
# ---------------------------------------------------------------------------


def _hook_target_paths(tool_input: dict[str, Any]) -> list[str]:
    return [tool_input[f] for f in ("file_path", "notebook_path", "path") if isinstance(tool_input.get(f), str)]


def validate_write(trial_dir: str | os.PathLike[str], hook_input: dict[str, Any]) -> dict[str, Any] | None:
    """Return a PreToolUse deny decision for writes to protected paths, else None (allow)."""
    if hook_input.get("tool_name") not in WRITE_TOOLS:
        return None
    trial = Path(trial_dir)
    base = Path(hook_input.get("cwd") or trial)
    for raw in _hook_target_paths(hook_input.get("tool_input") or {}):
        target = Path(raw) if Path(raw).is_absolute() else base / raw
        if is_run_state_path(trial, target):
            reason = (
                "state/run_state.json is owned by the validator and cannot be edited. "
                "Use `agentic-automl artifacts <begin-skill|save-artifact|set-trial-slug>` instead."
            )
        else:
            key = artifact_key_for_path(trial, target)
            if key is None:
                continue
            skills = [s for s, keys in SKILL_PRODUCES.items() if key in keys]
            reason = (
                f"{ARTIFACT_REGISTRY[key]['path']} is a registered artifact ({key}) and cannot be written "
                f"directly. Write the draft to work/{key}.json, then run: agentic-automl artifacts "
                f"save-artifact {key} --trial {trial} --payload work/{key}.json "
                f"--source-skill {skills[0] if len(skills) == 1 else '<' + '|'.join(skills) + '>'}"
            )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _emit(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _load_payload_arg(value: str) -> Any:
    try:
        if value == "-":
            return json.load(sys.stdin)
        return _read_json(Path(value))
    except FileNotFoundError:
        raise ArtifactError(f"Payload file not found: {value}")
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"Payload is not valid JSON: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-automl artifacts", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--trial", help="Trial directory or id (default: $AGENTIC_AUTOML_TRIAL or cwd)")
        return p

    add("begin-skill", "Check gates before starting a skill").add_argument("skill")
    p = add("save-artifact", "Validate and persist an artifact")
    p.add_argument("key")
    p.add_argument("--payload", required=True, help="JSON file path, or - for stdin")
    p.add_argument("--source-skill", required=True)
    add("next-step", "Report the next skill in the chain")
    add("status", "Trial snapshot")
    add("set-trial-slug", "Record a slug applied on next resume").add_argument("slug")
    add("developer-focus", "Record a developer focus note").add_argument("text")
    add("validate-write", "PreToolUse guard: reads hook JSON on stdin")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "validate-write":
            return _validate_write_cli(args.trial)
        trial = resolve_trial(args.trial)
        if args.cmd == "begin-skill":
            _emit(begin_skill(trial, args.skill))
        elif args.cmd == "save-artifact":
            _emit(save_artifact(trial, args.key, _load_payload_arg(args.payload), args.source_skill))
        elif args.cmd == "next-step":
            _emit(next_step(trial))
        elif args.cmd == "status":
            _emit(snapshot(trial))
        elif args.cmd == "set-trial-slug":
            _emit(set_trial_slug(trial, args.slug))
        elif args.cmd == "developer-focus":
            _emit(record_developer_focus(trial, args.text))
    except ArtifactError as exc:
        _emit(exc.payload)
        return 1
    return 0


def _validate_write_cli(trial_arg: str | None) -> int:
    try:
        hook_input = json.load(sys.stdin)
        trial = resolve_trial(trial_arg)
    except (json.JSONDecodeError, ArtifactError) as exc:
        # A malformed hook payload is a harness problem, not an agent bypass; don't wedge the session.
        print(f"agentic-automl validate-write: skipped ({exc})", file=sys.stderr)
        return 0
    decision = validate_write(trial, hook_input)
    if decision is not None:
        _emit(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
