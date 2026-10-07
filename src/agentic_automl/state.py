"""State helpers shared by the launcher: user config, trial discovery, Claude transcripts.

Trial run_state.json itself is owned by `artifacts.py`; this module only reads it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_automl.contracts import RUN_STATE_PATH, trials_dir

DEFAULT_RUNTIME_PYTHON = "/opt/conda/bin/python"

# ---------------------------------------------------------------------------
# User home + config (~/.agentic-automl)
# ---------------------------------------------------------------------------


def user_home() -> Path:
    env = os.environ.get("AGENTIC_AUTOML_HOME")
    return Path(env).expanduser() if env else Path.home() / ".agentic-automl"


def config_path() -> Path:
    return user_home() / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def set_config_value(key: str, value: Any) -> dict[str, Any]:
    config = load_config()
    config[key] = value
    save_config(config)
    return config


def runtime_python() -> str:
    """Interpreter the agent uses for ML work (env > config > RMR conda default)."""
    return os.environ.get("AGENTIC_AUTOML_PYTHON") or load_config().get("runtime_python") or DEFAULT_RUNTIME_PYTHON


def overrides_dir() -> Path:
    return user_home() / "skills" / "overrides"


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------


def new_trial_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def unique_trial_dir(base_name: str) -> Path:
    root = trials_dir()
    candidate = root / base_name
    n = 2
    while candidate.exists():
        candidate = root / f"{base_name}-{n}"
        n += 1
    return candidate


def read_state(trial_dir: Path) -> dict[str, Any] | None:
    path = trial_dir / RUN_STATE_PATH
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_trials() -> list[Path]:
    root = trials_dir()
    if not root.is_dir():
        return []
    return [p for p in root.iterdir() if p.is_dir() and (p / RUN_STATE_PATH).is_file()]


def last_activity(trial_dir: Path) -> float:
    """Most recent session start (epoch seconds), falling back to dir/state mtime."""
    state = read_state(trial_dir) or {}
    sessions = state.get("sessions") or []
    if sessions:
        try:
            return datetime.fromisoformat(sessions[-1]["started_at"]).timestamp()
        except (KeyError, ValueError):
            pass
    try:
        return max(trial_dir.stat().st_mtime, (trial_dir / RUN_STATE_PATH).stat().st_mtime)
    except OSError:
        return 0.0


def recent_trials(limit: int | None = None) -> list[Path]:
    trials = sorted(list_trials(), key=last_activity, reverse=True)
    return trials[:limit] if limit else trials


def find_trial(ref: str | None) -> Path | None:
    """Resolve a trial path, id, or unique id prefix; None -> most recent trial."""
    if ref is None:
        recent = recent_trials(1)
        return recent[0] if recent else None
    path = Path(ref).expanduser()
    if path.is_dir() and (path / RUN_STATE_PATH).is_file():
        return path.resolve()
    exact = trials_dir() / ref
    if exact.is_dir():
        return exact.resolve()
    matches = [p for p in list_trials() if p.name.startswith(ref) or p.name.split("--", 1)[-1] == ref]
    return matches[0].resolve() if len(matches) == 1 else None


# ---------------------------------------------------------------------------
# Claude Code transcripts (~/.claude/projects/<encoded cwd>/<session>.jsonl)
# ---------------------------------------------------------------------------


def claude_config_dir() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env).expanduser() if env else Path.home() / ".claude"


def encode_project_path(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(path).resolve()))


def claude_project_dir(trial_dir: Path) -> Path:
    return claude_config_dir() / "projects" / encode_project_path(trial_dir)


def transcripts(trial_dir: Path) -> list[Path]:
    """Transcript files for a trial, newest first."""
    project = claude_project_dir(trial_dir)
    if not project.is_dir():
        return []
    return sorted(project.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def latest_session_id(trial_dir: Path, since: float | None = None) -> str | None:
    for path in transcripts(trial_dir):
        if since is None or path.stat().st_mtime >= since:
            return path.stem
    return None


def migrate_transcripts(old_dir: Path, new_dir: Path) -> bool:
    """Keep `claude --resume` working after a trial directory is renamed."""
    src = claude_config_dir() / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(old_dir))
    dst = claude_project_dir(new_dir)
    if not src.is_dir() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dst)
    return True
