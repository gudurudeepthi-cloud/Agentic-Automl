"""Feedback episodes: `agentic-automl feedback-episodes <sub>`.

An append-only log at work/feedback_episodes.jsonl. Episode types:
  error     - a recoverable failure (captured by the async Bash hook)
  insight   - something the agent learned that the platform should know
  rationale - why a decision was made (often attached to a decision episode)
  decision  - proposal vs approved diff, written by save-artifact
  resolution- marks an earlier episode resolved (status is derived, never edited)

`filter` produces a sanitised copy for sharing (aaml-feedback skill).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

from agentic_automl.artifacts import ArtifactError, now_iso, resolve_trial

EPISODES_FILE = "work/feedback_episodes.jsonl"
LAYERS = ("skill", "schema", "gate", "eval", "domain")
MAX_TEXT = 2000


def episodes_path(trial: Path) -> Path:
    return trial / EPISODES_FILE


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def append(trial: Path, record: dict[str, Any]) -> dict[str, Any]:
    record = {"id": _new_id(), "at": now_iso(), **record}
    path = episodes_path(trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def read_all(trial: Path) -> list[dict[str, Any]]:
    path = episodes_path(trial)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn concurrent write; skip rather than fail
    return out


def episodes(trial: Path) -> list[dict[str, Any]]:
    """Episodes with derived status (resolution records folded in)."""
    records = read_all(trial)
    resolved = {r["resolves"]: r for r in records if r.get("type") == "resolution"}
    out = []
    for rec in records:
        if rec.get("type") == "resolution":
            continue
        rec = dict(rec)
        if rec["id"] in resolved:
            rec["status"] = "resolved"
            rec["resolution_note"] = resolved[rec["id"]].get("note")
        else:
            rec["status"] = "open"
        out.append(rec)
    return out


def _clip(text: str | None, limit: int = MAX_TEXT) -> str | None:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} chars clipped]"


# ---------------------------------------------------------------------------
# Hook capture
# ---------------------------------------------------------------------------

_ERROR_MARKERS = re.compile(r"Traceback \(most recent call last\)|\bError\b|command not found|No such file", re.I)


def failure_from_hook(hook: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a recoverable Bash failure from a PostToolUse / PostToolUseFailure payload."""
    if hook.get("tool_name") != "Bash":
        return None
    command = (hook.get("tool_input") or {}).get("command", "")
    if command.lstrip().startswith("agentic-automl feedback-episodes"):
        return None  # never capture ourselves
    if hook.get("is_interrupt"):
        return None  # user interrupt, not a failure
    error = hook.get("error")
    response = hook.get("tool_response")
    if error is None and isinstance(response, dict):
        if response.get("interrupted"):
            return None  # user interrupt, not a failure
        stderr = response.get("stderr") or ""
        exit_code = response.get("exit_code", response.get("returnCode"))
        if exit_code not in (None, 0) or (stderr and _ERROR_MARKERS.search(stderr)):
            error = stderr or f"exit code {exit_code}"
    if not error:
        return None
    if isinstance(error, (dict, list)):
        error = json.dumps(error)
    return {
        "type": "error",
        "source": "bash-hook",
        "command": _clip(command, 500),
        "error": _clip(str(error)),
        "hook_event": hook.get("hook_event_name"),
    }


def capture_from_stdin(trial_arg: str | None) -> int:
    """Hook entry point: always exits 0 and prints nothing (async + silent)."""
    try:
        hook = json.load(sys.stdin)
        trial = resolve_trial(trial_arg)
        record = failure_from_hook(hook)
        if record:
            append(trial, record)
    except Exception:  # noqa: BLE001 - a capture hook must never disturb the session
        pass
    return 0


# ---------------------------------------------------------------------------
# Sharing filter
# ---------------------------------------------------------------------------

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"gs://[^\s\"']+"), "gs://<redacted>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"(?:[A-Za-z]:)?[\\/](?:Users|home)[\\/][^\s\"']+"), "<path>"),
    (re.compile(r"\b\d{9,}\b"), "<number>"),
]


def sanitize(value: Any) -> Any:
    if isinstance(value, str):
        for pattern, repl in _REDACTIONS:
            value = pattern.sub(repl, value)
        return _clip(value, 1000)
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, dict):
        # Decision diffs can carry data values (labels, column names): keep paths, drop values.
        if {"proposed", "approved"} & set(value) and "path" in value:
            return {"path": value["path"], "change": value.get("change")}
        return {k: sanitize(v) for k, v in value.items()}
    return value


def filter_for_sharing(records: Iterable[dict[str, Any]], include_resolved: bool = False) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        if rec.get("status") == "resolved" and not include_resolved:
            continue
        if rec.get("type") == "error" and not rec.get("layer"):
            continue  # raw errors are only shared once triaged to a layer
        out.append(sanitize(rec))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-automl feedback-episodes", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name: str, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--trial", help="Trial directory or id (default: $TRIAL_DIR)")
        return p

    p = add("list", "List episodes")
    p.add_argument("--status", choices=["open", "resolved"])
    p.add_argument("--type", choices=["error", "insight", "rationale", "decision"])
    p = add("resolve", "Mark an episode resolved")
    p.add_argument("episode_id")
    p.add_argument("--note")
    p = add("insight", "Record an insight for the platform")
    p.add_argument("text")
    p.add_argument("--skill")
    p.add_argument("--layer", choices=LAYERS)
    p = add("rationale", "Record why a decision was made")
    p.add_argument("text")
    p.add_argument("--decision", help="Artifact key or episode id the rationale explains")
    p.add_argument("--skill")
    p = add("triage", "Tag an episode with the layer it belongs to")
    p.add_argument("episode_id")
    p.add_argument("--layer", choices=LAYERS, required=True)
    p.add_argument("--target", help="e.g. the skill or schema name")
    p = add("filter", "Sanitised episodes for sharing")
    p.add_argument("--include-resolved", action="store_true")
    add("capture", "Hook entry point (reads hook JSON on stdin; silent)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "capture":
        return capture_from_stdin(args.trial)
    try:
        trial = resolve_trial(args.trial)
        if args.cmd == "list":
            rows = [e for e in episodes(trial)
                    if (not args.status or e["status"] == args.status) and (not args.type or e["type"] == args.type)]
            result: Any = {"ok": True, "episodes": rows}
        elif args.cmd == "resolve":
            known = {e["id"] for e in episodes(trial)}
            if args.episode_id not in known:
                raise ArtifactError(f"Unknown episode: {args.episode_id}")
            result = {"ok": True, "episode": append(trial, {"type": "resolution", "resolves": args.episode_id,
                                                            "note": args.note})}
        elif args.cmd == "insight":
            result = {"ok": True, "episode": append(trial, {"type": "insight", "text": args.text,
                                                            "skill": args.skill, "layer": args.layer})}
        elif args.cmd == "rationale":
            result = {"ok": True, "episode": append(trial, {"type": "rationale", "text": args.text,
                                                            "decision": args.decision, "skill": args.skill})}
        elif args.cmd == "triage":
            known = {e["id"]: e for e in episodes(trial)}
            if args.episode_id not in known:
                raise ArtifactError(f"Unknown episode: {args.episode_id}")
            # Append-only: re-emit the episode with its layer under a new id that resolves the old one.
            tagged = {k: v for k, v in known[args.episode_id].items() if k not in ("id", "at", "status")}
            tagged.update(layer=args.layer, target=args.target, triaged_from=args.episode_id)
            new = append(trial, tagged)
            append(trial, {"type": "resolution", "resolves": args.episode_id, "note": f"triaged as {new['id']}"})
            result = {"ok": True, "episode": new}
        else:  # filter
            result = {"ok": True, "episodes": filter_for_sharing(episodes(trial), args.include_resolved)}
    except ArtifactError as exc:
        print(json.dumps(exc.payload, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
