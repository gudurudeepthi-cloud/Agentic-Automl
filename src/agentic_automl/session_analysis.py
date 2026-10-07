"""Read Claude Code transcripts for a trial: `agentic-automl session-analysis <sub>`.

Read-only. Used by `resume --summarize` to seed work/session_brief.md, and by
the agent to recall what happened in an earlier chat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from agentic_automl import state as st


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return ""


def iter_messages(transcript: Path) -> Iterator[dict[str, str]]:
    """User/assistant text turns (tool calls and tool results are skipped)."""
    with open(transcript, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") not in ("user", "assistant") or record.get("isMeta"):
                continue
            message = record.get("message") or {}
            text = _text_of(message.get("content")).strip()
            if text and not text.startswith("<command-"):
                yield {"role": message.get("role", record["type"]), "text": text, "at": record.get("timestamp", "")}


def transcript_for(trial_dir: Path, session_id: str | None = None) -> Path | None:
    files = st.transcripts(trial_dir)
    if session_id:
        files = [f for f in files if f.stem == session_id]
    return files[0] if files else None


def extract_text(trial_dir: Path, session_id: str | None = None, max_chars: int = 12000) -> str:
    """The tail of the most recent (or given) transcript as plain text, newest content kept."""
    path = transcript_for(trial_dir, session_id)
    if path is None:
        return ""
    blocks = [f"[{m['role']}] {m['text']}" for m in iter_messages(path)]
    out: list[str] = []
    used = 0
    for block in reversed(blocks):
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block) + 2
    return "\n\n".join(reversed(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-automl session-analysis", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_ in (("sessions", "List transcripts for the trial"), ("transcript", "Print transcript text")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--trial", help="Trial directory or id (default: $TRIAL_DIR or most recent)")
        if name == "transcript":
            p.add_argument("--session", help="Session id (default: most recent)")
            p.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args(argv)
    trial = st.find_trial(args.trial or os.environ.get("TRIAL_DIR"))
    if trial is None:
        print(json.dumps({"ok": False, "error": "Trial not found"}))
        return 1
    if args.cmd == "sessions":
        rows = [{"session_id": p.stem, "bytes": p.stat().st_size} for p in st.transcripts(trial)]
        print(json.dumps({"ok": True, "trial_dir": str(trial), "sessions": rows}, indent=2))
    else:
        sys.stdout.write(extract_text(trial, args.session, args.max_chars) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
