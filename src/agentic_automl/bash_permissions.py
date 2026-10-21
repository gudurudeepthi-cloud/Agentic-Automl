"""Bash auto-allow PreToolUse hook: `agentic-automl bash-permissions check`.

Reads the hook JSON on stdin. A single, un-chained command that starts with a
safe prefix gets `permissionDecision: allow`; anything else produces no output
so Claude Code's normal permission flow (settings.json allowlist / prompt)
applies. This hook never denies.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from typing import Any

from agentic_automl.state import runtime_python

# Validator/lifecycle commands, optionally via `uv run` (dev checkouts export UV_PROJECT).
_LIFECYCLE = ("agentic-automl artifacts", "agentic-automl session-analysis")

# Any of these means the command is chained, redirected, substituted or
# backgrounded, so a safe prefix says nothing about the rest of it.
_UNSAFE = re.compile(r"[;&|`<>\n\r]|\$\(")


def safe_prefixes() -> list[list[str]]:
    prefixes = [p.split() for p in _LIFECYCLE]
    prefixes += [["uv", "run", *p] for p in prefixes]
    prefixes.append([runtime_python()])
    prefixes.append(["mkdir", "-p"])
    return prefixes


def _is_template_copy(tokens: list[str]) -> bool:
    """`cp .claude/scripts/<file>... work/[<name>]`: copying step templates into the lab notebook."""
    if len(tokens) < 3 or tokens[0] != "cp":
        return False
    *sources, dest = tokens[1:]
    clean = lambda p: ".." not in p.split("/") and not p.startswith(("/", "~", "-"))  # noqa: E731
    return (all(clean(s) and s.removeprefix("./").startswith(".claude/scripts/") for s in sources)
            and clean(dest) and dest.removeprefix("./").split("/")[0] == "work")


def is_auto_allowed(command: str) -> bool:
    command = command.strip()
    if not command or _UNSAFE.search(command):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if _is_template_copy(tokens):
        return True
    # Require an argument after every prefix: a bare interpreter would open a REPL.
    return any(tokens[: len(p)] == p and len(tokens) > len(p) for p in safe_prefixes())


def decision(hook_input: dict[str, Any]) -> dict[str, Any] | None:
    if hook_input.get("tool_name") != "Bash":
        return None
    command = (hook_input.get("tool_input") or {}).get("command")
    if not isinstance(command, str) or not is_auto_allowed(command):
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": "agentic-automl safe command prefix",
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-automl bash-permissions", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="PreToolUse hook: read hook JSON on stdin")
    sub.add_parser("prefixes", help="Print the auto-allowed prefixes")
    args = parser.parse_args(argv)
    if args.cmd == "prefixes":
        print(json.dumps([" ".join(p) for p in safe_prefixes()], indent=2))
        return 0
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    result = decision(hook_input)
    if result is not None:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
