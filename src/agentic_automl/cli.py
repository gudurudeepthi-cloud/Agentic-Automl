"""`agentic-automl` entry point.

Launcher subcommands (new/resume/fork/status) land in build step 3; for now
this dispatches to the validator sub-tool.
"""

from __future__ import annotations

import sys

SUBTOOLS = {
    "artifacts": "agentic_automl.artifacts",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBTOOLS:
        import importlib

        return importlib.import_module(SUBTOOLS[argv[0]]).main(argv[1:])
    print("usage: agentic-automl {" + ",".join(SUBTOOLS) + "} ...", file=sys.stderr)
    return 2
