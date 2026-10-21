"""Inspect a built wheel for `uv tool` release readiness: `python scripts/inspect_tool_distribution.py dist/<wheel>.whl [--release-ready] [--json]`.

Checks:
  REQUIRED_WHEEL_MEMBERS  the launcher, schemas, runtime .claude tree and step templates are inside
  RELEASE_BLOCKING        nothing that must never ship (contributor tools, answer keys, trials,
                          local settings, caches, data)
  PACKAGE_UPDATE          (--release-ready) a final version: no .dev / local "+" segment, and the
                          wheel version matches pyproject.toml
"""
import argparse
import json
import re
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_WHEEL_MEMBERS = [
    "agentic_automl/__init__.py",
    "agentic_automl/cli.py",
    "agentic_automl/artifacts.py",
    "agentic_automl/_data/schemas/prediction_contract.schema.json",
    "agentic_automl/_data/schemas/run_state.schema.json",
    "agentic_automl/_data/claude/CLAUDE.md",
    "agentic_automl/_data/claude/hooks.json",
    "agentic_automl/_data/claude/settings.json",
    "agentic_automl/_data/claude/skills/SKILLS.md",
    "agentic_automl/_data/claude/skills/domains/registry.json",
    "agentic_automl/_data/steps/aaml_common.py",
    "agentic_automl/_data/steps/train.py",
]
RELEASE_BLOCKING = [
    (r"(^|/)settings\.local\.json$", "local Claude settings"),
    (r"ANSWER_KEY", "a synthetic dataset answer key"),
    (r"generate_synthetic_fraud\.py$", "the synthetic-data generator (reveals the generating process)"),
    (r"(^|/)(trials|results|data|benchmarks)/", "trial, results, data or benchmark files"),
    (r"__pycache__/|\.pyc$", "bytecode caches"),
    (r"(^|/)\.env$|\.pypirc$", "credentials"),
]
PACKAGE_UPDATE = [
    (r"\.dev\d*", "development version"),
    (r"\+", "local version segment"),
]


def inspect(wheel: Path, release_ready: bool) -> dict:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
        version = re.search(r"^Version: (.+)$", zf.read(meta).decode(), re.M).group(1) if meta else None
        entry = next((n for n in names if n.endswith(".dist-info/entry_points.txt")), None)
        entry_points = zf.read(entry).decode() if entry else ""
    problems, warnings = [], []
    missing = [m for m in REQUIRED_WHEEL_MEMBERS if m not in names]
    problems += [f"missing required member {m}" for m in missing]
    if "agentic-automl = agentic_automl.cli:main" not in entry_points.replace("  ", " "):
        problems.append("console script agentic-automl = agentic_automl.cli:main not declared")
    for pattern, why in RELEASE_BLOCKING:
        hits = [n for n in names if re.search(pattern, n)]
        if hits:
            problems.append(f"ships {why}: {hits[:5]}")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if version != pyproject:
        (problems if release_ready else warnings).append(f"wheel version {version} != pyproject.toml {pyproject}")
    for pattern, why in PACKAGE_UPDATE:
        if version and re.search(pattern, version):
            (problems if release_ready else warnings).append(f"{why} {version!r}")
    return {"wheel": str(wheel), "version": version, "members": len(names), "ok": not problems,
            "problems": problems, "warnings": warnings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wheel", nargs="?", type=Path, help="wheel to inspect (default: newest in dist/)")
    ap.add_argument("--release-ready", action="store_true", help="treat version anti-patterns as blocking")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    wheel = args.wheel or max((ROOT / "dist").glob("agentic_automl-*.whl"), key=lambda p: p.stat().st_mtime, default=None)
    if not wheel or not wheel.is_file():
        print("no wheel found: build one with `uv build --wheel` or scripts/check_tool_distribution.sh", file=sys.stderr)
        return 1
    r = inspect(wheel, args.release_ready)
    if args.json:
        print(json.dumps(r, indent=2))
    else:
        print(f"{wheel.name}: version {r['version']}, {r['members']} members -> {'OK' if r['ok'] else 'NOT READY'}")
        for p in r["problems"]:
            print(f"  BLOCKING: {p}")
        for w in r["warnings"]:
            print(f"  warning: {w}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
