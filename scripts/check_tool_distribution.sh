#!/usr/bin/env bash
# Build the wheel, inspect it, and (optionally) smoke-install it as a uv tool in a sandbox.
#
#   scripts/check_tool_distribution.sh [--out-dir dist] [--index URL] [--no-smoke]
#                                      [--smoke-extras phase1] [--release-ready]
#
# --index          extra package index for the smoke install (e.g. an internal mirror)
# --no-smoke       skip the uv tool install smoke test
# --smoke-extras   install with extras, e.g. phase1
# --release-ready  version anti-patterns (.dev, +local, pyproject mismatch) become blocking
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT/dist"
INDEX=""
SMOKE=1
EXTRAS=""
RELEASE_READY=""
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --index) INDEX="$2"; shift 2 ;;
    --no-smoke) SMOKE=0; shift ;;
    --smoke-extras) EXTRAS="$2"; shift 2 ;;
    --release-ready) RELEASE_READY="--release-ready"; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }

echo "== build wheel -> $OUT_DIR"
rm -f "$OUT_DIR"/agentic_automl-*.whl
uv build --wheel --out-dir "$OUT_DIR" "$ROOT"
WHEEL="$(ls -t "$OUT_DIR"/agentic_automl-*.whl | head -n 1)"

echo "== inspect $WHEEL"
"$PY" "$ROOT/scripts/inspect_tool_distribution.py" "$WHEEL" $RELEASE_READY

if [[ "$SMOKE" == 1 ]]; then
  echo "== uv tool smoke install (sandboxed)"
  SANDBOX="$(mktemp -d)"
  trap 'rm -rf "$SANDBOX"' EXIT
  export UV_TOOL_DIR="$SANDBOX/tools" UV_TOOL_BIN_DIR="$SANDBOX/bin"
  SPEC="$WHEEL"
  [[ -n "$EXTRAS" ]] && SPEC="agentic-automl[$EXTRAS] @ file://$WHEEL"
  ARGS=(tool install "$SPEC")
  [[ -n "$INDEX" ]] && ARGS+=(--index "$INDEX")
  uv "${ARGS[@]}"
  BIN="$UV_TOOL_BIN_DIR/agentic-automl"
  [[ -x "$BIN" ]] || BIN="$UV_TOOL_BIN_DIR/agentic-automl.exe"
  "$BIN" --version
  AGENTIC_AUTOML_HOME="$SANDBOX/home" AGENTIC_AUTOML_TRIALS_DIR="$SANDBOX/trials" \
    "$BIN" new --no-launch >/dev/null
  ls "$SANDBOX"/trials/*/.claude/scripts/aaml_common.py >/dev/null \
    || { echo "installed tool did not scaffold step templates into a trial" >&2; exit 1; }
  echo "smoke OK: installed tool scaffolds a trial with skills and step templates"
fi
echo "== distribution check passed: $WHEEL"
