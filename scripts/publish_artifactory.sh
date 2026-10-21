#!/usr/bin/env bash
# Build, release-check and upload the wheel to an Artifactory PyPI repository with twine.
#
#   scripts/publish_artifactory.sh [--dry-run] [--skip-build] [--skip-check] [--repository NAME]
#
# Credential modes:
#   CI         ARTIFACTORY_URL (repository upload URL, e.g. https://<host>/artifactory/api/pypi/<repo>)
#              plus ARTIFACTORY_USER and ARTIFACTORY_TOKEN in the environment
#   ~/.pypirc  --repository NAME selects a [NAME] section with repository/username/password
# --dry-run    build + check + `twine check`, print the upload command, upload nothing
# --skip-build reuse the newest wheel in dist/
# --skip-check skip scripts/check_tool_distribution.sh (not recommended)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
SKIP_BUILD=0
SKIP_CHECK=0
REPOSITORY=""
PY="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --repository) REPOSITORY="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$REPOSITORY" ]]; then
  : "${ARTIFACTORY_URL:?set ARTIFACTORY_URL (CI mode) or pass --repository NAME (~/.pypirc mode)}"
  : "${ARTIFACTORY_USER:?set ARTIFACTORY_USER}"
  : "${ARTIFACTORY_TOKEN:?set ARTIFACTORY_TOKEN}"
fi

if [[ "$SKIP_BUILD" == 0 && "$SKIP_CHECK" == 0 ]]; then
  "$ROOT/scripts/check_tool_distribution.sh" --release-ready
elif [[ "$SKIP_BUILD" == 0 ]]; then
  uv build --wheel --out-dir "$ROOT/dist" "$ROOT"
elif [[ "$SKIP_CHECK" == 0 ]]; then
  "$PY" "$ROOT/scripts/inspect_tool_distribution.py" --release-ready
fi
WHEEL="$(ls -t "$ROOT"/dist/agentic_automl-*.whl 2>/dev/null | head -n 1 || true)"
[[ -n "$WHEEL" ]] || { echo "no wheel in $ROOT/dist: drop --skip-build or run scripts/check_tool_distribution.sh" >&2; exit 1; }

if "$PY" -m twine --version >/dev/null 2>&1; then
  "$PY" -m twine check "$WHEEL"
elif [[ "$DRY_RUN" == 1 ]]; then
  echo "warning: twine not installed (uv pip install twine); a real upload needs it" >&2
else
  echo "twine missing: uv pip install twine" >&2; exit 1
fi

if [[ -n "$REPOSITORY" ]]; then
  UPLOAD=("$PY" -m twine upload --non-interactive --repository "$REPOSITORY" "$WHEEL")
else
  UPLOAD=("$PY" -m twine upload --non-interactive --repository-url "$ARTIFACTORY_URL"
          -u "$ARTIFACTORY_USER" -p "$ARTIFACTORY_TOKEN" "$WHEEL")
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "dry run: would upload $(basename "$WHEEL") to ${REPOSITORY:-${ARTIFACTORY_URL:-}}" \
       "as ${ARTIFACTORY_USER:-<~/.pypirc credentials>} (token not shown)"
  exit 0
fi
"${UPLOAD[@]}"
echo "published $(basename "$WHEEL")"
