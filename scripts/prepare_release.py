"""Bump the package version: `python scripts/prepare_release.py patch|minor|major|<X.Y.Z[...]> [--pre rc] [--dry-run]`.

Updates [project].version in pyproject.toml and __version__ in src/agentic_automl/__init__.py
(the two must agree). Accepts semver plus PEP 440 pre/post/dev segments:
  patch            0.1.0 -> 0.1.1          minor  0.1.3 -> 0.2.0          major  0.4.2 -> 1.0.0
  --pre rc         0.1.0 -> 0.1.1rc1       (a further --pre rc on 0.1.1rc1 -> 0.1.1rc2)
  final            0.1.1rc2 -> 0.1.1       (drop pre/post/dev)
  --post / --dev   0.1.1 -> 0.1.1.post1 / 0.1.2.dev1
  explicit         2.0.0b1
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "agentic_automl" / "__init__.py"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?(?:\.dev(\d+))?$")


def current() -> str:
    m = re.search(r'^version\s*=\s*"([^"]+)"', PYPROJECT.read_text(encoding="utf-8"), re.M)
    init = re.search(r'__version__\s*=\s*"([^"]+)"', INIT.read_text(encoding="utf-8")).group(1)
    if m.group(1) != init:
        raise SystemExit(f"pyproject.toml ({m.group(1)}) and __init__.py ({init}) disagree: fix by hand first")
    return m.group(1)


def bump(version: str, part: str, pre: str | None, post: bool, dev: bool) -> str:
    m = VERSION_RE.match(version)
    if not m:
        raise SystemExit(f"current version {version!r} is not X.Y.Z[{{a|b|rc}}N][.postN][.devN]")
    major, minor, patch = map(int, m.group(1, 2, 3))
    cur_pre, cur_pre_n = m.group(4), int(m.group(5) or 0)
    if VERSION_RE.match(part):
        return part
    if part == "final":
        return f"{major}.{minor}.{patch}"
    if post:
        n = int(m.group(6) or 0) + 1
        return f"{major}.{minor}.{patch}{cur_pre + str(cur_pre_n) if cur_pre else ''}.post{n}"
    if pre and cur_pre == pre and part == "patch":  # rc1 -> rc2 on the same release
        return f"{major}.{minor}.{patch}{pre}{cur_pre_n + 1}"
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        if not cur_pre:  # 0.1.1rc2 + patch -> 0.1.1 is "final"; plain patch bumps
            patch += 1
    else:
        raise SystemExit(f"unknown part {part!r}")
    out = f"{major}.{minor}.{patch}"
    if pre:
        out += f"{pre}1"
    if dev:
        out += ".dev1"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", help="patch | minor | major | final | explicit version")
    ap.add_argument("--pre", choices=["a", "b", "rc"])
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    old = current()
    new = bump(old, args.part, args.pre, args.post, args.dev)
    if not VERSION_RE.match(new):
        raise SystemExit(f"refusing to write invalid version {new!r}")
    print(f"{old} -> {new}{' (dry run)' if args.dry_run else ''}")
    if args.dry_run:
        return 0
    PYPROJECT.write_text(re.sub(r'^(version\s*=\s*)"[^"]+"', rf'\g<1>"{new}"', PYPROJECT.read_text("utf-8"),
                                count=1, flags=re.M), encoding="utf-8")
    INIT.write_text(re.sub(r'__version__\s*=\s*"[^"]+"', f'__version__ = "{new}"', INIT.read_text("utf-8")),
                    encoding="utf-8")
    print("updated pyproject.toml and src/agentic_automl/__init__.py; commit, tag v" + new + ", then publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
