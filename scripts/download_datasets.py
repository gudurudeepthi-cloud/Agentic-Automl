"""Fetch benchmark datasets into data/benchmarks/<id>/ (gitignored): `python scripts/download_datasets.py [--id ID] [--force]`.

Per registry `access`:
  public_download  HTTP(S) download of download_url / download_urls (stdlib urllib)
  generated        runs the entry's generator script (e.g. scripts/generate_synthetic_fraud.py)
  gcs              copies gcs_paths.staging with gcsfs (credentials from the environment)
  manual           prints where to put the file (e.g. Kaggle datasets that need an account)
Prints the sha256 of every file it fetches.
"""
import argparse
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_automl.benchmark_readiness import load_dataset_registry  # noqa: E402
from agentic_automl.eval_seeds import data_dir  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(entry: dict, force: bool) -> bool:
    d = data_dir(entry["id"], ROOT)
    d.mkdir(parents=True, exist_ok=True)
    access = entry["access"]
    if access == "public_download":
        urls = entry.get("download_urls") or [entry["download_url"]]
        for url in urls:
            dest = d / url.rsplit("/", 1)[-1]
            if dest.is_file() and not force:
                print(f"  {dest.relative_to(ROOT).as_posix()} exists (use --force)")
                continue
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as fh:  # noqa: S310 (registry URLs)
                fh.write(r.read())
            print(f"  -> {dest.relative_to(ROOT).as_posix()} sha256 {sha256(dest)}")
        return True
    if access == "generated":
        g = entry["generator"]
        dest = d / g["output"]
        if dest.is_file() and not force:
            print(f"  {dest.relative_to(ROOT).as_posix()} exists (use --force)")
            return True
        subprocess.run([sys.executable, str(ROOT / g["script"]), *g.get("args", []), "--out-dir", str(d)], check=True)
        print(f"  -> {dest.relative_to(ROOT).as_posix()} sha256 {sha256(dest)}")
        return True
    if access == "gcs":
        import gcsfs  # noqa: PLC0415  only needed for gcs datasets

        src = (entry.get("gcs_paths") or {}).get("staging")
        if not src:
            print("  no gcs_paths.staging in the registry entry")
            return False
        fs = gcsfs.GCSFileSystem()
        for remote in fs.glob(src.rstrip("/") + "/*") if fs.isdir(src) else [src]:
            dest = d / remote.rsplit("/", 1)[-1]
            if dest.is_file() and not force:
                continue
            fs.get(remote, str(dest))
            print(f"  -> {dest.relative_to(ROOT).as_posix()} sha256 {sha256(dest)}")
        return True
    print(f"  manual: {entry.get('source') or 'see the brief'}; put the file in {d.relative_to(ROOT).as_posix()}/")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    entries = [e for e in load_dataset_registry() if (args.id is None or e["id"] == args.id) and e["status"] != "retired"]
    if not entries:
        print(f"no dataset {args.id!r} in benchmarks/datasets.json", file=sys.stderr)
        return 1
    failed = 0
    for e in entries:
        print(f"{e['id']} ({e['access']})")
        try:
            ok = fetch(e, args.force)
        except Exception as exc:  # noqa: BLE001  report and continue with the next dataset
            print(f"  FAILED: {exc}")
            ok = False
        failed += (not ok) and e["access"] != "manual"
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
