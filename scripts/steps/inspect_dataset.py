"""Inspect the raw data before the contract exists -> work/intake_candidates.json.

Template (copy to work/ with aaml_common.py). Lists the files, columns, dtypes and null
rates, and proposes answers for the intake interview: candidate target, date/time,
entity-id and row-id columns, and columns worth asking about as possible leakage (known
only after the outcome). Everything here is a *proposal* for the user to confirm; the
target is never chosen from it.

    <runtime python> work/inspect_dataset.py <path> [<path> ...] [--sample-rows 200000]

A path may be a file or a directory of csv/parquet/jsonl files (gs:// works with gcsfs).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("paths", nargs="+")
ap.add_argument("--sample-rows", type=int, default=200_000, help="rows read per file for the column survey")
ap.add_argument("--out", default="work/intake_candidates.json")
args = ap.parse_args()

FORMATS = {"csv", "parquet", "pq", "jsonl", "ndjson", "json"}
# Names that often mean "recorded after the outcome". Only a prompt for a question, never a decision.
POST_OUTCOME = re.compile(r"(chargeback|refund|dispute|outcome|result|resolved|closed|cancel|default(ed)?_|"
                          r"churn(ed)?_date|end_date|paid_off|recover|collection|writeoff|write_off|label|status)",
                          re.I)


def list_files(path: str) -> list[str]:
    if path.startswith("gs://"):
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        found = [f"gs://{p}" for p in fs.find(path)] if fs.isdir(path) else [path]
    else:
        p = Path(path)
        found = [str(f) for f in sorted(p.rglob("*")) if f.is_file()] if p.is_dir() else [str(p)]
    return [f for f in found if Path(f).suffix.lstrip(".").lower() in FORMATS]


def read_sample(path: str) -> pd.DataFrame:
    fmt = Path(path).suffix.lstrip(".").lower()
    if fmt == "csv":
        return pd.read_csv(path, nrows=args.sample_rows)
    return A._read_one(path, fmt).head(args.sample_rows)


inventory, frames = [], []
for path in args.paths:
    files = list_files(path)
    assert files, f"no csv/parquet/jsonl files at {path}"
    fmts = sorted({Path(f).suffix.lstrip(".").lower() for f in files})
    inventory.append({"path": path, "file_count": len(files), "formats": fmts})
    frames.append(read_sample(files[0]))
    print(f"{path}: {len(files)} file(s) {fmts}; sampled {len(frames[-1])} rows from {Path(files[0]).name}")

df = frames[0]
if len(frames) > 1:
    common = [c for c in df.columns if all(c in f.columns for f in frames[1:])]
    if len(common) < len(df.columns):
        print(f"columns differ across paths; surveying the {len(common)} shared ones")
    df = pd.concat([f[common] for f in frames], ignore_index=True)
n = len(df)

columns, cand = [], {"target": [], "time": [], "entity_id": [], "row_id": [], "ask_if_post_outcome": [],
                     "numeric_as_text": [], "split_column": []}
for col in df.columns:
    s = df[col]
    nu = int(s.nunique(dropna=True))
    numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    entry = {"name": col, "dtype": str(s.dtype), "missing_fraction": round(float(s.isna().mean()), 4), "n_unique": nu,
             "examples": [str(v) for v in s.dropna().unique()[:3]]}
    columns.append(entry)
    if nu <= 1:
        continue
    if pd.api.types.is_datetime64_any_dtype(s):
        cand["time"].append(col)
    elif not numeric:
        sample = s.dropna().astype(str).head(1000)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if len(sample) and parsed.notna().mean() > 0.9 and not sample.str.fullmatch(r"\d+").all():
            cand["time"].append(col)
        elif A.numeric_text_share(s) > 0.95:
            cand["numeric_as_text"].append(col)
    if (nu >= 0.95 * n and (not numeric or pd.api.types.is_integer_dtype(s))
            and col not in cand["time"] + cand["numeric_as_text"]):
        cand["row_id"].append(col)
    elif not numeric and 0.001 * n < nu < 0.5 * n and re.search(r"(^|_)(id|key|code|no|number)$|id$", col, re.I):
        cand["entity_id"].append(col)  # repeating key: rows per customer/account/...
    if nu <= 20 and col not in cand["time"]:
        cand["target"].append({"column": col, "levels": nu,
                               "top": {str(k): int(v) for k, v in s.value_counts().head(3).items()}})
        if {str(v).lower() for v in s.dropna().unique()} <= {"train", "test", "valid", "validation", "oot"}:
            cand["split_column"].append(col)
    if POST_OUTCOME.search(col):
        cand["ask_if_post_outcome"].append(col)

A.write_json(args.out, {"inventory": inventory, "sampled_rows": n, "columns": columns, "candidates": cand})
print(f"\n{n} sampled rows x {len(columns)} columns")
for e in columns:
    print(f"  {e['name']:<28} {e['dtype']:<14} miss={e['missing_fraction']:<7} uniq={e['n_unique']:<8} e.g. {e['examples']}")
print("\nintake proposals (confirm each with the user; the target is always the user's call):")
for k, v in cand.items():
    shown = [f"{t['column']} ({t['levels']})" for t in v] if k == "target" else v
    print(f"  {k:<20} {', '.join(shown) if shown else '-'}")
print(f"\nwrote {args.out}")
