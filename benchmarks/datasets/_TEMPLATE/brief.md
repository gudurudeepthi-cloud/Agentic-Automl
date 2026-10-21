# <dataset id>

**What it is:** one paragraph: source, unit of observation (one row = ?), time span.

**Prediction task:** target column, what a positive means, and how the prediction is used.

**Why it's in the benchmark set:** which agent behaviour it exercises (leakage trap, imbalance,
temporal split, high-cardinality ids, messy types, domain overlay, ...).

**Known traps:** columns the agent must drop or handle, with the reason.

**Access:** how to get it (`scripts/download_datasets.py --id <id>`, generator, GCS, manual).

To add a dataset: copy this directory to `benchmarks/datasets/<id>/`, fill in `brief.md`,
`features.json` and `sizing.json`, add the registry entry to `benchmarks/datasets.json`, then
run `scripts/audit_benchmark_datasets.py` and `scripts/generate_dataset_seeds.py --dataset <id>`.
