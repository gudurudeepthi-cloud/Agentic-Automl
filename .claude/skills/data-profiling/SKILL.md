---
name: data-profiling
description: Profile the dataset and every column against the approved prediction contract, producing dataset_profile and column_profile (sizes, target balance, split counts, missingness, duplicates, column roles, leakage suspects). Use after the contract is approved, or for any "what's in this data" question.
---

# data-profiling

## What it does
Measures the data as the contract defines it, and assigns each column a role
so that planning works from facts rather than guesses.

## Inputs
- `prediction_contract.json` (target, segments, split).
- The files listed in `data_segments`.

## Output
- `dataset_profile` → `dataset_profile.json` (gate `dataset_profile_complete`)
- `column_profile` → `column_profile.json` (gate `column_profile_complete`)
- Both are required before `workflow-planning`.

## Key notes
- Profile **each split as the contract defines it**: apply `split_strategy`
  exactly as preprocessing will, so the counts match later steps.
- Column `role` is one of `feature, target, id, weight, split, time, drop, leakage_suspect`.
  Mark `leakage_suspect` for anything that could encode the target: post-outcome
  timestamps, near-perfect univariate AUC, names like `*_status`, `closed_*`.
- `semantic_type` is free text (`numeric`, `categorical`, `high_cardinality_categorical`,
  `free_text`, `date`, `boolean`, ...).
- Large data: profile a reproducible sample (fixed seed) and say so in `notes`.
  Count rows exactly if that's cheap.

## Before you start
```
agentic-automl artifacts begin-skill data-profiling
```

## How to persist
Write `work/dataset_profile.json` and `work/column_profile.json`, then:
```
agentic-automl artifacts save-artifact dataset_profile --payload work/dataset_profile.json --source-skill data-profiling
agentic-automl artifacts save-artifact column_profile --payload work/column_profile.json --source-skill data-profiling
```

## Steps
1. Write `work/profile_dataset.py`. It loads every segment, applies the
   contract split, and writes both JSON drafts. Run it with the runtime Python.
2. `dataset_profile` fields: `dataset_path`, `n_rows`, `n_columns`,
   `target_column`, `target_summary` (class counts and rates, or
   mean/std/quantiles), `split_summary` (rows and target rate per split),
   `missing_value_summary` (column → fraction, non-zero only),
   `duplicate_rows`, `data_quality_flags` (`{severity: info|warning|critical,
   message, columns}`), `profiling_script`. Optional fields: `segment_summary`,
   `generated_at`, `notes`.
3. `column_profile.columns[]`: `name`, `dtype`, `semantic_type`, `role`,
   `missing_fraction`, `n_unique`, plus optional `stats` (min/max/quantiles,
   or top categories) and `notes`.
4. Run cheap leakage probes on each feature: single-feature AUC (or
   correlation) with the target on train. Flag anything suspiciously
   predictive (e.g. AUC > 0.95) as `leakage_suspect` with a `critical` flag.
5. Flag `warning`/`critical` issues: target imbalance below 1%, test-split
   target rate far from train, constant columns, ID-like columns, heavy
   missingness (over 50%), unparseable dates.
6. Save both artifacts. Summarise for the user in a few bullets: size,
   balance, top issues, and leakage suspects that need a decision.
