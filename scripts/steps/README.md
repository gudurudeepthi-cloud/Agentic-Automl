# Step templates

Tested starting points for each step's script. They come from a full fraud trial and
are generalized to work from the contract and plan. Don't write these steps from
scratch. Copy the template you need into `work/`, adapt it to the trial, and run it:

```
cp templates/aaml_common.py work/
cp templates/<step>.py work/
<runtime python> work/<step>.py --help
```

Run scripts from the trial root. `aaml_common.py` must sit next to them in `work/`.

In the repo these live in `scripts/steps/`. The launcher copies them into every
trial as `templates/` on new and resume. The rest of `scripts/` is contributor
tooling and is never copied into a trial.

| Skill | Template | Reads | Writes |
|---|---|---|---|
| prediction-contract | `inspect_dataset.py` (takes the data paths) | data only (no contract yet) | `work/intake_candidates.json` (proposals for the intake interview) |
| data-profiling | `profile_dataset.py` | contract, data | `work/dataset_profile.json`, `work/column_profile.json` |
| workflow-planning | `features_template.py` → `work/features.py` (optional) | contract | engineered features (point-in-time) |
| workflow-planning | `model_family_screen.py` | contract, profiles, `work/features.py` | `work/model_family_advice.json` |
| workflow-planning | `vintage_analysis.py` (when rows have a date) | contract, profiles, train split | `work/vintage_analysis.json`, `outputs/report/vintage_analysis.png` (proposal for `vintage_selection`) |
| data-preprocessing | `preprocess.py` | contract, plan, profiles, `work/features.py` | parquet, `outputs/pipeline/preprocessor.joblib`, `work/preprocessing_summary.json` |
| training | `feature_selection.py` (`--smoke` first; `--filters-only` for the bivariate report) | preprocessed train, plan | `outputs/model/feature_selection.json`, `outputs/model/bivariate.json` |
| training | `train.py` | `feature_selection.json` | models, `cv_predictions.parquet`, `work/training_summary.json` |
| evaluation | `evaluate.py` | test parquet, models, profiles | plots, `work/evaluation_summary.json` |

Feature selection stages (all on train CV folds; required features exempt):
1. plan drops
2. Spearman redundancy
3. filters (`--filters`):
   - `quasi_constant`
   - `bivariate`: IV/WoE per bin for binary targets, Cramér's V for multiclass, eta for regression
   - `mutual_info`
   - `vif`
   - `l1`

   The filtered set is kept only if CV holds.
4. per-family tuning and permutation importance
5. keep the reduced set only if within 1 std of the full set

What `aaml_common.py` provides:
- **One split** (`split`). Profiling, the screen and preprocessing all get the same rows: exact duplicates are dropped and `work/features.py` runs on the time-sorted table before the split. The plan's `vintage_selection` then filters **train only** (`vintage=False` gives the full split, which profiling and the vintage analysis use).
- **One set of CV folds** (`cv_folds`):
  - temporal contract: expanding-window `TimeSeriesSplit`
  - repeated `entity_id_column`: group folds
  - otherwise stratified or plain K-fold
- **One encoder** (`build_encoder`). The family screen and preprocessing use the same median+indicator and one-hot encoding.
- **Metrics** (`score`): `pr_auc`, `roc_auc`, `log_loss`, `brier`, `accuracy`, `f1`, `f1_macro`, `balanced_accuracy`, `precision_at_top_<k>pct` / `recall_at_top_<k>pct` (e.g. `precision_at_top_0_5pct`), `rmse`, `mse`, `mae`, `r2`, `mape`. Lower-is-better metrics are handled. Anything else: add it to `score`.
- **Model families** (`make_model`): `logistic_regression`, `ridge`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `lightgbm`, `xgboost`. Linear models log-transform and scale inside their own pipeline, so each CV fold fits its own.

Rules the templates enforce (keep them when you adapt):
- Test is read only by `evaluate.py`, and only once. A re-run needs `--rerun-reason`, which is recorded.
- Thresholds come from CV predictions, never from test.
- Leakage suspects must be in `drop_columns`, and preprocessing fails otherwise.
- Intake answers are binding. Contract `excluded_columns` are profiled as role `drop` and can never be features. `required_features` survive feature selection, and preprocessing fails if the plan drops one.
- Every trust-check number is computed, including the test-metric bootstrap CI. Don't paste in numbers from elsewhere.
- Engineered features must be point-in-time (see `features_template.py`).

Adapt freely (a different encoder, extra families, custom segments, domain checks),
but say what you changed in the artifact's `notes`.
