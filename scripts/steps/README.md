# Step templates

Tested starting points for each step's script. They come from a full fraud trial and
are generalized to work from the contract and plan. Don't write these steps from
scratch. Copy the template you need into `work/`, adapt it to the trial, and run it:

```
cp .claude/scripts/aaml_common.py work/
cp .claude/scripts/<step>.py work/
<runtime python> work/<step>.py --help
```

Run scripts from the trial root. `aaml_common.py` must sit next to them in `work/`.

In the repo these live in `scripts/steps/`. The launcher copies them into every
trial as `.claude/scripts/` on new and resume. The rest of `scripts/` is contributor
tooling and is never copied into a trial.

| Skill | Template | Reads | Writes |
|---|---|---|---|
| data-profiling | `profile_dataset.py` | contract, data | `work/dataset_profile.json`, `work/column_profile.json` |
| workflow-planning | `features_template.py` → `work/features.py` (optional) | contract | engineered features (point-in-time) |
| workflow-planning | `model_family_screen.py` | contract, profiles, `work/features.py` | `work/model_family_advice.json` |
| data-preprocessing | `preprocess.py` | contract, plan, profiles, `work/features.py` | parquet, `outputs/pipeline/preprocessor.joblib`, `work/preprocessing_summary.json` |
| training | `feature_selection.py` (`--smoke` first) | preprocessed train, plan | `outputs/model/feature_selection.json` |
| training | `train.py` | `feature_selection.json` | models, `cv_predictions.parquet`, `work/training_summary.json` |
| evaluation | `evaluate.py` | test parquet, models, profiles | plots, `work/evaluation_summary.json` |

What `aaml_common.py` provides:
- **One split** (`split`). Profiling, the screen and preprocessing all get the same rows: exact duplicates are dropped and `work/features.py` runs on the time-sorted table before the split.
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
- Every trust-check number is computed, including the test-metric bootstrap CI. Don't paste in numbers from elsewhere.
- Engineered features must be point-in-time (see `features_template.py`).

Adapt freely (a different encoder, extra families, custom segments, domain checks),
but say what you changed in the artifact's `notes`.
