"""Evaluation: score the selected model ONCE on the held-out test split; trust checks; plots; narrative.

Template (copy to work/ with aaml_common.py). Open-source path (joblib model on the
preprocessed test parquet); on the parity-ml path, score raw test rows through the ONNX
graph instead and keep the rest.

Evaluate once: the script refuses to run when outputs/report/evaluation_summary.json
already exists, unless you pass --rerun-reason (recorded in the summary notes). Re-running
does not make it OK to change the model or threshold based on test results.

A threshold, if any, is chosen on the training CV predictions (outputs/model/cv_predictions.parquet),
never on test:
  --threshold-rule none      ranking use case: no threshold (default for binary)
  --threshold-rule f1        maximise F1 on CV predictions
  --threshold-rule top:0.01  flag the top 1% by score (rate taken from CV predictions)

Every number in the trust checks is computed here; the narrative is a draft for you to
rewrite in the business owner's terms (keep it factual).

    <runtime python> work/evaluate.py [--threshold-rule f1] [--compare-trial ../other-trial]
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aaml_common as A

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--threshold-rule", default="none")
ap.add_argument("--bootstrap", type=int, default=300, help="resamples for the test-metric confidence interval")
ap.add_argument("--compare-trial", default=None, help="another trial dir whose evaluation_summary to compare to")
ap.add_argument("--rerun-reason", default=None)
args = ap.parse_args()

FINAL = Path("outputs/report/evaluation_summary.json")
if FINAL.exists() and not args.rerun_reason:
    raise SystemExit("evaluation_summary already exists: test has been scored. Re-running needs "
                     "--rerun-reason '...' (and must not feed back into model or threshold choices).")

c, dprof, cprof = A.contract(), A.dataset_profile(), A.column_profile()
T, METRIC, TARGET = A.task(c), c["primary_metric"], c["target_column"]
training = A.read_json("outputs/model/training_summary.json")
research = A.read_json("outputs/model/auto_research_summary.json", default={})
selection = A.read_json("outputs/model/feature_selection.json", default={})
model_path = research.get("best_model_artifact_path") or training["model_artifact_path"]
bundle = joblib.load(model_path)
selected = bundle.get("family", training["selected_model"])

test = pd.read_parquet("preprocessed_data/test.parquet")
y = A.encode_target(test[TARGET], c)
s = A.predict_scores(bundle["model"], test[bundle["features"]], T)

# --- metrics ----------------------------------------------------------------------
extra = {"binary": ["pr_auc", "roc_auc", "brier", "log_loss"], "multiclass": ["accuracy", "f1_macro", "log_loss"],
         "regression": ["rmse", "mae", "r2"]}[T]
names = list(dict.fromkeys([METRIC, *c.get("report_metrics", []), *extra]))
metrics = {m: A.safe_score(m, y, s, T) for m in names}
assert metrics[METRIC] is not None, f"primary metric {METRIC} could not be computed"

rng = np.random.default_rng(A.SEED)
boot = []
for _ in range(args.bootstrap):
    if T == "binary":  # stratified, so every resample has positives
        idx = np.concatenate([rng.choice(np.flatnonzero(y == k), (y == k).sum()) for k in (0, 1)])
    else:
        idx = rng.integers(0, len(y), len(y))
    v = A.safe_score(METRIC, y[idx], s[idx], T)
    if v is not None:
        boot.append(v)
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None
ci_half = (ci[1] - ci[0]) / 2 if ci else 0.0

# --- threshold (from CV predictions, never test) -------------------------------------
threshold, confusion = None, None
cvp_path = Path("outputs/model/cv_predictions.parquet")
if T == "binary" and args.threshold_rule != "none":
    assert cvp_path.exists(), "no outputs/model/cv_predictions.parquet: re-run train.py without --no-cv-predictions"
    cvp = pd.read_parquet(cvp_path)
    if args.threshold_rule == "f1":
        from sklearn.metrics import precision_recall_curve
        p, r, th = precision_recall_curve(cvp["y"], cvp["score"])
        f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
        threshold = float(th[np.argmax(f1[:-1])])
    elif args.threshold_rule.startswith("top:"):
        threshold = float(np.quantile(cvp["score"], 1 - float(args.threshold_rule.split(":")[1])))
    else:
        raise SystemExit(f"unknown --threshold-rule {args.threshold_rule}")
    pred = (s >= threshold).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    confusion = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    metrics["precision_at_threshold"] = tp / max(1, tp + fp)
    metrics["recall_at_threshold"] = tp / max(1, tp + fn)

# --- baselines (information only; selection was fixed in training) ----------------------
if T == "binary":
    prior = np.full(len(y), A.encode_target(pd.read_parquet("preprocessed_data/train.parquet",
                                                            columns=[TARGET])[TARGET], c).mean())
    triv_name = "trivial_prior"
elif T == "multiclass":
    tr_y = A.encode_target(pd.read_parquet("preprocessed_data/train.parquet", columns=[TARGET])[TARGET], c)
    freq = np.bincount(tr_y, minlength=s.shape[1]) / len(tr_y)
    prior, triv_name = np.tile(freq, (len(y), 1)), "class_prior"
else:
    tr_y = A.encode_target(pd.read_parquet("preprocessed_data/train.parquet", columns=[TARGET])[TARGET], c)
    prior, triv_name = np.full(len(y), tr_y.mean()), "train_mean"
baseline = {triv_name: {METRIC: A.safe_score(METRIC, y, prior, T), "note": "constant prediction from train"}}
for m in training["models"]:
    if m.get("artifact_path") and m["artifact_path"] != model_path and Path(m["artifact_path"]).exists():
        other = joblib.load(m["artifact_path"])
        baseline[m["name"]] = {METRIC: A.safe_score(METRIC, y, A.predict_scores(other["model"],
                                                                                test[other["features"]], T), T),
                               "note": "training contender, scored after selection; not used to choose"}
if args.compare_trial:
    prev = A.read_json(Path(args.compare_trial) / "outputs/report/evaluation_summary.json")
    baseline[f"trial:{Path(args.compare_trial).name}"] = {prev["primary_metric"]: prev["primary_metric_value"],
                                                           "note": "earlier trial's test result"}

# --- trust checks -----------------------------------------------------------------------
used_raw = bundle.get("raw_features", [])
leak = [col["name"] for col in cprof["columns"] if col["role"] == "leakage_suspect"]
fam_sel = (selection.get("families") or {}).get(selected, {})
cv_res = (fam_sel.get("reduced") if fam_sel.get("use_reduced") else fam_sel.get("full")) or {}
cv_score = cv_res.get("score", next(m["cv_score"] for m in training["models"] if m["name"] == selected))
cv_std = cv_res.get("score_std") or next((m.get("cv_score_std") for m in training["models"]
                                          if m["name"] == selected), 0) or 0
imp = (fam_sel.get("full") or {}).get("permutation_importance", {})
pos_imp = {f: max(0.0, imp[f]["mean"]) for f in used_raw if f in imp}
top_feat = max(pos_imp, key=pos_imp.get) if pos_imp else None
top_share = pos_imp[top_feat] / sum(pos_imp.values()) if top_feat and sum(pos_imp.values()) > 0 else None

single = {}  # each numeric feature used alone as the score (both orientations) on test
if T == "binary":
    meta = joblib.load("outputs/pipeline/preprocessor.joblib")
    for f in used_raw:
        if f in meta["categorical"]:
            continue
        x = test[meta["raw_to_columns"][f][0]].to_numpy(dtype=float)
        v1, v2 = A.safe_score(METRIC, y, x, T), A.safe_score(METRIC, y, -x, T)
        vals = [v for v in (v1, v2) if v is not None]
        if vals:
            single[f] = max(vals) if A.greater_is_better(METRIC) else min(vals)
best_single = (max(single, key=single.get) if A.greater_is_better(METRIC) else min(single, key=single.get)) \
    if single else None

checks = [{"name": "leakage", "passed": not set(leak) & set(used_raw),
           "detail": f"Leakage suspects {leak or 'none'}; overlap with the {len(used_raw)} model features: "
                     f"{sorted(set(leak) & set(used_raw)) or 'none'}. Top permutation feature: {top_feat}."}]
mv = metrics[METRIC]
# A dominant feature is only suspicious if it also nearly separates the target on its own (profile probe;
# binary only: the multiclass probe takes the best one-vs-rest AUC and runs high). Otherwise it is reported.
probe = {col["name"]: (col.get("stats") or {}).get("univariate_signal_train") for col in cprof["columns"]}
dominant = T == "binary" and top_share is not None and top_share > 0.8 and (probe.get(top_feat) or 0) > 0.9
if best_single is not None:
    ratio = single[best_single] / mv if mv else None
    near_alone = A.greater_is_better(METRIC) and ratio is not None and ratio > 0.9
else:
    near_alone = False
near_perfect = ((T == "binary" and (metrics.get("roc_auc") or 0) > 0.995 and y.mean() >= 0.01)
                or (T == "regression" and (metrics.get("r2") or 0) > 0.99))
detail = f"{METRIC} {mv:.4f}"
if T == "binary":
    detail += f" vs base rate {y.mean():.4f} ({mv / max(y.mean(), 1e-12):.0f}x)" if METRIC == "pr_auc" else ""
if best_single is not None:
    detail += f"; best single feature {best_single} alone scores {single[best_single]:.4f}"
if top_share is not None:
    detail += f"; {top_feat} carries {top_share:.0%} of permutation importance"
checks.append({"name": "too_good_to_be_true", "passed": not (dominant or near_alone or near_perfect),
               "detail": detail + "."})
gap = mv - cv_score
tol = max(2 * cv_std, ci_half)
checks.append({"name": "cv_vs_test_gap", "passed": abs(gap) <= tol,
               "detail": f"Test {mv:.4f} vs CV {cv_score:.4f} +/- {cv_std:.4f} (gap {gap:+.4f}); tolerance "
                         f"max(2 x CV std, test 95% CI half-width {ci_half:.4f}) = {tol:.4f}."})
sp = dprof.get("split_summary", {}).get("test", {})
integ = len(test) == sp.get("rows", len(test)) and (T != "binary" or int(y.sum()) == sp.get("positives", int(y.sum())))
checks.append({"name": "split_integrity", "passed": bool(integ),
               "detail": f"Test rows {len(test)} (profile {sp.get('rows')})"
                         + (f", positives {int(y.sum())} (profile {sp.get('positives')})" if T == "binary" else "")})
if T == "binary":
    checks.append({"name": "test_sample_size", "passed": int(y.sum()) >= 100,
                   "detail": f"{int(y.sum())} test positives; {METRIC} 95% bootstrap CI "
                             f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "no CI"})

# --- per-segment ------------------------------------------------------------------------
seg = {}
seg_cols = [col for col in test.columns if col.startswith(A.SEGMENT_PREFIX)]
tcol = A.CARRY_PREFIX + c["time_column"] if c.get("time_column") else None
frame = test[seg_cols].copy()
if tcol and tcol in test:
    frame["month"] = pd.to_datetime(test[tcol]).dt.to_period("M").astype(str)
for col in frame.columns:
    name = col.replace(A.SEGMENT_PREFIX, "")
    seg[name] = {}
    for val, idx in frame.groupby(col).indices.items():
        entry = {"rows": int(len(idx))}
        if T == "binary":
            entry["positives"] = int(y[idx].sum())
        entry[METRIC] = A.safe_score(METRIC, y[idx], s[idx], T)
        seg[name][str(val)] = entry

# --- plots ------------------------------------------------------------------------------
plots = []
Path("outputs/report").mkdir(parents=True, exist_ok=True)
if T == "binary":
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    p, r, _ = precision_recall_curve(y, s)
    ax[0].plot(r, p, label=f"{selected} (AP {metrics.get('pr_auc', 0):.3f})")
    ax[0].axhline(y.mean(), ls="--", c="grey", label=f"base rate {y.mean():.4f}")
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision-recall (test)")
    ax[0].legend(fontsize=8)
    fpr, tpr, _ = roc_curve(y, s)
    ax[1].plot(fpr, tpr)
    ax[1].plot([0, 1], [0, 1], ls="--", c="grey")
    ax[1].set(xlabel="FPR", ylabel="TPR", title=f"ROC (AUC {metrics.get('roc_auc', 0):.3f})")
    fp_, mp_ = calibration_curve(y, s, n_bins=10, strategy="quantile")
    ax[2].plot(mp_, fp_, marker="o")
    lim = max(mp_.max(), fp_.max())
    ax[2].plot([0, lim], [0, lim], ls="--", c="grey")
    ax[2].set(xlabel="mean predicted", ylabel="observed rate", title="Calibration (deciles)")
    fig.tight_layout()
    fig.savefig("outputs/report/evaluation_curves.png", dpi=120)
    plots.append("outputs/report/evaluation_curves.png")
elif T == "regression":
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(y, s, s=4, alpha=0.3)
    lo, hi = min(y.min(), s.min()), max(y.max(), s.max())
    ax[0].plot([lo, hi], [lo, hi], ls="--", c="grey")
    ax[0].set(xlabel="actual", ylabel="predicted", title="Predicted vs actual (test)")
    ax[1].hist(y - s, bins=60)
    ax[1].set(xlabel="residual", title="Residuals (test)")
    fig.tight_layout()
    fig.savefig("outputs/report/evaluation_residuals.png", dpi=120)
    plots.append("outputs/report/evaluation_residuals.png")
if T == "multiclass" or confusion:
    from sklearn.metrics import ConfusionMatrixDisplay
    pred = np.argmax(s, axis=1) if T == "multiclass" else (s >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test)" + (f", threshold {threshold:.4f} from CV" if threshold else ""))
    fig.tight_layout()
    fig.savefig("outputs/report/confusion_matrix.png", dpi=120)
    plots.append("outputs/report/confusion_matrix.png")
if pos_imp:
    top = sorted(((f, imp[f]["mean"]) for f in used_raw if f in imp), key=lambda kv: kv[1])[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f for f, _ in top], [v for _, v in top])
    ax.set(xlabel=f"CV {METRIC} drop when permuted", title="Top features (permutation importance on CV, not test)")
    fig.tight_layout()
    fig.savefig("outputs/report/feature_importance.png", dpi=120)
    plots.append("outputs/report/feature_importance.png")

# --- summary ----------------------------------------------------------------------------
failed = [ch["name"] for ch in checks if not ch["passed"]]
triv = baseline[triv_name][METRIC]
findings = [f"Test {METRIC} {mv:.4f} (95% CI {ci[0]:.4f}-{ci[1]:.4f}); CV {cv_score:.4f} +/- {cv_std:.4f}."
            if ci else f"Test {METRIC} {mv:.4f}; CV {cv_score:.4f} +/- {cv_std:.4f}.",
            f"Trivial baseline ({triv_name}) {METRIC}: {triv:.4f}." if triv is not None else "Trivial baseline n/a.",
            *[f"{k} {METRIC}: {v[METRIC]:.4f} ({v['note']})." for k, v in baseline.items()
              if k != triv_name and v.get(METRIC) is not None],
            f"Trust checks failed: {failed}." if failed else "All trust checks passed."]
if top_feat:
    findings.append(f"Top driver on CV: {top_feat} ({top_share:.0%} of permutation importance).")
narrative = (  # DRAFT: rewrite for the business owner (what it does, how good vs baseline, trust, main caveat)
    f"The {selected} model scores {METRIC} {mv:.3f} on the held-out test data, against {triv:.3f} for a "
    f"constant prediction. " + ("All trust checks passed. " if not failed else f"Checks needing attention: "
                                f"{', '.join(failed)}. ")
    + (f"The test score is uncertain by about +/-{ci_half:.3f} (95% bootstrap interval). " if ci else ""))
summary = {
    "model_evaluated": selected,
    "split_evaluated": f"test ({c['split_strategy']} split)",
    "primary_metric": METRIC,
    "primary_metric_value": mv,
    "metrics": {**metrics, f"{METRIC}_ci_low": ci[0] if ci else None, f"{METRIC}_ci_high": ci[1] if ci else None},
    "baseline_comparison": baseline,
    "threshold": threshold,
    "per_segment": {**seg, **({"confusion_at_threshold": confusion} if confusion else {})},
    "trust_checks": checks,
    "narrative": narrative,
    "plots": plots,
    "findings": findings,
    "evaluation_script": "work/evaluate.py",
    "notes": (f"Model: {model_path}. Threshold rule: {args.threshold_rule}"
              + (" (chosen on CV predictions)." if threshold is not None else ".")
              + (f" RE-RUN: {args.rerun_reason}" if args.rerun_reason else "")),
}
A.write_json("work/evaluation_summary.json", summary)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))
print("baseline:", {k: v.get(METRIC) for k, v in baseline.items()})
for ch in checks:
    print(f"[{'PASS' if ch['passed'] else 'FAIL'}] {ch['name']}: {ch['detail']}")
print("\nDRAFT narrative (rewrite before saving):", narrative)
