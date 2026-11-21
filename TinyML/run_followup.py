#!/usr/bin/env python3
"""
GripSense TinyML - follow-up analysis (phase 2), existing data only.

Uses the ORIGINAL split manifest, feature extractor and saved Random Forest.
Writes only to reports/followup/, results/followup/, models/followup/.
No new data, no hardware.

Stages:
  1. Load + consistency checks against phase-1 artifacts
  2. Fit comparison models on TRAIN; score on VALIDATION
  3. Selection with pre-declared rules -> selection_freeze.json (before any test prediction)
  4. Held-out TEST evaluation of the frozen selections (+ all candidates, reported as secondary)
  5. Uncertainty: Wilson CIs, bootstrap macro-F1, exact McNemar tests
  6. Protocol analysis: DOUBLE_SQUEEZE anatomy, press onset in LIGHT/HARD, 20 flagged trials
  7. EXPLORATORY robustness (clearly labelled; NOT independent validation)
  8. Report + artifacts

    .venv/bin/python run_followup.py
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import run_experiment as rx  # phase-1 helpers + plot style (main() is not executed)
from gripsense_features import (LABEL_MEANINGS, LABELS, SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL,
                                build_feature_table, extract_trial_features)

SEED = 42
ROOT = Path(__file__).resolve().parent
REP, RES, MDL = ROOT / "reports" / "followup", ROOT / "results" / "followup", ROOT / "models" / "followup"
FIG = REP / "figures"
CC = rx.CLASS_COLORS
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
ACTIVE = 100  # ADC threshold for "pressed" (same constant as phase-1 features)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def segments(x: np.ndarray, th: float = ACTIVE) -> list[tuple[int, int]]:
    """[start, end) sample indices of contiguous runs above threshold."""
    a = np.r_[False, x > th, False].astype(int)
    d = np.diff(a)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def footprint(model, n_features: int) -> dict:
    """Rough on-device size estimate (flattened C arrays). Estimates, not measurements."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "estimators_"):
        nodes = sum(t.tree_.node_count for t in est.estimators_)
        used = sorted({int(f) for t in est.estimators_ for f in t.tree_.feature if f >= 0})
        return {"kind": f"{len(est.estimators_)} trees", "size_units": f"{nodes} nodes",
                "est_bytes": nodes * 12, "n_features_used": len(used), "features_used_idx": used}
    if hasattr(est, "tree_"):
        nodes = est.tree_.node_count
        used = sorted({int(f) for f in est.tree_.feature if f >= 0})
        return {"kind": "1 tree", "size_units": f"{nodes} nodes", "est_bytes": nodes * 12,
                "n_features_used": len(used), "features_used_idx": used}
    if hasattr(est, "coef_"):
        n = est.coef_.size + est.intercept_.size + 2 * n_features  # + scaler mean/scale
        return {"kind": "linear", "size_units": f"{n} float params", "est_bytes": n * 4,
                "n_features_used": n_features, "features_used_idx": list(range(n_features))}
    if hasattr(est, "centroids_"):
        return {"kind": "centroids", "size_units": f"{est.centroids_.size} floats",
                "est_bytes": est.centroids_.size * 4, "n_features_used": est.centroids_.shape[1],
                "features_used_idx": [0]}
    return {"kind": "constant", "size_units": "1 label", "est_bytes": 1, "n_features_used": 0,
            "features_used_idx": []}


def proba_or_none(model, X):
    try:
        return model.predict_proba(X)
    except Exception:
        return None


def bootstrap_macro_f1(y, yh, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, yh = np.asarray(y), np.asarray(yh)
    vals = [f1_score(y[i], yh[i], labels=LABELS, average="macro", zero_division=0)
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar_exact(y, a, b) -> dict:
    y, a, b = map(np.asarray, (y, a, b))
    a_only = int(np.sum((a == y) & (b != y)))
    b_only = int(np.sum((b == y) & (a != y)))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(min(a_only, b_only), n, 0.5).pvalue)
    return {"only_first_correct": a_only, "only_second_correct": b_only, "p_value_exact": p}


def main() -> None:
    for d in (REP, RES, MDL, FIG):
        d.mkdir(parents=True, exist_ok=True)
    stage_log = []

    def stage(name):
        stage_log.append({"stage": name, "utc": rx.now()})
        print(f"[{stage_log[-1]['utc']}] {name}")

    # ======================================================== 1. load + checks
    stage("1. load phase-1 artifacts + consistency checks")
    csv = ROOT / "gripsense_clean_data.csv"
    m1 = json.loads((ROOT / "reports" / "metrics.json").read_text())
    assert sha(csv) == m1["input"]["sha256"], "original CSV changed"
    df = pd.read_csv(csv)
    feats = build_feature_table(df)
    tf_saved = pd.read_csv(ROOT / "reports" / "trial_features.csv").sort_values("trial")
    fcols = [c for c in feats.columns if c not in ("trial", "label")]
    assert np.allclose(feats.sort_values("trial")[fcols].to_numpy(), tf_saved[fcols].to_numpy())
    manifest = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
    feats = feats.merge(manifest[["trial", "split"]], on="trial")
    part = {s: feats[feats["split"] == s].sort_values("trial").reset_index(drop=True)
            for s in ("train", "validation", "test")}
    ids = {s: set(p["trial"]) for s, p in part.items()}
    assert not (ids["train"] & ids["validation"] or ids["train"] & ids["test"]
                or ids["validation"] & ids["test"])
    bundle = joblib.load(ROOT / "models" / "gripsense_model.joblib")
    meta1 = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    assert set(meta1["training_trial_ids"]) == ids["train"]
    assert bundle["feature_columns"] == fcols
    Xtr, ytr = part["train"][fcols], part["train"]["label"].to_numpy()
    Xva, yva = part["validation"][fcols], part["validation"]["label"].to_numpy()
    Xte, yte = part["test"][fcols], part["test"]["label"].to_numpy()
    wave = {int(t): g.sort_values("elapsed_ms")["adc"].to_numpy() for t, g in df.groupby("trial")}
    label_of = feats.set_index("trial")["label"].to_dict()
    split_of = feats.set_index("trial")["split"].to_dict()
    flags = pd.read_csv(ROOT / "reports" / "trial_quality_flags.csv")
    flagged_ids = flags.loc[flags["flagged"], "trial"].tolist()

    # ======================================================== 2. candidates (train -> val)
    stage("2. fit candidates on train, score on validation")
    cands = {
        "B0_majority": ("Baseline", {}, 0, DummyClassifier(strategy="most_frequent"), fcols, False, True),
        "B1_mean_adc_centroid": ("Baseline", {"feature": "mean"}, 0, NearestCentroid(), ["mean"], False, True),
        "RF_existing": ("RandomForest", bundle["params"], 3, bundle["model"], fcols, True, False),
    }
    for C in (0.1, 1.0, 10.0):
        cands[f"LR_C{C:g}"] = ("LogisticRegression", {"C": C}, 2, make_pipeline(
            StandardScaler(), LogisticRegression(C=C, max_iter=5000)), fcols, True, True)
    for depth in (2, 3):
        cands[f"DT_depth{depth}"] = ("DecisionTree", {"max_depth": depth}, 1,
                                     DecisionTreeClassifier(max_depth=depth, random_state=SEED),
                                     fcols, True, True)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models, rows = {}, []
    for name, (fam, params, cx, est, cols, eligible, refit) in cands.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_f1 = cross_val_score(clone(est), Xtr[cols], ytr, cv=cv, scoring="f1_macro")
            model = clone(est).fit(Xtr[cols], ytr) if refit else est  # RF: saved model as-is
        models[name] = (model, cols)
        pv = model.predict(Xva[cols])
        pr = proba_or_none(model, Xva[cols])
        fp = footprint(model, len(cols))
        rows.append({
            "model": name, "family": fam, "params": json.dumps(params), "eligible": eligible,
            "complexity_rank": cx, "train_accuracy": accuracy_score(ytr, model.predict(Xtr[cols])),
            "train_cv5_macro_f1": cv_f1.mean(), "val_accuracy": accuracy_score(yva, pv),
            "val_macro_f1": f1_score(yva, pv, labels=LABELS, average="macro", zero_division=0),
            "val_log_loss": log_loss(yva, pr, labels=model.classes_) if pr is not None else np.nan,
            "val_errors": int((pv != yva).sum()),
            "footprint": fp["size_units"], "est_bytes": fp["est_bytes"],
            "n_features_used": fp["n_features_used"],
            "features_used": ", ".join(cols[i] for i in fp["features_used_idx"])
            if fp["n_features_used"] <= 10 else f"{fp['n_features_used']} features",
        })
    # The saved RF must still reproduce its phase-1 validation predictions.
    old_val = pd.read_csv(ROOT / "results" / "validation_predictions.csv").sort_values("trial")
    rf_val_match = list(models["RF_existing"][0].predict(Xva)) == list(old_val["predicted"])
    assert rf_val_match
    comp = pd.DataFrame(rows)

    # ======================================================== 3. selection + freeze
    stage("3. validation-based selection; freeze before any test prediction")
    el = comp[comp["eligible"]].copy()
    rule_a = el.sort_values(["val_macro_f1", "train_cv5_macro_f1", "val_log_loss", "complexity_rank"],
                            ascending=[False, False, True, True], kind="mergesort")
    best_f1 = el["val_macro_f1"].max()
    tied = el[np.isclose(el["val_macro_f1"], best_f1)]
    rule_b = tied.sort_values(["est_bytes", "train_cv5_macro_f1", "val_log_loss"],
                              ascending=[True, False, True], kind="mergesort")
    sel_a, sel_b = rule_a.iloc[0]["model"], rule_b.iloc[0]["model"]
    best_lr = rule_a[rule_a["family"] == "LogisticRegression"].iloc[0]["model"]
    best_dt = rule_a[rule_a["family"] == "DecisionTree"].iloc[0]["model"]
    freeze = {
        "frozen_utc": rx.now(),
        "test_predictions_computed_in_this_run_before_freeze": False,
        "rule_A_primary": "max val macro-F1 -> max train 5-fold CV macro-F1 -> min val log-loss -> "
                          "simpler family (identical to phase-1 rule)",
        "rule_B_esp32_deployment": "among eligible models tied at the best val macro-F1, smallest "
                                   "estimated on-device size -> max train CV macro-F1 -> min val log-loss",
        "rule_B_disclosure": "Rule B was formulated after the phase-1 validation table (all candidates "
                             "tied at 0.975) had been seen, but before any test prediction of the LR/DT "
                             "candidates.",
        "selected_rule_A": sel_a, "selected_rule_B": sel_b,
        "family_representatives_by_rule_A": {"LogisticRegression": best_lr, "DecisionTree": best_dt},
        "validation_table": comp.drop(columns=["features_used"]).to_dict(orient="records"),
    }
    freeze_path = RES / "selection_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, default=float))
    freeze_hash = sha(freeze_path)
    print(f"   rule A -> {sel_a}; rule B -> {sel_b}")
    for name in (sel_a, sel_b, best_lr, best_dt):
        m, cols = models[name]
        joblib.dump({"model": m, "feature_columns": cols, "labels": LABELS, "seed": SEED,
                     "trained_on": "phase-1 training split (120 trials)", "name": name},
                    MDL / f"{name}.joblib")
    dt_model = models[best_dt][0]
    (MDL / f"{best_dt}_rules.txt").write_text(export_text(dt_model, feature_names=fcols, decimals=3))

    # ======================================================== 4. test evaluation
    stage("4. held-out test evaluation (frozen selections; all candidates secondary)")
    pred_rows, evals = [], {}
    for name, (model, cols) in models.items():
        evals[name] = {}
        for split, X, y, p in (("validation", Xva, yva, part["validation"]),
                               ("test", Xte, yte, part["test"])):
            yh = model.predict(X[cols])
            pr = proba_or_none(model, X[cols])
            ev = rx.evaluate(y, yh)
            ev["misclassified_trials"] = p.loc[yh != y, "trial"].astype(int).tolist()
            ev["recall_wilson_95ci"] = {
                lab: rx.wilson_ci(int(np.sum((y == lab) & (yh == lab))), int(np.sum(y == lab)))
                for lab in LABELS}
            ev["macro_f1_bootstrap_95ci"] = bootstrap_macro_f1(y, yh)
            evals[name][split] = ev
            for i, t in enumerate(p["trial"]):
                pred_rows.append({"model": name, "split": split, "trial": int(t), "actual": y[i],
                                  "predicted": yh[i], "correct": bool(yh[i] == y[i]),
                                  "confidence": round(float(pr[i].max()), 4) if pr is not None else np.nan,
                                  "audit_flagged": int(t) in flagged_ids})
    preds = pd.DataFrame(pred_rows)
    preds.to_csv(RES / "predictions_all_models.csv", index=False)
    old_test = pd.read_csv(ROOT / "results" / "test_predictions.csv").sort_values("trial")
    rf_test_match = list(preds.query("model=='RF_existing' and split=='test'").sort_values("trial")
                         ["predicted"]) == list(old_test["predicted"])
    assert rf_test_match
    assert sha(freeze_path) == freeze_hash
    comp["test_accuracy"] = comp["model"].map(lambda n: evals[n]["test"]["accuracy"])
    comp["test_macro_f1"] = comp["model"].map(lambda n: evals[n]["test"]["macro_avg"]["f1"])
    comp["test_errors"] = comp["model"].map(lambda n: evals[n]["test"]["n_trials"] - evals[n]["test"]["n_correct"])
    comp["test_misclassified"] = comp["model"].map(lambda n: evals[n]["test"]["misclassified_trials"])
    comp["val_misclassified"] = comp["model"].map(lambda n: evals[n]["validation"]["misclassified_trials"])
    comp.to_csv(RES / "model_comparison.csv", index=False)

    # ======================================================== 5. uncertainty
    stage("5. uncertainty: paired exact McNemar tests on test set")
    mcn = {}
    for other in comp["model"]:
        if other == sel_a:
            continue
        a = preds.query("model==@sel_a and split=='test'").sort_values("trial")
        b = preds.query("model==@other and split=='test'").sort_values("trial")
        mcn[f"{sel_a} vs {other}"] = mcnemar_exact(a["actual"], a["predicted"], b["predicted"])
    # Smallest per-class recall that is still consistent with 10/10 at 95% (Wilson)
    lo10 = rx.wilson_ci(10, 10)[0]
    lo9 = rx.wilson_ci(9, 10)[0]

    # ======================================================== 6. protocol analysis
    stage("6. protocol analysis: DOUBLE_SQUEEZE, press onset, flagged trials")
    seg_rows = []
    for t, x in wave.items():
        s = segments(x)
        lab = label_of[t]
        durs = [(b - a) * SAMPLE_INTERVAL_MS for a, b in s]
        gaps = [(s[k + 1][0] - s[k][1]) * SAMPLE_INTERVAL_MS for k in range(len(s) - 1)]
        after2 = (np.nan if len(s) < 2 else
                  ((s[2][0] if len(s) > 2 else len(x)) - s[1][1]) * SAMPLE_INTERVAL_MS)
        seg_rows.append({
            "trial": t, "label": lab, "n_presses": len(s),
            "complete_presses": sum(1 for a, b in s if a > 0 and b < len(x)),
            "active_at_start": bool(x[0] > ACTIVE), "active_at_end": bool(x[-1] > ACTIVE),
            "first_onset_ms": s[0][0] * SAMPLE_INTERVAL_MS if s else np.nan,
            "median_press_ms": float(np.median(durs)) if durs else np.nan,
            "median_gap_ms": float(np.median(gaps)) if gaps else np.nan,
            "presses_per_s": len(s) / (len(x) * SAMPLE_INTERVAL_MS / 1000),
            "rest_after_2nd_press_ms": after2, "peak_adc": int(x.max()),
            "active_fraction": float(np.mean(x > ACTIVE)),
        })
    seg = pd.DataFrame(seg_rows)
    seg.to_csv(RES / "press_segments_per_trial.csv", index=False)
    ds = seg[seg["label"] == "DOUBLE_SQUEEZE"]
    rho_peak = spearmanr(ds["trial"], ds["peak_adc"])
    rho_n = spearmanr(ds["trial"], ds["n_presses"])
    ds_stats = {
        "n_trials": len(ds),
        "presses_distribution": ds["n_presses"].value_counts().sort_index().to_dict(),
        "trials_with_exactly_2_presses": int((ds["n_presses"] == 2).sum()),
        "complete_presses_median": float(ds["complete_presses"].median()),
        "median_press_ms": float(ds["median_press_ms"].median()),
        "median_gap_ms": float(ds["median_gap_ms"].median()),
        "presses_per_s_median": float(ds["presses_per_s"].median()),
        "rest_after_2nd_press_ms_median": float(ds["rest_after_2nd_press_ms"].median()),
        "rest_after_2nd_press_ms_max": float(ds["rest_after_2nd_press_ms"].max()),
        "trials_with_>=1000ms_rest_after_2nd_press": int((ds["rest_after_2nd_press_ms"] >= 1000).sum()),
        "active_at_window_end": int(ds["active_at_end"].sum()),
        "active_at_window_start": int(ds["active_at_start"].sum()),
        "spearman_trialid_vs_peak": [float(rho_peak.statistic), float(rho_peak.pvalue)],
        "spearman_trialid_vs_presses": [float(rho_n.statistic), float(rho_n.pvalue)],
        "peak_adc_median": float(ds["peak_adc"].median()),
    }
    onset_stats = {}
    for lab in ("LIGHT_PRESS", "HARD_PRESS"):
        s_ = seg[seg["label"] == lab]
        onset_stats[lab] = {"active_at_start": int(s_["active_at_start"].sum()),
                            "active_at_end": int(s_["active_at_end"].sum()),
                            "single_press": int((s_["n_presses"] == 1).sum()), "n": len(s_)}

    # Flagged trials: held-out prediction (val/test) or out-of-fold prediction (train)
    rf_cfg = clone(bundle["model"])
    oof = pd.Series(cross_val_predict(rf_cfg, Xtr, ytr, cv=cv), index=part["train"]["trial"])
    rf_preds = preds.query("model=='RF_existing'").set_index("trial")["predicted"]
    med = {lab: np.median(np.stack([wave[t] for t in wave if label_of[t] == lab]), axis=0)
           for lab in LABELS}
    fl_rows = []
    for t in flagged_ids:
        x, lab = wave[t], label_of[t]
        f = feats.set_index("trial").loc[t]
        if lab == "LIGHT_PRESS":
            pattern = ("brief peak into HARD_PRESS force range" if f["max"] > 1500 else
                       "contact lost / released during the window" if f["zero_fraction"] > 0.1 else
                       "force less steady than typical LIGHT_PRESS (larger fluctuations)")
        elif lab == "HARD_PRESS":
            mid = x[10:-10]
            dip = 1 - mid.min() / max(mid.max(), 1)
            pattern = ("released to zero during the window" if f["zero_fraction"] > 0 else
                       f"force not held steady (dips {dip:.0%} below its peak)")
        else:
            pattern = f"brief unintended contact (peak {int(f['max'])} ADC) during rest"
        nearest = min(LABELS, key=lambda l: float(np.mean((x - med[l]) ** 2)))
        if split_of[t] == "train":
            pred, src = oof.loc[t], "out-of-fold (5-fold CV within train)"
        else:
            pred, src = rf_preds.loc[t], f"held-out {split_of[t]}"
        fl_rows.append({"trial": t, "label": lab, "split": split_of[t], "pattern": pattern,
                        "peak_adc": int(f["max"]), "mean_adc": round(float(f["mean"]), 1),
                        "zero_fraction": round(float(f["zero_fraction"]), 3),
                        "n_presses": int(seg.set_index("trial").loc[t, "n_presses"]),
                        "closest_class_median_waveform": nearest,
                        "rf_prediction": pred, "prediction_source": src,
                        "rf_correct": pred == lab,
                        "audit_flags": flags.set_index("trial").loc[t, "flags"]})
    fl = pd.DataFrame(fl_rows)
    fl.to_csv(RES / "flagged_trials_review.csv", index=False)
    oof_all = pd.Series(oof.values == ytr, index=part["train"]["trial"])
    oof_acc = float(oof_all.mean())

    # ======================================================== 7. exploratory robustness
    stage("7. EXPLORATORY robustness analyses (not independent validation)")
    expl_models = {"RF": ("RF_existing", clone(bundle["model"])),
                   "LR": (best_lr, clone(models[best_lr][0])),
                   "DT": (best_dt, clone(models[best_dt][0]))}
    # E1 repeated stratified CV over all 200 trials
    Xall, yall = feats.sort_values("trial")[fcols], feats.sort_values("trial")["label"].to_numpy()
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    e1 = {}
    for short, (name, est) in expl_models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = cross_val_score(est, Xall, yall, cv=rcv, scoring="accuracy", n_jobs=-1)
        e1[short] = {"model": name, "n_folds": len(acc), "mean": float(acc.mean()),
                     "std": float(acc.std()), "min": float(acc.min()),
                     "p05": float(np.percentile(acc, 5)), "max": float(acc.max()), "_all": acc.tolist()}
    # E2 chronological within-class split (drift proxy)
    fs = feats.sort_values("trial")
    rank = fs.groupby("label")["trial"].rank(method="first")
    e2 = {}
    for direction, tr_mask in (("early->late", rank <= 30), ("late->early", rank > 20)):
        te_mask = ~tr_mask
        e2[direction] = {}
        for short, (name, est) in expl_models.items():
            mdl = clone(est).fit(fs.loc[tr_mask, fcols], fs.loc[tr_mask, "label"])
            yh = mdl.predict(fs.loc[te_mask, fcols])
            yt = fs.loc[te_mask, "label"].to_numpy()
            e2[direction][short] = {
                "train_trials": int(tr_mask.sum()), "test_trials": int(te_mask.sum()),
                "accuracy": float(accuracy_score(yt, yh)),
                "macro_f1": float(f1_score(yt, yh, labels=LABELS, average="macro")),
                "confusion_matrix": confusion_matrix(yt, yh, labels=LABELS).tolist(),
                "misclassified": fs.loc[te_mask, "trial"][yh != yt].astype(int).tolist()}
    # E3 perturbations + synthetic probes on held-out trials (val + test), phase-1-trained models
    fitted = {"RF": models["RF_existing"][0], "LR": models[best_lr][0], "DT": models[best_dt][0]}
    held = sorted(ids["validation"] | ids["test"])
    rng = np.random.default_rng(SEED)

    def predict_waves(ws):
        F = pd.DataFrame([extract_trial_features(w) for w in ws])[fcols]
        return {k: m.predict(F) for k, m in fitted.items()}

    pert_rows = []
    perturbations = ([("gain", g) for g in (0.7, 0.85, 1.0, 1.15, 1.3)] +
                     [("noise_while_pressed_sd", s) for s in (25, 50)] +
                     [("floor_offset", o) for o in (10, 30)])
    y_held = np.array([label_of[t] for t in held])
    for kind, v in perturbations:
        ws = []
        for t in held:
            x = wave[t].astype(float)
            if kind == "gain":
                x = x * v
            elif kind == "noise_while_pressed_sd":
                x = np.where(x > 0, x + rng.normal(0, v, len(x)), 0)
            else:
                x = x + v
            ws.append(np.clip(np.round(x), 0, 4095))
        out = predict_waves(ws)
        for k, yh in out.items():
            row = {"perturbation": kind, "value": v, "model": k,
                   "accuracy": float(accuracy_score(y_held, yh)),
                   "macro_f1": float(f1_score(y_held, yh, labels=LABELS, average="macro", zero_division=0))}
            for lab in LABELS:
                msk = y_held == lab
                row[f"recall_{lab}"] = float(np.mean(yh[msk] == lab))
            pert_rows.append(row)
    pert = pd.DataFrame(pert_rows)
    pert.to_csv(RES / "exploratory_perturbations.csv", index=False)

    probe_rows, probe_examples = [], {}
    held_ds = [t for t in held if label_of[t] == "DOUBLE_SQUEEZE"]
    held_lh = [t for t in held if label_of[t] in ("LIGHT_PRESS", "HARD_PRESS")]
    probes = {}
    ws = []
    for t in held_ds:
        x, s = wave[t].copy(), segments(wave[t])
        x[s[1][1]:] = 0
        ws.append(x)
    probes["P1_two_squeezes_then_rest"] = (held_ds, ws, "DOUBLE_SQUEEZE")
    ws = []
    for t in held_ds:
        x, s = wave[t].copy(), segments(wave[t])
        x[s[0][1]:] = 0
        ws.append(x)
    probes["P2_single_squeeze_then_rest"] = (held_ds, ws, "(none - not a defined gesture)")
    ws = []
    for t in held_lh:
        x = wave[t].copy()
        x[:25], x[-25:] = 0, 0
        ws.append(x)
    probes["P3_press_with_onset_and_release"] = (held_lh, ws, "original label")
    for pname, (tids, ws, intended) in probes.items():
        out = predict_waves(ws)
        probe_examples[pname] = (tids[0], ws[0])
        for k, yh in out.items():
            for t, p_ in zip(tids, yh):
                probe_rows.append({"probe": pname, "model": k, "trial": t,
                                   "original_label": label_of[t], "intended_label": intended
                                   if intended != "original label" else label_of[t], "predicted": p_})
    probe = pd.DataFrame(probe_rows)
    probe.to_csv(RES / "exploratory_probes.csv", index=False)
    probe_summary = (probe.groupby(["probe", "model", "original_label"])["predicted"]
                     .value_counts().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0))
    probe_summary.to_csv(RES / "exploratory_probe_summary.csv")

    # E4 confidence threshold for RF chosen on validation (accepted accuracy = 100%)
    rfv = preds.query("model=='RF_existing' and split=='validation'")
    rft = preds.query("model=='RF_existing' and split=='test'")
    thr_rows = []
    for th in np.round(np.arange(0.40, 1.0001, 0.05), 2):
        for sp, d in (("validation", rfv), ("test", rft)):
            acc_ = d[d["confidence"] >= th]
            thr_rows.append({"threshold": th, "split": sp, "coverage": len(acc_) / len(d),
                             "accepted_accuracy": float(acc_["correct"].mean()) if len(acc_) else np.nan,
                             "accepted_errors": int((~acc_["correct"]).sum())})
    thr = pd.DataFrame(thr_rows)
    thr.to_csv(RES / "exploratory_confidence_threshold.csv", index=False)
    vt = thr[(thr["split"] == "validation") & (thr["accepted_errors"] == 0)]
    chosen_thr = float(vt["threshold"].min())
    thr_val = thr[(thr["split"] == "validation") & (thr["threshold"] == chosen_thr)].iloc[0]
    thr_test = thr[(thr["split"] == "test") & (thr["threshold"] == chosen_thr)].iloc[0]
    held_flag = preds[preds["model"].isin(["RF_existing", best_lr, best_dt])]
    e5 = (held_flag.groupby(["model", "audit_flagged"])["correct"].agg(["sum", "count"])
          .reset_index())

    # ======================================================== 8. figures + metrics + report
    stage("8. figures, metrics, report")
    figs = make_figures(comp, sel_a, sel_b, evals, models, ds, seg, wave, fl, med, e1, pert,
                        probe_summary, probe_examples, thr, chosen_thr, label_of)
    metrics = {
        "generated_utc": rx.now(), "seed": SEED, "environment": m1["environment"] | {
            "scipy": __import__("scipy").__version__},
        "phase1_reference": {"metrics_generated_utc": m1["generated_utc"],
                             "input_sha256": m1["input"]["sha256"]},
        "split_counts": {s: len(p) for s, p in part.items()},
        "consistency": {"features_match_phase1": True, "rf_reproduces_phase1_val": rf_val_match,
                        "rf_reproduces_phase1_test": rf_test_match,
                        "freeze_unchanged_after_test": sha(freeze_path) == freeze_hash},
        "test_history_disclosure": (
            "Phase 1 already computed and reported RF_existing test results (39/40, trial 39 "
            "misclassified); the phase-1 run history also discloses a rerun after the test score was "
            "seen. The test set is therefore NOT completely untouched. In phase 2, selection rules "
            "were frozen (selection_freeze.json) before any test prediction of the new candidates. "
            "run_followup.py was executed more than once while the report text was being corrected; "
            "between runs only report wording changed - the split, candidates, selection rules and "
            "selected models were identical in every run."),
        "selection": {"rule_A": sel_a, "rule_B": sel_b, "best_lr": best_lr, "best_dt": best_dt,
                      "n_eligible": len(el), "n_tied_best_val": len(tied),
                      "best_val_macro_f1": float(best_f1), "freeze_sha256": freeze_hash},
        "comparison": comp.drop(columns=["test_misclassified", "val_misclassified"]).to_dict(orient="records"),
        "evaluations": evals,
        "mcnemar_test_vs_selected": mcn,
        "wilson_lower_bound_10_of_10": lo10, "wilson_lower_bound_9_of_10": lo9,
        "double_squeeze": ds_stats, "press_onset": onset_stats,
        "flagged": {"n": len(fl), "rf_correct": int(fl["rf_correct"].sum()),
                    "by_label": fl["label"].value_counts().to_dict(),
                    "train_oof_accuracy_all_train": oof_acc},
        "exploratory": {
            "E1_repeated_cv": {k: {kk: vv for kk, vv in v.items() if kk != "_all"} for k, v in e1.items()},
            "E2_chronological": e2,
            "E3_perturbations": pert.to_dict(orient="records"),
            "E3_probes": {f"{p}|{m}|{o}": r.to_dict() for (p, m, o), r in probe_summary.iterrows()},
            "E4_confidence_threshold": {"chosen_on_validation": chosen_thr,
                                        "val_coverage": float(thr_val["coverage"]),
                                        "test_coverage": float(thr_test["coverage"]),
                                        "test_accepted_accuracy": float(thr_test["accepted_accuracy"]),
                                        "test_accepted_errors": int(thr_test["accepted_errors"])},
            "E5_flagged_vs_unflagged": e5.to_dict(orient="records"),
        },
        "stage_log": stage_log, "figures": figs,
    }
    (REP / "followup_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (MDL / "followup_models_metadata.json").write_text(json.dumps({
        "models": {n: {"file": f"{n}.joblib", **{k: v for k, v in footprint(models[n][0], len(models[n][1])).items()
                                                  if k != "features_used_idx"}}
                   for n in {sel_a, sel_b, best_lr, best_dt}},
        "training_trial_ids": sorted(ids["train"]), "feature_columns": fcols,
        "feature_module": "gripsense_features.extract_trial_features",
        "note": "RF_existing is a copy of models/gripsense_model.joblib's estimator (not refitted).",
    }, indent=2, default=str))
    write_report(metrics, comp, fl, pert, probe_summary, e2, thr, m1, best_lr, best_dt, sel_a, sel_b, label_of)
    print("Done: reports/followup/GripSense_TinyML_Final_Report.md")


# ============================================================ figures
def make_figures(comp, sel_a, sel_b, evals, models, ds, seg, wave, fl, med, e1, pert, probe_summary,
                 probe_examples, thr, chosen_thr, label_of) -> dict:
    figs = {}
    t_ms = np.arange(SAMPLES_PER_TRIAL) * SAMPLE_INTERVAL_MS

    # Model comparison: validation (used for selection) vs test (secondary)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    names = list(comp["model"])
    y = np.arange(len(names))
    for off, col, colr, lab in ((-0.19, "val_accuracy", SERIES[0], "Validation (used for selection)"),
                                (0.19, "test_accuracy", SERIES[1], "Test (evaluated after freeze)")):
        ax.barh(y + off, comp[col], height=0.36, color=colr, label=lab)
        for yi, v in zip(y + off, comp[col]):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=7.5, color=rx.INK2)
    ax.set_yticks(y, [n + ("  [A]" if n == sel_a else "") + ("  [B]" if n == sel_b else "") for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Accuracy (40 trials each)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.set_title("Model comparison on the phase-1 split  ([A] rule-A selection, [B] ESP32 rule-B)",
                 loc="left")
    figs["model_comparison"] = rx.savefig(fig, FIG / "f01_model_comparison.png")

    for name, tag in ((sel_a, "selected_A"), (sel_b, "deployment_B")):
        figs[f"cm_test_{tag}"] = rx.plot_confusion(
            evals[name]["test"]["confusion_matrix"]["matrix"],
            f"{name} - held-out TEST (40 trials)", FIG / f"f02_cm_test_{name}.png")

    # DOUBLE_SQUEEZE anatomy
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax = axes[0, 0]
    vc = ds["n_presses"].value_counts().reindex(range(1, 8), fill_value=0)
    ax.bar(vc.index, vc.values, color=[rx.INK2 if i == 2 else CC["DOUBLE_SQUEEZE"] for i in vc.index],
           width=0.7)
    for i, v in vc.items():
        ax.text(i, v + 0.4, str(v), ha="center", fontsize=8, color=rx.INK2)
    ax.annotate("intended gesture = 2\n(0 trials)", (2, 0.3), xytext=(1.2, 14), fontsize=8,
                color=rx.INK, arrowprops=dict(arrowstyle="->", color=rx.INK2))
    ax.set_xlabel("Distinct presses in the 3 s window (> 100 ADC)")
    ax.set_ylabel("Trials")
    ax.grid(axis="x", visible=False)
    ax.set_title("Presses per DOUBLE_SQUEEZE trial (50 trials)", loc="left")
    ax = axes[0, 1]
    ax.hist(ds["median_press_ms"], bins=np.arange(0, 900, 60), color=CC["DOUBLE_SQUEEZE"],
            edgecolor=rx.SURFACE, label="median press duration")
    ax.hist(ds["median_gap_ms"].dropna(), bins=np.arange(0, 900, 60), color=rx.INK2, alpha=0.55,
            edgecolor=rx.SURFACE, label="median gap between presses")
    ax.set_xlabel("ms")
    ax.set_ylabel("Trials")
    ax.legend(fontsize=8)
    ax.set_title("Rhythm: press ~{:.0f} ms, gap ~{:.0f} ms (medians)".format(
        ds["median_press_ms"].median(), ds["median_gap_ms"].median()), loc="left")
    ax = axes[1, 0]
    t0, w_probe = probe_examples["P1_two_squeezes_then_rest"]
    ax.plot(t_ms, wave[t0], color=CC["DOUBLE_SQUEEZE"], label=f"recorded trial #{t0}")
    ax.plot(t_ms, w_probe, color=rx.INK, linewidth=1.2, linestyle="--",
            label="synthetic 'intended' probe P1 (after 2nd press set to 0)")
    for a, b in segments(wave[t0]):
        ax.axvspan(a * SAMPLE_INTERVAL_MS, b * SAMPLE_INTERVAL_MS, color=CC["DOUBLE_SQUEEZE"],
                   alpha=0.08, linewidth=0)
    ax.set_xlabel("Time within trial (ms)")
    ax.set_ylabel("ADC")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Recorded vs intended double squeeze", loc="left")
    ax = axes[1, 1]
    ax.scatter(ds["trial"], ds["peak_adc"], s=18, color=CC["DOUBLE_SQUEEZE"], edgecolors=rx.SURFACE)
    ax.set_xlabel("Trial ID (recording order)")
    ax.set_ylabel("Peak ADC")
    ax2_txt = ax.twinx()
    ax2_txt.set_visible(False)
    ax.set_title("Squeeze force changed during the DOUBLE_SQUEEZE block", loc="left")
    fig.suptitle("DOUBLE_SQUEEZE: what was recorded", x=0.01, ha="left", fontsize=12)
    figs["double_squeeze"] = rx.savefig(fig, FIG / "f03_double_squeeze_anatomy.png")

    # All flagged trials with class median reference
    n = len(fl)
    ncol = 5
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 2.5 * nrow), squeeze=False, sharex=True, sharey=True)
    for ax, (_, r) in zip(axes.flat, fl.iterrows()):
        ax.plot(t_ms, med[r["label"]], color=rx.INK2, linewidth=1, linestyle=":")
        ax.plot(t_ms, wave[r["trial"]], color=CC[r["label"]])
        mark = "ok" if r["rf_correct"] else "WRONG"
        ax.set_title(f"#{r['trial']} {r['label']} [{r['split']}]\nRF: {r['rf_prediction']} ({mark})",
                     loc="left", fontsize=7.5)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("All 20 flagged trials (solid) vs their class median waveform (dotted). "
                 "RF prediction: held-out for val/test, out-of-fold for train", x=0.01, ha="left",
                 fontsize=11)
    figs["flagged_all"] = rx.savefig(fig, FIG / "f04_flagged_trials_all.png")

    # E1 repeated CV distribution
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rng = np.random.default_rng(SEED)
    for i, (k, v) in enumerate(e1.items()):
        a = np.array(v["_all"])
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a, s=10, color=SERIES[i], alpha=0.6,
                   edgecolors="none")
        ax.plot([i - 0.25, i + 0.25], [a.mean()] * 2, color=rx.INK, linewidth=2)
        ax.text(i + 0.3, a.mean(), f"mean {a.mean():.3f}", va="center", fontsize=8, color=rx.INK2)
    ax.set_xticks(range(len(e1)), [f"{k}\n({v['model']})" for k, v in e1.items()])
    ax.set_xlim(-0.5, len(e1) - 0.1)
    ax.set_ylabel("Fold accuracy (40 trials)")
    ax.grid(axis="x", visible=False)
    ax.set_title("EXPLORATORY: 20x repeated 5-fold CV on all 200 trials", loc="left")
    figs["repeated_cv"] = rx.savefig(fig, FIG / "f05_exploratory_repeated_cv.png")

    # E3 gain perturbation
    fig, ax = plt.subplots(figsize=(7, 3.4))
    g = pert[pert["perturbation"] == "gain"]
    for i, k in enumerate(("RF", "LR", "DT")):
        s = g[g["model"] == k]
        ax.plot(s["value"], s["accuracy"], color=SERIES[i], marker="o", markersize=5, label=k)
        ax.text(s["value"].iloc[-1] + 0.01, s["accuracy"].iloc[-1], k, color=rx.INK2, fontsize=8,
                va="center")
    ax.set_xlabel("Simulated sensor gain (1.0 = as recorded)")
    ax.set_ylabel("Accuracy, 80 held-out trials")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("EXPLORATORY: sensitivity to force/gain scaling", loc="left")
    figs["gain"] = rx.savefig(fig, FIG / "f06_exploratory_gain_sensitivity.png")

    # Probes: RF predictions
    rf_pr = probe_summary.xs("RF", level="model")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    labels_y = [f"{p}\n(from {o})" for p, o in rf_pr.index]
    left = np.zeros(len(rf_pr))
    for lab in LABELS:
        vals = rf_pr[lab].to_numpy()
        ax.barh(range(len(rf_pr)), vals, left=left, color=CC[lab], label=lab, height=0.6,
                edgecolor=rx.SURFACE, linewidth=2)
        for i, (l_, v) in enumerate(zip(left, vals)):
            if v >= 1:
                ax.text(l_ + v / 2, i, str(int(v)), ha="center", va="center", fontsize=8, color=rx.INK)
        left += vals
    ax.set_yticks(range(len(rf_pr)), labels_y, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out trials")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.set_title("EXPLORATORY synthetic probes: what the RF predicts", loc="left")
    figs["probes"] = rx.savefig(fig, FIG / "f07_exploratory_probes_rf.png")

    # Confidence threshold
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, sp in enumerate(("validation", "test")):
        s = thr[thr["split"] == sp]
        ax.plot(s["threshold"], s["coverage"], color=SERIES[i], marker="o", markersize=4,
                label=f"{sp} coverage")
    ax.axvline(chosen_thr, color=rx.INK2, linestyle=":", linewidth=1)
    ax.text(chosen_thr + 0.01, 0.1, f"threshold chosen on\nvalidation = {chosen_thr:.2f}", fontsize=8,
            color=rx.INK2)
    ax.set_xlabel("RF confidence threshold (below -> 'uncertain, please repeat')")
    ax.set_ylabel("Fraction of trials accepted")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("EXPLORATORY: reject option for the RF", loc="left")
    figs["threshold"] = rx.savefig(fig, FIG / "f08_exploratory_confidence_threshold.png")
    return figs


# ============================================================ report
def write_report(m, comp, fl, pert, probe_summary, e2, thr, m1, best_lr, best_dt, sel_a, sel_b, label_of_report):
    ev = m["evaluations"]
    A, B = ev[sel_a], ev[sel_b]
    ds, on = m["double_squeeze"], m["press_onset"]
    ex = m["exploratory"]
    fig = m["figures"]
    audit = pd.read_csv(ROOT / "reports" / "dataset_audit.csv")
    warn = audit[audit["status"].isin(["WARN", "FAIL"])][["check", "observed", "status"]]
    n_pass = int((audit["status"] == "PASS").sum())

    def img(k, alt):
        return f"![{alt}](figures/{fig[k]})" if k in fig else ""

    def f3(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"

    ct = comp[["model", "family", "params", "train_cv5_macro_f1", "val_accuracy", "val_macro_f1",
               "val_log_loss", "val_misclassified", "footprint", "est_bytes", "n_features_used"]].copy()
    for c in ("train_cv5_macro_f1", "val_accuracy", "val_macro_f1", "val_log_loss"):
        ct[c] = ct[c].map(f3)
    ct["params"] = ct["params"].str.replace("|", "/")
    tt = comp[["model", "test_accuracy", "test_macro_f1", "test_errors", "test_misclassified"]].copy()
    tt["test_accuracy"] = tt["test_accuracy"].map(f3)
    tt["test_macro_f1"] = tt["test_macro_f1"].map(f3)
    tt["acc_95ci_wilson"] = comp["model"].map(
        lambda n: "{:.3f}-{:.3f}".format(*ev[n]["test"]["accuracy_95ci_wilson"]))
    tt["macro_f1_bootstrap_95ci"] = comp["model"].map(
        lambda n: "{:.3f}-{:.3f}".format(*ev[n]["test"]["macro_f1_bootstrap_95ci"]))

    def per_class(e):
        return rx.per_class_md(e)

    def recall_ci(e):
        rows = ["| Class | Recall | 95% Wilson CI |", "|---|---|---|"]
        for lab in LABELS:
            lo, hi = e["recall_wilson_95ci"][lab]
            rows.append(f"| {lab} | {e['per_class'][lab]['recall']:.3f} | {lo:.3f}-{hi:.3f} |")
        return "\n".join(rows)

    mcn_rows = ["| Comparison (test, paired) | only first correct | only second correct | exact p |",
                "|---|---|---|---|"]
    for k, v in m["mcnemar_test_vs_selected"].items():
        mcn_rows.append(f"| {k} | {v['only_first_correct']} | {v['only_second_correct']} | {v['p_value_exact']:.3g} |")

    flv = fl[["trial", "label", "split", "pattern", "peak_adc", "zero_fraction", "n_presses",
              "closest_class_median_waveform", "rf_prediction", "prediction_source"]]

    e1 = ex["E1_repeated_cv"]
    e1_rows = ["| Model | Folds | Mean acc | Std | 5th pct | Min |", "|---|---|---|---|---|---|"]
    for k, v in e1.items():
        e1_rows.append(f"| {k} ({v['model']}) | {v['n_folds']} | {v['mean']:.3f} | {v['std']:.3f} | {v['p05']:.3f} | {v['min']:.3f} |")
    e2_rows = ["| Direction | Model | Train / test trials | Accuracy | Macro-F1 | Misclassified |",
               "|---|---|---|---|---|---|"]
    for d, dd in e2.items():
        for k, v in dd.items():
            e2_rows.append(f"| {d} | {k} | {v['train_trials']} / {v['test_trials']} | {v['accuracy']:.3f} "
                           f"| {v['macro_f1']:.3f} | {v['misclassified']} |")
    pv = pert.pivot_table(index=["perturbation", "value"], columns="model", values="accuracy").reset_index()
    pv = pv[["perturbation", "value", "RF", "LR", "DT"]]
    for c in ("RF", "LR", "DT"):
        pv[c] = pv[c].map(f3)
    pr = probe_summary.reset_index()
    pr = pr[pr["model"].isin(["RF", "LR", "DT"])]
    thr_v = thr.pivot_table(index="threshold", columns="split", values=["coverage", "accepted_accuracy"])
    thr_v.columns = [f"{a}_{b}" for a, b in thr_v.columns]
    thr_v = thr_v.reset_index().map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
    e5 = pd.DataFrame(ex["E5_flagged_vs_unflagged"])
    e5["accuracy"] = (e5["sum"] / e5["count"]).map(f3)
    e5 = e5.rename(columns={"sum": "correct", "count": "trials"})
    e4 = ex["E4_confidence_threshold"]

    p1_rf = probe_summary.loc[("P1_two_squeezes_then_rest", "RF", "DOUBLE_SQUEEZE")]
    p2_rf = probe_summary.loc[("P2_single_squeeze_then_rest", "RF", "DOUBLE_SQUEEZE")]
    g = pert[(pert["perturbation"] == "gain")]
    gain_rf = g[g["model"] == "RF"].set_index("value")["accuracy"]
    off_rf = pert[(pert["perturbation"] == "floor_offset") & (pert["model"] == "RF")].set_index("value")["accuracy"]
    off_dt = pert[(pert["perturbation"] == "floor_offset") & (pert["model"] == "DT")].set_index("value")["accuracy"]
    rfB = next(r for r in m["comparison"] if r["model"] == sel_b)
    rfA = next(r for r in m["comparison"] if r["model"] == sel_a)
    lrr = next(r for r in m["comparison"] if r["model"] == best_lr)
    dt_rules = (MDL / f"{best_dt}_rules.txt").read_text()
    dt_feats = rfB["features_used"]
    p3_rf = probe_summary.xs(("P3_press_with_onset_and_release", "RF"), level=("probe", "model"))
    p3_dt = probe_summary.xs(("P3_press_with_onset_and_release", "DT"), level=("probe", "model"))
    p3_lr = probe_summary.xs(("P3_press_with_onset_and_release", "LR"), level=("probe", "model"))
    p3_rf_ds = int(p3_rf["DOUBLE_SQUEEZE"].sum())
    p3_n = int(p3_rf.to_numpy().sum())
    p2_dt = probe_summary.loc[("P2_single_squeeze_then_rest", "DT", "DOUBLE_SQUEEZE")]
    p1_n = int(p1_rf.sum())
    trained = [r["model"] for r in m["comparison"] if r["family"] != "Baseline"]
    val_err_sets = {n: tuple(ev[n]["validation"]["misclassified_trials"]) for n in trained}
    same_val = len(set(val_err_sets.values())) == 1
    val_sentence = (f"Every trained candidate makes the same validation error(s): trials {list(val_err_sets[trained[0]])}."
                    if same_val else
                    "Validation errors by model: " + "; ".join(f"{n}: {list(v)}" for n, v in val_err_sets.items()) + ".")
    flagged_set = set(fl["trial"])
    err_rows = []
    for n in trained:
        for sp in ("validation", "test"):
            for t in ev[n][sp]["misclassified_trials"]:
                err_rows.append((n, sp, t, label_of_report[t], t in flagged_set))
    err_df = pd.DataFrame(err_rows, columns=["model", "split", "trial", "actual", "audit_flagged"])
    err_trials = err_df.drop_duplicates("trial")
    n_err_flag = int(err_trials["audit_flagged"].sum())
    err_summary = ", ".join(f"#{r.trial} {r.actual}{' (flagged)' if r.audit_flagged else ' (not flagged)'}"
                            for r in err_trials.sort_values("trial").itertuples())
    sig = {k: v["p_value_exact"] < 0.05 for k, v in m["mcnemar_test_vs_selected"].items()}
    sig_models = [k.split(" vs ")[1] for k, v in sig.items() if v]
    nonsig_trained = [k.split(" vs ")[1] for k, v in sig.items() if not v and "B" != k.split(" vs ")[1][0]]
    rft_ = pd.read_csv(RES / "predictions_all_models.csv").query("model=='RF_existing' and split=='test'")
    acc_err = rft_[(rft_["confidence"] >= e4["chosen_on_validation"]) & (~rft_["correct"])]
    rej_txt = ", ".join(f"#{r.trial} {r.actual}->{r.predicted} (confidence {r.confidence:.2f})"
                        for r in acc_err.itertuples()) or "none"
    rho_p, rho_pp = ds["spearman_trialid_vs_peak"]
    rho_n, rho_np = ds["spearman_trialid_vs_presses"]
    trend_txt = (f"Peak squeeze force {'rises' if rho_p > 0 else 'falls'} over the recording block "
                 f"(Spearman rho {rho_p:+.2f}, p = {rho_pp:.3g}{'' if rho_pp < 0.05 else ', not significant'}); "
                 f"press count shows {'a significant' if rho_np < 0.05 else 'no significant'} trend "
                 f"(rho {rho_n:+.2f}, p = {rho_np:.3g}).")

    rep = f"""# GripSense TinyML - Final Report (phase 2 follow-up)

_Generated {m['generated_utc']} by `run_followup.py` (seed {m['seed']}). All numbers are read from
`reports/followup/followup_metrics.json`, `results/followup/*.csv` and the phase-1 artifacts, and were
cross-checked by `validate_followup.py`. Phase-1 outputs (`reports/`, `results/`, `models/` top level) are
unchanged; their SHA-256 hashes were snapshotted before this work and re-verified afterwards._

> **No new data was collected and no hardware testing was performed.** Everything below comes from the
> existing `gripsense_clean_data.csv` (SHA-256 `{m['phase1_reference']['input_sha256'][:16]}...`), offline. The ESP32 was
> not connected. Exploratory analyses (Section 8) reuse or synthetically modify the same recordings
> and are **not** independent or real-world validation.

## Executive summary

- **Dataset (audited before training):** 30,000 readings, 200 trials x 150 samples at 20 ms, 50 trials per
  class, structurally clean ({n_pass} PASS checks). It has protocol limitations: classes were recorded
  in contiguous blocks; **no DOUBLE_SQUEEZE trial contains a double squeeze** ({ds['presses_distribution']}
  presses: trials); and {on['LIGHT_PRESS']['active_at_start']}/50 LIGHT_PRESS and {on['HARD_PRESS']['active_at_start']}/50 HARD_PRESS windows start mid-press.
- **Model comparison (same phase-1 split, 120/40/40 trials):** {m['selection']['n_tied_best_val']} of {m['selection']['n_eligible']} trained candidates tie at validation macro-F1
  {m['selection']['best_val_macro_f1']:.3f}. Baselines: majority {ev['B0_majority']['validation']['accuracy']:.3f}, mean-ADC centroid {ev['B1_mean_adc_centroid']['validation']['accuracy']:.3f} validation accuracy.
- **Rule A (primary, unchanged from phase 1) selects `{sel_a}`.** Test accuracy **{A['test']['accuracy']:.3f}**
  ({A['test']['n_correct']}/40; 95% CI {A['test']['accuracy_95ci_wilson'][0]:.3f}-{A['test']['accuracy_95ci_wilson'][1]:.3f}), macro-F1 **{A['test']['macro_avg']['f1']:.3f}**, misclassified {A['test']['misclassified_trials']}.
- **Rule B (ESP32 size) selects `{sel_b}`** ({rfB['footprint']}, ~{rfB['est_bytes']} bytes, {rfB['n_features_used']} features vs RF ~{rfA['est_bytes']:,} bytes).
  Test accuracy **{B['test']['accuracy']:.3f}** ({B['test']['n_correct']}/40), macro-F1 **{B['test']['macro_avg']['f1']:.3f}**, misclassified {B['test']['misclassified_trials']}.
- **Uncertainty dominates.** With 10 test trials per class, a perfect 10/10 recall is consistent with a true recall as low as
  {m['wilson_lower_bound_10_of_10']:.2f} (95% Wilson). On the paired test set, no trained model differs significantly from the RF; only the two
  baselines are significantly worse (exact McNemar, p < 0.05).
- **Test-set history:** the phase-1 RF test result had already been inspected, so the test set is **not completely
  untouched** (Section 5.1).
- **Biggest practical risk (exploratory, synthetic probes):** the models are sensitive to how a press sits in the window.
  When LIGHT/HARD presses are edited to include a realistic onset and release (rest at both ends), the RF calls
  **{p3_rf_ds}/{p3_n}** of them DOUBLE_SQUEEZE. A *single* squeeze is also called DOUBLE_SQUEEZE ("Washroom") {int(p2_rf['DOUBLE_SQUEEZE'])}/{int(p2_rf.sum())} times
  (tree: {int(p2_dt['DOUBLE_SQUEEZE'])}/{int(p2_dt.sum())}). "Two squeezes then rest" is recognised {int(p1_rf['DOUBLE_SQUEEZE'])}/{p1_n} times.

## 1. Objectives and scope

1. Re-use the phase-1 data, split, features and model, without changing trial assignments.
2. Compare the existing Random Forest with baselines, logistic regression and shallow decision trees, using train for
   fitting and validation for selection, and then evaluate on the held-out test set.
3. Quantify uncertainty, analyse errors, the 20 flagged trials and the DOUBLE_SQUEEZE protocol mismatch.
4. Run clearly labelled exploratory robustness analyses on the existing data.
5. Give practical ESP32 deployment recommendations, with no new data or hardware.

## 2. Dataset and quality audit (performed BEFORE training, phase 1)

The phase-1 audit (`reports/dataset_audit.csv`, unchanged) ran before any split or model fitting.
Integrity checks all passed: no missing values or duplicates, one label per trial, exactly 150 samples per
trial at 20 ms, ADC 0-2608 (no saturation), and 50 trials per class. Non-PASS items:

{rx.md_table(warn)}

Flagged trials (kept, not relabelled): {m['flagged']['n']} ({m['flagged']['by_label']}). Phase-1 report: `reports/GripSense_TinyML_Report.md`.

## 3. Split (unchanged from phase 1)

`reports/split_manifest.csv` was re-used without modification: train {m['split_counts']['train']}, validation
{m['split_counts']['validation']}, test {m['split_counts']['test']} trials (30/10/10 per class; 18,000/6,000/6,000 readings), seed 42, stratified
by label at trial level. There is no trial overlap. All new models were fitted on the 120 training trials only
(trial IDs stored in `models/followup/followup_models_metadata.json`). The features were recomputed from the CSV and match
`reports/trial_features.csv` exactly. Trial ID is not a feature.

## 4. Existing baselines and model comparison (validation-based selection)

All candidates are fitted on train and scored on validation. `RF_existing` is the saved phase-1 model, used as-is; it
reproduces its phase-1 validation and test predictions exactly. Model sizes are **rough estimates** for flattened C
arrays (12 bytes per tree node, 4 bytes per float), not measurements on an ESP32.

{rx.md_table(ct)}

**Pre-declared selection rules** (frozen in `results/followup/selection_freeze.json` before any test prediction in this phase):
- **Rule A (primary):** max validation macro-F1 -> max train 5-fold CV macro-F1 -> min validation log-loss -> simpler
  family. This is identical to phase 1. -> **`{sel_a}`**
- **Rule B (ESP32 deployment):** among models tied at the best validation macro-F1, the smallest estimated size -> train
  CV -> log-loss. -> **`{sel_b}`**
- Disclosure: Rule B was written after the phase-1 validation table (all candidates tied) had been seen, but before any
  test prediction of the logistic-regression or decision-tree candidates.

{val_sentence} Validation therefore cannot rank the trained models. It only shows that they all clearly beat both
baselines.

{img('model_comparison', 'Model comparison')}

## 5. Final held-out test results

### 5.1 Test-set history (disclosure)

{m['test_history_disclosure']}

### 5.2 Rule-A model: `{sel_a}` (primary)

**Accuracy {A['test']['accuracy']:.3f}** ({A['test']['n_correct']}/{A['test']['n_trials']}; Wilson 95% CI {A['test']['accuracy_95ci_wilson'][0]:.3f}-{A['test']['accuracy_95ci_wilson'][1]:.3f});
macro-F1 {A['test']['macro_avg']['f1']:.3f} (bootstrap 95% CI {A['test']['macro_f1_bootstrap_95ci'][0]:.3f}-{A['test']['macro_f1_bootstrap_95ci'][1]:.3f}).

{per_class(A['test'])}

{rx.cm_md(A['test']['confusion_matrix']['matrix'])}

Misclassified test trials: {A['test']['misclassified_trials']}. Validation: {A['validation']['misclassified_trials']}.

{img('cm_test_selected_A', 'Confusion matrix A')}

### 5.3 Rule-B model: `{sel_b}` (ESP32 deployment candidate)

**Accuracy {B['test']['accuracy']:.3f}** ({B['test']['n_correct']}/{B['test']['n_trials']}; Wilson 95% CI {B['test']['accuracy_95ci_wilson'][0]:.3f}-{B['test']['accuracy_95ci_wilson'][1]:.3f});
macro-F1 {B['test']['macro_avg']['f1']:.3f} (bootstrap 95% CI {B['test']['macro_f1_bootstrap_95ci'][0]:.3f}-{B['test']['macro_f1_bootstrap_95ci'][1]:.3f}).

{per_class(B['test'])}

{rx.cm_md(B['test']['confusion_matrix']['matrix'])}

Misclassified test trials: {B['test']['misclassified_trials']}. Validation: {B['validation']['misclassified_trials']}.

{img('cm_test_deployment_B', 'Confusion matrix B')}

Decision rules of `{best_dt}` (the whole model):

```
{dt_rules}```

### 5.4 All candidates on test (secondary; NOT used for selection)

{rx.md_table(tt)}

## 6. Error analysis and uncertainty

**Recall uncertainty for `{sel_a}` (test, 10 trials per class):**

{recall_ci(A['test'])}

- With n = 10, 10/10 correct gives a 95% lower bound of {m['wilson_lower_bound_10_of_10']:.3f}, and 9/10 gives {m['wilson_lower_bound_9_of_10']:.3f}. One trial moves
  per-class recall by 0.10 and overall accuracy by 0.025.
- Paired comparisons against `{sel_a}` on the same 40 test trials:

{chr(10).join(mcn_rows)}

  Significantly different from `{sel_a}` (p < 0.05): {', '.join(sig_models) or 'none'}. Not significantly different:
  {', '.join(nonsig_trained) or 'none'}. With 40 paired trials, differences of 1-2 trials cannot be distinguished from chance.
- **Where errors fall.** Distinct trials misclassified by any trained model (validation or test): {err_summary}.
  {n_err_flag} of {len(err_trials)} were flagged by the pre-training audit. The test error shared by every trained model is #39,
  an atypically variable LIGHT_PRESS predicted as HARD_PRESS. For this application that is a **false "Pain"** signal.
- Held-out accuracy split by audit flag (exploratory, val+test pooled):

{rx.md_table(e5)}

## 7. Dataset protocol limitations

### 7.1 DOUBLE_SQUEEZE does not match the intended gesture

The intended "Washroom" gesture is two squeezes followed by rest. The recordings show something else:

| Property (50 DOUBLE_SQUEEZE trials) | Measured |
|---|---|
| Presses per 3 s window (> 100 ADC) | {ds['presses_distribution']} (presses: trials) |
| Trials with exactly 2 presses | **{ds['trials_with_exactly_2_presses']}** |
| Median press duration / gap | {ds['median_press_ms']:.0f} ms / {ds['median_gap_ms']:.0f} ms |
| Median press rate | {ds['presses_per_s_median']:.2f} presses per second |
| Rest after the 2nd press (median / max) | {ds['rest_after_2nd_press_ms_median']:.0f} ms / {ds['rest_after_2nd_press_ms_max']:.0f} ms |
| Trials with >= 1 s rest after the 2nd press | {ds['trials_with_>=1000ms_rest_after_2nd_press']} |
| Window starts / ends mid-press | {ds['active_at_window_start']} / {ds['active_at_window_end']} trials |
| Peak ADC vs recording order (Spearman rho, p) | {ds['spearman_trialid_vs_peak'][0]:+.2f}, p = {ds['spearman_trialid_vs_peak'][1]:.3g} |
| Press count vs recording order (Spearman rho, p) | {ds['spearman_trialid_vs_presses'][0]:+.2f}, p = {ds['spearman_trialid_vs_presses'][1]:.3g} |

The recordings are continuous rhythmic squeezing (~{ds['presses_per_s_median']:.1f} Hz) that fills the window and is
often cut off at the window edges. They are not one discrete double squeeze. {trend_txt} The label is internally consistent, so I did **not** relabel or remove anything.
The class should be read as "repeated squeezing". A real double squeeze is not represented in the training data. Probe P1 (Section 8) suggests the models still
mostly accept it, but probe P2 shows they also accept a single squeeze, so the class is not specific to "two".

{img('double_squeeze', 'DOUBLE_SQUEEZE anatomy')}

### 7.2 LIGHT_PRESS / HARD_PRESS windows start mid-press

{on['LIGHT_PRESS']['active_at_start']}/50 LIGHT_PRESS and {on['HARD_PRESS']['active_at_start']}/50 HARD_PRESS trials are already above 100 ADC at t = 0, and
{on['LIGHT_PRESS']['active_at_end']}/50 and {on['HARD_PRESS']['active_at_end']}/50 are still pressed at the end. The windows capture a steady hold, not the
approach, onset and release of a real press. On a device that triggers on onset, the window will contain rest + ramp-up,
which differs from the training data (probe P3).

### 7.3 Other limitations

- **Recording-order confound:** each class is one contiguous block of trials, so session drift is aligned with the label.
  A random trial split cannot separate gesture from session (exploratory E2 gives a partial view).
- **Probably one participant, one sensor placement, one session per class.** No cross-user or cross-day data exists.
- **NO_PRESS is almost entirely exact zeros.** Zero/contact features are among the top RF importances. In E3a, constant
  floor offsets of +10/+30 ADC (below the 100-ADC "pressed" threshold) changed accuracy only slightly. Offsets near or above
  100 ADC, or a drifting floor, were not tested.
- **Nominal timestamps** (always 20 ms), so sampling jitter is unknown.

### 7.4 The 20 flagged trials (inspected, kept)

{rx.md_table(flv)}

RF correct on {m['flagged']['rf_correct']}/{m['flagged']['n']} flagged trials (held-out predictions for val/test; 5-fold
out-of-fold predictions within the training set for train trials; out-of-fold accuracy on all 120 training trials:
{m['flagged']['train_oof_accuracy_all_train']:.3f}). "Closest class median waveform" is the class whose median
waveform has the smallest mean squared distance to the trial. It is a hint, not a relabelling.

{img('flagged_all', 'Flagged trials')}

## 8. EXPLORATORY robustness analyses (not independent validation)

These analyses re-use the same 200 recordings (from one participant/session context) and, for E3, synthetically modify
them. They help find weaknesses. They do **not** measure real-world accuracy. E1 and E2 pool test trials into
training folds and were run after the final test evaluation, so they had no influence on selection.

**E1 - Repeated 5-fold CV on all 200 trials (20 repeats; fixed hyperparameters).**

{chr(10).join(e1_rows)}

{img('repeated_cv', 'Repeated CV')}

**E2 - Chronological within-class split (drift proxy).** Train on the earliest 30 trials of each class and test on the
latest 20, and the reverse.

{chr(10).join(e2_rows)}

Accuracy stays within a few trials of the random split in both directions, so drift *within* a class block has little
effect on these features. This cannot test the more important *between-class* confound (each class is its own block).

**E3a - Signal perturbations on the 80 held-out (val+test) trials, using models trained on the training split.** Gain =
multiply ADC (sensitivity/placement change); noise only while pressed; floor offset = add a constant to every reading
(sensor no longer reads exactly 0 at rest).

{rx.md_table(pv)}

RF accuracy at gain 0.7 / 1.3: {gain_rf.loc[0.7]:.3f} / {gain_rf.loc[1.3]:.3f}. With a floor offset of +10 / +30 ADC:
RF {off_rf.loc[10]:.3f} / {off_rf.loc[30]:.3f}, DT {off_dt.loc[10]:.3f} / {off_dt.loc[30]:.3f}.

{img('gain', 'Gain sensitivity')}

**E3b - Synthetic gesture probes (held-out trials).** P1: DOUBLE_SQUEEZE trials with everything after the 2nd press set to
0, approximating the intended gesture. P2: the same with only the 1st press kept. P3: LIGHT/HARD trials with the first and
last 500 ms set to 0, approximating onset and release. These are crude edits of real signals, not new recordings.

{rx.md_table(pr)}

Interpretation (RF unless stated):
- P1 "two squeezes then rest" -> DOUBLE_SQUEEZE {int(p1_rf['DOUBLE_SQUEEZE'])}/{p1_n}.
- P2 single squeeze -> {', '.join(f"{k} x{int(v)}" for k, v in p2_rf.items() if v)}. There is no "single squeeze" class, so roughly half
  are called "Washroom".
- P3 LIGHT/HARD with onset and release -> RF: {', '.join(f"{k} x{int(v)}" for k, v in p3_rf.sum().items() if v)}; LR: {', '.join(f"{k} x{int(v)}" for k, v in p3_lr.sum().items() if v)};
  DT: {', '.join(f"{k} x{int(v)}" for k, v in p3_dt.sum().items() if v)}. Any rest inside the window looks like the gaps of
  DOUBLE_SQUEEZE to the RF/LR. The tree instead misreads HARD as LIGHT, because its force split uses `segment_1_mean`
  (300-600 ms), which this probe sets to rest. P3 is a crude edit (instant steps to 0), so the size of the effect is uncertain,
  but its direction is a clear warning about window alignment.

{img('probes', 'Probes')}

**E4 - Reject option for the RF (threshold chosen on validation).** Smallest threshold with zero accepted validation
errors: **{e4['chosen_on_validation']:.2f}** (validation coverage {e4['val_coverage']:.3f}). On test it accepts
{e4['test_coverage']:.3f} of trials with accepted accuracy {e4['test_accepted_accuracy']:.3f} ({e4['test_accepted_errors']} accepted errors).
Accepted test errors at this threshold: {rej_txt}. A confidence threshold alone does not reliably catch the false-"Pain" case.

{rx.md_table(thr_v)}

{img('threshold', 'Confidence threshold')}

## 9. Practical ESP32 deployment recommendations

1. **Model choice.** On this data the {best_dt} (~{rfB['est_bytes']} bytes, {rfB['n_features_used']} features) and
   {best_lr} (~{lrr['est_bytes']:,} bytes, all 49 features) match the RF on validation. The 300-tree RF (~{rfA['est_bytes']:,} bytes
   estimated) is far larger for no measured gain. The tree uses only {dt_feats}. Its LIGHT/HARD decision depends
   on the mean force in a fixed 300-600 ms slice of the window, so it is only valid if device windows are aligned like
   the training windows (probe P3). Logistic regression gives smoother confidence scores but needs all 49 features.
2. **Feature parity.** Port only the features the chosen model uses, and unit-test the C implementation against
   `reports/trial_features.csv` (same 150-sample inputs, outputs within a tolerance) before trusting on-device results.
3. **Buffering and timing.** 150 x uint16 = 300 bytes per window at 50 Hz (20 ms timer-driven sampling, not `delay()`).
   Classify on a sliding window (e.g. every 0.5 s), or after an onset trigger plus 3 s.
4. **Onset/segmentation (most important).** Training windows start mid-press (Section 7.2). Probe P3 shows that adding
   rest at the window edges flips the RF's LIGHT/HARD predictions to DOUBLE_SQUEEZE. On the device, classify only windows that
   resemble training conditions (e.g. start classifying once force has been above threshold for ~0.5 s), and require
   two consecutive identical predictions before acting.
5. **Reject option.** Output "uncertain - please repeat" below a confidence threshold. On this data the RF threshold
   chosen on validation was {e4['chosen_on_validation']:.2f}. Re-tune it on device data before relying on it.
6. **Safety for "Pain".** The observed failure mode is LIGHT -> HARD (false Pain). Use an explicit confirmation (LED/
   vibration + repeat or hold) before sending an alert, and never make the device the only channel for pain reporting.
7. **DOUBLE_SQUEEZE.** The model's "Washroom" class means "intermittent squeezing", and it also fires on a single squeeze
   (probe P2). Either instruct users to "squeeze repeatedly" (matching the data), or decide this class with an explicit
   rule-based press counter (e.g. exactly two presses within ~1.5 s, then rest) instead of the classifier.
8. **Calibration.** Force-level classes shift with gain (E3a). Fix the ADC attenuation/width (e.g. 11 dB, 12-bit) and the
   divider resistor exactly as used during recording, and consider a per-user calibration step that scales readings.

## 10. Conclusion

On the existing split, all trained models (RF, logistic regression, depth-2/3 trees) reach the same validation score
({m['selection']['best_val_macro_f1']:.3f} macro-F1) and clearly beat the baselines. On the held-out test set, the rule-A RF scores
{A['test']['accuracy']:.3f} and the rule-B tree {B['test']['accuracy']:.3f}. The test set was already partly inspected in
phase 1, and with 10 trials per class these numbers carry wide uncertainty. The main obstacles to deployment are not model
accuracy but **data protocol**: DOUBLE_SQUEEZE is "repeated squeezing", presses are captured mid-hold, classes are confounded
with recording session, there is probably one user, and synthetic probes show strong sensitivity to window alignment. These results show that the classes are separable in this recording.
They do not show real-world reliability.

## 11. Recommended next software-only step

Port `gripsense_features.py` (only the features `{best_dt}` uses, plus the press counter) and the {best_dt} rules to
plain C. Then build a host-side parity test, compiled with the Mac's `cc`, that feeds all 200 recorded trials from the CSV
through the C code and checks that the features and predictions exactly match `reports/trial_features.csv` and the Python
model. This needs no hardware and de-risks the eventual ESP32 port.

## 12. Reproducibility

```bash
cd "/Users/zenith/Desktop/GripSense-TinyML "
.venv/bin/python run_followup.py
.venv/bin/python validate_followup.py
```

Seed {m['seed']}; Python {m['environment']['python']}, numpy {m['environment']['numpy']}, pandas {m['environment']['pandas']},
scikit-learn {m['environment']['scikit-learn']}, scipy {m['environment']['scipy']}, matplotlib {m['environment']['matplotlib']}.
Selection-freeze SHA-256: `{m['selection']['freeze_sha256']}`.

Artifacts: `reports/followup/` (this report, metrics, figures, validation checks), `results/followup/` (comparison,
predictions for every model/split, freeze file, protocol and exploratory tables), `models/followup/` (fitted
comparison models, tree rules, metadata).
"""
    (REP / "GripSense_TinyML_Final_Report.md").write_text(rep)


if __name__ == "__main__":
    main()
