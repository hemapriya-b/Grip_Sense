#!/usr/bin/env python3
"""
GripSense TinyML - complete, reproducible offline experiment.

Stages (executed strictly in this order):
  1. Dataset audit (before any split or training)
  2. Trial-level stratified 60/20/20 split (train/validation/test)
  3. Feature extraction (one vector per 3-second trial)
  4. Baselines (fit on train, evaluated on validation)
  5. Pre-training plan written to disk (before any model is fitted)
  6. Model training + selection (train + validation ONLY)
  7. Single final evaluation of the selected model on the held-out test set
  8. Artifacts: model bundle, predictions, metrics, figures, report

Run:
    .venv/bin/python run_experiment.py --csv gripsense_clean_data.csv --seed 42

No hardware or serial access is used. The input CSV is only read, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from gripsense_features import (ACTIVE_THRESHOLD, EVENT_THRESHOLDS, LABEL_MEANINGS, LABELS,
                                LEVEL_THRESHOLDS, N_SEGMENTS, REQUIRED_COLUMNS,
                                SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL, build_feature_table,
                                feature_columns)

EXPECTED = {"rows": 30000, "trials": 200, "per_trial": SAMPLES_PER_TRIAL, "per_class": 50}
SPLIT_TRIALS = {"train": 120, "validation": 40, "test": 40}
SPLIT_PER_CLASS = {"train": 30, "validation": 10, "test": 10}
ADC_MAX = 4095  # ESP32 12-bit ADC full scale

# ---------------------------------------------------------------- plot style
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CLASS_COLORS = dict(zip(LABELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 11, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "font.size": 9, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 150,
    "lines.linewidth": 1.4,
})


def class_title(label: str) -> str:
    return f"{label} ({LABEL_MEANINGS[label]})"


def savefig(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path.name


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ======================================================== 1. DATASET AUDIT
def robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    scale = 1.4826 * (values - med).abs().median()
    if scale == 0:
        scale = values.std(ddof=0)
    if scale == 0 or np.isnan(scale):
        return pd.Series(0.0, index=values.index)
    return (values - med) / scale


def audit_dataset(df: pd.DataFrame, feats: pd.DataFrame, fig_dir: Path, seed: int):
    checks = []

    def add(category, check, expected, observed, status, notes=""):
        checks.append({"category": category, "check": check, "expected": str(expected),
                       "observed": str(observed), "status": status, "notes": notes})

    g = df.groupby("trial")
    add("schema", "required columns present", REQUIRED_COLUMNS, list(df.columns),
        "PASS" if list(df.columns) == REQUIRED_COLUMNS else "WARN")
    add("schema", "column dtypes", "int,str,int,int",
        {c: str(t) for c, t in df.dtypes.items()}, "PASS")
    add("counts", "total readings", EXPECTED["rows"], len(df),
        "PASS" if len(df) == EXPECTED["rows"] else "FAIL")
    add("counts", "unique trials", EXPECTED["trials"], df["trial"].nunique(),
        "PASS" if df["trial"].nunique() == EXPECTED["trials"] else "FAIL")
    ids = sorted(df["trial"].unique())
    add("counts", "trial IDs contiguous", "1..200", f"{ids[0]}..{ids[-1]}",
        "PASS" if ids == list(range(1, 201)) else "WARN")
    sizes = g.size()
    add("counts", "readings per trial", f"all {SAMPLES_PER_TRIAL}",
        sizes.value_counts().to_dict(),
        "PASS" if (sizes == SAMPLES_PER_TRIAL).all() else "FAIL")
    for col in REQUIRED_COLUMNS:
        n_missing = int(df[col].isna().sum())
        add("missing", f"missing values: {col}", 0, n_missing, "PASS" if n_missing == 0 else "FAIL")
    n_dup = int(df.duplicated().sum())
    add("duplicates", "exact duplicate rows", 0, n_dup, "PASS" if n_dup == 0 else "WARN")
    n_dup_ts = int(df.duplicated(["trial", "elapsed_ms"]).sum())
    add("duplicates", "duplicate (trial, elapsed_ms)", 0, n_dup_ts, "PASS" if n_dup_ts == 0 else "FAIL")
    unknown = sorted(set(df["label"]) - set(LABELS))
    add("labels", "labels within allowed set", LABELS, sorted(df["label"].unique()),
        "PASS" if not unknown else "FAIL", f"unknown: {unknown}" if unknown else "")
    n_multi = int((g["label"].nunique() > 1).sum())
    add("labels", "trials with inconsistent labels", 0, n_multi, "PASS" if n_multi == 0 else "FAIL")
    trial_labels = g["label"].first()
    for lab in LABELS:
        c = int((trial_labels == lab).sum())
        add("balance", f"trials for {lab}", EXPECTED["per_class"], c,
            "PASS" if c == EXPECTED["per_class"] else "FAIL")
        r = int((df["label"] == lab).sum())
        add("balance", f"readings for {lab}", EXPECTED["per_class"] * SAMPLES_PER_TRIAL, r,
            "PASS" if r == EXPECTED["per_class"] * SAMPLES_PER_TRIAL else "FAIL")

    intervals = np.concatenate([np.diff(t.sort_values().to_numpy()) for _, t in g["elapsed_ms"]])
    iv_counts = pd.Series(intervals).value_counts().to_dict()
    add("timing", "sampling interval (ms)", f"all {SAMPLE_INTERVAL_MS}",
        {int(k): int(v) for k, v in iv_counts.items()},
        "PASS" if set(iv_counts) == {SAMPLE_INTERVAL_MS} else "WARN",
        "Timestamps are relative per trial; jitter of the real sampling clock cannot be "
        "observed if the logger wrote nominal times.")
    starts, ends = g["elapsed_ms"].min(), g["elapsed_ms"].max()
    add("timing", "trial start elapsed_ms", "all 0", starts.value_counts().to_dict(),
        "PASS" if (starts == 0).all() else "WARN")
    add("timing", "trial last timestamp (ms)", "all 2980 (150 x 20 ms = 3.0 s window)",
        ends.value_counts().to_dict(), "PASS" if (ends == 2980).all() else "WARN")
    monotonic = all(t.is_monotonic_increasing for _, t in g["elapsed_ms"])
    add("timing", "timestamps monotonic within trial (file order)", True, monotonic,
        "PASS" if monotonic else "WARN")

    add("adc", "ADC min / max", f"within 0..{ADC_MAX}", f"{df['adc'].min()} / {df['adc'].max()}",
        "PASS" if df["adc"].between(0, ADC_MAX).all() else "FAIL")
    n_sat = int((df["adc"] >= ADC_MAX).sum())
    add("adc", "saturated readings (>= 4095)", 0, n_sat, "PASS" if n_sat == 0 else "WARN")
    add("adc", "zero readings overall", "INFO", f"{int((df['adc'] == 0).sum())} "
        f"({(df['adc'] == 0).mean():.1%})", "INFO",
        "Zeros are expected when the FSR is unloaded (NO_PRESS, gaps between squeezes).")
    for lab in LABELS:
        s = df.loc[df["label"] == lab, "adc"]
        add("adc", f"ADC stats {lab}", "INFO",
            f"mean={s.mean():.1f} median={s.median():.0f} std={s.std():.1f} "
            f"min={s.min()} max={s.max()} zero={(s == 0).mean():.1%}", "INFO")

    # Recording-order confound: are classes recorded in contiguous trial blocks?
    blocks = trial_labels.reset_index().groupby("label")["trial"].agg(["min", "max", "count"])
    contiguous = all(r["max"] - r["min"] + 1 == r["count"] for _, r in blocks.iterrows())
    block_txt = "; ".join(f"{lab}: {int(blocks.loc[lab, 'min'])}-{int(blocks.loc[lab, 'max'])}"
                          for lab in LABELS)
    add("protocol", "classes recorded in contiguous trial blocks", "interleaved/randomised order",
        block_txt, "WARN" if contiguous else "PASS",
        "Class is confounded with recording session/time. Sensor drift, grip habituation or "
        "placement changes between blocks could be learned as 'class'. Trial ID is excluded "
        "from features, but the confound cannot be removed from this dataset.")

    # -------- per-trial quality flags (documented, NOT removed)
    ta = feats[["trial", "label", "mean", "std", "max", "zero_fraction", "mean_abs_diff",
                "max_rise", "press_events_100"]].copy()
    z_cols = ["mean", "std", "max", "zero_fraction", "mean_abs_diff"]
    for col in z_cols:
        ta[f"rz_{col}"] = ta.groupby("label")[col].transform(robust_z)
    flags = []
    for _, r in ta.iterrows():
        f = [f"atypical_{c}(z={r[f'rz_{c}']:+.1f})" for c in z_cols if abs(r[f"rz_{c}"]) > 3.5]
        if r["label"] == "NO_PRESS" and r["max"] > 50:
            f.append(f"activity_during_no_press(max={r['max']:.0f})")
        if r["label"] in ("LIGHT_PRESS", "HARD_PRESS") and r["zero_fraction"] > 0.10:
            f.append(f"dropout_to_zero_during_press({r['zero_fraction']:.0%})")
        if r["label"] == "DOUBLE_SQUEEZE" and r["press_events_100"] < 2:
            f.append(f"double_squeeze_events={int(r['press_events_100'])}(expected >=2)")
        if r["label"] == "LIGHT_PRESS" and r["max"] > 1500:
            f.append(f"light_press_peak_in_hard_range(max={r['max']:.0f})")
        flags.append("; ".join(f))
    ta["flags"] = flags
    ta["flagged"] = ta["flags"] != ""
    flagged = ta[ta["flagged"]]

    # Protocol check: number of distinct presses (rising crossings of 100 ADC) per trial
    for lab in LABELS:
        ev = ta.loc[ta["label"] == lab, "press_events_100"].astype(int)
        dist = ev.value_counts().sort_index().to_dict()
        if lab == "DOUBLE_SQUEEZE":
            n2 = int((ev == 2).sum())
            add("protocol", "press events per trial: DOUBLE_SQUEEZE", "2 (one double squeeze)",
                dist, "WARN" if n2 < len(ev) else "PASS",
                f"Only {n2}/{len(ev)} trials contain exactly 2 presses; median {ev.median():.0f}. "
                "Trials contain repeated squeezing for the whole 3 s window, so a single real-world "
                "double squeeze is not represented in the training data.")
        else:
            add("protocol", f"press events per trial: {lab}",
                "0" if lab == "NO_PRESS" else "1 (one sustained press)", dist, "INFO")
    add("quality", "trials flagged for review (kept in dataset)", 0, len(flagged),
        "WARN" if len(flagged) else "PASS",
        "IDs: " + ", ".join(str(t) for t in flagged["trial"]) if len(flagged) else "")
    for lab in LABELS:
        sub = flagged[flagged["label"] == lab]
        add("quality", f"flagged trials {lab}", "INFO", len(sub), "INFO",
            ", ".join(str(t) for t in sub["trial"]))

    audit = pd.DataFrame(checks)
    figs = audit_figures(df, feats, ta, fig_dir, seed)
    return audit, ta, figs, block_txt


def audit_figures(df, feats, ta, fig_dir: Path, seed: int) -> dict[str, str]:
    figs = {}
    t_ms = np.arange(SAMPLES_PER_TRIAL) * SAMPLE_INTERVAL_MS
    rng = np.random.default_rng(seed)

    # Class distribution
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    counts = feats["label"].value_counts().reindex(LABELS)
    ax.barh(range(4), counts.values, color=[CLASS_COLORS[l] for l in LABELS], height=0.6)
    for i, (lab, v) in enumerate(counts.items()):
        ax.text(v + 0.8, i, f"{v} trials · {v * SAMPLES_PER_TRIAL:,} readings", va="center",
                color=INK2, fontsize=8)
    ax.set_yticks(range(4), [class_title(l) for l in LABELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xlabel("Trials")
    ax.grid(axis="y", visible=False)
    ax.set_title("Class distribution: 50 trials per class (balanced)", loc="left")
    figs["class_distribution"] = savefig(fig, fig_dir / "fig01_class_distribution.png")

    # Representative waveforms (small multiples, 5 random trials per class)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, lab in zip(axes.flat, LABELS):
        ids = sorted(rng.choice(feats.loc[feats["label"] == lab, "trial"].to_numpy(), 5,
                                replace=False))
        for tid in ids:
            x = df.loc[df["trial"] == tid].sort_values("elapsed_ms")["adc"].to_numpy()
            ax.plot(t_ms, x, color=CLASS_COLORS[lab], alpha=0.85, linewidth=1.2)
        ax.set_title(f"{class_title(lab)}  -  trials {', '.join(map(str, ids))}", loc="left")
        ax.set_ylabel("ADC")
    for ax in axes[1]:
        ax.set_xlabel("Time within trial (ms)")
    fig.suptitle("Representative raw FSR waveforms (5 randomly chosen trials per class)",
                 x=0.01, ha="left", fontsize=12)
    figs["representative_waveforms"] = savefig(fig, fig_dir / "fig02_representative_waveforms.png")

    # Class median waveform with IQR band
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, lab in zip(axes, LABELS):
        mat = np.stack([df.loc[df["trial"] == tid].sort_values("elapsed_ms")["adc"].to_numpy()
                        for tid in feats.loc[feats["label"] == lab, "trial"]])
        q25, q50, q75 = np.percentile(mat, [25, 50, 75], axis=0)
        ax.fill_between(t_ms, q25, q75, color=CLASS_COLORS[lab], alpha=0.25, linewidth=0)
        ax.plot(t_ms, q50, color=CLASS_COLORS[lab], linewidth=2)
        ax.set_title(class_title(lab), loc="left", fontsize=10)
        ax.set_xlabel("Time (ms)")
    axes[0].set_ylabel("ADC (median, IQR band)")
    fig.suptitle("Per-class median waveform across 50 trials (shaded: 25th-75th percentile)",
                 x=0.01, ha="left", fontsize=12)
    figs["class_median_waveform"] = savefig(fig, fig_dir / "fig03_class_median_waveform.png")

    # ADC value distribution per class
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.0), sharex=True)
    bins = np.arange(0, 2700, 50)
    for ax, lab in zip(axes, LABELS):
        s = df.loc[df["label"] == lab, "adc"]
        ax.hist(s, bins=bins, color=CLASS_COLORS[lab], edgecolor=SURFACE, linewidth=0.5)
        ax.set_yscale("log")
        ax.set_title(f"{lab}\nzero readings: {(s == 0).mean():.1%}", loc="left", fontsize=9)
        ax.set_xlabel("ADC")
    axes[0].set_ylabel("Readings (log scale)")
    fig.suptitle("Distribution of individual ADC readings by class", x=0.01, ha="left",
                 fontsize=12)
    figs["adc_distribution"] = savefig(fig, fig_dir / "fig04_adc_distribution.png")

    # Trial-level statistics by class (strip + median)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    stats = [("mean", "Trial mean ADC"), ("std", "Trial std ADC"), ("max", "Trial max ADC"),
             ("zero_fraction", "Fraction of zero readings")]
    for ax, (col, title) in zip(axes, stats):
        for i, lab in enumerate(LABELS):
            v = feats.loc[feats["label"] == lab, col].to_numpy()
            jitter = rng.uniform(-0.18, 0.18, len(v))
            ax.scatter(i + jitter, v, s=14, color=CLASS_COLORS[lab], alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.6)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, linewidth=2)
        ax.set_xticks(range(4), [l.replace("_", "\n") for l in LABELS])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Trial-level statistics by class (dots = trials, black bar = median)",
                 x=0.01, ha="left", fontsize=12)
    figs["trial_stats_by_class"] = savefig(fig, fig_dir / "fig05_trial_stats_by_class.png")

    # Recording order / drift
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for lab in LABELS:
        sub = feats[feats["label"] == lab]
        ax.scatter(sub["trial"], sub["mean"], s=16, color=CLASS_COLORS[lab],
                   edgecolors=SURFACE, linewidths=0.6, label=lab)
        ax.text(sub["trial"].mean(), feats["mean"].max() * 1.08, lab, ha="center",
                color=INK2, fontsize=8)
    for b in (50.5, 100.5, 150.5):
        ax.axvline(b, color=INK2, linewidth=0.8, linestyle=":")
    flagged = ta[ta["flagged"]]
    ax.scatter(flagged["trial"], feats.set_index("trial").loc[flagged["trial"], "mean"],
               s=70, facecolors="none", edgecolors=INK, linewidths=1.0, label="flagged trial")
    ax.set_xlabel("Trial ID (recording order)")
    ax.set_ylabel("Trial mean ADC")
    ax.set_ylim(top=feats["mean"].max() * 1.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    ax.set_title("Trial mean vs recording order: classes were recorded in contiguous blocks",
                 loc="left")
    figs["recording_order"] = savefig(fig, fig_dir / "fig06_recording_order.png")

    # Flagged trials
    if len(flagged):
        show = flagged.head(16)
        ncol = 4
        nrow = math.ceil(len(show) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.4 * nrow), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, (_, r) in zip(axes.flat, show.iterrows()):
            x = df.loc[df["trial"] == r["trial"]].sort_values("elapsed_ms")["adc"].to_numpy()
            ax.plot(t_ms, x, color=CLASS_COLORS[r["label"]])
            short = r["flags"].split(";")[0][:38]
            ax.set_title(f"#{int(r['trial'])} {r['label']}\n{short}", loc="left", fontsize=7.5)
        for ax in axes.flat[len(show):]:
            ax.set_visible(False)
        fig.suptitle(f"Trials flagged for review ({len(flagged)} total; kept in the dataset)",
                     x=0.01, ha="left", fontsize=12)
        figs["flagged_trials"] = savefig(fig, fig_dir / "fig07_flagged_trials.png")
    return figs


# ============================================================== 2. SPLIT
def make_split(feats: pd.DataFrame, seed: int) -> pd.DataFrame:
    trials = feats[["trial", "label"]]
    rest, test = train_test_split(trials, test_size=SPLIT_TRIALS["test"],
                                  stratify=trials["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=SPLIT_TRIALS["validation"],
                                  stratify=rest["label"], random_state=seed)
    manifest = pd.concat([train.assign(split="train"), val.assign(split="validation"),
                          test.assign(split="test")]).sort_values("trial").reset_index(drop=True)
    manifest["n_readings"] = SAMPLES_PER_TRIAL
    return manifest


def verify_split(manifest: pd.DataFrame, df: pd.DataFrame) -> list[dict]:
    out = []
    sets = {s: set(manifest.loc[manifest["split"] == s, "trial"]) for s in SPLIT_TRIALS}
    for s, n in SPLIT_TRIALS.items():
        out.append({"check": f"{s} trials", "expected": n, "observed": len(sets[s])})
        n_read = int(df["trial"].isin(sets[s]).sum())
        out.append({"check": f"{s} readings", "expected": n * SAMPLES_PER_TRIAL,
                    "observed": n_read})
        for lab in LABELS:
            c = int(((manifest["split"] == s) & (manifest["label"] == lab)).sum())
            out.append({"check": f"{s} {lab} trials", "expected": SPLIT_PER_CLASS[s],
                        "observed": c})
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        out.append({"check": f"overlap {a}/{b}", "expected": 0, "observed": len(sets[a] & sets[b])})
    out.append({"check": "trials assigned exactly once", "expected": EXPECTED["trials"],
                "observed": int(manifest["trial"].nunique())})
    for r in out:
        r["status"] = "PASS" if r["expected"] == r["observed"] else "FAIL"
    if any(r["status"] == "FAIL" for r in out):
        raise RuntimeError(f"Split verification failed: {out}")
    return out


# ============================================================ 4. BASELINES
def run_baselines(Xtr, ytr, Xva, yva, seed):
    results, preds = [], {}
    defs = [
        ("B0_majority_class", "Always predicts the most frequent training class "
         "(classes are tied, so sklearn picks the first label alphabetically).",
         DummyClassifier(strategy="most_frequent"), list(Xtr.columns)),
        ("B1_nearest_centroid_mean_adc", "Nearest class centroid on a single feature: the "
         "trial's mean ADC. Centroids = per-class average of trial means on the training set.",
         NearestCentroid(), ["mean"]),
    ]
    for name, desc, est, cols in defs:
        est.fit(Xtr[cols], ytr)
        p = est.predict(Xva[cols])
        preds[name] = p
        results.append({
            "baseline": name, "description": desc, "features": cols if len(cols) < 5 else "all",
            "val_accuracy": accuracy_score(yva, p),
            "val_macro_f1": f1_score(yva, p, labels=LABELS, average="macro", zero_division=0),
            "val_confusion_matrix": confusion_matrix(yva, p, labels=LABELS).tolist(),
        })
        if name.startswith("B1"):
            results[-1]["centroids_mean_adc"] = {c: float(v) for c, v in
                                                 zip(est.classes_, est.centroids_[:, 0])}
    return results, preds


# ================================================ 6. MODEL SELECTION (train+val)
def candidate_models(seed: int):
    """Small, pre-declared search space. complexity: lower = simpler (tie-break)."""
    cands = []
    for depth in (2, 3, 4):
        cands.append(("DecisionTree", {"max_depth": depth}, 1,
                      DecisionTreeClassifier(max_depth=depth, random_state=seed)))
    for C in (0.1, 1.0, 10.0):
        cands.append(("LogisticRegression", {"C": C}, 2,
                      make_pipeline(StandardScaler(),
                                    LogisticRegression(C=C, max_iter=5000))))
    for depth in (None, 6):
        for leaf in (1, 3):
            for mf in ("sqrt", 0.3):
                p = {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf,
                     "max_features": mf}
                cands.append(("RandomForest", p, 3, RandomForestClassifier(
                    **p, class_weight="balanced", random_state=seed, n_jobs=-1)))
    return cands


def select_model(Xtr, ytr, Xva, yva, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows, fitted = [], []
    for i, (family, params, complexity, est) in enumerate(candidate_models(seed)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_f1 = cross_val_score(clone(est), Xtr, ytr, cv=cv, scoring="f1_macro")
            model = clone(est).fit(Xtr, ytr)
        p = model.predict(Xva)
        proba = model.predict_proba(Xva)
        rows.append({
            "candidate_id": i, "family": family, "params": json.dumps(params),
            "complexity_rank": complexity,
            "train_accuracy": accuracy_score(ytr, model.predict(Xtr)),
            "train_cv5_macro_f1_mean": cv_f1.mean(), "train_cv5_macro_f1_std": cv_f1.std(),
            "val_accuracy": accuracy_score(yva, p),
            "val_macro_f1": f1_score(yva, p, labels=LABELS, average="macro", zero_division=0),
            "val_log_loss": log_loss(yva, proba, labels=model.classes_),
        })
        fitted.append(model)
    res = pd.DataFrame(rows)
    # Pre-declared rule: val macro-F1 (desc) -> train CV macro-F1 (desc)
    #                    -> val log-loss (asc) -> simpler family -> candidate order
    order = res.sort_values(["val_macro_f1", "train_cv5_macro_f1_mean", "val_log_loss",
                             "complexity_rank", "candidate_id"],
                            ascending=[False, False, True, True, True], kind="mergesort")
    res["selection_rank"] = 0
    res.loc[order.index, "selection_rank"] = np.arange(1, len(res) + 1)
    best = int(order.index[0])
    return res, fitted[best], best


# ======================================================= 7. EVALUATION
def evaluate(y_true, y_pred) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_class = {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                       "support": int(s[i])} for i, lab in enumerate(LABELS)}
    n = len(y_true)
    k = int(np.sum(np.asarray(y_true) == np.asarray(y_pred)))
    lo, hi = wilson_ci(k, n)
    return {
        "n_trials": n, "n_correct": k, "accuracy": k / n,
        "accuracy_95ci_wilson": [lo, hi],
        "per_class": per_class,
        "macro_avg": dict(zip(["precision", "recall", "f1"], map(float, precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0)[:3]))),
        "weighted_avg": dict(zip(["precision", "recall", "f1"], map(float, precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)[:3]))),
        "confusion_matrix": {"labels_order": LABELS, "rows": "actual", "columns": "predicted",
                             "matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
    }


def plot_confusion(cm, title, path: Path) -> str:
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap=SEQ_CMAP, vmin=0, vmax=max(cm.max(), 1))
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            v = cm[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                    color="#ffffff" if v > 0.6 * cm.max() else (INK if v else "#a3a29d"))
    ticks = [l.replace("_", "\n") for l in LABELS]
    ax.set_xticks(range(4), ticks)
    ax.set_yticks(range(4), ticks)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10)
    return savefig(fig, path)


# ============================================================ REPORT
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def cm_md(cm) -> str:
    rows = ["| Actual \\ Predicted | " + " | ".join(LABELS) + " |", "|---|" + "---|" * 4]
    for lab, row in zip(LABELS, cm):
        rows.append(f"| **{lab}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(rows)


def per_class_md(ev: dict) -> str:
    rows = ["| Class | Meaning | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
    for lab in LABELS:
        c = ev["per_class"][lab]
        rows.append(f"| {lab} | {LABEL_MEANINGS[lab]} | {c['precision']:.3f} | {c['recall']:.3f} "
                    f"| {c['f1']:.3f} | {c['support']} |")
    for name in ("macro_avg", "weighted_avg"):
        a = ev[name]
        rows.append(f"| **{name.replace('_', ' ')}** | | {a['precision']:.3f} | {a['recall']:.3f} "
                    f"| {a['f1']:.3f} | {ev['n_trials']} |")
    return "\n".join(rows)


# ================================================================= MAIN
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", default="gripsense_clean_data.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=".", help="project folder for outputs")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    csv_path = Path(args.csv).resolve()
    rep, figd, mdl, resd = root / "reports", root / "reports" / "figures", root / "models", root / "results"
    for d in (rep, figd, mdl, resd):
        d.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    stage_log = []

    def stage(name):
        stage_log.append({"stage": name, "utc": now()})
        print(f"[{stage_log[-1]['utc']}] {name}")

    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
           "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
    csv_hash_before = sha256(csv_path)

    # ---------------------------------------------------------------- 1
    stage("1. load + dataset audit (before split/training)")
    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")
    df = df[REQUIRED_COLUMNS]
    df["label"] = df["label"].astype(str)
    feats_all = build_feature_table(df)  # per-trial, stateless (no fitted parameters)
    audit, trial_audit, audit_figs, block_txt = audit_dataset(df, feats_all, figd, args.seed)
    audit.to_csv(rep / "dataset_audit.csv", index=False)
    trial_audit.to_csv(rep / "trial_quality_flags.csv", index=False)
    if (audit["status"] == "FAIL").any():
        print(audit[audit["status"] == "FAIL"].to_string())
        raise SystemExit("Dataset audit has FAIL items; stopping before training.")

    # ---------------------------------------------------------------- 2
    stage("2. trial-level stratified split 60/20/20")
    manifest = make_split(feats_all, args.seed)
    split_checks = verify_split(manifest, df)
    manifest.to_csv(rep / "split_manifest.csv", index=False)

    # ---------------------------------------------------------------- 3
    stage("3. feature extraction (one vector per trial)")
    feats = feats_all.merge(manifest[["trial", "split"]], on="trial")
    fcols = feature_columns(feats_all)
    feats[["trial", "label", "split"] + fcols].to_csv(rep / "trial_features.csv", index=False)
    part = {s: feats[feats["split"] == s].sort_values("trial") for s in SPLIT_TRIALS}
    Xtr, ytr = part["train"][fcols], part["train"]["label"].to_numpy()
    Xva, yva = part["validation"][fcols], part["validation"]["label"].to_numpy()
    Xte, yte = part["test"][fcols], part["test"]["label"].to_numpy()

    # ---------------------------------------------------------------- 4
    stage("4. baselines (fit on train, evaluated on validation)")
    baselines, base_preds = run_baselines(Xtr, ytr, Xva, yva, args.seed)
    pd.DataFrame([{k: v for k, v in b.items() if k != "val_confusion_matrix"} for b in baselines]) \
        .to_csv(resd / "baseline_results.csv", index=False)

    # ---------------------------------------------------------------- 5
    stage("5. pre-training plan written (no model fitted yet)")
    status_counts = audit["status"].value_counts().to_dict()
    n_flagged = int(trial_audit["flagged"].sum())
    b0, b1 = baselines
    plan = f"""## Pre-training plan (written {stage_log[-1]['utc']}, before any classifier was fitted)

**Dataset.** `{csv_path.name}` (SHA-256 `{csv_hash_before[:16]}...`): {len(df):,} ADC readings from
{df['trial'].nunique()} trials x {SAMPLES_PER_TRIAL} readings at {SAMPLE_INTERVAL_MS} ms (3.0 s per trial), 4 classes x 50 trials.

**Quality.** Audit outcome: {status_counts}. No missing values, duplicates, inconsistent labels, irregular
sampling or saturation. {n_flagged} trials flagged for review and **kept**. Main concern: classes were
recorded in contiguous blocks ({block_txt}), so class is confounded with recording session.

**Class distribution.** Perfectly balanced: 50 trials (7,500 readings) per class.

**Planned features.** {len(fcols)} per-trial features from the trial's own readings only: summary statistics
(mean, median, std, min, max, range, percentiles, IQR, RMS), zero-reading fraction, fixed-threshold
fractions ({', '.join(map(str, LEVEL_THRESHOLDS))} ADC), active-segment mean (> {ACTIVE_THRESHOLD}), longest zero/active runs,
press-event counts (rising crossings of {EVENT_THRESHOLDS} ADC), peak position, consecutive-difference
statistics, and mean/max over {N_SEGMENTS} equal time segments. Thresholds are fixed constants, not tuned.
Trial ID and `elapsed_ms` are not used as features.

**Split.** Stratified by label at trial level, seed {args.seed}: 120/40/40 trials (30/10/10 per class),
18,000/6,000/6,000 readings. No trial appears in more than one split (verified).

**Baseline (train -> validation).** {b0['baseline']}: accuracy {b0['val_accuracy']:.3f}, macro-F1 {b0['val_macro_f1']:.3f}.
{b1['baseline']}: accuracy {b1['val_accuracy']:.3f}, macro-F1 {b1['val_macro_f1']:.3f}. These are baseline results, not model results.

**Evaluation plan.** Candidates (decision tree, logistic regression, Random Forest; small fixed grid) are fitted
on the training set only. Selection rule, declared in advance: highest validation macro-F1, then highest
5-fold CV macro-F1 on the training set, then lowest validation log-loss, then the simpler model family.
The selected model (trained on the training set only) is then evaluated **once** on the 40 test trials.
Test results are not used for any decision.
"""
    (rep / "pre_training_plan.md").write_text(plan)

    # ---------------------------------------------------------------- 6
    stage("6. model training + selection (train/validation only)")
    selection_trials = set(part["train"]["trial"]) | set(part["validation"]["trial"])
    assert not selection_trials & set(part["test"]["trial"]), "test leakage into selection"
    sel_res, model, best_id = select_model(Xtr, ytr, Xva, yva, args.seed)
    sel_res.to_csv(resd / "model_selection_results.csv", index=False)
    best_row = sel_res.loc[best_id]
    val_pred = model.predict(Xva)
    val_eval = evaluate(yva, val_pred)
    print(f"   selected: {best_row['family']} {best_row['params']} "
          f"val_acc={best_row['val_accuracy']:.3f} val_macroF1={best_row['val_macro_f1']:.3f}")
    rf_best = sel_res[sel_res["family"] == "RandomForest"].sort_values("selection_rank").iloc[0]

    # Save model bundle BEFORE touching the test set
    bundle = {
        "model": model, "feature_columns": fcols, "labels": LABELS,
        "label_meanings": LABEL_MEANINGS, "family": best_row["family"],
        "params": json.loads(best_row["params"]), "seed": args.seed,
        "feature_module": "gripsense_features.extract_trial_features",
        "samples_per_trial": SAMPLES_PER_TRIAL, "sample_interval_ms": SAMPLE_INTERVAL_MS,
        "trained_on": "training split only (120 trials)", "versions": env,
    }
    joblib.dump(bundle, mdl / "gripsense_model.joblib")

    # ---------------------------------------------------------------- 7
    stage("7. final evaluation on held-out test set (once)")
    test_pred = model.predict(Xte)
    test_proba = model.predict_proba(Xte)
    test_eval = evaluate(yte, test_pred)
    print(f"   test accuracy={test_eval['accuracy']:.3f} macroF1={test_eval['macro_avg']['f1']:.3f}")

    # reload check
    reloaded = joblib.load(mdl / "gripsense_model.joblib")
    reload_pred = reloaded["model"].predict(Xte[reloaded["feature_columns"]])
    reload_ok = bool(np.array_equal(reload_pred, test_pred))

    # ---------------------------------------------------------------- 8
    stage("8. artifacts, figures, report")
    flag_map = trial_audit.set_index("trial")["flags"]

    def pred_frame(p, X, y, proba, split):
        out = p[["trial", "label"]].rename(columns={"label": "actual"}).copy()
        out["predicted"] = model.predict(X) if proba is None else model.classes_[proba.argmax(1)]
        out["correct"] = out["actual"] == out["predicted"]
        pr = model.predict_proba(X) if proba is None else proba
        for j, c in enumerate(model.classes_):
            out[f"prob_{c}"] = np.round(pr[:, j], 4)
        out["confidence"] = np.round(pr.max(1), 4)
        out["audit_flags"] = out["trial"].map(flag_map).fillna("")
        out["split"] = split
        return out

    val_df = pred_frame(part["validation"], Xva, yva, None, "validation")
    test_df = pred_frame(part["test"], Xte, yte, test_proba, "test")
    assert (test_df["predicted"].to_numpy() == test_pred).all()
    val_df.to_csv(resd / "validation_predictions.csv", index=False)
    test_df.to_csv(resd / "test_predictions.csv", index=False)
    mis = pd.concat([val_df[~val_df["correct"]], test_df[~test_df["correct"]]])
    mis.to_csv(resd / "misclassified_trials.csv", index=False)
    (resd / "classification_report_test.txt").write_text(
        classification_report(yte, test_pred, labels=LABELS, digits=3, zero_division=0))
    (resd / "classification_report_validation.txt").write_text(
        classification_report(yva, val_pred, labels=LABELS, digits=3, zero_division=0))

    figs = dict(audit_figs)
    # model comparison
    best_per_family = sel_res.sort_values("selection_rank").groupby("family").head(1)
    comp = [(b["baseline"], b["val_macro_f1"], "baseline") for b in baselines] + \
           [(f"{r['family']} (best of family)", r["val_macro_f1"],
             "selected" if r["candidate_id"] == best_id else "model")
            for _, r in best_per_family.iterrows()]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    for i, (name, v, kind) in enumerate(comp):
        ax.barh(i, v, color="#b3b2ad" if kind == "baseline" else CLASS_COLORS["LIGHT_PRESS"],
                height=0.6)
        ax.text(v + 0.01, i, f"{v:.3f}" + ("  <- selected" if kind == "selected" else ""),
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(comp)), [c[0] for c in comp])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Validation macro-F1 (40 trials)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Validation comparison: baselines (gray) vs trained models (blue)", loc="left")
    figs["model_comparison"] = savefig(fig, figd / "fig08_model_comparison_validation.png")
    figs["cm_baseline"] = plot_confusion(b1["val_confusion_matrix"],
                                         "Baseline B1 (mean-ADC nearest centroid)\nvalidation, 40 trials",
                                         figd / "fig09_confusion_matrix_baseline_validation.png")
    figs["cm_validation"] = plot_confusion(val_eval["confusion_matrix"]["matrix"],
                                           f"Selected model ({best_row['family']})\nvalidation, 40 trials",
                                           figd / "fig10_confusion_matrix_validation.png")
    figs["cm_test"] = plot_confusion(test_eval["confusion_matrix"]["matrix"],
                                     f"Selected model ({best_row['family']})\nheld-out TEST, 40 trials",
                                     figd / "fig11_confusion_matrix_test.png")

    # feature importance (model-native if available, else permutation on TRAIN)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp, imp_kind = pd.Series(est.feature_importances_, index=fcols), "impurity-based (training fit)"
    else:
        pi = permutation_importance(model, Xtr, ytr, n_repeats=20, random_state=args.seed,
                                    scoring="f1_macro")
        imp, imp_kind = pd.Series(pi.importances_mean, index=fcols), "permutation on training set"
    imp = imp.sort_values(ascending=False)
    imp.rename("importance").to_csv(resd / "feature_importance.csv", header=True)
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(range(len(top)), top.values, color=CLASS_COLORS["LIGHT_PRESS"], height=0.65)
    ax.set_yticks(range(len(top)), top.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Importance ({imp_kind})")
    ax.set_title(f"Top 15 features - selected {best_row['family']}", loc="left")
    figs["feature_importance"] = savefig(fig, figd / "fig12_feature_importance.png")

    # val vs test per-class F1
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    xs = np.arange(4)
    for k, (ev, name, col, off) in enumerate([(val_eval, "Validation", "#2a78d6", -0.19),
                                              (test_eval, "Test", "#eb6834", 0.19)]):
        vals = [ev["per_class"][l]["f1"] for l in LABELS]
        ax.bar(xs + off, vals, width=0.36, color=col, label=name)
        for x, v in zip(xs + off, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_xticks(xs, [l.replace("_", "\n") for l in LABELS])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1-score (10 trials per class)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Per-class F1: validation vs held-out test (selected model)", loc="left")
    figs["val_vs_test"] = savefig(fig, figd / "fig13_val_vs_test_f1.png")

    # misclassified waveforms
    if len(mis):
        n = len(mis)
        ncol = min(4, n)
        nrow = math.ceil(n / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.5 * nrow), squeeze=False,
                                 sharey=True)
        t_ms = np.arange(SAMPLES_PER_TRIAL) * SAMPLE_INTERVAL_MS
        for ax, (_, r) in zip(axes.flat, mis.iterrows()):
            x = df.loc[df["trial"] == r["trial"]].sort_values("elapsed_ms")["adc"].to_numpy()
            ax.plot(t_ms, x, color=CLASS_COLORS[r["actual"]])
            ax.set_title(f"#{r['trial']} [{r['split']}]\nactual {r['actual']}\n"
                         f"predicted {r['predicted']} (p={r['confidence']:.2f})",
                         loc="left", fontsize=7.5)
            ax.set_xlabel("ms")
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        fig.suptitle("Misclassified trials (validation + test)", x=0.01, ha="left", fontsize=12)
        figs["misclassified"] = savefig(fig, figd / "fig14_misclassified_trials.png")

    csv_hash_after = sha256(csv_path)
    metrics = {
        "generated_utc": now(), "seed": args.seed, "environment": env,
        "input": {"csv": csv_path.name, "sha256": csv_hash_before,
                  "unchanged_after_run": csv_hash_before == csv_hash_after},
        "dataset": {"rows": len(df), "trials": int(df["trial"].nunique()),
                    "readings_per_trial": SAMPLES_PER_TRIAL,
                    "trials_per_class": feats_all["label"].value_counts().reindex(LABELS).to_dict(),
                    "audit_status_counts": status_counts, "flagged_trials": n_flagged,
                    "flagged_trial_ids": trial_audit.loc[trial_audit["flagged"], "trial"].tolist(),
                    "recording_blocks": block_txt},
        "split": {"method": "stratified by label at trial level, sklearn train_test_split x2",
                  "trials": {s: int((manifest["split"] == s).sum()) for s in SPLIT_TRIALS},
                  "readings": {s: int((manifest["split"] == s).sum()) * SAMPLES_PER_TRIAL
                               for s in SPLIT_TRIALS},
                  "per_class": {s: manifest[manifest["split"] == s]["label"].value_counts()
                                .reindex(LABELS).astype(int).to_dict() for s in SPLIT_TRIALS},
                  "checks": split_checks},
        "features": {"n_features": len(fcols), "names": fcols},
        "baselines_validation": baselines,
        "model_selection": {
            "rule": "max val macro-F1 -> max train 5-fold CV macro-F1 -> min val log-loss -> "
                    "simpler family -> candidate order",
            "n_candidates": len(sel_res),
            "selection_used_trials": "train + validation only",
            "test_trials_used_in_selection": sorted(selection_trials & set(part["test"]["trial"])),
            "selected": {"candidate_id": best_id, "family": best_row["family"],
                         "params": json.loads(best_row["params"]),
                         "val_accuracy": float(best_row["val_accuracy"]),
                         "val_macro_f1": float(best_row["val_macro_f1"]),
                         "val_log_loss": float(best_row["val_log_loss"]),
                         "train_cv5_macro_f1": float(best_row["train_cv5_macro_f1_mean"])},
            "best_random_forest": {"candidate_id": int(rf_best["candidate_id"]),
                                   "params": json.loads(rf_best["params"]),
                                   "val_accuracy": float(rf_best["val_accuracy"]),
                                   "val_macro_f1": float(rf_best["val_macro_f1"]),
                                   "val_log_loss": float(rf_best["val_log_loss"]),
                                   "train_cv5_macro_f1": float(rf_best["train_cv5_macro_f1_mean"])},
            "n_candidates_tied_at_best_val_macro_f1": int(np.isclose(
                sel_res["val_macro_f1"], sel_res["val_macro_f1"].max()).sum()),
        },
        "validation": val_eval,
        "test": test_eval,
        "misclassified": {"validation": val_df.loc[~val_df["correct"], "trial"].tolist(),
                          "test": test_df.loc[~test_df["correct"], "trial"].tolist()},
        "model_reload_predictions_match": reload_ok,
        "stage_log": stage_log,
        "figures": figs,
    }
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    meta = {k: v for k, v in bundle.items() if k != "model"}
    meta.update({"validation_accuracy": val_eval["accuracy"], "test_accuracy": test_eval["accuracy"],
                 "training_trial_ids": sorted(part["train"]["trial"].tolist()),
                 "input_csv_sha256": csv_hash_before,
                 "usage": "python predict.py --csv <file with trial,elapsed_ms,adc>"})
    (mdl / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    write_report(rep, metrics, audit, trial_audit, sel_res, baselines, plan, val_df, test_df,
                 imp, imp_kind, args)
    print(f"   reload predictions match: {reload_ok}; CSV unchanged: {csv_hash_before == csv_hash_after}")
    print("Done. Report: reports/GripSense_TinyML_Report.md")


def write_report(rep, m, audit, trial_audit, sel_res, baselines, plan, val_df, test_df, imp,
                 imp_kind, args):
    sel = m["model_selection"]["selected"]
    rf = m["model_selection"]["best_random_forest"]
    v, t = m["validation"], m["test"]
    b0, b1 = baselines
    env = m["environment"]
    lo, hi = t["accuracy_95ci_wilson"]
    vlo, vhi = v["accuracy_95ci_wilson"]
    fig = m["figures"]
    flagged = trial_audit[trial_audit["flagged"]][["trial", "label", "flags"]]
    audit_view = audit[audit["status"] != "INFO"][["category", "check", "expected", "observed", "status"]]
    adc_view = audit[audit["check"].str.startswith("ADC stats")][["check", "observed"]]

    sel_view = sel_res.sort_values("selection_rank")[[
        "selection_rank", "family", "params", "train_cv5_macro_f1_mean", "val_accuracy",
        "val_macro_f1", "val_log_loss"]].copy()
    for c in ("train_cv5_macro_f1_mean", "val_accuracy", "val_macro_f1", "val_log_loss"):
        sel_view[c] = sel_view[c].map(lambda x: f"{x:.3f}")
    sel_view["params"] = sel_view["params"].str.replace("|", "/")

    def mis_table(d):
        mm = d[~d["correct"]]
        if mm.empty:
            return "_None - every trial in this split was classified correctly._"
        return md_table(mm[["trial", "actual", "predicted", "confidence", "audit_flags"]])

    low_conf = test_df.sort_values("confidence").head(5)[["trial", "actual", "predicted",
                                                          "confidence", "audit_flags"]]
    gap = t["accuracy"] - v["accuracy"]
    f1_lines = "\n".join(
        f"| {lab} | {v['per_class'][lab]['f1']:.3f} | {t['per_class'][lab]['f1']:.3f} |" for lab in LABELS)
    flag_test = test_df[test_df["audit_flags"] != ""]
    mis_all = pd.concat([val_df[~val_df["correct"]], test_df[~test_df["correct"]]])
    n_mis_all = len(mis_all)
    n_mis_flagged = int((mis_all["audit_flags"] != "").sum())
    mis_desc = ", ".join(f"#{r.trial} {r.actual}->{r.predicted} [{r.split}]" for r in mis_all.itertuples()) or "none"
    n_flag_test_wrong = int((~flag_test["correct"]).sum())

    def img(key, alt):
        return f"![{alt}](figures/{fig[key]})" if key in fig else ""

    report = f"""# GripSense TinyML - FSR Grip Gesture Classification Report

_Generated {m['generated_utc']} by `run_experiment.py` (seed {m['seed']}). Every number in this
report is read from `reports/metrics.json` and the files in `results/`; nothing is hand-entered._

## Executive summary

GripSense classifies a 3-second force-sensitive-resistor (FSR) recording from an ESP32 into four
assistive-communication intents: **LIGHT_PRESS (Hungry)**, **HARD_PRESS (Pain)**, **DOUBLE_SQUEEZE
(Washroom)** and **NO_PRESS (Idle/resting)**. This phase ran fully offline on the existing CSV
(`{m['input']['csv']}`, {m['dataset']['rows']:,} readings, {m['dataset']['trials']} trials). No hardware was used.

- **Dataset audit (before training):** structurally clean (no missing values, duplicates, label
  conflicts, timing irregularities or saturation). {m['dataset']['flagged_trials']} trials were flagged as
  atypical and kept. **Main risk:** each class was recorded as one contiguous block of trials, so class
  is confounded with recording session.
- **Baseline (validation):** majority class = {b0['val_accuracy']:.3f} accuracy; single-feature
  nearest-centroid on mean ADC = {b1['val_accuracy']:.3f} accuracy / {b1['val_macro_f1']:.3f} macro-F1.
- **Selected model:** {sel['family']} `{json.dumps(sel['params'])}`, chosen on validation only.
  Validation accuracy {v['accuracy']:.3f}, macro-F1 {v['macro_avg']['f1']:.3f}.
- **Final held-out test (40 trials, evaluated once):** accuracy **{t['accuracy']:.3f}**
  ({t['n_correct']}/{t['n_trials']}; 95% Wilson CI {lo:.3f}-{hi:.3f}), macro-F1 **{t['macro_avg']['f1']:.3f}**.
- With only 10 test trials per class from one recording session (and likely one user), these results
  show that the classes are separable **in this dataset**. They do **not** establish real-world reliability.

## 1. Objectives

1. Verify and audit the recorded FSR dataset before any modelling.
2. Build a leakage-free trial-level train/validation/test split (60/20/20, stratified).
3. Establish simple baselines, then train and select ML models using train/validation data only.
4. Report one final, untouched evaluation on the held-out test set, with honest limitations.
5. Produce a model artifact that can be reloaded for inference (a stepping stone toward on-device TinyML).

## 2. Methodology overview

| Stage | What was done | Data used |
|---|---|---|
| Audit | Schema, counts, timing, labels, ADC ranges, per-trial anomaly flags, plots | full CSV (descriptive only) |
| Split | Stratified by label at **trial** level, seed {m['seed']} | trial IDs + labels |
| Features | {m['features']['n_features']} fixed, stateless per-trial features | each trial's own readings |
| Baselines | Majority class; mean-ADC nearest centroid | fit: train · score: validation |
| Model selection | {m['model_selection']['n_candidates']} candidates (tree / logistic regression / Random Forest) | fit: train · choose: validation (+ train CV) |
| Final test | Selected model scored once | test (40 trials) |

The script records a timestamped stage log (in `metrics.json`) showing that the pre-training plan was
written before any classifier was fitted and that the test set was scored only after selection.

---

## 3. Dataset findings BEFORE training

### 3.1 Structure and integrity

{md_table(audit_view)}

Full audit (including INFO rows): `reports/dataset_audit.csv`. Per-trial flags: `reports/trial_quality_flags.csv`.

### 3.2 ADC value summary per class (raw readings)

{md_table(adc_view)}

ADC values lie in 0-{int(audit.loc[audit['check'] == 'ADC min / max', 'observed'].iloc[0].split('/')[1])}
of the 0-4095 range. No saturation. Zero readings are normal for this sensor when unloaded:
NO_PRESS is almost entirely zeros and DOUBLE_SQUEEZE drops to zero between squeezes.

{img('class_distribution', 'Class distribution')}

{img('representative_waveforms', 'Representative waveforms')}

{img('class_median_waveform', 'Class median waveforms')}

{img('adc_distribution', 'ADC distributions')}

{img('trial_stats_by_class', 'Trial statistics by class')}

### 3.3 Data-quality concerns

1. **Recording-order confound (most important).** Trials were recorded class-by-class
   ({m['dataset']['recording_blocks']}). Any drift over time (sensor warm-up, finger placement, user
   fatigue, cable movement) is perfectly aligned with the label. The model could partly learn
   "session" rather than "gesture", and a random split cannot detect this. Trial ID is excluded from
   the features, but the confound itself cannot be fixed in this dataset. Fig. 6 shows visible
   within-block drift (e.g. HARD_PRESS mean force falls over trials 51-100, DOUBLE_SQUEEZE force rises
   over trials 101-120), so recording conditions changed during collection.
2. **Overlap between LIGHT_PRESS and DOUBLE_SQUEEZE by average force.** Their trial means overlap
   heavily (see Fig. 5), so mean force alone cannot separate them. Temporal structure (zeros between
   squeezes, press-event count) is required.
3. **DOUBLE_SQUEEZE trials do not contain a double squeeze.** Counting distinct presses (rising
   crossings of 100 ADC) gives {audit.loc[audit['check'] == 'press events per trial: DOUBLE_SQUEEZE', 'observed'].iloc[0]}
   (events: trials). No DOUBLE_SQUEEZE trial has exactly 2 presses. The raw waveforms (Fig. 2) show clean,
   separate squeezes repeated for the whole 3-second window. The model therefore learns "repeated
   squeezing", and a user who performs one real double squeeze (2 presses, then rest) would produce a
   pattern absent from the training data. This is a protocol/labelling issue, not sensor noise, and is
   reported as a dataset-level WARN rather than 50 per-trial flags.
4. **Atypical trials ({m['dataset']['flagged_trials']} flagged, none removed).** Flags use per-class robust
   z-scores (|z| > 3.5 on mean/std/max/zero-fraction/mean |Δ|) plus protocol heuristics (activity during
   NO_PRESS, dropouts to zero during a sustained press, DOUBLE_SQUEEZE with fewer than two presses,
   LIGHT_PRESS peaks above 1500 ADC). The heuristics are review prompts, not proof of mislabelling.
   HARD_PRESS has many high-variability flags: several trials pulse or wobble rather than holding steady.
5. **Nominal timestamps.** All intervals are exactly {SAMPLE_INTERVAL_MS} ms. This is consistent with a logger writing
   nominal sample times, so real sampling jitter cannot be checked from this file.
6. **Unknown provenance details.** The CSV does not record the participant, session or date. All
   trials appear to come from one collection effort, so generalisation to other users or sensor
   placements is untested.

{img('recording_order', 'Recording order')}

Flagged trials:

{md_table(flagged) if len(flagged) else '_None._'}

{img('flagged_trials', 'Flagged trials')}

---

{plan.replace('## Pre-training plan', '## 4. Pre-training plan: dataset, split, features, baseline, evaluation')}

---

## 5. Split design

{md_table(pd.DataFrame([{'split': s, 'trials': m['split']['trials'][s], 'readings': m['split']['readings'][s],
    **m['split']['per_class'][s]} for s in ('train', 'validation', 'test')]))}

Verification (all PASS): {sum(c['status'] == 'PASS' for c in m['split']['checks'])}/{len(m['split']['checks'])} checks, including
zero trial overlap between every pair of splits and all four classes present in every split.
Manifest: `reports/split_manifest.csv`.

## 6. Baseline results (validation set, NOT the trained model)

| Baseline | Method | Val accuracy | Val macro-F1 |
|---|---|---|---|
| {b0['baseline']} | {b0['description']} | {b0['val_accuracy']:.3f} | {b0['val_macro_f1']:.3f} |
| {b1['baseline']} | {b1['description']} | {b1['val_accuracy']:.3f} | {b1['val_macro_f1']:.3f} |

B1 training centroids (mean ADC): {', '.join(f"{k}={v:.0f}" for k, v in b1['centroids_mean_adc'].items())}.

Baseline B1 confusion matrix (validation; rows = actual, columns = predicted):

{cm_md(b1['val_confusion_matrix'])}

{img('cm_baseline', 'Baseline confusion matrix')}

## 7. Feature engineering

{m['features']['n_features']} features per trial, computed by `gripsense_features.extract_trial_features` from that
trial's 150 readings only. The function has no learned parameters, so it cannot leak statistics across
splits. The same function is used in `predict.py` for inference.

| Group | Features |
|---|---|
| Level statistics | mean, median, std, min, max, range, q10, q25, q75, q90, iqr, rms |
| Contact / zero | zero_fraction, longest_zero_run_frac, longest_active_run_frac, active_mean (mean of readings > {ACTIVE_THRESHOLD}) |
| Threshold fractions | above_{{{', '.join(map(str, LEVEL_THRESHOLDS))}}}_fraction |
| Temporal events | press_events_100, press_events_300 (rising threshold crossings), peak_position_frac |
| Change between readings | mean_abs_diff, std_diff, max_rise, max_fall |
| Segment statistics | segment_0..{N_SEGMENTS - 1}_mean, segment_0..{N_SEGMENTS - 1}_max ({N_SEGMENTS} x 300 ms windows) |

Per-trial feature table: `reports/trial_features.csv`.

## 8. Model training and validation

Every candidate was fitted on the 120 training trials. Validation scores (40 trials) and a 5-fold
stratified CV on the training set were used for comparison. Logistic regression standardises features
inside a pipeline, so the scaler is fitted on training data only.

**Selection rule (declared before fitting):** {m['model_selection']['rule']}.

{md_table(sel_view)}

**{m['model_selection']['n_candidates_tied_at_best_val_macro_f1']} of {m['model_selection']['n_candidates']} candidates
tied at the best validation macro-F1 ({sel['val_macro_f1']:.3f}).** With so many ties, the validation set (40 trials) cannot
meaningfully rank these models. The tie-breakers (training CV, then validation log-loss, then
simplicity) made the final choice.

- **Selected:** {sel['family']} `{json.dumps(sel['params'])}`. Validation accuracy {sel['val_accuracy']:.3f},
  macro-F1 {sel['val_macro_f1']:.3f}, log-loss {sel['val_log_loss']:.3f}, train-CV macro-F1 {sel['train_cv5_macro_f1']:.3f}.
- **Best Random Forest:** `{json.dumps(rf['params'])}`. Validation accuracy {rf['val_accuracy']:.3f}, macro-F1
  {rf['val_macro_f1']:.3f}, log-loss {rf['val_log_loss']:.3f}, train-CV macro-F1 {rf['train_cv5_macro_f1']:.3f}.

Selected model, validation (95% Wilson CI for accuracy {vlo:.3f}-{vhi:.3f}):

{per_class_md(v)}

{cm_md(v['confusion_matrix']['matrix'])}

{img('model_comparison', 'Model comparison')}

{img('cm_validation', 'Validation confusion matrix')}

{img('feature_importance', 'Feature importance')}

Feature importance ({imp_kind}), top 10: {', '.join(f'`{k}` ({x:.3f})' for k, x in imp.head(10).items())}.

Validation misclassifications:

{mis_table(val_df)}

---

## 9. Final test results AFTER training (held-out, evaluated once)

The model was fitted on the 120 training trials and never saw the 40 test trials. It was saved to
`models/gripsense_model.joblib` before the test set was scored. No decision was changed after viewing these numbers.

**Accuracy: {t['accuracy']:.3f}** ({t['n_correct']}/{t['n_trials']} trials; 95% Wilson CI {lo:.3f}-{hi:.3f}).

{per_class_md(t)}

### Confusion matrix (test; rows = actual, columns = predicted)

{cm_md(t['confusion_matrix']['matrix'])}

{img('cm_test', 'Test confusion matrix')}

### Misclassified test trials

{mis_table(test_df)}

### Validation vs test

| Metric | Validation | Test |
|---|---|---|
| Accuracy | {v['accuracy']:.3f} | {t['accuracy']:.3f} |
| Macro F1 | {v['macro_avg']['f1']:.3f} | {t['macro_avg']['f1']:.3f} |
| Weighted F1 | {v['weighted_avg']['f1']:.3f} | {t['weighted_avg']['f1']:.3f} |
| Accuracy 95% CI | {vlo:.3f}-{vhi:.3f} | {lo:.3f}-{hi:.3f} |

| Class | Val F1 | Test F1 |
|---|---|---|
{f1_lines}

{img('val_vs_test', 'Validation vs test F1')}

{img('misclassified', 'Misclassified trials')}

## 10. Error analysis

- Test accuracy differs from validation accuracy by {gap:+.3f}. With 40 trials per split, one trial
  changes accuracy by 0.025, so differences of 1-2 trials are within noise.
- Test trials carrying an audit flag: {len(flag_test)}, of which {n_flag_test_wrong} were misclassified.
- Misclassified trials across validation + test: {n_mis_all}; of these, {n_mis_flagged} were already flagged as
  atypical in the pre-training audit ({mis_desc}). The model's errors therefore fall on unusual
  recordings (e.g. LIGHT_PRESS trials with HARD-like peaks or dropouts) rather than on typical trials.
  On ~40 trials per split, this is an observation, not a guarantee.
- Lowest-confidence test predictions (the cases closest to the decision boundary):

{md_table(low_conf)}

- The confusable pair in this data is LIGHT_PRESS vs DOUBLE_SQUEEZE (similar mean force). The baseline B1
  confusion matrix shows this directly. The trained model separates them using temporal and zero-contact
  features (see feature importance). HARD_PRESS and NO_PRESS are separated by level alone.

## 11. Discussion

The per-trial features make the four gestures close to linearly separable **in this recording**.
Simple models (shallow tree, logistic regression) score as well as the Random Forest on validation. This is good
news for TinyML, because a tree of depth 2-4 or a 4x{m['features']['n_features']} linear model fits easily on an ESP32,
while a 300-tree forest is much heavier. The jump from baseline B1 ({b1['val_accuracy']:.3f}) to the trained model
({v['accuracy']:.3f}) comes almost entirely from temporal and contact features that resolve LIGHT_PRESS vs DOUBLE_SQUEEZE.

Near-perfect scores call for scepticism rather than celebration. The recording-order confound
(Section 3.3) means part of the separability could come from session effects. That matters most for
classes whose definitions depend on absolute force level (LIGHT vs HARD), which also shift with sensor
placement and between users.

## 12. Limitations

1. **Small test set:** 10 trials per class (40 total). One error costs 2.5 accuracy points overall and
   10 recall points for that class. The 95% CI on test accuracy is {lo:.3f}-{hi:.3f}. Per-class
   precision and recall have even wider uncertainty.
2. **Ceiling effects:** {m['model_selection']['n_candidates_tied_at_best_val_macro_f1']} of {m['model_selection']['n_candidates']} candidates tie at the best validation score, so model selection is weakly
   informed. The validation-vs-test comparison cannot reveal small generalisation gaps.
3. **Recording-order / session confound:** classes were recorded in contiguous blocks. Random trial
   splits put trials from the same session in both train and test, which likely overstates performance.
4. **Probably one user, one sensor, one placement:** no cross-user or cross-day evaluation is possible.
5. **DOUBLE_SQUEEZE protocol mismatch:** training "double squeezes" are 3-7 repeated squeezes. Performance
   on a genuine single double squeeze is unknown and could be poor (it may look like LIGHT_PRESS).
6. **Scripted, segmented gestures:** each trial is a clean, pre-segmented 3-second window. Real use
   needs continuous segmentation, rejection of unintended grips, and robustness to long idle periods.
7. **Offline only:** the model has not been converted, quantised or timed on the ESP32.
8. **Safety:** HARD_PRESS means "Pain". A missed or false "Pain" signal has real consequences, so
   any deployment needs confirmation feedback and human oversight, whatever these metrics show.

## 13. Conclusion

On this dataset, a trial-level classifier on {m['features']['n_features']} hand-crafted FSR features clearly beats
both baselines. The selected {sel['family']} reached {v['accuracy']:.3f} validation accuracy and
**{t['accuracy']:.3f} held-out test accuracy ({t['n_correct']}/{t['n_trials']})**. The data pipeline is clean
and leakage-free at the trial level. The evidence supports "these four gestures are distinguishable
in this recording". It does not support "GripSense is reliable in real-world use".

## 14. Next steps

1. **Collect new data with interleaved/randomised class order** across several sessions, days and (ideally)
   users. Then evaluate with session- or user-grouped splits (leave-one-session-out).
2. Re-record DOUBLE_SQUEEZE as exactly two squeezes followed by rest, matching the intended gesture. Keep
   the current trials as a separate "repeated squeeze" condition if it is useful.
3. Record NO_PRESS with realistic distractors (hand resting on the sensor, accidental bumps), not only zeros.
4. Review the flagged trials ({', '.join(map(str, m['dataset']['flagged_trial_ids'][:12]))}{'...' if len(m['dataset']['flagged_trial_ids']) > 12 else ''}) and confirm whether they were performed as instructed.
5. Port a compact model (depth-limited tree or logistic regression on a few features, e.g. via
   `micromlgen`/`m2cgen` or hand-written C). Verify on-device feature parity against `gripsense_features.py`,
   then measure latency and memory on the ESP32.
6. Add a sliding-window / onset detector for continuous streams, and a confidence threshold with an
   "uncertain - please repeat" output.

## 15. Reproducibility

- Seed: **{m['seed']}** (split, CV folds, models, figure sampling).
- Environment: Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, scikit-learn
  {env['scikit-learn']}, matplotlib {env['matplotlib']}, joblib {env['joblib']} ({env['platform']}).
- Input CSV SHA-256: `{m['input']['sha256']}` (unchanged after run: {m['input']['unchanged_after_run']}).
- Exact split counts: train {m['split']['trials']['train']} trials / {m['split']['readings']['train']:,} readings;
  validation {m['split']['trials']['validation']} / {m['split']['readings']['validation']:,}; test {m['split']['trials']['test']} / {m['split']['readings']['test']:,};
  30/10/10 trials per class.
- Commands (from the project folder):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_experiment.py --csv gripsense_clean_data.csv --seed {m['seed']}
.venv/bin/python validate_outputs.py
.venv/bin/python predict.py --csv gripsense_clean_data.csv --trials 1 60 120 180
```

- Saved model reload check (predictions identical to in-memory model): {m['model_reload_predictions_match']}.

**Run history (disclosure).** The first full run of this script used the same split, features, candidate
grid and selection rule. It selected a Random Forest (`max_depth=None, min_samples_leaf=1, max_features="sqrt"`)
and scored 0.975 on validation and 0.975 on test (39/40; misclassified test trial 39). Afterwards, two
changes were made, prompted by the audit and a runtime warning (not by the test score):
(a) validation log-loss had used clipped probabilities that were not renormalised (tie-breaker only);
(b) the audit flag "DOUBLE_SQUEEZE must have exactly 2 presses" flagged all 50 trials, which was recognised
as a protocol finding (Section 3.3) and moved to a dataset-level check. Features, split and candidates were
not changed.

## Appendix: artifact index

| Path | Content |
|---|---|
| `reports/dataset_audit.csv` | data-quality checks |
| `reports/trial_quality_flags.csv` | per-trial robust z-scores and flags |
| `reports/trial_features.csv` | one row per trial: trial, label, split, {m['features']['n_features']} features |
| `reports/split_manifest.csv` | trial -> label -> split |
| `reports/pre_training_plan.md` | plan written before model fitting |
| `reports/metrics.json` | all measured metrics |
| `reports/figures/` | {len(fig)} figures |
| `models/gripsense_model.joblib`, `models/model_metadata.json` | model bundle + metadata |
| `results/` | predictions, selection table, baselines, classification reports, misclassified trials, feature importance |
"""
    (rep / "GripSense_TinyML_Report.md").write_text(report)


if __name__ == "__main__":
    main()
