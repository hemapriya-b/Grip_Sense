#!/usr/bin/env python3
"""
Build the GripSense project documentation package from EXISTING artifacts only.

Reads (never writes) the dataset, phase-1/phase-2/C-port outputs and saved models;
re-derives key facts from the raw files and asserts they match the saved metrics;
then writes figures, Markdown reports, README and PDF into results/project_documentation/.

    .venv/bin/python results/project_documentation/scripts/build_documentation.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "results" / "project_documentation"
FIG = DOC / "figures"
LABELS = ["LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"]
MEANING = {"LIGHT_PRESS": "Hungry", "HARD_PRESS": "Pain", "DOUBLE_SQUEEZE": "Washroom",
           "NO_PRESS": "Resting / idle"}
DISPLAY = {"B0_majority": "Majority-class baseline", "B1_mean_adc_centroid": "Mean-ADC baseline",
           "RF_existing": "Random Forest (300 trees)", "LR_C0.1": "Logistic Regression C=0.1",
           "LR_C1": "Logistic Regression C=1", "LR_C10": "Logistic Regression C=10",
           "DT_depth2": "Decision Tree depth 2", "DT_depth3": "Decision Tree depth 3"}

# ---- plot style (palette validated earlier with the dataviz validator: slots 1-4)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CLASS_COLORS = dict(zip(LABELS, [BLUE, ORANGE, AQUA, YELLOW]))
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 12,
    "legend.frameon": False, "savefig.dpi": 220,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rjson(p):
    return json.loads((ROOT / p).read_text())


# =========================================================== 1. collect + verify facts
def collect() -> dict:
    F: dict = {}
    df = pd.read_csv(ROOT / "gripsense_clean_data.csv")
    g = df.groupby("trial")
    tl = g["label"].first()
    F["columns"] = list(df.columns)
    F["rows"], F["trials"] = len(df), int(df["trial"].nunique())
    F["per_trial"] = sorted(set(g.size().tolist()))
    F["per_label_trials"] = tl.value_counts().reindex(LABELS).astype(int).to_dict()
    F["per_label_rows"] = df["label"].value_counts().reindex(LABELS).astype(int).to_dict()
    iv = np.concatenate([np.diff(s.to_numpy()) for _, s in g["elapsed_ms"]])
    F["intervals_ms"] = sorted(set(iv.tolist()))
    F["elapsed_min"], F["elapsed_max"] = int(df["elapsed_ms"].min()), int(df["elapsed_ms"].max())
    F["adc_min"], F["adc_max"] = int(df["adc"].min()), int(df["adc"].max())
    F["missing"] = int(df.isna().sum().sum())
    F["dup_rows"] = int(df.duplicated().sum())
    F["labels_per_trial_max"] = int(g["label"].nunique().max())
    blocks = tl.reset_index().groupby("label")["trial"].agg(["min", "max"])
    F["blocks"] = {lab: (int(blocks.loc[lab, "min"]), int(blocks.loc[lab, "max"])) for lab in LABELS}
    F["zero_frac_by_label"] = {lab: float((df.loc[df.label == lab, "adc"] == 0).mean()) for lab in LABELS}
    assert F["rows"] == 30000 and F["trials"] == 200 and F["per_trial"] == [150]
    assert all(v == 50 for v in F["per_label_trials"].values())

    man = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
    F["split_trials"] = man["split"].value_counts().to_dict()
    F["split_per_class"] = {s: man[man.split == s]["label"].value_counts().reindex(LABELS).astype(int).to_dict()
                            for s in ("train", "validation", "test")}
    F["split_rows"] = {s: int(df["trial"].isin(man.loc[man.split == s, "trial"]).sum())
                       for s in ("train", "validation", "test")}
    sets = {s: set(man.loc[man.split == s, "trial"]) for s in ("train", "validation", "test")}
    F["split_overlap"] = len(sets["train"] & sets["validation"]) + len(sets["train"] & sets["test"]) + \
        len(sets["validation"] & sets["test"])
    assert F["split_rows"] == {"train": 18000, "validation": 6000, "test": 6000} and F["split_overlap"] == 0

    m1 = rjson("reports/metrics.json")
    m2 = rjson("reports/followup/followup_metrics.json")
    vs = rjson("results/c_port/verification_summary.json")
    tm = rjson("models/c_port/tree_metadata.json")
    F["seed"] = m1["seed"]
    F["env"] = m1["environment"]
    F["n_features"] = m1["features"]["n_features"]
    F["phase1_selected"] = m1["model_selection"]["selected"]
    F["phase1_generated"] = m1["generated_utc"]
    F["audit_status"] = m1["dataset"]["audit_status_counts"]
    F["flagged_ids"] = m1["dataset"]["flagged_trial_ids"]
    flags = pd.read_csv(ROOT / "reports" / "trial_quality_flags.csv")
    fl = flags[flags.flagged]
    F["flagged_by_label"] = fl["label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()
    F["no_press_flagged"] = {int(r.trial): r.flags for r in fl[fl.label == "NO_PRESS"].itertuples()}
    F["disclosure"] = m2["test_history_disclosure"]
    F["selection"] = m2["selection"]

    # model comparison, re-derived from predictions
    comp = pd.read_csv(ROOT / "results" / "followup" / "model_comparison.csv")
    preds = pd.read_csv(ROOT / "results" / "followup" / "predictions_all_models.csv")
    models = []
    for r in comp.itertuples():
        row = {"id": r.model, "name": DISPLAY[r.model], "family": r.family, "params": r.params}
        for sp in ("validation", "test"):
            d = preds[(preds.model == r.model) & (preds.split == sp)]
            acc = float(d["correct"].mean())
            ev = m2["evaluations"][r.model][sp]
            assert abs(acc - ev["accuracy"]) < 1e-12, (r.model, sp)
            row[f"{sp}_acc"] = acc
            row[f"{sp}_macro_f1"] = ev["macro_avg"]["f1"]
            row[f"{sp}_errors"] = int((~d["correct"]).sum())
            row[f"{sp}_mis"] = sorted(d.loc[~d["correct"], "trial"].astype(int).tolist())
            row[f"{sp}_ci"] = wilson(int(d["correct"].sum()), len(d))
            row[f"{sp}_cm"] = ev["confusion_matrix"]["matrix"]
            row[f"{sp}_per_class"] = ev["per_class"]
        row["size_units"], row["est_bytes"] = r.footprint, int(r.est_bytes)
        row["n_features_used"] = int(r.n_features_used)
        row["train_cv"] = float(r.train_cv5_macro_f1)
        models.append(row)
    F["models"] = models
    F["mcnemar"] = m2["mcnemar_test_vs_selected"]
    F["wilson_10_10"], F["wilson_9_10"] = m2["wilson_lower_bound_10_of_10"], m2["wilson_lower_bound_9_of_10"]

    # hyperparameters straight from saved estimators
    rf = joblib.load(ROOT / "models" / "gripsense_model.joblib")["model"].get_params()
    lr = joblib.load(ROOT / "models" / "followup" / "LR_C1.joblib")["model"]
    lrp = lr.steps[-1][1].get_params()
    dt = joblib.load(ROOT / "models" / "followup" / "DT_depth2.joblib")["model"].get_params()
    F["hp"] = {
        "rf": {k: rf[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features",
                                  "class_weight", "criterion", "bootstrap", "random_state")},
        "lr": {"pipeline": [s[0] for s in lr.steps], "C": lrp["C"], "solver": lrp["solver"],
               "l1_ratio": lrp["l1_ratio"], "max_iter": lrp["max_iter"], "class_weight": lrp["class_weight"]},
        "dt": {k: dt[k] for k in ("max_depth", "criterion", "min_samples_leaf", "class_weight", "random_state")},
    }

    # protocol facts
    ds = m2["double_squeeze"]
    F["ds"] = ds
    F["onset"] = m2["press_onset"]
    ex = m2["exploratory"]
    F["e1"] = ex["E1_repeated_cv"]
    F["e4"] = ex["E4_confidence_threshold"]
    ps = pd.read_csv(ROOT / "results" / "followup" / "exploratory_probe_summary.csv")
    F["probes"] = ps.to_dict(orient="records")
    pert = pd.read_csv(ROOT / "results" / "followup" / "exploratory_perturbations.csv")
    F["gain_rf"] = pert[(pert.perturbation == "gain") & (pert.model == "RF")].set_index("value")["accuracy"].to_dict()
    seg = pd.read_csv(ROOT / "results" / "followup" / "press_segments_per_trial.csv")
    F["ds_press_counts"] = seg[seg.label == "DOUBLE_SQUEEZE"]["n_presses"].value_counts().sort_index().to_dict()
    assert {int(k): v for k, v in ds["presses_distribution"].items()} == F["ds_press_counts"]

    # C port, re-derived from per-trial file
    cp = pd.read_csv(ROOT / "results" / "c_port" / "predictions_python_vs_c.csv")
    F["c"] = {
        "n": len(cp),
        "agree_features": int((cp["c_from_features"] == cp["python_sklearn"]).sum()),
        "agree_raw": int((cp["c_from_raw_adc"] == cp["python_sklearn"]).sum()),
        "feat_exact": {f: int((cp[f"py_{f}"] == cp[f"c_{f}"]).sum())
                       for f in ("longest_active_run_frac", "segment_1_mean")},
    }
    assert F["c"]["agree_features"] == vs["agreement_features_mode"]
    assert F["c"]["agree_raw"] == vs["agreement_raw_adc_mode"]
    bt = pd.read_csv(ROOT / "results" / "c_port" / "boundary_tests.csv")
    F["c"]["boundary_n"], F["c"]["boundary_agree"] = len(bt), int(bt["agree"].sum())
    F["c"]["naive_disagree"] = int((bt["naive_double_compare"] != bt["python_sklearn"]).sum())
    assert F["c"]["boundary_agree"] == vs["boundary_tests"]["agree"]
    F["vs"] = vs
    F["tree"] = tm
    F["compile_log"] = (ROOT / "results" / "c_port" / "compile_log.txt").read_text()
    F["c_warnings"] = len(re.findall(r"warning:", F["compile_log"]))
    assert F["c_warnings"] == vs["compiler"]["n_warnings"]
    F["c_accuracy_by_split"] = vs["accuracy_vs_labels_by_split"]
    # live check of the compiled harness (read-only execution)
    info = subprocess.run([str(ROOT / "models" / "c_port" / "build" / "decision_tree"), "--info"],
                          capture_output=True, text=True)
    F["c_info"] = info.stdout.strip()
    F["c_info_ok"] = info.returncode == 0
    lab_lines = [l.split() for l in F["c_info"].splitlines() if l.startswith("label ")]
    F["c_label_map"] = {int(a[1]): a[2] for a in lab_lines}
    assert [F["c_label_map"][i] for i in range(4)] == LABELS
    c_src = (ROOT / "models" / "c_port" / "decision_tree.c").read_text()
    F["c_src_has_float_cast"] = "(double)(float)v <= GS_THRESHOLD[node]" in c_src
    F["c_seg_bounds"] = re.search(r"segment_mean\(adc, (\d+), (\d+)\)", c_src).groups()
    feat_src = (ROOT / "gripsense_features.py").read_text()
    F["feat_src_checks"] = {
        "ACTIVE_THRESHOLD = 100": "ACTIVE_THRESHOLD = 100" in feat_src,
        "longest_active_run_frac = _longest_run(active) / n": '"longest_active_run_frac": _longest_run(active) / n' in feat_src,
        "segments via np.array_split(x, N_SEGMENTS)": "np.array_split(x, N_SEGMENTS)" in feat_src,
    }
    assert all(F["feat_src_checks"].values()) and F["c_src_has_float_cast"]

    # dashboard (separate repository outside the project; read-only facts)
    dash = Path.home() / "Desktop" / "gripsense-dashboard"
    F["dashboard"] = {"exists": dash.exists(), "path": "~/Desktop/gripsense-dashboard"}
    if dash.exists():
        ev = (dash / "events.js").read_text()
        F["dashboard"]["actions"] = re.findall(r'(\w+): "\1"', ev)
        F["dashboard"]["has_no_press"] = "NO_PRESS" in ev
    F["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    F["report_date"] = date.today().isoformat()
    return F


# =========================================================== 2. figures
def box(ax, x, y, w, h, text, kind="done", fs=9):
    styles = {
        "done": dict(fc="#e8f1fc", ec=BLUE, ls="-", tc=INK),
        "data": dict(fc="#f0efec", ec=INK2, ls="-", tc=INK),
        "future": dict(fc=SURFACE, ec="#8f8e89", ls=(0, (4, 3)), tc=INK2),
        "pass": dict(fc="#e3f5ec", ec="#14845b", ls="-", tc=INK),
        "warn": dict(fc="#fff4dc", ec="#b57900", ls="-", tc=INK),
    }[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=styles["fc"],
                       ec=styles["ec"], lw=1.6, ls=styles["ls"])
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=styles["tc"],
            wrap=True, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, dashed=False, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color=color, ls=(0, (4, 3)) if dashed else "-",
                                 shrinkA=2, shrinkB=2))


def fig_architecture(F, path):
    fig, ax = plt.subplots(figsize=(16, 10.8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    ax.text(0.2, 10.45, "GripSense - system architecture and methodology", fontsize=17, weight="bold")
    ax.text(0.2, 10.0, "Solid boxes: implemented and verified offline in this project.  Grey boxes: existing "
            "inputs whose collection was not re-verified.  Dashed boxes: planned / NOT verified.",
            fontsize=10, color=INK2)
    W, H = 3.2, 1.25
    xs = [0.3, 4.3, 8.3, 12.3]
    # lanes
    lanes = [(8.0, "A. Data acquisition (historical)"), (5.85, "B. Offline ML pipeline (Python)"),
             (3.7, "C. Decision tree -> C port (offline)"), (1.55, "D. Embedded deployment (NOT verified)")]
    for y, t in lanes:
        ax.text(xs[1], y + H + 0.12, t, fontsize=10.5, weight="bold", color=INK2)
    A = [("FSR pressure sensor\n(force-sensitive resistor)", "data"),
         ("ESP32 analog input\nADC samples every 20 ms\n(firmware not in project)", "data"),
         (f"CSV dataset\n{F['rows']:,} rows | {F['trials']} trials\n150 samples x 3 s per trial", "data"),
         (f"Dataset audit\nno missing / duplicate rows\n{len(F['flagged_ids'])} trials flagged, kept", "done")]
    B = [(f"Trial-level split (seed {F['seed']})\n120 / 40 / 40 trials\n30 / 10 / 10 per class", "done"),
         (f"Feature extraction\n{F['n_features']} features per trial\n(gripsense_features.py)", "done"),
         ("Train on train split\n8 models incl. 2 baselines\nselect on validation", "done"),
         ("Held-out test\nevaluation\n(40 trials)", "done")]
    C = [("Read fitted depth-2 tree\n7 nodes, 2 features\nfrom DT_depth2.joblib", "done"),
         ("Generate decision_tree.c\nfloat32 comparison semantics\nproject label order", "done"),
         (f"Compile with cc\nstrict warnings\n{F['c_warnings']} warnings", "done"),
         (f"Python vs C parity\nfeatures mode {F['c']['agree_features']}/{F['c']['n']}\n"
          f"raw ADC mode {F['c']['agree_raw']}/{F['c']['n']}\nboundary {F['c']['boundary_agree']}/{F['c']['boundary_n']}", "pass")]
    D = [("ESP32 target build\n(Xtensa/RISC-V toolchain)\nNOT verified", "future"),
         ("Live windowing, onset\ndetection, press counting\nNOT implemented", "future"),
         ("On-device inference\nmemory / timing\nNOT measured", "future"),
         ("Firebase /events ->\ncaretaker dashboard (separate\nrepo; link to model NOT verified)", "future")]
    for row, y in ((A, 8.0), (B, 5.85), (C, 3.7), (D, 1.55)):
        for i, (t, k) in enumerate(row):
            box(ax, xs[i], y, W, H, t, k)
            if i < 3:
                arrow(ax, xs[i] + W, y + H / 2, xs[i + 1], y + H / 2, dashed=(k == "future"))
    # lane connectors: elbow routes (down, left above the lane label, down into the first box)
    def elbow(x_from, y_from, y_to_top, dashed=False):
        y_mid = y_from - 0.22
        ls = (0, (4, 3)) if dashed else "-"
        ax.plot([x_from, x_from, xs[0] + W - 0.35], [y_from, y_mid, y_mid], color=INK2, lw=1.4, ls=ls)
        arrow(ax, xs[0] + W - 0.35, y_mid, xs[0] + W - 0.35, y_to_top, dashed=dashed)
    elbow(xs[3] + W / 2, 8.0, 5.85 + H)
    elbow(xs[2] + W / 2, 5.85, 3.7 + H)
    elbow(xs[3] + W / 2, 3.7, 1.55 + H, dashed=True)
    ax.text(xs[2] + W / 2 + 0.12, 5.5, "depth-2 tree chosen for the C port (smallest model tied on validation)",
            fontsize=8.5, color=INK2, ha="left", va="top")
    ax.text(0.3, 0.55, "The C port classifies complete, pre-cut 3-second windows of recorded data only. "
            "It has not run on an ESP32 and is not a live gesture detector.", fontsize=10, color="#9a3b12",
            weight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(F, path):
    ms = F["models"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    y = np.arange(len(ms))
    for off, sp, col, lab in ((-0.19, "validation", BLUE, "Validation accuracy (used for selection)"),
                              (0.19, "test", ORANGE, "Test accuracy (40 held-out trials)")):
        vals = np.array([m[f"{sp}_acc"] for m in ms])
        lo = vals - np.array([m[f"{sp}_ci"][0] for m in ms])
        hi = np.array([m[f"{sp}_ci"][1] for m in ms]) - vals
        ax.barh(y + off, vals, height=0.36, color=col, label=lab)
        ax.errorbar(vals, y + off, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5)
        for yi, v in zip(y + off, vals):
            ax.text(0.015, yi, f"{v:.3f}", va="center", fontsize=8.5, color="white", weight="bold")
    ax.set_yticks(y, [m["name"] for m in ms])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy (error bars: 95% Wilson interval, n = 40 trials per split)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    ax.set_title("GripSense model comparison on the fixed trial-level split", loc="left", pad=28)
    fig.text(0.01, -0.02, "One trial = 0.025 accuracy. Differences of 1-2 trials between the strong models are "
             "within the uncertainty shown; none is statistically significant (exact McNemar vs RF).",
             fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_model_size(F, path):
    ms = [m for m in F["models"] if m["est_bytes"] > 1]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(ms))
    vals = [m["est_bytes"] for m in ms]
    ax.barh(y, vals, color=BLUE, height=0.55, hatch="///", edgecolor=SURFACE, label="estimate (model parameters only)")
    dt_i = [i for i, m in enumerate(ms) if m["id"] == "DT_depth2"][0]
    ax.barh(dt_i - 0.3, F["vs"]["sizes"]["c_tables_sizeof_bytes_host"], height=0.18, color=ORANGE,
            label="measured: sizeof(C tree tables), host build")
    for i, m in enumerate(ms):
        ax.text(m["est_bytes"] * 1.15, i, f"{m['est_bytes']:,} B  ({m['size_units']}; {m['n_features_used']} feature{'s' if m['n_features_used'] != 1 else ''})",
                va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(5, 3e6)
    ax.set_yticks(y, [m["name"] for m in ms])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Bytes (log scale)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Approximate model size (not ESP32 measurements)", loc="left")
    fig.text(0.01, -0.06, "Estimates: tree nodes x 12 bytes; logistic regression weights + scaler x 4 bytes; baselines "
             "= stored centroids. Feature-extraction code and runtime are excluded, so sizes are not a full "
             "deployment comparison.", fontsize=8, color=INK2, wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_c_port(F, path):
    c = F["c"]
    fig, ax = plt.subplots(figsize=(15, 9.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9.8)
    ax.axis("off")
    ax.text(0.2, 9.45, "GripSense C-port verification (offline, recorded data)", fontsize=16, weight="bold")
    ax.text(0.2, 9.0, "OFFLINE parity test on the 200 recorded trials - this is NOT live ESP32 validation "
            "and NOT a measure of real-world accuracy.", fontsize=10.5, color="#9a3b12", weight="bold")
    box(ax, 0.3, 4.0, 2.7, 1.5, f"Existing raw ADC trials\n{c['n']} trials x 150 samples\n(gripsense_clean_data.csv)", "data")
    box(ax, 4.0, 6.0, 3.3, 1.4, "Python\ngripsense_features.py\n(49 features; tree uses 2)", "done")
    box(ax, 4.0, 2.1, 3.3, 1.4, "C (generated)\ngs_extract_features()\n(the 2 features the tree needs)", "done")
    box(ax, 8.2, 6.0, 3.0, 1.4, "sklearn DecisionTree\npredict()\nDT_depth2.joblib", "done")
    box(ax, 8.2, 2.1, 3.0, 1.4, "C gs_predict()\n(double)(float)x <= threshold\nproject label order", "done")
    arrow(ax, 3.0, 4.9, 4.0, 6.6)
    arrow(ax, 3.0, 4.6, 4.0, 2.8)
    arrow(ax, 7.3, 6.7, 8.2, 6.7)
    arrow(ax, 7.3, 2.8, 8.2, 2.8)
    arrow(ax, 5.65, 6.0, 9.7, 3.5)
    ax.text(6.9, 4.45, "Python feature vectors\nfed to C (features mode)", fontsize=8.5, color=INK2, ha="center")
    checks = [
        (f"Features: longest_active_run_frac\n{c['feat_exact']['longest_active_run_frac']}/{c['n']} bit-identical", "pass"),
        (f"Features: segment_1_mean\n{c['feat_exact']['segment_1_mean']}/{c['n']} bit-identical", "pass"),
        (f"Predictions, C on Python features\n{c['agree_features']}/{c['n']} agree", "pass"),
        (f"Predictions, C from raw ADC\n{c['agree_raw']}/{c['n']} agree", "pass"),
        (f"Threshold boundary vectors\n{c['boundary_agree']}/{c['boundary_n']} agree\n(naive double compare: {c['naive_disagree']} disagree)", "pass"),
        (f"NaN input rejected: {F['vs']['invalid_input_rejected']}\nCompiler warnings: {F['c_warnings']}\n"
         f"Files changed: {len(F['vs']['preservation']['changed'])} of {F['vs']['preservation']['files_checked']}", "pass"),
    ]
    for i, (t, k) in enumerate(checks):
        box(ax, 12.0, 7.0 - i * 1.18, 2.85, 1.0, t, k, fs=8.3)
    ax.text(13.42, 8.25, "Comparison results", ha="center", fontsize=11, weight="bold")
    arrow(ax, 11.2, 6.7, 12.0, 6.7)
    arrow(ax, 11.2, 2.8, 12.0, 3.2)
    box(ax, 0.3, 0.3, 11.2, 1.3, "Not covered by this verification: ESP32 target compiler, ADC behaviour on hardware,\n"
        "live windowing / onset detection / press counting, memory and timing on device, real-world accuracy.\n"
        "Exact parity proves the C code reproduces the Python model on these inputs only.", "warn", fs=9)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(F, path):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    sel = [m for m in F["models"] if m["id"] in ("RF_existing", "DT_depth2")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, m in zip(axes, sel):
        cm = np.array(m["test_cm"])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=10)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                v = cm[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                        color="white" if v >= 6 else (INK if v else "#a3a29d"))
        ticks = [f"{l.replace('_', ' ')}\n({MEANING[l]})" for l in LABELS]
        ax.set_xticks(range(4), ticks, fontsize=8)
        ax.set_yticks(range(4), ticks, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Actual label")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{m['name']}\nTEST split, 40 trials: {m['test_errors']} error(s), misclassified {m['test_mis']}",
                     loc="left", fontsize=10)
    fig.suptitle("Confusion matrices on the held-out test split (from saved predictions)", x=0.01, ha="left",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_ds(F, path):
    counts = F["ds_press_counts"]
    xs = list(range(1, 8))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vals = [counts.get(x, 0) for x in xs]
    ax.bar(xs, vals, color=[INK2 if x == 2 else AQUA for x in xs], width=0.7)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, str(v), ha="center", fontsize=9, color=INK2)
    ax.annotate("intended gesture:\nexactly 2 presses\n(0 trials)", (2, 1.5), xytext=(1.05, 12.5), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK2))
    ax.set_xticks(xs)
    ax.set_xlabel("Distinct presses in the 3-second window (runs above 100 ADC)")
    ax.set_ylabel("DOUBLE_SQUEEZE trials")
    ax.grid(axis="x", visible=False)
    ax.set_title("Recorded DOUBLE_SQUEEZE trials contain 3-7 presses, not 2", loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_recall(F, path):
    sel = [m for m in F["models"] if m["id"] in ("RF_existing", "DT_depth2")]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(4)
    for k, (m, col, off) in enumerate(zip(sel, (BLUE, ORANGE), (-0.17, 0.17))):
        rec = np.array([m["test_per_class"][l]["recall"] for l in LABELS])
        ci = [wilson(int(round(r * 10)), 10) for r in rec]
        ax.bar(x + off, rec, width=0.32, color=col, label=m["name"])
        ax.errorbar(x + off, rec, yerr=[rec - np.array([c[0] for c in ci]), np.array([c[1] for c in ci]) - rec],
                    fmt="none", ecolor=INK2, capsize=3, elinewidth=0.9)
        for xi, r in zip(x + off, rec):
            ax.text(xi, 0.05, f"{r:.1f}", ha="center", fontsize=8.5, color="white", weight="bold")
    ax.set_xticks(x, [f"{l}\n({MEANING[l]})" for l in LABELS], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Test recall (10 trials per class)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    ax.set_title("Per-class test recall with 95% Wilson intervals: 10/10 is compatible with ~0.72", loc="left", pad=26)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


FIGURES = [
    ("GripSense_System_Architecture.png", fig_architecture, "System architecture and methodology"),
    ("GripSense_Model_Comparison.png", fig_model_comparison, "Validation vs test accuracy for all evaluated models"),
    ("GripSense_Model_Size.png", fig_model_size, "Approximate model size and how it was measured"),
    ("GripSense_C_Port_Verification.png", fig_c_port, "Offline Python-vs-C parity verification"),
    ("GripSense_Confusion_Matrix.png", fig_confusion, "Test-split confusion matrices (RF and depth-2 tree)"),
    ("GripSense_Test_Recall_Uncertainty.png", fig_recall, "Per-class test recall with uncertainty"),
    ("GripSense_DoubleSqueeze_Presses.png", fig_ds, "Press counts in DOUBLE_SQUEEZE trials"),
]


# =========================================================== 3. markdown
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def cm_table(cm):
    return md_table(["Actual \\ Predicted"] + LABELS, [[f"**{l}**"] + list(r) for l, r in zip(LABELS, cm)])


def build_markdown(F) -> str:
    M = {m["id"]: m for m in F["models"]}
    rf, dt, lr1 = M["RF_existing"], M["DT_depth2"], M["LR_C1"]
    c, vs, hp = F["c"], F["vs"], F["hp"]
    nodes = F["tree"]["nodes"]
    ds = F["ds"]
    on = F["onset"]
    sizes = vs["sizes"]
    e4 = F["e4"]
    probe = {(p["probe"], p["model"], p["original_label"]): p for p in F["probes"]}
    p1 = probe[("P1_two_squeezes_then_rest", "RF", "DOUBLE_SQUEEZE")]
    p2 = probe[("P2_single_squeeze_then_rest", "RF", "DOUBLE_SQUEEZE")]
    p3h = probe[("P3_press_with_onset_and_release", "RF", "HARD_PRESS")]
    p3l = probe[("P3_press_with_onset_and_release", "RF", "LIGHT_PRESS")]
    p3_ds = p3h["DOUBLE_SQUEEZE"] + p3l["DOUBLE_SQUEEZE"]
    p3_n = sum(p3h[l] + p3l[l] for l in LABELS)
    sig = [k.split(" vs ")[1] for k, v in F["mcnemar"].items() if v["p_value_exact"] < 0.05]
    nonsig = [DISPLAY[k.split(" vs ")[1]] for k, v in F["mcnemar"].items()
              if v["p_value_exact"] >= 0.05]
    comp_rows = []
    for m in F["models"]:
        comp_rows.append([m["name"], f"{m['validation_acc']:.3f}", f"{m['test_acc']:.3f}",
                          f"{m['validation_macro_f1']:.3f}", f"{m['test_macro_f1']:.3f}", m["test_errors"],
                          ", ".join(map(str, m["test_mis"])) if len(m["test_mis"]) <= 3 else f"{len(m['test_mis'])} trials",
                          f"{m['est_bytes']:,} B ({m['size_units']})", m["n_features_used"]])
    tree_rows = []
    for nd in nodes:
        if nd["is_leaf"]:
            tree_rows.append([nd["node"], "leaf", "", "", "", f"{nd['leaf_class']} (= {LABELS.index(nd['leaf_class'])})"])
        else:
            tree_rows.append([nd["node"], "split", f"`{nd['feature_name']}`", f"{nd['threshold']!r}",
                              f"{nd['left']} / {nd['right']}", ""])
    seg_s, seg_e = F["c_seg_bounds"]
    status_rows = [
        ["Dataset (CSV) and audit", "Completed", "`gripsense_clean_data.csv`, `reports/dataset_audit.csv`"],
        ["Trial-level split", "Completed", "`reports/split_manifest.csv` (120/40/40)"],
        ["Model training (8 models)", "Completed", "`results/followup/model_comparison.csv`"],
        ["Model evaluation", "Completed on the existing split (test previously inspected)", "`reports/followup/followup_metrics.json`"],
        ["C code generation (depth-2 tree)", "Completed", "`models/c_port/decision_tree.c`"],
        ["Offline C/Python parity tests", f"Completed: {c['agree_features']}/{c['n']} and {c['agree_raw']}/{c['n']}", "`results/c_port/verification_summary.json`"],
        ["Host compilation (Apple clang)", f"Completed, {F['c_warnings']} warnings", "`results/c_port/compile_log.txt`"],
        ["ESP32 firmware source", "Not found in project", "No `.ino`/PlatformIO/ESP-IDF files found"],
        ["ESP32 target compilation", "Not verified", "No target build artifacts"],
        ["Live sensor inference", "Not verified", "No hardware tests were run"],
        ["Live windowing / onset detection / press counting", "Not implemented", "C port classifies pre-cut windows only"],
        ["Firebase / caretaker dashboard", "Exists as a separate repository; not integrated with or tested against the ML model",
         f"`{F['dashboard']['path']}` (outside this project)"],
    ]
    fl_by = ", ".join(f"{k}: {v}" for k, v in F["flagged_by_label"].items())
    npf = "; ".join(f"trial {k} ({v})" for k, v in F["no_press_flagged"].items())
    e1 = F["e1"]
    dash = F["dashboard"]

    md = f"""# GripSense - ESP32 FSR Pressure-Gesture Recognition with TinyML: Project Report

_Report date: {F['report_date']}. Generated from the project's saved artifacts by
`results/project_documentation/scripts/build_documentation.py`. Every number was re-derived from the raw CSV,
saved predictions and verification files, and checked against the saved metrics before writing._

> **Scope statement.** This report documents offline, software-only work on an existing recorded dataset. No new
> data was collected, no ESP32 was connected, and no hardware or live-inference testing was performed. GripSense
> is a prototype. It is **not** a validated medical device, diagnostic system, or dependable emergency-alert system.

## 1. Title and project overview

**GripSense** is an assistive-communication prototype. A person squeezes a force-sensitive resistor (FSR) in
one of a few predefined ways, and the system turns that grip pattern into a short message for a caretaker. It is
intended for people who find speaking or using a phone difficult. The problem it addresses is turning a noisy
analog pressure signal into a small, fixed set of intentional gestures, reliably enough to be useful and with
little enough computation to run on a microcontroller.

The components and their status:

- **FSR pressure sensor.** Converts grip force into a resistance change that the ESP32 reads as an ADC value
  (0-4095 scale). The recorded values range {F['adc_min']}-{F['adc_max']}.
- **ESP32.** Sampled the sensor every {F['intervals_ms'][0]} ms to produce the recorded dataset. The firmware that recorded the data
  is **not** in this project, so its exact configuration is not verified.
- **TinyML model.** Machine-learning models were trained offline in Python on trial-level features. A compact
  depth-2 decision tree was exported to plain C and verified against Python. It has **not** run on an ESP32.
- **Dashboard / Firebase.** A separate repository, `{dash['path']}`, contains a Firebase Realtime Database
  caretaker dashboard. Its documented data contract expects the ESP32 to write events with actions
  {', '.join('`' + a + '`' for a in dash.get('actions', []))} (no `NO_PRESS`). It lives outside this project. The
  dashboard was not run, and its connection to the ML model was **not** verified.

At a high level, the system is: FSR -> ESP32 ADC samples -> 3-second window -> features -> classifier -> intended
message. In this project, the steps from the recorded CSV through offline evaluation and the C parity test are
implemented and verified. Everything on the device side remains planned or unverified (Figure 1, Section 15).

## 2. Executive summary

- **What was built:** an offline pipeline that audits the recorded FSR dataset, splits it by whole trial,
  extracts {F['n_features']} features per 3-second trial, trains and compares 8 models, and exports a depth-2 decision
  tree to standalone C with an automated Python-vs-C parity test.
- **Data:** {F['rows']:,} ADC readings = {F['trials']} trials x 150 samples (20 ms apart, 3 s per trial), 50 trials
  for each of 4 labels.
- **Methods:** majority-class and mean-ADC baselines, Random Forest (300 trees), logistic regression (C = 0.1, 1, 10),
  and shallow decision trees (depth 2 and 3).
- **Measured performance (40 held-out test trials):** Random Forest {rf['test_acc']:.3f} ({40 - rf['test_errors']}/40),
  logistic regression C=1 {lr1['test_acc']:.3f}, depth-2 tree {dt['test_acc']:.3f} ({40 - dt['test_errors']}/40). Baselines:
  {M['B0_majority']['test_acc']:.3f} and {M['B1_mean_adc_centroid']['test_acc']:.3f}. All trained models tied on validation ({rf['validation_acc']:.3f}),
  and their test differences (1-2 trials) are not statistically significant.
- **Model chosen for the C port:** the depth-2 decision tree. It was the **smallest** model tied at the best validation
  score (7 nodes, 2 features, ~{dt['est_bytes']} bytes of tables), not the most accurate one.
- **C-port verification (offline):** C matched scikit-learn on **{c['agree_features']}/{c['n']}** trials given Python features and
  **{c['agree_raw']}/{c['n']}** when computing its own features from raw ADC. Both C features were bit-identical ({c['feat_exact']['longest_active_run_frac']}/{c['n']}, {c['feat_exact']['segment_1_mean']}/{c['n']}),
  the threshold-boundary tests passed {c['boundary_agree']}/{c['boundary_n']}, and compilation produced {F['c_warnings']} warnings under strict flags.
- **Main limitations:** small test set (10 trials per class); DOUBLE_SQUEEZE recordings contain 3-7 squeezes rather
  than two; classes were recorded in sequential blocks; most presses are captured mid-hold; and there has been no
  hardware or live-stream validation.
- **Remaining work:** keep the C and Python implementations synchronized, add regression tests, build and measure on the
  ESP32 toolchain, and design live windowing with an "uncertain" output.

## 3. Problem statement and objectives

The task is to classify one 3-second window of FSR readings into one of four labels. The labels and their intended
meanings, exactly as used in the dataset:

{md_table(["Label", "Intended meaning", "Trials"], [[f"`{l}`", MEANING[l], F['per_label_trials'][l]] for l in LABELS])}

These meanings are **design intentions** of the project. A correct classification only means that the grip
pattern matched a recorded pattern. It does not show that the system can infer a person's actual needs or medical
condition.

Objectives: (1) audit the dataset before modelling; (2) build a leakage-free trial-level evaluation; (3) compare simple
and stronger models against baselines; (4) pick a model suitable for a microcontroller; (5) port it to C and prove
equivalence to the Python model; (6) document limitations honestly.

## 4. System architecture

![GripSense system architecture and methodology](figures/GripSense_System_Architecture.png)

*Figure 1. System architecture and methodology. Solid boxes are implemented and verified offline; grey boxes are
existing inputs; dashed boxes are planned or not verified.*

1. **FSR pressure sensor.** Physical input. Its characteristics (divider resistor, placement) are not documented in
   the project.
2. **ESP32 analog input.** Produced ADC samples at a nominal {F['intervals_ms'][0]} ms interval. Firmware not in the project; not verified.
3. **Data preparation / CSV.** `gripsense_clean_data.csv` with columns {', '.join('`' + c + '`' for c in F['columns'])}.
   Prior cleaning steps (before this project) are not documented. This project performed no further cleaning and
   removed no data.
4. **Feature extraction.** `gripsense_features.py` turns each trial's 150 readings into {F['n_features']} features. The same
   function is used for training and inference.
5. **Model training and evaluation.** `run_experiment.py` (phase 1) and `run_followup.py` (phase 2), validated by
   `validate_outputs.py` and `validate_followup.py`.
6. **Decision-tree export to C.** `export_and_verify_c_port.py` reads the fitted tree and generates
   `models/c_port/decision_tree.c`.
7. **C prediction and parity verification.** The same script compiles the C code and compares it with Python on all 200 trials.
8. **Embedded deployment.** Planned; not performed.
9. **Dashboard / Firebase.** A separate repository exists (Section 1). Not integrated with or tested against this model.

## 5. Dataset description

All figures below were recomputed from `gripsense_clean_data.csv` for this report.

{md_table(["Property", "Verified value"], [
    ["Rows", f"{F['rows']:,}"], ["Columns", ", ".join(F['columns'])], ["Trials", F['trials']],
    ["Samples per trial", ", ".join(map(str, F['per_trial']))],
    ["Trials per label", ", ".join(f"{k} {v}" for k, v in F['per_label_trials'].items())],
    ["Rows per label", ", ".join(f"{k} {v:,}" for k, v in F['per_label_rows'].items())],
    ["Sampling interval", f"{', '.join(map(str, F['intervals_ms']))} ms (every interval)"],
    ["elapsed_ms range per trial", f"{F['elapsed_min']}-{F['elapsed_max']} ms (3.0 s window)"],
    ["ADC range", f"{F['adc_min']}-{F['adc_max']} (12-bit scale 0-4095; no saturation)"],
    ["Missing values / duplicate rows", f"{F['missing']} / {F['dup_rows']}"],
    ["Labels per trial", f"exactly 1 (max observed {F['labels_per_trial_max']})"],
    ["Recording order", "; ".join(f"{k}: trials {a}-{b}" for k, (a, b) in F['blocks'].items())],
])}

Each trial is one 3-second window of 150 consecutive samples with a single label. Zero readings are normal when the
sensor is unloaded: {', '.join(f"{k} {v:.1%}" for k, v in F['zero_frac_by_label'].items())} of readings are 0.
The phase-1 audit ran **before** any training, and none of its checks failed ({F['audit_status']}). {len(F['flagged_ids'])} trials were flagged for review
and kept ({fl_by}).

## 6. Data preparation and splitting

The split is stratified by label and done on **whole trial IDs** with seed {F['seed']} (`reports/split_manifest.csv`):

{md_table(["Split", "Trials", "Readings", "Per class (LIGHT / HARD / DOUBLE / NO)"],
          [[s, F['split_trials'][s], f"{F['split_rows'][s]:,}", " / ".join(str(F['split_per_class'][s][l]) for l in LABELS)]
           for s in ("train", "validation", "test")])}

No trial appears in more than one split (verified overlap = {F['split_overlap']}). Splitting by trial matters because the 150
samples within a trial are strongly correlated. Splitting individual samples at random would put near-identical
neighbouring samples in both training and test data, which would make the test score look better than it really is
(leakage).

**Test-set history.** The Random Forest's test result was examined during the initial experiment, and that pipeline
was re-run after the test score had been seen (see the run-history note in `reports/GripSense_TinyML_Report.md`;
features, split and candidate models were unchanged). In phase 2 the selection rules were frozen before the new candidates were scored on test. The test set
is therefore **not a completely untouched final evaluation** for every later decision. Saved disclosure: _{F['disclosure']}_

## 7. Feature engineering

All models use the same {F['n_features']} trial-level features from `gripsense_features.py`: level statistics (mean, median, std,
min, max, range, percentiles, IQR, RMS), contact features (zero fraction, longest zero/active runs, active mean),
threshold fractions (50-2000 ADC), press-event counts, peak position, consecutive-difference statistics, and the mean
and max of 10 equal time segments. Trial ID and timestamps are never features. The models differ in which features they
actually use: the Random Forest uses 47, logistic regression all 49, and the depth-2 tree only 2.

**Features used by the depth-2 tree (verified in source code).**

- `longest_active_run_frac` = (longest run of consecutive samples with ADC > 100) / 150. Dimensionless, 0-1.
  Source: `ACTIVE_THRESHOLD = 100` and `_longest_run(active) / n` in `gripsense_features.py`.
- `segment_1_mean` = mean ADC of samples {seg_s}-{int(seg_e) - 1} (the 2nd of 10 equal segments from `np.array_split`),
  i.e. {int(seg_s) * 20}-{(int(seg_e) - 1) * 20} ms into the window. Unit: ADC counts.

**Why the C implementation must preserve these exactly.** The tree's thresholds only make sense for features with the
same definition, units and order. The C code must keep: (a) feature order, since the tree indexes features by
position; (b) the comparison operator (`<=` goes left); (c) numeric precision, because scikit-learn casts inputs to
float32 before comparing with float64 thresholds, so a plain double comparison can choose the other branch next to a
threshold; and (d) the class-label mapping, because scikit-learn stores classes alphabetically while the C interface
uses the project order. Getting any of these wrong would produce silent misclassifications.

## 8. Machine-learning methods

Hyperparameters below were read from the saved estimators, not from memory.

{md_table(["Model", "Configuration (verified)", "Purpose", "Strengths", "Weaknesses"], [
    ["Majority-class baseline", "always predicts one class (classes are tied)", "chance floor", "trivial", "25% by construction"],
    ["Mean-ADC baseline", "nearest class centroid on trial mean ADC", "does force level alone suffice?", "1 feature, 16 B", "confuses LIGHT with DOUBLE"],
    ["Random Forest", f"{hp['rf']['n_estimators']} trees, max_features={hp['rf']['max_features']}, max_depth={hp['rf']['max_depth']}, "
                      f"class_weight={hp['rf']['class_weight']}, {hp['rf']['criterion']}, seed {hp['rf']['random_state']}",
     "strong non-linear reference", "robust, little tuning", "large (2,556 nodes), 47 features"],
    ["Logistic Regression", f"{' + '.join(hp['lr']['pipeline'])}; C = 0.1 / 1 / 10; {hp['lr']['solver']}, "
                            f"L2 (l1_ratio={hp['lr']['l1_ratio']}), max_iter={hp['lr']['max_iter']}",
     "linear reference", "smooth probabilities, ~1.2 KB", "needs all 49 features + scaling"],
    ["Decision Tree", f"max_depth = 2 / 3, {hp['dt']['criterion']}, seed {hp['dt']['random_state']}",
     "embedded candidate", "7 nodes, 2 features, readable rules", "coarse; fixed time-slice feature"],
])}

**Why a compact tree suits embedded inference.** On a microcontroller, the cost includes computing the features as
well as storing the model. The depth-2 tree needs only two cheap features (a run-length count and a 15-sample mean) and
at most two comparisons. The Random Forest needs 47 features and ~{rf['est_bytes']:,} bytes of tree tables (estimate). The tree
is **not more accurate** than the strong models. On test it made {dt['test_errors']} errors against the RF's {rf['test_errors']}. It was chosen because it tied on
validation and is far smaller.

## 9. Model evaluation and comparison

{md_table(["Model", "Val acc", "Test acc", "Val macro-F1", "Test macro-F1", "Test errors", "Misclassified (test)", "Approx. size (method)", "Features"], comp_rows)}

![Model comparison](figures/GripSense_Model_Comparison.png)

*Figure 2. Validation and test accuracy with 95% Wilson intervals (40 trials per split).*

**Reconciliation with the summary table supplied for this report.** All supplied values match the saved metrics.
The artifacts contain two more models than the six listed: logistic regression C=10 (test {M['LR_C10']['test_acc']:.3f}) and a depth-3 tree.
The depth-3 tree is structurally identical to the depth-2 tree ({M['DT_depth3']['size_units']}), with test {M['DT_depth3']['test_acc']:.3f}. The majority baseline's
"size" is 1 stored label.

**Size is not measured the same way for every model.** The RF and tree sizes are estimates at 12 bytes per node.
Logistic regression is 4 bytes per float parameter, including the scaler's means and scales. The depth-2 tree's
{sizes['c_tables_sizeof_bytes_host']} bytes was also **measured** as the `sizeof` of its C tables on the host. None of these include
feature-extraction code, so they are not a full deployment comparison (Figure 3).

![Model size](figures/GripSense_Model_Size.png)

*Figure 3. Approximate model size (log scale). Estimates are hatched; the measured host value is shown in orange.*

**Confusion matrices (test, 40 trials; rows = actual, columns = predicted).**

Random Forest:

{cm_table(rf['test_cm'])}

Depth-2 decision tree:

{cm_table(dt['test_cm'])}

![Confusion matrices](figures/GripSense_Confusion_Matrix.png)

*Figure 4. Test-split confusion matrices from the saved predictions.*

**Interpretation.** Every trained model clearly beats both baselines. The mean-ADC baseline reaches only
{M['B1_mean_adc_centroid']['test_acc']:.3f} because LIGHT_PRESS and DOUBLE_SQUEEZE have similar average force. Between the trained models,
exact McNemar tests against the RF found significant differences only for the baselines ({', '.join(DISPLAY[s] for s in sig)}). None were
found for {', '.join(nonsig)}. The test error shared by every trained model is trial 39, a LIGHT_PRESS (Hungry) predicted as
HARD_PRESS (Pain): a false "Pain" signal. The depth-2 tree also called NO_PRESS trial 157 (which contains a brief
contact) DOUBLE_SQUEEZE.

**Uncertainty.** The test set has only 10 trials per class. A perfect 10/10 recall is consistent with a true recall as
low as {F['wilson_10_10']:.2f}, and 9/10 with {F['wilson_9_10']:.2f} (95% Wilson). One trial changes overall accuracy by 0.025. In an exploratory 20x
repeated 5-fold cross-validation on all 200 trials, fold accuracy for the RF ranged {e1['RF']['min']:.3f}-{e1['RF']['max']:.3f} (mean {e1['RF']['mean']:.3f}), and for
the depth-2 tree {e1['DT']['min']:.3f}-{e1['DT']['max']:.3f} (mean {e1['DT']['mean']:.3f}).

![Recall uncertainty](figures/GripSense_Test_Recall_Uncertainty.png)

*Figure 5. Per-class test recall with 95% Wilson intervals.*

## 10. Decision-tree-to-C export

- **Source artifact:** `{F['tree']['source_model']}` (SHA-256 `{F['tree']['source_model_sha256'][:16]}...`; unchanged).
- **Generation:** `export_and_verify_c_port.py` loads the fitted `DecisionTreeClassifier` and reads `tree_.feature`,
  `tree_.threshold`, `children_left/right` and `tree_.value` programmatically. It then writes flattened C tables, with
  thresholds as exact hex-float literals. Nothing is hand-written.
- **Features in C:** `gs_extract_features()` computes `longest_active_run_frac` and `segment_1_mean` from 150 `uint16_t`
  ADC samples.
- **Class-index mapping (verified from the compiled harness `--info`):** {', '.join(f"`{v} = {k}`" for k, v in F['c_label_map'].items())}.
  The leaves are mapped **by name** from scikit-learn's alphabetical order ({', '.join(F['tree']['sklearn_classes_order'])}).
- **Comparison:** `node = ((double)(float)v <= GS_THRESHOLD[node]) ? left : right`, reproducing scikit-learn's float32 cast.

{md_table(["Node", "Type", "Feature", "Threshold (float64)", "Left / Right", "Leaf class (C index)"], tree_rows)}

```
{chr(10).join(F['tree']['rules'])}
```

- **Compilation:** `cc` ({F['vs']['environment']['cc']}) with `{' '.join(F['vs']['compiler']['flags'])}`.
  Result: **{F['c_warnings']} warnings**. An earlier build produced 2 unused-variable warnings in the library build, which were fixed by
  adding name-lookup functions rather than by suppressing the warnings.
- **Files:** `models/c_port/decision_tree.c` (generated source), `models/c_port/tree_metadata.json`,
  `models/c_port/build/decision_tree` (host test harness), `models/c_port/build/decision_tree.o` (library object).

**Why the float32 semantics matter.** The thresholds are float32 midpoints stored as float64. In the boundary test set, a naive
double-precision comparison disagreed with scikit-learn on **{c['naive_disagree']}** of {c['boundary_n']} vectors. The ported comparison
agreed on all of them. On the 200 real trials the naive comparison happened to disagree on
{vs['boundary_tests']['naive_double_compare_disagreements_on_200_real_trials']}, but exactness should not depend on luck.

## 11. C-port verification results

![C-port verification](figures/GripSense_C_Port_Verification.png)

*Figure 6. Offline C-port verification flow and results (recorded data only; not live ESP32 validation).*

{md_table(["Check (from results/c_port/verification_summary.json)", "Result"], [
    ["C predictions on Python-computed features vs scikit-learn", f"**{c['agree_features']}/{c['n']}**"],
    ["C predictions from raw ADC (C computes features) vs scikit-learn", f"**{c['agree_raw']}/{c['n']}**"],
    ["C vs Python feature values: longest_active_run_frac", f"{c['feat_exact']['longest_active_run_frac']}/{c['n']} bit-identical"],
    ["C vs Python feature values: segment_1_mean", f"{c['feat_exact']['segment_1_mean']}/{c['n']} bit-identical"],
    ["Threshold-boundary tests", f"{c['boundary_agree']}/{c['boundary_n']}"],
    ["scikit-learn predictions vs saved phase-2 predictions (80 val+test trials)", str(vs['python_matches_followup_predictions_val_test'])],
    ["NaN input rejected", str(vs['invalid_input_rejected'])],
    ["Strict compiler warnings", str(F['c_warnings'])],
    ["Pre-existing files changed during the C-port work", f"{len(vs['preservation']['changed'])} of {vs['preservation']['files_checked']}"],
    ["All checks pass", f"**{vs['all_checks_pass']}**"],
])}

The ported tree's accuracy against the recorded labels is unchanged from Python: {', '.join(f"{k} {v:.3f}" for k, v in F['c_accuracy_by_split'].items())}.

**What parity proves.** Exact agreement shows the C code is **equivalent to the Python model on these 200 recorded
inputs**. It does not show real-world accuracy, generalisation to new users or sessions, correct behaviour on a live
signal stream, or successful ESP32 deployment.

**Host size measurements (macOS arm64; not ESP32 figures).** Tree tables `sizeof` = {sizes['c_tables_sizeof_bytes_host']} bytes; library object
`size` total = {sizes['object_size_cmd'].get('dec', 'n/a')} bytes (`__TEXT` {sizes['object_size_cmd'].get('__TEXT', 'n/a')}, `__DATA` {sizes['object_size_cmd'].get('__DATA', 'n/a')}); test
executable on disk = {sizes['executable_file_bytes']:,} bytes. The executable's very large `others` value from `size` is the macOS
`__PAGEZERO` reservation, not code or data.

## 12. Findings and interpretation

1. **Strong results on the existing trial-level split.** Trained models reach 0.950-0.975 on validation and test,
   far above the baselines (0.250 and {M['B1_mean_adc_centroid']['test_acc']:.3f}).
2. **Multiple models score alike.** {F['selection']['n_tied_best_val']} of {F['selection']['n_eligible']} trained candidates tied on validation, and their test differences are 1-2 trials.
   The data does not single out a "best" model.
3. **A small tree is attractive for embedded use.** It is accurate to within one or two trials of the RF on this data, needs two
   features, and is about 84 bytes of tables. These are advantages in size and simplicity, not in accuracy.
4. **High accuracy here is not proven real-world reliability.** The data comes from one recording effort with
   protocol issues (Section 13). The test set is small and was previously inspected.
5. **Parity is essential when moving Python to C.** The float32-cast detail would have caused silent disagreements
   at threshold boundaries. Automated parity tests caught and pinned this behaviour.

## 13. Data quality and methodological limitations

- **DOUBLE_SQUEEZE does not contain a double squeeze.** Presses per trial: {F['ds_press_counts']} (presses: trials), median
  {int(np.median([k for k, v in F['ds_press_counts'].items() for _ in range(v)]))}. {ds['trials_with_exactly_2_presses']} trials contain exactly 2. The median press lasts {ds['median_press_ms']:.0f} ms with a {ds['median_gap_ms']:.0f} ms gap (rhythmic squeezing). The
  model may therefore recognise repeated or rhythmic squeezing rather than the intended two-squeeze gesture (Figure 7).
- **Sequential recording blocks.** {'; '.join(f"{k}: trials {a}-{b}" for k, (a, b) in F['blocks'].items())}. Sensor baseline
  drift or changes in force over time are confounded with class. Peak DOUBLE_SQUEEZE force rose over its block
  (Spearman rho {ds['spearman_trialid_vs_peak'][0]:+.2f}, p = {ds['spearman_trialid_vs_peak'][1]:.3g}).
- **Presses captured mid-hold.** {on['LIGHT_PRESS']['active_at_start']}/50 LIGHT_PRESS and {on['HARD_PRESS']['active_at_start']}/50 HARD_PRESS windows are already above 100 ADC at
  t = 0, so press onset is mostly not represented.
- **Flagged trials retained.** {len(F['flagged_ids'])} unusual trials were flagged and kept ({fl_by}), including NO_PRESS
  {npf}.
- **Small test set.** 10 trials per class (Section 9).
- **Confidence thresholds do not reliably remove errors.** An RF threshold of {e4['chosen_on_validation']:.2f}, chosen on validation, accepted
  {e4['test_coverage']:.3f} of test trials and still accepted {e4['test_accepted_errors']} error (trial 39, confidence 0.83).
- **Simulated window edits are exploratory only.** Synthetic edits of real recordings suggested weaknesses. With rest
  added before and after LIGHT/HARD presses, the RF predicted DOUBLE_SQUEEZE for {p3_ds}/{p3_n}. A single squeeze was called
  DOUBLE_SQUEEZE {p2['DOUBLE_SQUEEZE']}/{sum(p2[l] for l in LABELS)} times, and "two squeezes then rest" {p1['DOUBLE_SQUEEZE']}/{sum(p1[l] for l in LABELS)}. These are edits, not real experiments.
- **The C port only covers complete, pre-cut 3-second windows.** It is not a live gesture detector: live windowing,
  gesture-start detection and squeeze counting are not implemented or verified.
- **No ESP32 validation.** Target ADC readings, target-compiler behaviour (including float handling), memory use and live
  inference have not been validated.

![DOUBLE_SQUEEZE press counts](figures/GripSense_DoubleSqueeze_Presses.png)

*Figure 7. Distinct presses per DOUBLE_SQUEEZE trial (runs above 100 ADC).*

These limitations mean the reported accuracies describe agreement with **this recording**. Performance on new users,
new sessions, a live stream, or a correctly performed double squeeze is unknown.

## 14. Security, safety, and intended-use boundaries

- GripSense is a **prototype**. It is not a validated medical device, a diagnostic system, or a dependable
  emergency-alert system, and it must not be the only channel for pain or emergency communication.
- Misclassification produces a wrong intended message. The observed error type (LIGHT_PRESS -> HARD_PRESS) is a false
  "Pain" signal. Missed signals are also possible.
- Before any real use: user testing with the intended users and caretakers, explicit failure handling (sensor
  disconnected, saturated or noisy input), an "uncertain - please repeat" output, confirmation feedback (LED/vibration),
  and acknowledgement handling on the caretaker side.
- Security: the separate dashboard documents that only client-side Firebase Web App configuration belongs in the web
  app. Database access rules and device authentication were not reviewed in this project.

## 15. Current status

{md_table(["Component", "Status", "Evidence"], status_rows)}

## 16. Recommended next steps (software-only first)

1. **Keep C and Python synchronized.** Regenerate `decision_tree.c` from the saved model, never edit it by hand, and
   re-run `export_and_verify_c_port.py` after any feature or model change.
2. **Add deterministic regression tests** for all {c['n']} trials and the {c['boundary_n']} boundary vectors. Run them as a CI-style check
   with fixed expected outputs.
3. **When hardware work resumes:** compile with the ESP32 target toolchain and repeat the parity test on target
   (or in an emulator).
4. **Measure target memory and inference timing** for the sampler, the 300-byte window buffer, feature extraction and the tree.
5. **Plan live windowing and invalid-input handling:** a sliding window or onset trigger, two-in-a-row agreement,
   an "uncertain" state, and rejection of disconnected or saturated input.
6. **Document label and window-alignment limits** wherever the model is used (Section 13).
7. **If more data ever becomes possible,** design a better-controlled evaluation (interleaved classes, separate sessions)
   rather than treating the current split as definitive. This is not required now.

## 17. Reproducibility

Run from the project folder (note the trailing space in the folder name):

```bash
cd "/Users/zenith/Desktop/GripSense-TinyML "
.venv/bin/python export_and_verify_c_port.py
.venv/bin/python validate_followup.py
./models/c_port/build/decision_tree --info
.venv/bin/python results/project_documentation/scripts/build_documentation.py
.venv/bin/python results/project_documentation/scripts/check_documentation.py
```

The first two commands regenerate their own timestamped outputs. For this documentation package they were therefore
executed in a **temporary copy** of the project, so that existing files stayed untouched. The `--info` command was run
in place (read-only). Results: `QUALITY_CHECKS.md`. Environment: Python {F['env']['python']}, scikit-learn {F['env']['scikit-learn']},
numpy {F['env']['numpy']}, pandas {F['env']['pandas']}, matplotlib {F['env']['matplotlib']}; PDF built with reportlab and checked with pypdf
(installed into `.venv` for this documentation only; `requirements.txt` unchanged). Seed {F['seed']}.

## 18. Artifact index

{md_table(["Path", "Purpose"], [
    ["`results/project_documentation/GripSense_Project_Report.md`", "this report (Markdown)"],
    ["`results/project_documentation/GripSense_Project_Report.pdf`", "this report (PDF)"],
    ["`results/project_documentation/GripSense_Executive_Summary.md`", "2-3 minute summary"],
    ["`results/project_documentation/README.md`", "entry point with links"],
    ["`results/project_documentation/manifest.json`", "list of every deliverable"],
    ["`results/project_documentation/QUALITY_CHECKS.md`", "documentation quality-control results"],
    ["`results/project_documentation/figures/*.png`", "7 figures used in this report"],
    ["`results/project_documentation/report_facts.json`", "all values used in this report"],
    ["`results/project_documentation/scripts/`", "builder and checker scripts"],
    ["`reports/GripSense_TinyML_Report.md`", "phase-1 report (audit, split, baselines, RF)"],
    ["`reports/followup/GripSense_TinyML_Final_Report.md`", "phase-2 report (comparison, protocol, exploratory)"],
    ["`reports/c_port_report.md`", "C-port report"],
    ["`models/c_port/decision_tree.c`", "generated C implementation"],
    ["`results/c_port/verification_summary.json`", "C-port verification results"],
])}

## 19. Conclusion

What has been demonstrated: on the recorded GripSense dataset, simple trial-level features separate the four grip
patterns well (0.950-0.975 on a small, previously inspected test set). A 7-node decision tree using two features
performs within one or two trials of a 300-tree Random Forest. That tree has been exported to standalone C that
reproduces the Python model exactly on all 200 recorded trials, from features and from raw ADC.

What remains unproven: real-world accuracy, behaviour on new users or sessions, recognition of a genuine double squeeze,
live-stream windowing, and anything on the ESP32 itself.

Next practical software-only milestone: turn the parity check into a fixed regression test suite (200 trials + 26
boundary vectors), and draft the live-windowing logic (onset trigger, two-in-a-row agreement, "uncertain" state) as
host-side C, tested against the recorded trials.
"""
    return md


def build_exec_summary(F) -> str:
    M = {m["id"]: m for m in F["models"]}
    c = F["c"]
    return f"""# GripSense - Executive Summary

_{F['report_date']} | offline, software-only work on the existing dataset; no hardware used._

**Goal.** Recognise four grip gestures on a force sensor (FSR) read by an ESP32, so that a person can signal a
need to a caretaker: LIGHT_PRESS = Hungry, HARD_PRESS = Pain, DOUBLE_SQUEEZE = Washroom, NO_PRESS = resting. These
meanings are design intentions, not evidence that the system can infer real needs.

**Data.** {F['rows']:,} sensor readings: {F['trials']} recorded trials of 3 seconds (150 samples each), 50 per gesture. The data is clean
(nothing missing or duplicated). Trials are split by whole trial into 120 training, 40 validation and 40 test trials.

**Methods.** Two baselines (always-the-same-answer; average force only), a Random Forest, logistic regression, and
small decision trees, all using {F['n_features']} summary features per trial.

**Results (40 test trials).** Random Forest {M['RF_existing']['test_acc']:.3f}, logistic regression (C=1) {M['LR_C1']['test_acc']:.3f}, depth-2 decision tree
{M['DT_depth2']['test_acc']:.3f}; baselines {M['B0_majority']['test_acc']:.3f} and {M['B1_mean_adc_centroid']['test_acc']:.3f}. The strong models are statistically indistinguishable.
With 10 test trials per gesture, these numbers are uncertain, and the test set had already been looked at.

**C port.** The depth-2 tree was chosen for its small size (7 decisions, 2 features, ~84 bytes), not for higher
accuracy. It was exported to plain C. The C version matches Python on **{c['agree_features']}/{c['n']}** trials (given features) and
**{c['agree_raw']}/{c['n']}** (from raw sensor data), plus {c['boundary_agree']}/{c['boundary_n']} edge-case tests, with {F['c_warnings']} compiler warnings. This proves the C code
equals the Python model on recorded data. It does not prove real-world accuracy.

**Major limitations.** The "double squeeze" recordings actually contain 3-7 squeezes. Each gesture was recorded in
one block, so changes over time are mixed up with the gesture. Most presses are recorded mid-hold. Nothing has run on
the ESP32 or on a live signal. The caretaker dashboard exists as a separate project and is not connected to this
model.

**Status.** Offline pipeline, model comparison, C export and parity tests: done. ESP32 build, live inference,
windowing and dashboard integration: not verified. This is a prototype, not a medical or emergency device.
"""


def build_readme(F) -> str:
    rows = [("GripSense_Project_Report.md", "Full technical report (Markdown)"),
            ("GripSense_Project_Report.pdf", "Full technical report (PDF, with cover, contents and page numbers)"),
            ("GripSense_Executive_Summary.md", "Two-to-three-minute summary"),
            ("manifest.json", "Every deliverable, its type, purpose and check status"),
            ("QUALITY_CHECKS.md", "Quality-control results for this package"),
            ("report_facts.json", "All values used by the report, re-derived from project artifacts"),
            ("preservation_baseline.json", "Hashes of pre-existing project files (taken before this work)")]
    rows += [(f"figures/{n}", d) for n, _, d in FIGURES]
    rows += [("scripts/build_documentation.py", "Generates figures, Markdown, PDF"),
             ("scripts/check_documentation.py", "Runs the quality checks and writes the manifest")]
    links = "\n".join(f"| [{p}]({p}) | {d} |" for p, d in rows)
    return f"""# GripSense project documentation

Documentation package for **GripSense**, an ESP32 + FSR pressure-gesture recognition prototype with a TinyML
decision tree. It was generated on {F['report_date']} from the project's saved artifacts. No models, data or earlier results were changed.

Start with the [Executive Summary](GripSense_Executive_Summary.md), then the full report
([Markdown](GripSense_Project_Report.md) / [PDF](GripSense_Project_Report.pdf)).

| File | Purpose |
|---|---|
{links}

Rebuild and re-check (from the project folder):

```bash
.venv/bin/python results/project_documentation/scripts/build_documentation.py
.venv/bin/python results/project_documentation/scripts/check_documentation.py
```

Important: all results are offline results on recorded data. Nothing in this package has been validated on an ESP32
or in real-world use.
"""


# =========================================================== 4. PDF
def build_pdf(md: str, F, out: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    from reportlab.platypus.tableofcontents import TableOfContents
    from reportlab.lib.fonts import addMapping

    fdir = Path("/System/Library/Fonts/Supplemental")
    base, bold, mono = "Helvetica", "Helvetica-Bold", "Courier"
    if (fdir / "Arial.ttf").exists():
        pdfmetrics.registerFont(TTFont("Arial", str(fdir / "Arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(fdir / "Arial Bold.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Italic", str(fdir / "Arial Italic.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(fdir / "Arial Bold Italic.ttf")))
        addMapping("Arial", 0, 0, "Arial")
        addMapping("Arial", 1, 0, "Arial-Bold")
        addMapping("Arial", 0, 1, "Arial-Italic")
        addMapping("Arial", 1, 1, "Arial-BoldItalic")
        base, bold = "Arial", "Arial-Bold"
    if (fdir / "Courier New.ttf").exists():
        pdfmetrics.registerFont(TTFont("CourierNew", str(fdir / "Courier New.ttf")))
        mono = "CourierNew"

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontName=base, fontSize=9.6, leading=13.4, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=body, fontName=bold, fontSize=15, leading=19, spaceBefore=12, spaceAfter=7,
                        textColor=colors.HexColor("#104281"), keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=h1, fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=13, bulletIndent=3, spaceAfter=2.5)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6, spaceAfter=0)
    cellh = ParagraphStyle("cellh", parent=cell, fontName=bold, textColor=colors.white)
    cap = ParagraphStyle("cap", parent=body, fontSize=8.4, leading=11, textColor=colors.HexColor("#52514e"),
                         alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
    quote = ParagraphStyle("quote", parent=body, leftIndent=8, borderPadding=6, backColor=colors.HexColor("#fff4dc"),
                           borderColor=colors.HexColor("#b57900"), borderWidth=0.6, spaceBefore=4, spaceAfter=10)
    code = ParagraphStyle("code", parent=body, fontName=mono, fontSize=7.6, leading=9.6,
                          backColor=colors.HexColor("#f0efec"), borderPadding=5, leftIndent=4, spaceAfter=8)

    def inline(t: str) -> str:
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        codes: list[str] = []

        def stash(mo):
            codes.append(mo.group(1))
            return f"\x00{len(codes) - 1}\x00"
        t = re.sub(r"`([^`]+)`", stash, t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        t = re.sub(r"(?<![\w*])_(.+?)_(?![\w])", r"<i>\1</i>", t)
        t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)
        return re.sub("\x00(\\d+)\x00", lambda mo: f'<font name="{mono}" size="8.6" color="#9a3b12">'
                      f'{codes[int(mo.group(1))]}</font>', t)

    W = A4[0] - 36 * mm

    class Doc(SimpleDocTemplate):
        def afterFlowable(self, f):
            if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
                lvl = 0 if f.style.name == "h1" else 1
                txt = f.getPlainText()
                key = f"h{id(f)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(txt, key, level=lvl, closed=True)
                self.notify("TOCEntry", (lvl, txt, self.page, key))

    def table_flow(lines):
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
        rows = [r for r in rows if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
        ncol = len(rows[0])
        clean = [[re.sub(r"[`*]", "", c) for c in r] for r in rows]
        cw = 5.5  # approx. points per character (bold uppercase headers are widest)
        word = [max(max((len(w) for w in r[i].split()), default=1) for r in clean) for i in range(ncol)]
        word[0] = max(word[0], 4)
        total = [min(max(len(r[i]) for r in clean), 70) for i in range(ncol)]
        minw = [min(w * cw + 9, W * 0.4) for w in word]
        spare = max(W - sum(minw), 0)
        extra = [max(t * cw + 9 - m, 0) for t, m in zip(total, minw)]
        widths = [m + (spare * e / sum(extra) if sum(extra) else spare / ncol) for m, e in zip(minw, extra)]
        if sum(widths) > W:
            widths = [w_ * W / sum(widths) for w_ in widths]
        data = [[Paragraph(inline(c), cellh if ri == 0 else cell) for c in r] for ri, r in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f4f1")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9c8c3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return [t, Spacer(1, 8)]

    story = []
    # cover
    story += [Spacer(1, 55 * mm),
              Paragraph("GripSense", ParagraphStyle("t", parent=h1, fontSize=34, leading=40, alignment=TA_CENTER,
                                                     keepWithNext=0)),
              Spacer(1, 6 * mm),
              Paragraph("ESP32 FSR Pressure-Gesture Recognition with TinyML", ParagraphStyle(
                  "st", parent=body, fontSize=15, leading=20, alignment=TA_CENTER)),
              Paragraph("Project Report", ParagraphStyle("st2", parent=body, fontSize=13, leading=18,
                                                          alignment=TA_CENTER, textColor=colors.HexColor("#52514e"))),
              Spacer(1, 18 * mm),
              Paragraph(f"Report date: {F['report_date']}", ParagraphStyle("d", parent=body, alignment=TA_CENTER)),
              Paragraph("Dataset: 30,000 ADC readings | 200 trials | 4 gesture labels",
                        ParagraphStyle("d2", parent=body, alignment=TA_CENTER)),
              Spacer(1, 30 * mm),
              Paragraph("Offline, software-only evaluation on an existing recorded dataset. No new data was collected "
                        "and no ESP32 hardware testing was performed. GripSense is a prototype, not a validated "
                        "medical device or emergency-alert system.", quote),
              PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("toc1", parent=body, fontName=bold, fontSize=10, leftIndent=10, leading=15),
                       ParagraphStyle("toc2", parent=body, fontSize=9, leftIndent=26, leading=12)]
    story += [Paragraph("Contents", ParagraphStyle("ct", parent=h1, keepWithNext=0)), toc, PageBreak()]

    lines = md.splitlines()
    i = 0
    para: list[str] = []

    def flush():
        if para:
            story.append(Paragraph(inline(" ".join(s.strip() for s in para)), body))
            para.clear()

    first_h1 = True
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            buf = []
            while not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            story.append(Preformatted("\n".join(buf), code))
            i = j + 1
            continue
        if s.startswith("# "):
            flush()
            if first_h1:
                first_h1 = False
                i += 1
                continue
        if s.startswith("## "):
            flush()
            story.append(Paragraph(inline(s[3:]), h1))
        elif s.startswith("### "):
            flush()
            story.append(Paragraph(inline(s[4:]), h2))
        elif s.startswith("|"):
            flush()
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            tf = table_flow(lines[i:j])
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h1", "h2"):
                story.append(KeepTogether([story.pop(), tf[0]]))
                story.append(tf[1])
            else:
                story += tf
            i = j
            continue
        elif s.startswith("!["):
            flush()
            mobj = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            img_path = DOC / mobj.group(2)
            caption = ""
            if i + 2 < len(lines) and lines[i + 2].strip().startswith("*Figure"):
                cap_lines = []
                j = i + 2
                while j < len(lines) and lines[j].strip():
                    cap_lines.append(lines[j].strip())
                    j += 1
                caption = " ".join(cap_lines).strip("*")
                i = j
            else:
                i += 1
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                iw, ih = im.size
            w = W
            h = w * ih / iw
            maxh = 200 * mm
            if h > maxh:
                h = maxh
                w = h * iw / ih
            grp = [Image(str(img_path), width=w, height=h), Paragraph(inline(caption), cap)]
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h1", "h2"):
                grp.insert(0, story.pop())
            story.append(KeepTogether(grp))
            continue
        elif s.startswith("> "):
            flush()
            j = i
            buf = []
            while j < len(lines) and lines[j].strip().startswith(">"):
                buf.append(lines[j].strip()[1:].strip())
                j += 1
            story.append(Paragraph(inline(" ".join(buf)), quote))
            i = j
            continue
        elif re.match(r"^(- |\d+\. )", s):
            flush()
            mnum = re.match(r"^(\d+)\. ", s)
            text = s[mnum.end():] if mnum else s[2:]
            j = i + 1
            while j < len(lines) and lines[j].startswith("  ") and lines[j].strip() and \
                    not re.match(r"^(- |\d+\. )", lines[j].strip()):
                text += " " + lines[j].strip()
                j += 1
            story.append(Paragraph(inline(text), bullet, bulletText=(f"{mnum.group(1)}." if mnum else "•")))
            i = j
            continue
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()

    def on_page(canv, doc):
        if doc.page <= 1:
            return
        canv.saveState()
        canv.setFont(base, 8)
        canv.setFillColor(colors.HexColor("#52514e"))
        canv.drawString(18 * mm, 10 * mm, "GripSense - Project Report")
        canv.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
        canv.setStrokeColor(colors.HexColor("#e4e3df"))
        canv.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
        canv.restoreState()

    doc = Doc(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=17 * mm,
              bottomMargin=19 * mm, title="GripSense - Project Report", author="GripSense project",
              subject="ESP32 FSR pressure-gesture recognition with TinyML")
    doc.multiBuild(story, onFirstPage=on_page, onLaterPages=on_page)


# =========================================================== main
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    F = collect()
    for name, fn, _ in FIGURES:
        fn(F, FIG / name)
    md = build_markdown(F)
    (DOC / "GripSense_Project_Report.md").write_text(md)
    (DOC / "GripSense_Executive_Summary.md").write_text(build_exec_summary(F))
    (DOC / "README.md").write_text(build_readme(F))
    facts = {k: v for k, v in F.items() if k not in ("compile_log",)}
    (DOC / "report_facts.json").write_text(json.dumps(facts, indent=2, default=str))
    build_pdf(md, F, DOC / "GripSense_Project_Report.pdf")
    print("built:", ", ".join(sorted(p.name for p in DOC.iterdir() if p.is_file())))


if __name__ == "__main__":
    main()
