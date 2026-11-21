#!/usr/bin/env python3
"""
Validate the phase-2 follow-up outputs and confirm phase-1 artifacts are intact.

1. Runs the phase-1 validator (validate_outputs.py) and stores its output in reports/followup/.
2. Re-hashes every pre-existing file against reports/followup/preservation_baseline.json.
3. Recomputes follow-up metrics from saved predictions, reloads saved models, checks the
   selection freeze, leakage guards, and that the report quotes the measured numbers.

    .venv/bin/python validate_followup.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from gripsense_features import LABELS, build_feature_table

ROOT = Path(__file__).resolve().parent
REP, RES, MDL = ROOT / "reports" / "followup", ROOT / "results" / "followup", ROOT / "models" / "followup"
results: list[dict] = []


def check(name, ok, detail=""):
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    # ---- 1. phase-1 validator
    proc = subprocess.run([sys.executable, str(ROOT / "validate_outputs.py")], cwd=ROOT,
                          capture_output=True, text=True)
    (REP / "phase1_validator_output.txt").write_text(proc.stdout + proc.stderr)
    last = (proc.stdout.strip().splitlines() or [""])[-1]
    check("phase-1 validator passes", proc.returncode == 0, last)

    # ---- 2. preservation
    base = json.loads((REP / "preservation_baseline.json").read_text())
    changed = [f for f, h in base["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    check(f"all {len(base['files'])} pre-existing files unchanged", not changed, changed)

    # ---- 3. follow-up artifacts
    m = json.loads((REP / "followup_metrics.json").read_text())
    for name in ("GripSense_TinyML_Final_Report.md", "followup_metrics.json"):
        check(f"exists reports/followup/{name}", (REP / name).exists())
    for name in ("model_comparison.csv", "predictions_all_models.csv", "selection_freeze.json",
                 "press_segments_per_trial.csv", "flagged_trials_review.csv",
                 "exploratory_perturbations.csv", "exploratory_probes.csv",
                 "exploratory_probe_summary.csv", "exploratory_confidence_threshold.csv"):
        check(f"exists results/followup/{name}", (RES / name).exists() and (RES / name).stat().st_size > 0)
    for f in m["figures"].values():
        p = REP / "figures" / f
        check(f"figure {f}", p.exists() and p.stat().st_size > 5000)

    man = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
    sets = {s: set(man.loc[man["split"] == s, "trial"]) for s in ("train", "validation", "test")}
    check("split counts 120/40/40", [len(sets[s]) for s in ("train", "validation", "test")] == [120, 40, 40])
    check("no trial overlap", not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                                   or sets["validation"] & sets["test"]))
    meta = json.loads((MDL / "followup_models_metadata.json").read_text())
    check("follow-up models trained on phase-1 train trials only", set(meta["training_trial_ids"]) == sets["train"])

    # freeze before test
    freeze = json.loads((RES / "selection_freeze.json").read_text())
    check("freeze hash matches metrics", sha(RES / "selection_freeze.json") == m["selection"]["freeze_sha256"])
    st = {s["stage"][:2]: datetime.fromisoformat(s["utc"]) for s in m["stage_log"]}
    check("freeze written before test stage",
          datetime.fromisoformat(freeze["frozen_utc"]) <= st["4."], freeze["frozen_utc"])
    check("frozen selections match reported selections",
          freeze["selected_rule_A"] == m["selection"]["rule_A"]
          and freeze["selected_rule_B"] == m["selection"]["rule_B"])
    check("freeze contains no test metrics", "test" not in json.dumps(freeze["validation_table"]))

    # recompute metrics from predictions
    preds = pd.read_csv(RES / "predictions_all_models.csv")
    for (model, split), d in preds.groupby(["model", "split"]):
        e = m["evaluations"][model][split]
        y, yh = d["actual"], d["predicted"]
        ok = (set(d["trial"]) == sets[split]
              and abs((y == yh).mean() - e["accuracy"]) < 1e-12
              and abs(f1_score(y, yh, labels=LABELS, average="macro", zero_division=0) - e["macro_avg"]["f1"]) < 1e-12
              and confusion_matrix(y, yh, labels=LABELS).tolist() == e["confusion_matrix"]["matrix"]
              and sorted(d.loc[~d["correct"], "trial"]) == sorted(e["misclassified_trials"]))
        pr, rc, f1, _ = precision_recall_fscore_support(y, yh, labels=LABELS, zero_division=0)
        ok &= all(abs(pr[i] - e["per_class"][l]["precision"]) < 1e-12 and abs(rc[i] - e["per_class"][l]["recall"]) < 1e-12
                  and abs(f1[i] - e["per_class"][l]["f1"]) < 1e-12 for i, l in enumerate(LABELS))
        check(f"metrics recomputed: {model} / {split}", ok)
    comp = pd.read_csv(RES / "model_comparison.csv")
    check("model_comparison.csv matches evaluations", all(
        abs(r.test_accuracy - m["evaluations"][r.model]["test"]["accuracy"]) < 1e-12
        and abs(r.val_accuracy - m["evaluations"][r.model]["validation"]["accuracy"]) < 1e-12
        for r in comp.itertuples()))

    # RF unchanged vs phase 1; saved follow-up models reproduce predictions
    old = pd.read_csv(ROOT / "results" / "test_predictions.csv").sort_values("trial")
    new = preds.query("model=='RF_existing' and split=='test'").sort_values("trial")
    check("RF_existing reproduces phase-1 test predictions", list(old["predicted"]) == list(new["predicted"]))
    feats = build_feature_table(pd.read_csv(ROOT / "gripsense_clean_data.csv"))
    te = feats[feats["trial"].isin(sets["test"])].sort_values("trial")
    for f in sorted(MDL.glob("*.joblib")):
        b = joblib.load(f)
        p = b["model"].predict(te[b["feature_columns"]])
        saved = preds.query("model==@b['name'] and split=='test'").sort_values("trial")["predicted"]
        check(f"saved model reloads + reproduces test predictions: {f.name}", list(p) == list(saved))
    check("trial ID not a feature", "trial" not in meta["feature_columns"])

    # ---- report consistency
    rep = (REP / "GripSense_TinyML_Final_Report.md").read_text()
    needles = {"no-new-data statement": "No new data was collected and no hardware testing was performed",
               "test not untouched disclosure": "not completely",
               "exploratory label": "EXPLORATORY robustness analyses (not independent validation)"}
    for tag, name in (("A", m["selection"]["rule_A"]), ("B", m["selection"]["rule_B"])):
        e = m["evaluations"][name]["test"]
        needles[f"{tag} test accuracy"] = f"**Accuracy {e['accuracy']:.3f}** ({e['n_correct']}/{e['n_trials']}"
        needles[f"{tag} misclassified"] = f"Misclassified test trials: {e['misclassified_trials']}"
        for lab, row in zip(LABELS, e["confusion_matrix"]["matrix"]):
            needles[f"{tag} CM {lab}"] = f"| **{lab}** | " + " | ".join(map(str, row)) + " |"
        for lab in LABELS:
            c = e["per_class"][lab]
            needles[f"{tag} per-class {lab}"] = f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f}"
    for r in comp.itertuples():
        needles[f"comparison row {r.model}"] = f"| {r.model} | {r.test_accuracy:.3f} | {r.test_macro_f1:.3f} |"
    ds = m["double_squeeze"]
    needles["DS distribution"] = str({int(k): v for k, v in ds["presses_distribution"].items()})
    needles["DS exactly-2 count"] = f"| Trials with exactly 2 presses | **{ds['trials_with_exactly_2_presses']}** |"
    e4 = m["exploratory"]["E4_confidence_threshold"]
    needles["threshold"] = f"**{e4['chosen_on_validation']:.2f}** (validation coverage {e4['val_coverage']:.3f})"
    for k, v in m["exploratory"]["E1_repeated_cv"].items():
        needles[f"E1 {k}"] = f"| {k} ({v['model']}) | {v['n_folds']} | {v['mean']:.3f} |"
    for name, needle in needles.items():
        check(f"report contains {name}", needle in rep, needle[:70])
    for section in ("Dataset and quality audit (performed BEFORE training", "Model comparison",
                    "Final held-out test results", "Error analysis and uncertainty",
                    "Dataset protocol limitations", "Practical ESP32 deployment recommendations"):
        check(f"report section: {section}", section in rep)

    pd.DataFrame(results).to_csv(REP / "validation_checks_followup.csv", index=False)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
