#!/usr/bin/env python3
"""
Independent post-run validation of every GripSense artifact.

Recomputes counts, split properties and metrics from the raw CSV and the saved
prediction files, reloads the saved model, and checks that the report quotes
the measured numbers. Exits non-zero if any check fails.

    .venv/bin/python validate_outputs.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from gripsense_features import LABELS, build_feature_table

ROOT = Path(__file__).resolve().parent
results: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    rep, figd, mdl, res = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models", ROOT / "results"
    required = [
        rep / "GripSense_TinyML_Report.md", rep / "dataset_audit.csv", rep / "trial_features.csv",
        rep / "split_manifest.csv", rep / "metrics.json", rep / "pre_training_plan.md",
        rep / "trial_quality_flags.csv", mdl / "gripsense_model.joblib", mdl / "model_metadata.json",
        res / "test_predictions.csv", res / "validation_predictions.csv",
        res / "model_selection_results.csv", res / "baseline_results.csv",
        res / "misclassified_trials.csv", res / "classification_report_test.txt",
        res / "feature_importance.csv", ROOT / "run_experiment.py", ROOT / "README.md",
        ROOT / "requirements.txt",
    ]
    for p in required:
        check(f"exists: {p.relative_to(ROOT)}", p.exists() and p.stat().st_size > 0)

    m = json.loads((rep / "metrics.json").read_text())
    for name in m["figures"].values():
        p = figd / name
        check(f"figure exists: {name}", p.exists() and p.stat().st_size > 5000)

    # ---- original CSV untouched + expected counts
    csv = ROOT / "gripsense_clean_data.csv"
    check("original CSV unchanged (SHA-256)",
          hashlib.sha256(csv.read_bytes()).hexdigest() == m["input"]["sha256"])
    df = pd.read_csv(csv)
    check("30,000 readings", len(df) == 30000, str(len(df)))
    check("200 trials", df["trial"].nunique() == 200)
    check("150 readings per trial", (df.groupby("trial").size() == 150).all())
    check("50 trials per class",
          (df.groupby("trial")["label"].first().value_counts().reindex(LABELS) == 50).all())

    # ---- split
    man = pd.read_csv(rep / "split_manifest.csv")
    check("manifest covers 200 unique trials", man["trial"].nunique() == 200 and len(man) == 200)
    for s, n, per in [("train", 120, 30), ("validation", 40, 10), ("test", 40, 10)]:
        sub = man[man["split"] == s]
        check(f"{s}: {n} trials", len(sub) == n, str(len(sub)))
        check(f"{s}: {per} trials per class",
              (sub["label"].value_counts().reindex(LABELS).fillna(0) == per).all())
        check(f"{s}: {n * 150} readings", int(df["trial"].isin(sub["trial"]).sum()) == n * 150)
    sets = {s: set(man.loc[man["split"] == s, "trial"]) for s in ("train", "validation", "test")}
    check("no trial overlap across splits",
          not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
               or sets["validation"] & sets["test"]))
    lab = df.groupby("trial")["label"].first()
    check("manifest labels match CSV", (man.set_index("trial")["label"] == lab.loc[man["trial"]]).all())

    # ---- features reproducible from raw CSV
    tf = pd.read_csv(rep / "trial_features.csv")
    check("trial_features: 200 rows", len(tf) == 200)
    fresh = build_feature_table(df).sort_values("trial").reset_index(drop=True)
    fcols = [c for c in fresh.columns if c not in ("trial", "label")]
    tf = tf.sort_values("trial").reset_index(drop=True)
    check("trial_features match recomputation", np.allclose(tf[fcols].to_numpy(), fresh[fcols].to_numpy()))
    check("trial_features split column matches manifest",
          (tf.set_index("trial")["split"] == man.set_index("trial")["split"]).all())

    # ---- leakage
    meta = json.loads((mdl / "model_metadata.json").read_text())
    check("model trained only on train trials",
          set(meta["training_trial_ids"]) == sets["train"])
    check("no test trials used in model selection",
          m["model_selection"]["test_trials_used_in_selection"] == []
          and m["model_selection"]["selection_used_trials"] == "train + validation only")
    stages = [s["stage"] for s in m["stage_log"]]
    order = [next(i for i, s in enumerate(stages) if s.startswith(k)) for k in ("1.", "4.", "5.", "6.", "7.")]
    check("stage order: audit < baseline < plan < selection < test", order == sorted(order))
    check("trial ID not used as a feature", "trial" not in meta["feature_columns"])

    # ---- metrics recomputed from saved predictions
    for split, key in (("test", "test"), ("validation", "validation")):
        p = pd.read_csv(res / f"{split}_predictions.csv")
        check(f"{split} predictions: 40 rows, ids match manifest", set(p["trial"]) == sets[split])
        y, yh = p["actual"], p["predicted"]
        acc = float((y == yh).mean())
        check(f"{split} accuracy matches metrics.json", abs(acc - m[key]["accuracy"]) < 1e-12,
              f"{acc:.4f}")
        check(f"{split} macro-F1 matches metrics.json",
              abs(f1_score(y, yh, labels=LABELS, average="macro", zero_division=0)
                  - m[key]["macro_avg"]["f1"]) < 1e-12)
        pr, rc, f1, _ = precision_recall_fscore_support(y, yh, labels=LABELS, zero_division=0)
        check(f"{split} per-class P/R/F1 match metrics.json", all(
            abs(pr[i] - m[key]["per_class"][l]["precision"]) < 1e-12
            and abs(rc[i] - m[key]["per_class"][l]["recall"]) < 1e-12
            and abs(f1[i] - m[key]["per_class"][l]["f1"]) < 1e-12 for i, l in enumerate(LABELS)))
        cm = confusion_matrix(y, yh, labels=LABELS).tolist()
        check(f"{split} confusion matrix matches metrics.json",
              cm == m[key]["confusion_matrix"]["matrix"])
        check(f"{split} misclassified IDs match metrics.json",
              sorted(p.loc[~p["correct"], "trial"]) == sorted(m["misclassified"][split]))

    # ---- saved model loads and reproduces predictions
    bundle = joblib.load(mdl / "gripsense_model.joblib")
    te = fresh[fresh["trial"].isin(sets["test"])].sort_values("trial")
    pred = bundle["model"].predict(te[bundle["feature_columns"]])
    tp = pd.read_csv(res / "test_predictions.csv").sort_values("trial")
    check("saved model reloads and reproduces test predictions",
          list(pred) == list(tp["predicted"]))

    # ---- report quotes the measured numbers
    report = (rep / "GripSense_TinyML_Report.md").read_text()
    t, v = m["test"], m["validation"]
    needles = {
        "test accuracy": f"**{t['accuracy']:.3f}**",
        "test n_correct": f"{t['n_correct']}/{t['n_trials']}",
        "test macro-F1": f"macro-F1 **{t['macro_avg']['f1']:.3f}**",
        "validation accuracy": f"Validation accuracy {v['accuracy']:.3f}",
        "baseline B1 accuracy": f"{m['baselines_validation'][1]['val_accuracy']:.3f}",
        "seed": f"Seed: **{m['seed']}**",
    }
    for lab_, row in zip(LABELS, t["confusion_matrix"]["matrix"]):
        needles[f"test CM row {lab_}"] = f"| **{lab_}** | " + " | ".join(map(str, row)) + " |"
    for lab_ in LABELS:
        c = t["per_class"][lab_]
        needles[f"test per-class {lab_}"] = f"{c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f}"
    for name, s in needles.items():
        check(f"report contains {name}", s in report, s)
    for section in ("Dataset findings BEFORE training", "Baseline results", "Model training and validation",
                    "Final test results AFTER training", "Limitations", "Conclusion", "Next steps"):
        check(f"report section: {section}", section in report)

    pd.DataFrame(results).to_csv(rep / "validation_checks.csv", index=False)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
