#!/usr/bin/env python3
"""
Offline inference with the saved GripSense model (no hardware required).

Input CSV needs columns: trial, elapsed_ms, adc  (label is optional; if present
it is shown next to the prediction). Each trial should hold 150 readings
sampled every 20 ms (one 3-second window), like the training data.

    .venv/bin/python predict.py --csv gripsense_clean_data.csv --trials 1 60 120 180
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from gripsense_features import extract_trial_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="models/gripsense_model.joblib")
    ap.add_argument("--trials", type=int, nargs="*", help="only these trial IDs")
    ap.add_argument("--out", help="optional CSV path for predictions")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model, cols = bundle["model"], bundle["feature_columns"]
    df = pd.read_csv(args.csv)
    if args.trials:
        df = df[df["trial"].isin(args.trials)]
    rows = []
    for tid, g in df.groupby("trial", sort=True):
        g = g.sort_values("elapsed_ms")
        if len(g) != bundle["samples_per_trial"]:
            print(f"warning: trial {tid} has {len(g)} readings "
                  f"(model trained on {bundle['samples_per_trial']})")
        feats = pd.DataFrame([extract_trial_features(g["adc"].to_numpy())])[cols]
        proba = model.predict_proba(feats)[0]
        pred = model.classes_[proba.argmax()]
        row = {"trial": tid, "predicted": pred, "meaning": bundle["label_meanings"][pred],
               "confidence": round(float(proba.max()), 3)}
        if "label" in g:
            row["actual"] = g["label"].iloc[0]
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
