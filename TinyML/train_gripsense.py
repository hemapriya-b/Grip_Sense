#!/usr/bin/env python3
"""
GripSense FSR TinyML baseline
- Input CSV columns: trial,label,elapsed_ms,adc
- Extracts one feature vector per complete trial (150 samples expected)
- Splits by whole trial, stratified by label (80/20; 40 train + 10 test trials per class)
- Trains a Random Forest and reports accuracy, per-class metrics, and confusion matrix

Install:
    pip install pandas numpy scikit-learn joblib

Run:
    python train_gripsense.py --csv gripsense_clean_data.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

LABELS = ["LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"]
REQUIRED_COLUMNS = {"trial", "label", "elapsed_ms", "adc"}


def extract_trial_features(group: pd.DataFrame) -> dict[str, float]:
    """Convert one time-ordered trial into a fixed-length feature vector."""
    group = group.sort_values("elapsed_ms")
    x = group["adc"].astype(float).to_numpy()

    features = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "range": float(np.max(x) - np.min(x)),
        "q10": float(np.percentile(x, 10)),
        "q25": float(np.percentile(x, 25)),
        "q75": float(np.percentile(x, 75)),
        "q90": float(np.percentile(x, 90)),
        "zero_fraction": float(np.mean(x == 0)),
        "above_100_fraction": float(np.mean(x > 100)),
        "above_500_fraction": float(np.mean(x > 500)),
        "above_1000_fraction": float(np.mean(x > 1000)),
        "above_1500_fraction": float(np.mean(x > 1500)),
        "above_2000_fraction": float(np.mean(x > 2000)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "mean_abs_difference": float(np.mean(np.abs(np.diff(x)))),
    }

    # Preserve broad timing information by summarizing 10 equal time segments.
    for i, chunk in enumerate(np.array_split(x, 10)):
        features[f"segment_{i}_mean"] = float(np.mean(chunk))
        features[f"segment_{i}_max"] = float(np.max(chunk))

    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="gripsense_clean_data.csv",
                        help="Path to the cleaned CSV")
    parser.add_argument("--out", default="gripsense_rf_model.joblib",
                        help="Where to save the trained model bundle")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["trial", "label", "elapsed_ms", "adc"]).copy()
    df["label"] = df["label"].astype(str)

    unknown = set(df["label"].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {sorted(unknown)}")

    # Validate each trial has exactly one label and 150 samples.
    rows = []
    for trial_id, group in df.groupby("trial", sort=True):
        labels = group["label"].unique()
        if len(labels) != 1:
            raise ValueError(f"Trial {trial_id} has inconsistent labels: {labels}")
        if len(group) != 150:
            raise ValueError(
                f"Trial {trial_id} has {len(group)} rows, expected 150. "
                "Check the CSV before training."
            )
        row = extract_trial_features(group)
        row["trial"] = trial_id
        row["label"] = labels[0]
        rows.append(row)

    trials = pd.DataFrame(rows)
    counts = trials["label"].value_counts()
    for label in LABELS:
        if counts.get(label, 0) != 50:
            raise ValueError(
                f"Expected 50 trials for {label}; found {counts.get(label, 0)}"
            )

    # Split the 200 trial-level records, not the 30,000 individual readings.
    train_trials, test_trials = train_test_split(
        trials[["trial", "label"]],
        test_size=0.20,
        random_state=42,
        stratify=trials["label"],
    )
    train = trials[trials["trial"].isin(train_trials["trial"])].copy()
    test = trials[trials["trial"].isin(test_trials["trial"])].copy()

    feature_columns = [
        col for col in trials.columns if col not in {"trial", "label"}
    ]
    X_train, y_train = train[feature_columns], train["label"]
    X_test, y_test = test[feature_columns], test["label"]

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Trials total: {len(trials)}")
    print(f"Training trials: {len(train)} ({train['label'].value_counts().to_dict()})")
    print(f"Testing trials:  {len(test)} ({test['label'].value_counts().to_dict()})")
    print(f"Features per trial: {len(feature_columns)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=LABELS, zero_division=0
    ))
    print("Confusion matrix (rows = true labels; columns = predicted labels):")
    print("Label order:", LABELS)
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "labels": LABELS,
        "note": "Random Forest baseline trained on trial-level engineered features.",
    }
    joblib.dump(bundle, args.out)
    print(f"\nSaved model bundle to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
