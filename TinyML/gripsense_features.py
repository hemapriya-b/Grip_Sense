"""
GripSense trial-level feature extraction (shared by training and inference).

Every feature is computed from a single trial's own 150 ADC readings using
fixed, pre-declared constants. Nothing is fitted or learned from data, so the
function behaves identically on train, validation, test, and live data and
cannot leak information across splits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"]
LABEL_MEANINGS = {
    "LIGHT_PRESS": "Hungry",
    "HARD_PRESS": "Pain",
    "DOUBLE_SQUEEZE": "Washroom",
    "NO_PRESS": "Idle/resting",
}
REQUIRED_COLUMNS = ["trial", "label", "elapsed_ms", "adc"]
SAMPLES_PER_TRIAL = 150
SAMPLE_INTERVAL_MS = 20

# Fixed thresholds (ADC counts, ESP32 12-bit scale 0-4095). Chosen a priori
# from the sensor's range, never tuned on validation or test results.
LEVEL_THRESHOLDS = [50, 100, 500, 1000, 1500, 2000]
ACTIVE_THRESHOLD = 100   # reading counts as "pressed" above this
EVENT_THRESHOLDS = [100, 300]  # rising crossings = number of distinct presses
N_SEGMENTS = 10


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def _rising_crossings(x: np.ndarray, threshold: float) -> int:
    above = x > threshold
    # Count a press already in progress at sample 0 as one event.
    return int(above[0]) + int(np.sum(~above[:-1] & above[1:]))


def extract_trial_features(adc: np.ndarray) -> dict[str, float]:
    """Convert one time-ordered trial (1-D array of ADC readings) to features."""
    x = np.asarray(adc, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("expected a 1-D array with at least 2 readings")
    n = len(x)
    d = np.diff(x)
    active = x > ACTIVE_THRESHOLD

    f: dict[str, float] = {
        "mean": np.mean(x),
        "median": np.median(x),
        "std": np.std(x),
        "min": np.min(x),
        "max": np.max(x),
        "range": np.ptp(x),
        "q10": np.percentile(x, 10),
        "q25": np.percentile(x, 25),
        "q75": np.percentile(x, 75),
        "q90": np.percentile(x, 90),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "rms": np.sqrt(np.mean(x * x)),
        "zero_fraction": np.mean(x == 0),
        "active_mean": np.mean(x[active]) if active.any() else 0.0,
        "longest_zero_run_frac": _longest_run(x == 0) / n,
        "longest_active_run_frac": _longest_run(active) / n,
        "peak_position_frac": np.argmax(x) / (n - 1),
        "mean_abs_diff": np.mean(np.abs(d)),
        "std_diff": np.std(d),
        "max_rise": np.max(d),
        "max_fall": -np.min(d),
    }
    for t in LEVEL_THRESHOLDS:
        f[f"above_{t}_fraction"] = np.mean(x > t)
    for t in EVENT_THRESHOLDS:
        f[f"press_events_{t}"] = _rising_crossings(x, t)
    for i, chunk in enumerate(np.array_split(x, N_SEGMENTS)):
        f[f"segment_{i}_mean"] = np.mean(chunk)
        f[f"segment_{i}_max"] = np.max(chunk)
    return {k: float(v) for k, v in f.items()}


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: trial, label, then features (sorted by elapsed_ms)."""
    rows = []
    for trial_id, g in df.groupby("trial", sort=True):
        g = g.sort_values("elapsed_ms")
        row = {"trial": int(trial_id), "label": str(g["label"].iloc[0])}
        row.update(extract_trial_features(g["adc"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(table: pd.DataFrame) -> list[str]:
    # Trial ID is deliberately excluded: trials were recorded in class blocks,
    # so the ID alone would perfectly (and meaninglessly) reveal the label.
    return [c for c in table.columns if c not in ("trial", "label")]
