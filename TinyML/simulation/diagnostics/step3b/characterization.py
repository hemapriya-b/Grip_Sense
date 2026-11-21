"""
Investigation 1: training-data characterization (train split only for distributions; all splits
shown separately for counts). Uses the existing features and the existing press-segment table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gripsense_features import LABELS

from .common import PRESS_SEGMENTS, W, trial_features

FEATURES = ["longest_active_run_frac", "segment_1_mean", "press_events_100", "press_events_300",
            "active_mean", "max", "above_100_fraction", "longest_zero_run_frac", "peak_position_frac"]
PRESS_TABLE_COLS = ["n_presses", "active_at_start", "active_at_end", "first_onset_ms", "median_press_ms"]
# Tree thresholds in samples: node 1 = 7/150, root = 53/150 (models/c_port/tree_metadata.json)
RUN_BINS = [-1, 7, 53, W]


def feature_quantiles(split: str = "train") -> pd.DataFrame:
    tf = trial_features()
    tf = tf[tf["split"] == split]
    rows = []
    for f in FEATURES:
        for lab in LABELS:
            v = tf.loc[tf["label"] == lab, f]
            rows.append({"feature": f, "label": lab, "n": len(v), "min": v.min(), "q10": v.quantile(.1),
                         "median": v.median(), "q90": v.quantile(.9), "max": v.max()})
    return pd.DataFrame(rows)


def range_overlap(split: str = "train") -> pd.DataFrame:
    """Fraction of class A trials whose value lies inside class B's [min, max] range (per feature)."""
    tf = trial_features()
    tf = tf[tf["split"] == split]
    rows = []
    for f in FEATURES:
        for a in LABELS:
            for b in LABELS:
                if a == b:
                    continue
                vb = tf.loc[tf["label"] == b, f]
                va = tf.loc[tf["label"] == a, f]
                rows.append({"feature": f, "class_a": a, "class_b": b,
                             "frac_a_inside_b_range": float(((va >= vb.min()) & (va <= vb.max())).mean())})
    return pd.DataFrame(rows)


def run_length_bins() -> pd.DataFrame:
    tf = trial_features().copy()
    tf["run_samples"] = (tf["longest_active_run_frac"] * W).round().astype(int)
    tf["run_bin"] = pd.cut(tf["run_samples"], RUN_BINS, labels=["0-7 (NO_PRESS leaf)", "8-53 (DS leaf)",
                                                                "54-150 (L/H branch)"])
    return pd.crosstab([tf["split"], tf["label"]], tf["run_bin"])


def press_event_counts(feature: str = "press_events_100") -> pd.DataFrame:
    tf = trial_features()
    return pd.crosstab([tf["split"], tf["label"]], tf[feature].astype(int))


def press_segment_summary(split: str = "train") -> pd.DataFrame:
    """From the existing results/followup/press_segments_per_trial.csv (read-only)."""
    ps = pd.read_csv(PRESS_SEGMENTS)
    tf = trial_features()[["trial", "split"]]
    ps = ps.merge(tf, on="trial")
    ps = ps[ps["split"] == split]
    return ps.groupby("label").agg(n=("trial", "size"), n_presses_min=("n_presses", "min"),
                                   n_presses_median=("n_presses", "median"), n_presses_max=("n_presses", "max"),
                                   exactly_two_presses=("n_presses", lambda s: int((s == 2).sum())),
                                   active_at_start=("active_at_start", "sum"),
                                   median_press_ms_median=("median_press_ms", "median")).reset_index()


def ds_runs_vs_partial_press_runs() -> dict:
    """DS trials' longest runs vs the run lengths a partial sustained press can produce in a window."""
    tf = trial_features()
    ds = (tf.loc[tf["label"] == "DOUBLE_SQUEEZE", "longest_active_run_frac"] * W).round().astype(int)
    return {"ds_longest_run_samples_min": int(ds.min()), "ds_longest_run_samples_max": int(ds.max()),
            "ds_trials_in_8_53": int(((ds >= 8) & (ds <= 53)).sum()), "ds_trials": int(len(ds)),
            "partial_sustained_press_can_produce_runs": "any length 1..150 depending on window offset; "
                                                        "8..53 lands in the DS leaf",
            "ds_press_events_100_min": int(tf.loc[tf["label"] == "DOUBLE_SQUEEZE", "press_events_100"].min()),
            "non_ds_with_press_events_100_ge_2": int(((tf["label"] != "DOUBLE_SQUEEZE")
                                                     & (tf["press_events_100"] >= 2)).sum()),
            "non_ds_with_press_events_100_ge_3": int(((tf["label"] != "DOUBLE_SQUEEZE")
                                                     & (tf["press_events_100"] >= 3)).sum())}
