"""
Shared data access, sequence construction, windowing and episode bookkeeping for Step 3B.

Read-only: the CSV, split manifest and saved model are only read. Splits are the existing
trial-level manifest (reports/split_manifest.csv: 120 train / 40 validation / 40 test trials,
30/10/10 per class); no window or trial is ever shuffled across splits.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from gripsense_features import LABELS, SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL, build_feature_table, extract_trial_features

from ... import ROOT
from ...stream import load_trials

W = SAMPLES_PER_TRIAL
IDLE = "NO_PRESS"
PRESS_LABELS = [l for l in LABELS if l != IDLE]
MANIFEST = ROOT / "reports" / "split_manifest.csv"
PRESS_SEGMENTS = ROOT / "results" / "followup" / "press_segments_per_trial.csv"


@lru_cache(maxsize=None)
def trials():
    return load_trials()


@lru_cache(maxsize=None)
def manifest() -> dict[int, str]:
    m = pd.read_csv(MANIFEST)
    return {int(t): str(s) for t, s in zip(m["trial"], m["split"])}


@lru_cache(maxsize=None)
def trial_features() -> pd.DataFrame:
    """Recomputed from the raw CSV with the existing feature definitions, plus the split column."""
    df = build_feature_table(pd.read_csv(ROOT / "gripsense_clean_data.csv")).sort_values("trial").reset_index(drop=True)
    df["split"] = df["trial"].map(manifest())
    return df


def feature_columns() -> list[str]:
    return [c for c in trial_features().columns if c not in ("trial", "label", "split")]


def trial_ids(split: str, label: str | None = None) -> list[int]:
    tf = trial_features()
    sel = tf[tf["split"] == split]
    if label is not None:
        sel = sel[sel["label"] == label]
    return [int(t) for t in sel["trial"]]


# ---------------------------------------------------------------- sequences
@dataclass(frozen=True)
class SeqSpec:
    name: str
    split: str          # trials drawn only from this split
    order: tuple[int, ...]


def rest_separated_order(split: str, seed: int) -> tuple[int, ...]:
    """rest, P1, rest, P2, ..., Pn, rest: every non-rest trial of the split once (shuffled by seed);
    rest trials are the split's NO_PRESS trials, shuffled and reused cyclically."""
    rng = random.Random(seed)
    press = [t for lab in PRESS_LABELS for t in trial_ids(split, lab)]
    rest = trial_ids(split, IDLE)
    rng.shuffle(press)
    rng.shuffle(rest)
    order = []
    for i, p in enumerate(press):
        order += [rest[i % len(rest)], p]
    order.append(rest[len(press) % len(rest)])
    return tuple(order)


def stream_of(order) -> np.ndarray:
    return np.concatenate([np.array(trials()[t].adc, dtype=np.int64) for t in order])


def window_starts(n_samples: int, hop: int) -> list[int]:
    """Same schedule as the production simulator: first window at samples 0..149, then every `hop`."""
    return list(range(0, n_samples - W + 1, hop))


@dataclass(frozen=True)
class Episode:
    label: str
    first_segment: int
    last_segment: int
    start_index: int      # first stream sample of the episode
    end_index: int        # last stream sample of the episode


def episodes(order) -> list[Episode]:
    """Maximal runs of consecutive non-rest segments sharing one label = one expected event each.
    Runs of consecutive non-rest segments with different labels are not produced by our sequence
    builder; if present they raise (no defensible expectation)."""
    out, seg = [], 0
    labs = [trials()[t].label for t in order]
    while seg < len(order):
        if labs[seg] == IDLE:
            seg += 1
            continue
        end = seg
        while end + 1 < len(order) and labs[end + 1] != IDLE:
            if labs[end + 1] != labs[seg]:
                raise ValueError(f"adjacent press segments with different labels at {end}/{end + 1}")
            end += 1
        out.append(Episode(labs[seg], seg, end, seg * W, end * W + W - 1))
        seg = end + 1
    return out


def window_table(order, hop: int) -> pd.DataFrame:
    """One row per window: indices, segments, mixed flag, and a label ONLY for single-trial windows."""
    n = len(order) * W
    rows = []
    for k, s in enumerate(window_starts(n, hop)):
        e = s + W - 1
        segs = tuple(range(s // W, e // W + 1))
        mixed = len(segs) > 1
        rows.append({"window": k, "start": s, "end": e, "end_ms": e * SAMPLE_INTERVAL_MS,
                     "segments": segs, "trials": tuple(order[x] for x in segs), "mixed": mixed,
                     "aligned": (not mixed) and s % W == 0,
                     "true_label": None if mixed else trials()[order[segs[0]]].label})
    return pd.DataFrame(rows)


def window_feature_frame(stream: np.ndarray, starts) -> pd.DataFrame:
    """Existing feature definitions applied to each 150-sample window (no new formulas)."""
    return pd.DataFrame([extract_trial_features(stream[s:s + W]) for s in starts])[feature_columns()]
