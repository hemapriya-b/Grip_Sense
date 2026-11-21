"""
Replay recorded trials from the original CSV (read-only) as a simulated ADC stream.

Each trial keeps its own sample order (sorted by elapsed_ms). Trials are emitted in the order
requested by the caller; every emitted sample is tagged with its segment (position in that order),
trial ID and index within the trial, so trial boundaries are never lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import pandas as pd

from gripsense_features import LABELS, SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL

from . import ROOT

DEFAULT_CSV = ROOT / "gripsense_clean_data.csv"


@dataclass(frozen=True)
class TrialRecord:
    trial: int
    label: str
    elapsed_ms: tuple[int, ...]
    adc: tuple[int, ...]


@dataclass(frozen=True)
class Sample:
    segment: int        # position of the trial in the replay order (0-based)
    trial: int          # trial ID from the CSV
    index: int          # 0..149 within the trial
    elapsed_ms: int     # as recorded within the trial
    stream_index: int   # 0-based position in the whole simulated stream
    stream_ms: int      # simulated stream time = stream_index * 20 ms
    adc: int
    label: str          # recorded label of the source trial


def load_trials(csv_path=DEFAULT_CSV) -> dict[int, TrialRecord]:
    """Read the CSV (never written) and validate the per-trial structure."""
    df = pd.read_csv(csv_path)
    out: dict[int, TrialRecord] = {}
    for tid, g in df.groupby("trial", sort=True):
        g = g.sort_values("elapsed_ms", kind="stable")
        labels = g["label"].unique()
        if len(labels) != 1 or labels[0] not in LABELS:
            raise ValueError(f"trial {tid}: expected one known label, got {list(labels)}")
        if len(g) != SAMPLES_PER_TRIAL:
            raise ValueError(f"trial {tid}: {len(g)} samples, expected {SAMPLES_PER_TRIAL}")
        el = [int(v) for v in g["elapsed_ms"]]
        if el != [i * SAMPLE_INTERVAL_MS for i in range(SAMPLES_PER_TRIAL)]:
            raise ValueError(f"trial {tid}: elapsed_ms is not 0, 20, ..., 2980")
        out[int(tid)] = TrialRecord(int(tid), str(labels[0]), tuple(el), tuple(int(v) for v in g["adc"]))
    return out


def replay(trials: dict[int, TrialRecord], order: Sequence[int]) -> Iterator[Sample]:
    """Yield samples of the requested trials, one trial after another, in recorded order."""
    stream_index = 0
    for segment, tid in enumerate(order):
        rec = trials[int(tid)]
        for i, (el, a) in enumerate(zip(rec.elapsed_ms, rec.adc)):
            yield Sample(segment, rec.trial, i, el, stream_index, stream_index * SAMPLE_INTERVAL_MS, a, rec.label)
            stream_index += 1


def segments(samples: Iterable[Sample]) -> list[int]:
    return sorted({s.segment for s in samples})
