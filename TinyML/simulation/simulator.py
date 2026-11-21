"""
Wire the stream, window buffer, inference backend(s) and event decider together.

Modes
    isolated    The buffer is reset at every trial boundary, so each trial yields exactly one full
                window: the aligned window that equals the recorded trial. Inference only; the event
                decider is not run (one window per trial can never satisfy K>=2 agreement, and running
                it across independent trials would invent a continuous stream).
    continuous  The requested trials are concatenated in order and a sliding window runs over the
                result every `hop` samples. Windows spanning two trials are flagged `mixed` and have
                no ground-truth label. Concatenated recordings are NOT real continuous recordings.

A window is `aligned` when it is exactly one whole recorded trial (single segment, sample indices
0..149). Only aligned windows correspond to the reference predictions in results/c_port/.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import pandas as pd

from gripsense_features import SAMPLES_PER_TRIAL

from . import ROOT
from .decision import DeciderConfig, EventDecider, Event
from .stream import TrialRecord, replay
from .window_buffer import DEFAULT_WINDOW, WindowBuffer

DEFAULT_HOP = 10   # samples (200 ms) - initial simulation setting, not validated
REFERENCE_CSV = ROOT / "results" / "c_port" / "predictions_python_vs_c.csv"


@dataclass(frozen=True)
class WindowResult:
    window_index: int
    stream_start_index: int
    stream_end_index: int
    stream_end_ms: int
    segments: tuple[int, ...]
    source_trials: tuple[int, ...]
    start: tuple[int, int]          # (trial, index) of the oldest sample
    end: tuple[int, int]            # (trial, index) of the newest sample
    mixed: bool
    aligned: bool
    true_label: Optional[str]       # recorded label, only for single-trial windows
    label: Optional[str]            # primary backend prediction
    reference_label: Optional[str]  # reference backend prediction (if run)
    backends_agree: Optional[bool]
    state: Optional[str]            # decider state after this window (continuous mode)
    event_label: Optional[str]


@dataclass
class SimulationResult:
    mode: str
    trial_order: list[int]
    window_size: int
    hop: int
    decider: Optional[dict]
    windows: list[WindowResult] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    suppressed_label_changes: int = 0

    def windows_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(w) for w in self.windows])


def load_reference_predictions(path=REFERENCE_CSV) -> dict[int, str]:
    """Stored sklearn predictions for the 200 whole trials (read-only)."""
    df = pd.read_csv(path)
    return {int(t): str(p) for t, p in zip(df["trial"], df["python_sklearn"])}


def simulate(trials: dict[int, TrialRecord], order: Sequence[int], mode: str, backend,
             reference=None, window_size: int = DEFAULT_WINDOW, hop: int = DEFAULT_HOP,
             decider_config: Optional[DeciderConfig] = None) -> SimulationResult:
    if mode not in ("isolated", "continuous"):
        raise ValueError(f"unknown mode {mode!r}")
    if isinstance(hop, bool) or not isinstance(hop, int) or hop < 1:
        raise ValueError(f"hop must be an integer >= 1, got {hop!r}")
    if window_size != SAMPLES_PER_TRIAL:
        raise ValueError(f"the verified model needs {SAMPLES_PER_TRIAL}-sample windows, got {window_size}")
    if not order:
        raise ValueError("no trials requested")

    buf = WindowBuffer(window_size)
    decider = EventDecider(decider_config or DeciderConfig()) if mode == "continuous" else None
    res = SimulationResult(mode, [int(t) for t in order], window_size, hop,
                           asdict(decider.config) if decider else None)
    prev_segment = None
    for s in replay(trials, order):
        if mode == "isolated" and s.segment != prev_segment:
            buf.reset()
        prev_segment = s.segment
        buf.push(s)
        if not buf.is_full() or (buf.pushed_since_reset - window_size) % hop != 0:
            continue
        win = buf.samples()
        adc = buf.adc_array()
        segs = tuple(sorted({x.segment for x in win}))
        mixed = len(segs) > 1
        aligned = not mixed and win[0].index == 0 and win[-1].index == SAMPLES_PER_TRIAL - 1
        pred = backend.predict(adc)
        ref = reference.predict(adc).label if reference is not None else None
        trial_seq = []
        for x in win:
            if not trial_seq or trial_seq[-1][0] != x.segment:
                trial_seq.append((x.segment, x.trial))
        ctx = dict(window_index=len(res.windows), stream_end_index=win[-1].stream_index,
                   stream_end_ms=win[-1].stream_ms, source_trials=tuple(t for _, t in trial_seq), mixed=mixed)
        step = decider.update(pred.label, ctx) if decider else None
        res.windows.append(WindowResult(
            window_index=len(res.windows), stream_start_index=win[0].stream_index,
            stream_end_index=win[-1].stream_index, stream_end_ms=win[-1].stream_ms, segments=segs,
            source_trials=tuple(t for _, t in trial_seq), start=(win[0].trial, win[0].index),
            end=(win[-1].trial, win[-1].index), mixed=mixed, aligned=aligned,
            true_label=None if mixed else win[0].label, label=pred.label, reference_label=ref,
            backends_agree=None if reference is None else ref == pred.label,
            state=step.state.value if step else None,
            event_label=step.event.label if step and step.event else None))
    if decider:
        res.events = list(decider.events)
        res.suppressed_label_changes = decider.suppressed_label_changes
    return res


def compare_aligned_to_reference(res: SimulationResult, reference: dict[int, str]) -> tuple[int, int, list]:
    """(matches, aligned windows, mismatches) against the stored whole-trial predictions."""
    aligned = [w for w in res.windows if w.aligned]
    mism = [(w.window_index, w.source_trials[0], w.label, reference[w.source_trials[0]])
            for w in aligned if w.label != reference[w.source_trials[0]]]
    return len(aligned) - len(mism), len(aligned), mism
