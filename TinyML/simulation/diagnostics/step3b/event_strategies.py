"""
Investigation 4: event/state strategies run on RECORDED window label sequences.

  latch        production EventDecider (simulation/decision.py, unchanged): K agreeing windows emit,
               latched until N NO_PRESS windows. Default K=2, N=1; K is also varied.
  ds_slower    same latch, but DOUBLE_SQUEEZE needs agreement lasting D ms (other labels K=2)
  correctable  latch K=2, N=1, but while ACTIVE a different non-idle label held for M ms emits a
               correction event (counted as an additional event)
  on_release   no event during the press; when N=1 NO_PRESS window ends the episode, emit the label
               with the longest consecutive streak in the episode (needs a streak >= 2 windows)

Parameters (K, D, M) are chosen on VALIDATION sequences only and then reported on TEST sequences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from gripsense_features import SAMPLE_INTERVAL_MS

from ...decision import DeciderConfig, EventDecider

IDLE = "NO_PRESS"


@dataclass(frozen=True)
class Ev:
    label: str
    first_window: int
    emit_window: int
    kind: str = "event"       # "event" or "correction"


def windows_for_ms(ms: int, hop: int) -> int:
    return max(2, math.ceil(ms / (hop * SAMPLE_INTERVAL_MS)))


def latch(labels: Sequence[Optional[str]], k: int = 2, n: int = 1) -> list[Ev]:
    d = EventDecider(DeciderConfig(k, n))
    for lab in labels:
        d.update(lab)
    return [Ev(e.label, e.first_window, e.emit_window) for e in d.events]


def generic_latch(labels, k_for: dict, n: int = 1, correction_windows: Optional[int] = None) -> list[Ev]:
    """Latch with per-label K and optional correction. With uniform K and no correction it is
    behaviourally identical to the production EventDecider (checked in tests)."""
    out: list[Ev] = []
    active = cand = None
    count = idle = 0
    start = -1
    c_lab, c_count, c_start = None, 0, -1
    for w, lab in enumerate(labels):
        if active is not None:
            if lab == IDLE:
                idle += 1
                c_lab, c_count = None, 0
                if idle >= n:
                    active, cand, count = None, None, 0
                continue
            idle = 0
            if lab is None or lab == active:
                c_lab, c_count = None, 0
                continue
            if correction_windows is None:
                continue
            if lab == c_lab:
                c_count += 1
            else:
                c_lab, c_count, c_start = lab, 1, w
            if c_count >= correction_windows:
                out.append(Ev(lab, c_start, w, "correction"))
                active, c_lab, c_count = lab, None, 0
            continue
        if lab == IDLE or lab is None:
            cand, count = None, 0
            continue
        if lab == cand:
            count += 1
        else:
            cand, count, start = lab, 1, w
        if count >= k_for.get(lab, 2):
            out.append(Ev(lab, start, w))
            active, idle, cand, count = lab, 0, None, 0
    return out


def on_release(labels, n: int = 1, min_streak: int = 2) -> list[Ev]:
    out: list[Ev] = []
    ep: list[tuple[int, str]] = []
    idle = 0

    def flush(emit_w):
        best_lab, best_len, run_lab, run_len = None, 0, None, 0
        for _, lab in ep:
            run_len = run_len + 1 if lab == run_lab else 1
            run_lab = lab
            if run_len >= best_len:          # ties -> later streak
                best_lab, best_len = run_lab, run_len
        if best_len >= min_streak:
            out.append(Ev(best_lab, ep[0][0], emit_w))

    for w, lab in enumerate(labels):
        if lab == IDLE or lab is None:
            if ep:
                idle += 1
                if idle >= n:
                    flush(w)
                    ep, idle = [], 0
            continue
        idle = 0
        ep.append((w, lab))
    if ep:
        flush(len(labels) - 1)               # stream ended mid-episode
    return out


def strategy_grid(hop: int) -> list[tuple[str, dict, callable]]:
    g = [(f"latch_K{k}", {"k": k}, (lambda k: lambda L: latch(L, k, 1))(k)) for k in (2, 3, 4, 6)]
    for ms in (400, 800, 1200):
        kd = windows_for_ms(ms, hop)
        g.append((f"ds_slower_{ms}ms", {"ds_ms": ms, "ds_windows": kd},
                  (lambda kd: lambda L: generic_latch(L, {"DOUBLE_SQUEEZE": kd}, 1))(kd)))
    for ms in (400, 800, 1200):
        m = windows_for_ms(ms, hop)
        g.append((f"correctable_{ms}ms", {"corr_ms": ms, "corr_windows": m},
                  (lambda m: lambda L: generic_latch(L, {}, 1, correction_windows=m))(m)))
    g.append(("on_release", {}, lambda L: on_release(L)))
    return g


def family(name: str) -> str:
    return name.split("_K")[0] if name.startswith("latch") else name.rsplit("_", 1)[0] if "ms" in name else name
