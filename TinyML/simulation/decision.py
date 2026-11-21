"""
Event/decision logic. Consumes one window label at a time; knows nothing about the model,
features, ADC samples, or C code. No confidence score is used or produced.

States
    IDLE       no event in progress (last window NO_PRESS, or start)
    CANDIDATE  a non-idle label has been seen in 1..K-1 consecutive windows; nothing emitted yet
    UNCERTAIN  the latest non-idle window disagreed with the previous one, or the classifier
               rejected the window (invalid input). Based on window disagreement, not probability.
    ACTIVE     an event was emitted; latched until re-armed

Rules (INITIAL SIMULATION SETTINGS - not validated on real use)
    * Emit an event after K consecutive windows agree on the same non-idle label (default K=2).
    * While ACTIVE, emit nothing - not even for a different non-idle label - until N consecutive
      NO_PRESS windows re-arm the detector (default N=1). Latching on a label change is a design
      choice to revisit after testing; suppressed label changes are counted for transparency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from gripsense_features import LABEL_MEANINGS, LABELS

IDLE_LABEL = "NO_PRESS"
DOUBLE_SQUEEZE_NOTE = ("Repeated-squeeze pattern. The DOUBLE_SQUEEZE training trials contain 3-7 presses "
                       "and no exactly-two-press examples; exact two-squeeze recognition is NOT validated.")
SIMULATION_NOTE = "Software simulation on replayed recordings; not validated on hardware or in real use."


class State(str, Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    UNCERTAIN = "UNCERTAIN"
    ACTIVE = "ACTIVE"


@dataclass(frozen=True)
class DeciderConfig:
    k_agree: int = 2
    n_rearm: int = 1

    def __post_init__(self):
        for name in ("k_agree", "n_rearm"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an integer >= 1, got {v!r}")


@dataclass(frozen=True)
class Event:
    label: str
    meaning: str
    first_window: int           # first window of the agreeing streak
    emit_window: int            # window at which the event was emitted
    k_agree: int
    notes: tuple[str, ...]
    context: Any = None         # caller-supplied metadata of the emitting window


@dataclass(frozen=True)
class Step:
    window: int
    input_label: Optional[str]
    state: State
    event: Optional[Event]


@dataclass
class EventDecider:
    config: DeciderConfig = field(default_factory=DeciderConfig)

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        self.state = State.IDLE
        self.window = -1
        self._cand: Optional[str] = None
        self._count = 0
        self._streak_start = -1
        self._active: Optional[str] = None
        self._idle_streak = 0
        self.suppressed_label_changes = 0
        self.events: list[Event] = []

    def update(self, label: Optional[str], context: Any = None) -> Step:
        if label is not None and label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        self.window += 1
        event = None
        k, n = self.config.k_agree, self.config.n_rearm

        if self.state is State.ACTIVE:
            if label == IDLE_LABEL:
                self._idle_streak += 1
                if self._idle_streak >= n:
                    self._clear()
                    self.state = State.IDLE
            else:
                self._idle_streak = 0          # invalid or non-idle window keeps the latch
                if label is not None and label != self._active:
                    self.suppressed_label_changes += 1
        elif label == IDLE_LABEL:
            self._clear()
            self.state = State.IDLE
        elif label is None:
            self._clear()
            self.state = State.UNCERTAIN
        else:
            disagreed = self._cand is not None and label != self._cand
            if self._cand == label:
                self._count += 1
            else:
                self._cand, self._count, self._streak_start = label, 1, self.window
            if self._count >= k:
                notes = (SIMULATION_NOTE,) + ((DOUBLE_SQUEEZE_NOTE,) if label == "DOUBLE_SQUEEZE" else ())
                event = Event(label, LABEL_MEANINGS[label], self._streak_start, self.window, k, notes, context)
                self.events.append(event)
                self._active, self._idle_streak = label, 0
                self._cand, self._count = None, 0
                self.state = State.ACTIVE
            else:
                self.state = State.UNCERTAIN if disagreed else State.CANDIDATE
        return Step(self.window, label, self.state, event)

    def _clear(self) -> None:
        self._cand, self._count, self._streak_start = None, 0, -1
        self._active, self._idle_streak = None, 0
