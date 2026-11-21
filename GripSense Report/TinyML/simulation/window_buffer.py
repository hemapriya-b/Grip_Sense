"""
Fixed-size ring buffer of ADC samples (default 150 = 3 s at 20 ms, the existing trial window).

The buffer size is configurable, but the verified model and C feature code are defined for exactly
150 samples; inference rejects any other window length (see inference.py).
"""
from __future__ import annotations

from collections import deque
from numbers import Integral

import numpy as np

from gripsense_features import SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL

ADC_MIN, ADC_MAX = 0, 4095   # ESP32 12-bit ADC scale (documented in gripsense_features)
DEFAULT_WINDOW = SAMPLES_PER_TRIAL


def validate_adc(value) -> int:
    """Accept only integer ADC counts in 0..4095. Never clips or rounds."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"ADC value must be an integer, got {value!r}")
    v = int(value)
    if not ADC_MIN <= v <= ADC_MAX:
        raise ValueError(f"ADC value {v} outside {ADC_MIN}..{ADC_MAX}")
    return v


class WindowBuffer:
    """Holds the most recent `size` samples (objects with an `.adc` attribute), oldest first."""

    def __init__(self, size: int = DEFAULT_WINDOW):
        if isinstance(size, bool) or not isinstance(size, Integral) or size < 1:
            raise ValueError(f"window size must be a positive integer, got {size!r}")
        self.size = int(size)
        self._buf: deque = deque(maxlen=self.size)
        self.pushed_since_reset = 0

    @property
    def duration_ms(self) -> int:
        return self.size * SAMPLE_INTERVAL_MS

    def push(self, sample) -> None:
        validate_adc(sample.adc)
        self._buf.append(sample)
        self.pushed_since_reset += 1

    def reset(self) -> None:
        self._buf.clear()
        self.pushed_since_reset = 0

    def is_full(self) -> bool:
        return len(self._buf) == self.size

    def __len__(self) -> int:
        return len(self._buf)

    def samples(self) -> list:
        return list(self._buf)

    def adc_array(self) -> np.ndarray:
        return np.fromiter((s.adc for s in self._buf), dtype=np.uint16, count=len(self._buf))
