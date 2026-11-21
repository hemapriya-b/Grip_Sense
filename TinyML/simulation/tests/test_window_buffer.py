"""Window size, ordering, reset and ADC validation of the ring buffer."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from simulation.inference import as_window
from simulation.window_buffer import DEFAULT_WINDOW, WindowBuffer, validate_adc


def s(adc):
    return SimpleNamespace(adc=adc)


class TestWindowBuffer(unittest.TestCase):
    def test_default_is_150_samples_3_seconds(self):
        b = WindowBuffer()
        self.assertEqual(DEFAULT_WINDOW, 150)
        self.assertEqual(b.size, 150)
        self.assertEqual(b.duration_ms, 3000)

    def test_full_only_after_150_pushes(self):
        b = WindowBuffer()
        for i in range(149):
            b.push(s(i))
        self.assertFalse(b.is_full())
        b.push(s(149))
        self.assertTrue(b.is_full())
        self.assertEqual(len(b), 150)

    def test_ring_drops_oldest_and_keeps_order(self):
        b = WindowBuffer()
        for i in range(160):
            b.push(s(i))
        self.assertEqual(len(b), 150)
        self.assertEqual(b.adc_array().tolist(), list(range(10, 160)))
        self.assertEqual(b.pushed_since_reset, 160)

    def test_adc_array_is_uint16(self):
        b = WindowBuffer(3)
        for v in (0, 4095, 7):
            b.push(s(v))
        a = b.adc_array()
        self.assertEqual(a.dtype, np.uint16)
        self.assertEqual(a.tolist(), [0, 4095, 7])

    def test_reset_empties_buffer_and_counter(self):
        b = WindowBuffer()
        for i in range(200):
            b.push(s(i))
        b.reset()
        self.assertEqual(len(b), 0)
        self.assertFalse(b.is_full())
        self.assertEqual(b.pushed_since_reset, 0)
        for i in range(149):
            b.push(s(1000 + i))
        self.assertFalse(b.is_full())     # nothing from before the reset is counted
        self.assertEqual(b.adc_array().tolist(), list(range(1000, 1149)))

    def test_configurable_size(self):
        b = WindowBuffer(5)
        for i in range(7):
            b.push(s(i))
        self.assertEqual(b.adc_array().tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(b.duration_ms, 100)

    def test_invalid_sizes_rejected(self):
        for bad in (0, -1, 1.5, True, "150"):
            with self.subTest(size=bad), self.assertRaises(ValueError):
                WindowBuffer(bad)

    def test_invalid_adc_rejected_not_clipped(self):
        b = WindowBuffer()
        for bad in (-1, 4096, 1.5, float("nan"), True, None, "12"):
            with self.subTest(adc=bad), self.assertRaises(ValueError):
                b.push(s(bad))
        self.assertEqual(len(b), 0)
        self.assertEqual(validate_adc(np.int64(4095)), 4095)

    def test_inference_requires_exactly_150_integer_samples(self):
        for bad in (np.zeros(149, dtype=int), np.zeros(151, dtype=int), np.zeros((2, 75), dtype=int),
                    np.zeros(150, dtype=float), np.full(150, 4096)):
            with self.subTest(shape=bad.shape, dtype=bad.dtype), self.assertRaises(ValueError):
                as_window(bad)
        self.assertEqual(as_window(np.arange(150)).dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
