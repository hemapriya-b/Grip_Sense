"""CSV replay ordering, trial boundaries, trial isolation and mixed-window tagging."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gripsense_features import LABELS
from simulation.inference import Prediction
from simulation.simulator import simulate
from simulation.stream import DEFAULT_CSV, load_trials, replay


class EchoBackend:
    """Test double: records every window it is given; always predicts NO_PRESS."""
    def __init__(self):
        self.windows = []

    def predict(self, adc):
        self.windows.append(np.array(adc))
        return Prediction("NO_PRESS", 3)


class TestStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = load_trials()
        cls.df = pd.read_csv(DEFAULT_CSV)

    def test_load_all_trials(self):
        self.assertEqual(sorted(self.trials), list(range(1, 201)))
        for rec in self.trials.values():
            self.assertEqual(len(rec.adc), 150)
            self.assertEqual(rec.elapsed_ms, tuple(range(0, 3000, 20)))
            self.assertIn(rec.label, LABELS)

    def test_trial_samples_match_csv_order(self):
        for tid in (1, 77, 123, 200):
            g = self.df[self.df["trial"] == tid].sort_values("elapsed_ms")
            self.assertEqual(self.trials[tid].adc, tuple(int(v) for v in g["adc"]))
            self.assertEqual(self.trials[tid].label, g["label"].iloc[0])

    def test_replay_preserves_requested_order_and_boundaries(self):
        order = [5, 151, 5, 120]
        out = list(replay(self.trials, order))
        self.assertEqual(len(out), 600)
        for seg, tid in enumerate(order):
            chunk = out[seg * 150:(seg + 1) * 150]
            self.assertTrue(all(x.segment == seg and x.trial == tid for x in chunk))
            self.assertEqual([x.index for x in chunk], list(range(150)))
            self.assertEqual(tuple(x.adc for x in chunk), self.trials[tid].adc)
        self.assertEqual([x.stream_index for x in out], list(range(600)))
        self.assertEqual(out[-1].stream_ms, 599 * 20)

    def test_isolated_one_aligned_window_per_trial_any_hop(self):
        order = [1, 60, 110, 160]
        for hop in (1, 10, 37, 150, 1000):
            with self.subTest(hop=hop):
                be = EchoBackend()
                res = simulate(self.trials, order, "isolated", be, hop=hop)
                self.assertEqual(len(res.windows), 4)
                for w, tid, adc in zip(res.windows, order, be.windows):
                    self.assertEqual(w.source_trials, (tid,))
                    self.assertTrue(w.aligned and not w.mixed)
                    self.assertEqual((w.start, w.end), ((tid, 0), (tid, 149)))
                    self.assertEqual(tuple(adc.tolist()), self.trials[tid].adc)   # trial isolation
                    self.assertEqual(w.true_label, self.trials[tid].label)
                self.assertEqual(res.events, [])

    def test_continuous_window_count_and_tagging(self):
        order, hop = [1, 151, 2], 10
        be = EchoBackend()
        res = simulate(self.trials, order, "continuous", be, hop=hop)
        self.assertEqual(len(res.windows), (450 - 150) // hop + 1)
        flat = np.concatenate([np.array(self.trials[t].adc) for t in order])
        for w, adc in zip(res.windows, be.windows):
            self.assertEqual(adc.tolist(), flat[w.stream_start_index:w.stream_end_index + 1].tolist())
            self.assertEqual(w.stream_end_index - w.stream_start_index, 149)
            self.assertEqual(w.mixed, len(w.segments) > 1)
            if w.mixed:
                self.assertIsNone(w.true_label)
        self.assertEqual([w.source_trials for w in res.windows if w.aligned], [(1,), (151,), (2,)])
        self.assertEqual(res.windows[1].source_trials, (1, 151))

    def test_continuous_hop_not_dividing_150_has_fewer_aligned_windows(self):
        res = simulate(self.trials, [1, 2, 3], "continuous", EchoBackend(), hop=7)
        # windows start at stream positions 0, 7, 14, ...; aligned only where start % 150 == 0
        self.assertEqual([w.stream_start_index for w in res.windows if w.aligned], [0])

    def test_invalid_arguments(self):
        for kw in ({"mode": "live"}, {"hop": 0}, {"window_size": 100}):
            args = {"mode": "continuous", **kw}
            with self.subTest(**kw), self.assertRaises(ValueError):
                simulate(self.trials, [1], backend=EchoBackend(), **args)


if __name__ == "__main__":
    unittest.main()
