"""
End-to-end simulator runs with the real C backend on replayed recordings, plus file preservation.

Continuous runs concatenate recorded trials; they are not real continuous recordings and mixed
windows have no ground truth. Assertions on mixed windows are characterization (current behaviour),
not correctness claims.
"""
from __future__ import annotations

import subprocess
import sys
import unittest

from simulation import ROOT
from simulation.decision import DeciderConfig
from simulation.inference import CTreeBackend, SklearnReferenceBackend
from simulation.run_simulation import resolve_out_dir
from simulation.simulator import compare_aligned_to_reference, load_reference_predictions, simulate
from simulation.stream import load_trials

from . import BASELINE, snapshot

L, H, D, N = "LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"


class TestSimulatorEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = CTreeBackend()
        cls.ref = SklearnReferenceBackend()
        cls.trials = load_trials()
        cls.stored = load_reference_predictions()

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def test_isolated_all_200_match_stored_reference(self):
        res = simulate(self.trials, sorted(self.trials), "isolated", self.c, reference=self.ref)
        self.assertEqual(compare_aligned_to_reference(res, self.stored), (200, 200, []))
        self.assertTrue(all(w.backends_agree for w in res.windows))
        self.assertEqual(res.events, [])

    def test_continuous_aligned_windows_match_reference_and_backends_agree(self):
        for order, hop in (([1, 2, 3, 151, 152, 4], 10), ([60, 120, 180, 39, 7], 25), ([101, 102], 1)):
            with self.subTest(order=order, hop=hop):
                res = simulate(self.trials, order, "continuous", self.c, reference=self.ref, hop=hop)
                self.assertEqual(compare_aligned_to_reference(res, self.stored), (len(order), len(order), []))
                self.assertTrue(all(w.backends_agree for w in res.windows))

    def test_continuous_output_labels_valid(self):
        res = simulate(self.trials, [1, 60, 110, 160, 2], "continuous", self.c, hop=10)
        self.assertTrue({w.label for w in res.windows} <= {L, H, D, N})
        self.assertTrue({e.label for e in res.events} <= {L, H, D})

    def test_dedup_on_real_stream(self):
        res = simulate(self.trials, [1, 2, 3, 4, 5], "continuous", self.c, hop=10)
        non_idle = sum(w.label != N for w in res.windows)
        self.assertGreater(non_idle, 50)
        self.assertEqual(len(res.events), 1)       # one latched event over 61 overlapping windows

    def test_events_require_rearm_between_them(self):
        for order in ([1, 2, 3, 151, 152, 4], [60, 151, 110, 152, 30], [1, 151, 51, 152, 101, 153]):
            res = simulate(self.trials, order, "continuous", self.c, hop=10)
            emits = [e.emit_window for e in res.events]
            for a, b in zip(emits, emits[1:]):
                with self.subTest(order=order, between=(a, b)):
                    self.assertIn(N, [w.label for w in res.windows[a + 1:b]])

    def test_characterization_onset_window_misfires_as_double_squeeze(self):
        """Current behaviour, NOT desired: windows mixing rest with the start of LIGHT_PRESS trial 4
        are classified DOUBLE_SQUEEZE, so the latch emits DOUBLE_SQUEEZE and suppresses LIGHT_PRESS."""
        res = simulate(self.trials, [1, 2, 3, 151, 152, 4], "continuous", self.c, hop=10)
        self.assertEqual([(e.label, e.emit_window) for e in res.events], [(L, 1), (D, 64)])
        self.assertEqual(res.events[1].context["source_trials"], (152, 4))
        self.assertTrue(res.events[1].context["mixed"])
        self.assertEqual(res.windows[-1].label, L)     # the aligned trial-4 window itself is LIGHT_PRESS
        self.assertGreater(res.suppressed_label_changes, 0)

    def test_k_and_n_change_behaviour(self):
        order = [1, 2, 3, 151, 152, 4]
        strict = simulate(self.trials, order, "continuous", self.c, hop=10,
                          decider_config=DeciderConfig(k_agree=6, n_rearm=1))
        self.assertEqual([e.label for e in strict.events], [L, L])

    def test_out_dir_inside_project_refused(self):
        for p in (ROOT, ROOT / "results", ROOT / "models" / "x", ROOT / "simulation" / "out"):
            with self.subTest(path=p), self.assertRaises(SystemExit):
                resolve_out_dir(str(p))

    def test_cli_runs_isolated(self):
        r = subprocess.run([sys.executable, "-B", "-m", "simulation.run_simulation", "--mode", "isolated"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("aligned windows matching stored reference predictions: 200/200", r.stdout)


class TestPreservation(unittest.TestCase):
    """Runs last (see simulation/tests/__init__.py load_tests)."""

    def test_project_files_outside_simulation_unchanged(self):
        now = snapshot()
        self.assertIn("gripsense_clean_data.csv", BASELINE)
        self.assertIn("tests/test_preservation.py", BASELINE)
        changed = sorted(f for f in BASELINE if f in now and now[f] != BASELINE[f])
        self.assertEqual((changed, sorted(set(BASELINE) - set(now)), sorted(set(now) - set(BASELINE))),
                         ([], [], []))


if __name__ == "__main__":
    unittest.main()
