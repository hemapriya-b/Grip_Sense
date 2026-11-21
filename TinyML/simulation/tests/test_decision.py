"""Event state machine: idle state, K-agreement, deduplication, re-arm, latching, uncertainty."""
from __future__ import annotations

import subprocess
import sys
import unittest

from simulation import ROOT
from simulation.decision import DOUBLE_SQUEEZE_NOTE, DeciderConfig, Event, EventDecider, State

L, H, D, N = "LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"


def run(labels, **cfg):
    d = EventDecider(DeciderConfig(**cfg))
    steps = [d.update(x) for x in labels]
    return d, steps, [e.label for e in d.events]


class TestDecision(unittest.TestCase):
    def test_defaults_are_k2_n1(self):
        self.assertEqual((DeciderConfig().k_agree, DeciderConfig().n_rearm), (2, 1))

    def test_starts_idle_and_no_press_stays_idle(self):
        d, steps, ev = run([N] * 10)
        self.assertEqual(EventDecider().state, State.IDLE)
        self.assertTrue(all(s.state is State.IDLE for s in steps))
        self.assertEqual(ev, [])

    def test_single_window_is_only_a_candidate(self):
        d, steps, ev = run([L])
        self.assertEqual(steps[-1].state, State.CANDIDATE)
        self.assertEqual(ev, [])

    def test_two_agreeing_windows_emit_once(self):
        d, steps, ev = run([L, L])
        self.assertEqual(ev, [L])
        self.assertIs(steps[1].event, d.events[0])
        self.assertEqual((d.events[0].first_window, d.events[0].emit_window), (0, 1))
        self.assertEqual(steps[1].state, State.ACTIVE)

    def test_overlapping_windows_deduplicated(self):
        d, steps, ev = run([L] * 50)
        self.assertEqual(ev, [L])
        self.assertEqual(sum(s.event is not None for s in steps), 1)

    def test_rearm_after_one_no_press(self):
        self.assertEqual(run([L, L, L, N, L, L])[2], [L, L])
        self.assertEqual(run([L, L, N, H, H, N, N, D, D])[2], [L, H, D])

    def test_rearm_needs_n_consecutive_no_press(self):
        self.assertEqual(run([L, L, N, N, L, L], n_rearm=3)[2], [L])
        self.assertEqual(run([L, L, N, N, N, L, L], n_rearm=3)[2], [L, L])
        self.assertEqual(run([L, L, N, N, L, N, L, L], n_rearm=3)[2], [L])   # streak broken by L

    def test_latched_while_active_different_label_not_emitted(self):
        d, steps, ev = run([L, L, H, H, H, D, D])
        self.assertEqual(ev, [L])
        self.assertEqual(d.suppressed_label_changes, 5)
        self.assertTrue(all(s.state is State.ACTIVE for s in steps[1:]))

    def test_disagreement_is_uncertain_not_event(self):
        d, steps, ev = run([L, H])
        self.assertEqual(steps[-1].state, State.UNCERTAIN)
        self.assertEqual(ev, [])
        self.assertEqual(run([L, H, H])[2], [H])
        self.assertEqual(run([L, H, L, H, L, D])[2], [])   # alternating never emits

    def test_invalid_window_is_uncertain(self):
        d, steps, ev = run([L, None, L])
        self.assertEqual(steps[1].state, State.UNCERTAIN)
        self.assertEqual(ev, [])
        d, steps, ev = run([L, L, None, N, L, L])      # invalid while active keeps latch; N re-arms
        self.assertEqual(ev, [L, L])

    def test_k_configurable(self):
        self.assertEqual(run([L], k_agree=1)[2], [L])
        self.assertEqual(run([L, L], k_agree=3)[2], [])
        self.assertEqual(run([L, L, L], k_agree=3)[2], [L])

    def test_invalid_config_and_labels(self):
        for kw in ({"k_agree": 0}, {"n_rearm": 0}, {"k_agree": 1.5}, {"n_rearm": True}):
            with self.subTest(**kw), self.assertRaises(ValueError):
                DeciderConfig(**kw)
        with self.assertRaises(ValueError):
            EventDecider().update("SQUEEZE")

    def test_event_labels_meanings_and_no_confidence(self):
        d, _, ev = run([L, L, N, H, H, N, D, D])
        self.assertEqual([e.meaning for e in d.events], ["Hungry", "Pain", "Washroom"])
        fields = set(Event.__dataclass_fields__)
        self.assertFalse(fields & {"confidence", "probability", "proba", "score"})
        self.assertIn(DOUBLE_SQUEEZE_NOTE, d.events[2].notes)
        self.assertNotIn(DOUBLE_SQUEEZE_NOTE, d.events[0].notes)
        self.assertIn("NOT validated", DOUBLE_SQUEEZE_NOTE)

    def test_reset(self):
        d, _, _ = run([L, L])
        d.reset()
        self.assertEqual((d.state, d.events, d.window), (State.IDLE, [], -1))

    def test_decision_module_independent_of_model(self):
        code = ("import sys, simulation.decision; "
                "print(sorted(m for m in ('sklearn','joblib','ctypes','numpy','simulation.inference') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        # gripsense_features imports numpy/pandas for its constants; the model stack must stay out.
        # ('ctypes' is not checked here: numpy itself imports ctypes, so its presence proves nothing.)
        self.assertNotIn("sklearn", out.stdout)
        self.assertNotIn("joblib", out.stdout)
        self.assertNotIn("simulation.inference", out.stdout)
        # decision.py's own imports: only stdlib typing/dataclasses/enum and gripsense_features constants
        import ast
        tree = ast.parse((ROOT / "simulation" / "decision.py").read_text())
        mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(mods, {"__future__", "dataclasses", "enum", "typing", "gripsense_features"})


if __name__ == "__main__":
    unittest.main()
