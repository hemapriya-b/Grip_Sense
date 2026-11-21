"""Window/trial-boundary accounting and reporting of the Step 3A transition diagnostics."""
from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from simulation import ROOT
from simulation.diagnostics import transition_report as tr
from simulation.inference import CTreeBackend, SklearnReferenceBackend
from simulation.stream import load_trials

from . import BASELINE, snapshot

ORDER = [1, 2, 3, 151, 152, 4]
L, D, N = "LIGHT_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"


class TestTransitionReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = CTreeBackend()
        cls.trials = load_trials()
        cls.res, cls.bounds, cls.df = tr.build_rows(cls.trials, ORDER, hop=10, context=10,
                                                    c=cls.c, ref=SklearnReferenceBackend())

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    # ------------------------------------------------------------ boundaries and selection
    def test_label_changing_boundaries(self):
        self.assertEqual([(b.before_trial, b.after_trial, b.stream_index) for b in self.bounds],
                         [(3, 151, 450), (152, 4, 750)])
        all_b = tr.boundaries(self.trials, ORDER, only_label_changes=False)
        self.assertEqual([b.stream_index for b in all_b], [150, 300, 450, 600, 750])

    def test_window_selection_counts(self):
        counts = self.df.groupby(["boundary", "position"]).size().to_dict()
        self.assertEqual(counts[("3->151", "before")], 10)
        self.assertEqual(counts[("3->151", "straddling")], 14)   # (150 - hop) / hop windows straddle
        self.assertEqual(counts[("3->151", "after")], 10)
        self.assertEqual(counts[("152->4", "before")], 10)
        self.assertEqual(counts[("152->4", "straddling")], 14)
        self.assertEqual(counts[("152->4", "after")], 1)         # stream ends: only one window starts at/after 750

    def test_positions_relative_to_boundary(self):
        for _, r in self.df.iterrows():
            b = r["boundary_stream_index"]
            if r["position"] == "before":
                self.assertLess(r["end_index"], b)
            elif r["position"] == "after":
                self.assertGreaterEqual(r["start_index"], b)
            else:
                self.assertTrue(r["start_index"] < b <= r["end_index"])

    # ------------------------------------------------------------ index / time / proportion accounting
    def test_indices_and_timestamps(self):
        for _, r in self.df.iterrows():
            self.assertEqual(r["end_index"] - r["start_index"], 149)
            self.assertEqual(r["start_ms"], r["start_index"] * 20)
            self.assertEqual(r["end_ms"], r["end_index"] * 20)
            self.assertEqual(r["start_index"] % 10, 0)                 # hop 10 from stream start
            seg, trial, idx = tr.locate(ORDER, r["start_index"])
            self.assertEqual(r["start_trial_idx"], f"{trial}[{idx}]")
            seg, trial, idx = tr.locate(ORDER, r["end_index"])
            self.assertEqual(r["end_trial_idx"], f"{trial}[{idx}]")

    def test_proportions_sum_and_match_arithmetic(self):
        for _, r in self.df.iterrows():
            parts = [(int(p.split(":")[0]), int(p.split(":")[1].split("/")[0])) for p in r["proportions"].split()]
            self.assertEqual(sum(n for _, n in parts), 150)
            self.assertEqual("+".join(str(t) for t, _ in parts), r["source_trials"])
            self.assertEqual(r["kind"] == "MIXED", len(parts) > 1)
            if len(parts) == 2:
                first_n = 150 - (r["start_index"] % 150)
                self.assertEqual(parts[0][1], first_n)

    def test_window_samples_reconstructed_exactly(self):
        stream = tr.stream_array(self.trials, ORDER)
        self.assertEqual(len(stream), 900)
        for _, r in self.df.iterrows():
            self.assertEqual(r["c_label"], r["c_label_recomputed"])   # same window the simulator classified

    # ------------------------------------------------------------ labels
    def test_no_ground_truth_on_mixed_windows(self):
        mixed = self.df[self.df["kind"] == "MIXED"]
        self.assertGreater(len(mixed), 0)
        self.assertTrue(mixed["true_label"].isna().all())
        aligned = self.df[self.df["kind"] == "ALIGNED"]
        self.assertEqual(sorted(aligned["source_trials"].unique()), ["151", "152", "3", "4"])
        for _, r in aligned.iterrows():
            self.assertEqual(r["true_label"], self.trials[int(r["source_trials"])].label)

    def test_c_reference_and_path_agree(self):
        self.assertTrue((self.df["c_label"] == self.df["reference_label"]).all())
        self.assertTrue((self.df["c_label"] == self.df["path_leaf"]).all())

    def test_decision_path_matches_stored_boundary_vectors(self):
        """The diagnostic tree walk reproduces sklearn/C on the 26 stored threshold-boundary vectors."""
        bt = pd.read_csv(ROOT / "results" / "c_port" / "boundary_tests.csv", float_precision="round_trip")
        nodes = tr.load_nodes()
        for _, r in bt.iterrows():
            leaf, _ = tr.decision_path(nodes, {"longest_active_run_frac": r["longest_active_run_frac"],
                                               "segment_1_mean": r["segment_1_mean"]})
            self.assertEqual(leaf, r["python_sklearn"])

    # ------------------------------------------------------------ features
    def test_aligned_window_features_equal_complete_trial_features(self):
        ref = self.df.attrs["complete_trial_features"]
        for _, r in self.df[self.df["kind"] == "ALIGNED"].iterrows():
            f = ref[int(r["source_trials"])]
            self.assertEqual(r["longest_active_run_frac"], f["longest_active_run_frac"])
            self.assertEqual(r["segment_1_mean"], f["segment_1_mean"])

    def test_run_length_matches_feature(self):
        for _, r in self.df.iterrows():
            self.assertEqual(r["longest_run_samples"] / 150, r["longest_active_run_frac"])

    def test_segment1_source_is_window_offset_15_to_29(self):
        self.assertEqual(tr.SEG1, (15, 29))
        for _, r in self.df.iterrows():
            a = tr.locate(ORDER, r["start_index"] + 15)
            b = tr.locate(ORDER, r["start_index"] + 29)
            self.assertEqual(r["segment_1_source"], f"{a[1]}[{a[2]}]..{b[1]}[{b[2]}]")

    # ------------------------------------------------------------ observed cause (characterization)
    def test_double_squeeze_windows_have_run_between_tree_thresholds(self):
        """Every DOUBLE_SQUEEZE window here has a longest active run of 8..53 samples, i.e. between the
        node-1 threshold (7/150) and the root threshold (53/150)."""
        ds = self.df[self.df["c_label"] == D]
        self.assertEqual(sorted(ds["window"].unique().tolist()), [40, 41, 42, 43, 44, 63, 64, 65, 66, 67])
        self.assertTrue(((ds["longest_run_samples"] > 7) & (ds["longest_run_samples"] <= 53)).all())
        self.assertTrue((ds["kind"] == "MIXED").all())
        others = self.df[self.df["c_label"] != D]
        self.assertFalse(((others["longest_run_samples"] > 7) & (others["longest_run_samples"] <= 53)).any())

    def test_ds_runs_come_from_one_light_trial(self):
        ds = self.df[self.df["c_label"] == D]
        for _, r in ds.iterrows():
            loc = r["longest_run_location"]
            trial = "3" if r["boundary"] == "3->151" else "4"
            self.assertTrue(loc.startswith(f"{trial}[") and f"..{trial}[" in loc, loc)

    def test_window_67_run_is_exactly_at_root_threshold(self):
        r = self.df[(self.df["window"] == 67)].iloc[0]
        self.assertEqual(r["longest_run_samples"], 53)
        thr = tr.load_nodes()[0]["threshold"]
        self.assertLessEqual(float(np.float32(53 / 150)), thr)
        self.assertGreater(float(np.float32(54 / 150)), thr)

    def test_event_rows(self):
        ev = self.df.dropna(subset=["event"]).drop_duplicates("window")
        self.assertEqual(ev[["window", "event"]].values.tolist(), [[64, D]])
        self.assertEqual(int(ev.iloc[0]["event_streak_first_window"]), 63)
        self.assertEqual(int(ev.iloc[0]["event_streak_start_index"]), 630)
        self.assertEqual([(e.label, e.emit_window) for e in self.res.events], [(L, 1), (D, 64)])

    # ------------------------------------------------------------ CLI
    def test_cli_report(self):
        r = subprocess.run([sys.executable, "-B", "-m", "simulation.diagnostics.transition_report"],
                           cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("=== Boundary 3 (LIGHT_PRESS) -> 151 (NO_PRESS) at stream index 450", r.stdout)
        self.assertIn("=== Boundary 152 (NO_PRESS) -> 4 (LIGHT_PRESS) at stream index 750", r.stdout)
        self.assertIn("truth=(none: mixed)", r.stdout)
        self.assertNotIn("nan", r.stdout)
        self.assertNotIn("FAIL", r.stdout)

    def test_csv_inside_project_refused(self):
        with self.assertRaises(SystemExit):
            tr.main(["--csv", str(ROOT / "results" / "x.csv")])


class TestPreservation(unittest.TestCase):
    """Runs last (see simulation/diagnostics/tests/__init__.py)."""

    def test_files_outside_diagnostics_unchanged(self):
        now = snapshot()
        for must in ("gripsense_clean_data.csv", "simulation/simulator.py", "simulation/decision.py",
                     "tests/test_preservation.py", "models/c_port/decision_tree.c"):
            self.assertIn(must, BASELINE)
        changed = sorted(f for f in BASELINE if f in now and now[f] != BASELINE[f])
        self.assertEqual((changed, sorted(set(BASELINE) - set(now)), sorted(set(now) - set(BASELINE))),
                         ([], [], []))


if __name__ == "__main__":
    unittest.main()
