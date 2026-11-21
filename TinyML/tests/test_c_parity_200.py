"""Python (sklearn) vs C prediction parity on all 200 recorded trials."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from . import _support as S


class TestCParity200(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="gripsense_parity_")
        cls.build = S.compile_c(Path(cls._tmp.name))
        cls.feats = S.features()
        cls.cols = S.bundle()["feature_columns"]
        cls.used = S.tree()["used_names"]
        cls.py_pred = list(S.bundle()["model"].predict(cls.feats[cls.cols]))
        cls.trials = [int(t) for t in cls.feats["trial"]]
        cls.stored = pd.read_csv(S.CPORT_RESULTS / "predictions_python_vs_c.csv",
                                 float_precision="round_trip").sort_values("trial").reset_index(drop=True)
        cls.c_feat = cls.c_raw = None
        if cls.build["exe_rc"] == 0:
            fv = "".join(f"{t} " + " ".join(repr(float(cls.feats.loc[i, c])) for c in cls.used) + "\n"
                         for i, t in enumerate(cls.trials))
            cls.r_feat = S.run_harness(cls.build["exe"], "features", fv)
            waves = {int(t): g.sort_values("elapsed_ms")["adc"].to_numpy() for t, g in S.raw_df().groupby("trial")}
            raw = "".join(f"{t} " + " ".join(map(str, waves[t])) + "\n" for t in cls.trials)
            cls.r_raw = S.run_harness(cls.build["exe"], "raw", raw)
            cls.c_feat = [line.split() for line in cls.r_feat.stdout.splitlines()]
            cls.c_raw = [line.split() for line in cls.r_raw.stdout.splitlines()]

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        if self.build["exe_rc"] != 0 or self.build["obj_rc"] != 0:
            self.fail(f"compilation failed:\n{self.build['diagnostics']}")

    # ------------------------------------------------------------ build
    def test_compiles_with_zero_warnings(self):
        self.assertEqual(self.build["n_warnings"], 0, self.build["diagnostics"])
        self.assertEqual(self.build["diagnostics"], "")

    def test_harness_info_matches_tree(self):
        r = S.subprocess.run([str(self.build["exe"]), "--info"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        lines = r.stdout.splitlines()
        self.assertIn("n_nodes 7", lines)
        self.assertEqual([l.split()[2] for l in lines if l.startswith("feature ")], self.used)

    # ------------------------------------------------------------ inputs
    def test_200_trials(self):
        self.assertEqual(len(self.trials), 200)
        self.assertEqual(len(set(self.trials)), 200)

    def test_python_features_equal_saved_trial_features(self):
        saved = pd.read_csv(S.ROOT / "reports" / "trial_features.csv",
                            float_precision="round_trip").sort_values("trial").reset_index(drop=True)
        self.assertTrue(np.array_equal(self.feats[self.cols].to_numpy(), saved[self.cols].to_numpy()))

    # ------------------------------------------------------------ parity
    def test_features_mode_200_of_200(self):
        self.assertEqual(self.r_feat.returncode, 0, self.r_feat.stderr)
        self.assertEqual(len(self.c_feat), 200)
        self.assertEqual([int(r[0]) for r in self.c_feat], self.trials)
        mism = [(t, p, r[2]) for t, p, r in zip(self.trials, self.py_pred, self.c_feat) if p != r[2]]
        self.assertEqual(mism, [], "features-mode disagreements (trial, python, c)")

    def test_raw_adc_mode_200_of_200(self):
        nf = len(self.used)
        self.assertEqual(self.r_raw.returncode, 0, self.r_raw.stderr)
        self.assertEqual(len(self.c_raw), 200)
        self.assertEqual([int(r[0]) for r in self.c_raw], self.trials)
        mism = [(t, p, r[2 + nf]) for t, p, r in zip(self.trials, self.py_pred, self.c_raw) if p != r[2 + nf]]
        self.assertEqual(mism, [], "raw-mode disagreements (trial, python, c)")

    def test_c_features_bit_exact(self):
        for k, name in enumerate(self.used):
            with self.subTest(feature=name):
                diffs = [(t, float(self.feats.loc[i, name]), float(r[1 + k]))
                         for i, (t, r) in enumerate(zip(self.trials, self.c_raw))
                         if float(self.feats.loc[i, name]) != float(r[1 + k])]
                self.assertEqual(diffs, [])

    def test_class_indices_consistent_with_names(self):
        from gripsense_features import LABELS
        for r in self.c_feat:
            self.assertEqual(LABELS[int(r[1])], r[2])

    # ------------------------------------------------------------ vs recorded results
    def test_matches_stored_predictions_python_vs_c(self):
        st = self.stored
        nf = len(self.used)
        self.assertEqual(list(st["trial"]), self.trials)
        self.assertEqual(list(st["python_sklearn"]), self.py_pred)
        self.assertEqual(list(st["c_from_features"]), [r[2] for r in self.c_feat])
        self.assertEqual(list(st["c_from_raw_adc"]), [r[2 + nf] for r in self.c_raw])
        for k, name in enumerate(self.used):
            self.assertEqual(list(st[f"py_{name}"]), [float(v) for v in self.feats[name]])
            self.assertEqual(list(st[f"c_{name}"]), [float(r[1 + k]) for r in self.c_raw])
        self.assertTrue(st["agree_features_mode"].all() and st["agree_raw_mode"].all())

    def test_matches_followup_predictions(self):
        fu = pd.read_csv(S.ROOT / "results" / "followup" / "predictions_all_models.csv")
        fu = fu[fu["model"] == "DT_depth2"].set_index("trial")["predicted"]
        self.assertEqual(len(fu), 80)
        mine = dict(zip(self.trials, self.py_pred))
        self.assertEqual({int(t): mine[int(t)] for t in fu.index}, {int(t): p for t, p in fu.items()})

    def test_matches_verification_summary(self):
        vs = S.read_json(S.CPORT_RESULTS / "verification_summary.json")
        self.assertEqual(vs["agreement_features_mode"], 200)
        self.assertEqual(vs["agreement_raw_adc_mode"], 200)
        self.assertTrue(vs["all_checks_pass"])


if __name__ == "__main__":
    unittest.main()
