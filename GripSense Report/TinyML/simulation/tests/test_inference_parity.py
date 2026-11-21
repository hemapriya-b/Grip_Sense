"""C (ctypes) and sklearn backends vs the stored reference predictions on all 200 whole trials."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from gripsense_features import LABELS
from simulation import ROOT
from simulation.inference import C_SRC, CTreeBackend, SklearnReferenceBackend
from simulation.stream import load_trials


class TestInferenceParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = CTreeBackend()
        cls.ref = SklearnReferenceBackend()
        cls.trials = load_trials()
        cls.stored = pd.read_csv(ROOT / "results" / "c_port" / "predictions_python_vs_c.csv",
                                 float_precision="round_trip").set_index("trial")
        cls.windows = {t: np.array(r.adc, dtype=np.int64) for t, r in cls.trials.items()}
        cls.c_pred = {t: cls.c.predict(w) for t, w in cls.windows.items()}

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def test_library_built_in_temp_dir_with_zero_warnings(self):
        self.assertEqual(self.c.n_warnings, 0, self.c.compile_diagnostics)
        self.assertNotIn(str(ROOT), str(self.c.lib_path))
        self.assertTrue(self.c.lib_path.exists())

    def test_c_matches_stored_reference_200_of_200(self):
        mism = [(t, p.label, self.stored.loc[t, "python_sklearn"]) for t, p in self.c_pred.items()
                if p.label != self.stored.loc[t, "python_sklearn"]]
        self.assertEqual(len(self.c_pred), 200)
        self.assertEqual(mism, [])

    def test_c_matches_stored_c_raw_mode(self):
        self.assertEqual({t: p.label for t, p in self.c_pred.items()},
                         {t: self.stored.loc[t, "c_from_raw_adc"] for t in self.c_pred})

    def test_sklearn_reference_backend_matches_stored(self):
        for t, w in self.windows.items():
            self.assertEqual(self.ref.predict(w).label, self.stored.loc[t, "python_sklearn"], t)

    def test_c_features_bit_exact_vs_stored(self):
        for t, w in self.windows.items():
            f = self.c.features(w)
            self.assertEqual(f, (self.stored.loc[t, "c_longest_active_run_frac"],
                                 self.stored.loc[t, "c_segment_1_mean"]), t)

    def test_matches_followup_predictions(self):
        fu = pd.read_csv(ROOT / "results" / "followup" / "predictions_all_models.csv")
        fu = fu[fu["model"] == "DT_depth2"]
        self.assertEqual(len(fu), 80)
        for t, p in zip(fu["trial"], fu["predicted"]):
            self.assertEqual(self.c_pred[int(t)].label, p)

    def test_output_labels_and_indices(self):
        self.assertEqual(LABELS, ["LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"])
        for p in self.c_pred.values():
            self.assertIn(p.label, LABELS)
            self.assertEqual(LABELS[p.class_index], p.label)
        self.assertEqual({p.label for p in self.c_pred.values()}, set(LABELS))

    def test_non_finite_features_rejected_by_c(self):
        for vec in ((float("nan"), 100.0), (0.5, float("nan")), (float("inf"), 100.0)):
            p = self.c.predict_features(vec)
            self.assertIsNone(p.label)
            self.assertEqual(p.class_index, -1)

    def test_no_confidence_in_predictions(self):
        self.assertEqual(set(type(next(iter(self.c_pred.values()))).__dataclass_fields__), {"label", "class_index"})
        # why: every leaf is pure, so predict_proba is always exactly 1.0 (uninformative)
        t = self.ref.model.tree_
        leaves = t.children_left == -1
        vals = t.value[leaves].reshape(leaves.sum(), -1)
        self.assertTrue(np.all(np.sort(vals, axis=1)[:, -1] == vals.sum(axis=1)))

    def test_source_hash_guard(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            fake = Path(td) / "decision_tree.c"
            fake.write_text(C_SRC.read_text().replace("<= GS_THRESHOLD", "< GS_THRESHOLD"))
            with self.assertRaises(RuntimeError):
                CTreeBackend(c_src=fake)


if __name__ == "__main__":
    unittest.main()
