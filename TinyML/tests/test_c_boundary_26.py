"""
Threshold-boundary checks: 26 synthetic feature vectors that probe each split threshold with its
float64/float32 neighbours and the nearest achievable feature values.

The vectors are (a) read from the stored results/c_port/boundary_tests.csv and (b) re-derived
independently here with the same rules as export_and_verify_c_port.py; both are checked.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gripsense_features import SAMPLES_PER_TRIAL

from . import _support as S


def derive_boundary_vectors(model, nodes, feats, cols) -> np.ndarray:
    """Re-derivation of export_and_verify_c_port.py step 4c (full 49-feature vectors)."""
    base = {}
    for _, r in feats.iterrows():
        vec = r[cols].to_numpy(dtype=float)
        for nd in model.decision_path(pd.DataFrame([vec], columns=cols)).indices:
            base.setdefault(int(nd), vec)
    out = []
    for nd in nodes:
        if nd["is_leaf"]:
            continue
        thr = nd["threshold"]
        f32 = np.float32(thr)
        cands = {thr, np.nextafter(thr, -np.inf), np.nextafter(thr, np.inf), float(f32),
                 float(np.nextafter(f32, np.float32(-np.inf))), float(np.nextafter(f32, np.float32(np.inf)))}
        if nd["feature_name"] == "longest_active_run_frac":
            k = int(round(thr * SAMPLES_PER_TRIAL))
            cands |= {(k - 1) / SAMPLES_PER_TRIAL, k / SAMPLES_PER_TRIAL, (k + 1) / SAMPLES_PER_TRIAL}
        else:
            cands |= {round(thr, 1), round(thr, 1) + 0.1, np.floor(thr * 15) / 15, np.ceil(thr * 15) / 15}
        for v in sorted(cands):
            vec = base[nd["node"]].copy()
            vec[nd["feature_orig_index"]] = v
            out.append(vec)
    return np.array(out)


def naive_double_predict(nodes, vec_by_name: dict) -> str:
    """Tree walk WITHOUT the float32 cast (diagnostic; expected to differ from sklearn at boundaries)."""
    node = 0
    while not nodes[node]["is_leaf"]:
        n = nodes[node]
        node = n["left"] if vec_by_name[n["feature_name"]] <= n["threshold"] else n["right"]
    return nodes[node]["leaf_class"]


class TestCBoundary26(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="gripsense_boundary_")
        cls.build = S.compile_c(Path(cls._tmp.name))
        cls.model = S.bundle()["model"]
        cls.cols = S.bundle()["feature_columns"]
        cls.nodes = S.tree()["nodes"]
        cls.used = S.tree()["used_names"]
        cls.derived_full = derive_boundary_vectors(cls.model, cls.nodes, S.features(), cls.cols)
        cls.derived = pd.DataFrame(cls.derived_full, columns=cls.cols)[cls.used].reset_index(drop=True)
        cls.stored = pd.read_csv(S.CPORT_RESULTS / "boundary_tests.csv", float_precision="round_trip")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        if self.build["exe_rc"] != 0:
            self.fail(f"compilation failed:\n{self.build['diagnostics']}")

    def c_predict(self, vecs: pd.DataFrame) -> list[str]:
        lines = "".join(f"b{i} " + " ".join(repr(float(vecs.loc[i, c])) for c in self.used) + "\n"
                        for i in range(len(vecs)))
        r = S.run_harness(self.build["exe"], "features", lines)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = [line.split() for line in r.stdout.splitlines()]
        self.assertEqual([o[0] for o in out], [f"b{i}" for i in range(len(vecs))])
        return [o[2] for o in out]

    def sklearn_predict_two_feature(self, vecs: pd.DataFrame) -> list[str]:
        """sklearn on the stored 2-feature vectors. Columns the tree never reads are set to 0;
        test_unused_columns_do_not_matter checks that this does not change predictions."""
        X = pd.DataFrame(np.zeros((len(vecs), len(self.cols))), columns=self.cols)
        for c in self.used:
            X[c] = vecs[c].to_numpy()
        return list(self.model.predict(X))

    # ------------------------------------------------------------ stored vs re-derived
    def test_count_is_26(self):
        self.assertEqual(len(self.stored), 26)
        self.assertEqual(len(self.derived), 26)

    def test_rederived_vectors_equal_stored_bit_exact(self):
        for c in self.used:
            with self.subTest(feature=c):
                self.assertEqual(list(self.derived[c]), list(self.stored[c]))

    def test_every_threshold_probed_both_sides(self):
        """For each split node, at least one vector goes left and one goes right (float32 semantics)."""
        for nd in self.nodes:
            if nd["is_leaf"]:
                continue
            v = self.derived[nd["feature_name"]].to_numpy()
            left = np.float32(v).astype(float) <= nd["threshold"]
            with self.subTest(node=nd["node"]):
                self.assertTrue(left.any() and (~left).any())

    # ------------------------------------------------------------ C vs sklearn
    def test_c_agrees_with_sklearn_on_rederived_vectors(self):
        py = list(self.model.predict(pd.DataFrame(self.derived_full, columns=self.cols)))
        c = self.c_predict(self.derived)
        self.assertEqual(sum(a == b for a, b in zip(py, c)), 26)

    def test_c_agrees_with_sklearn_on_stored_vectors(self):
        py = self.sklearn_predict_two_feature(self.stored)
        c = self.c_predict(self.stored)
        self.assertEqual(sum(a == b for a, b in zip(py, c)), 26)

    def test_matches_stored_prediction_columns(self):
        self.assertEqual(self.sklearn_predict_two_feature(self.stored), list(self.stored["python_sklearn"]))
        self.assertEqual(self.c_predict(self.stored), list(self.stored["c"]))
        self.assertTrue(self.stored["agree"].all())

    def test_unused_columns_do_not_matter(self):
        full = list(self.model.predict(pd.DataFrame(self.derived_full, columns=self.cols)))
        self.assertEqual(self.sklearn_predict_two_feature(self.derived), full)

    # ------------------------------------------------------------ float32-cast semantics
    def test_naive_double_compare_disagrees_on_exactly_2(self):
        """Confirms the boundary set really exercises the float32 cast: a naive double walk differs on 2."""
        py = list(self.model.predict(pd.DataFrame(self.derived_full, columns=self.cols)))
        naive = [naive_double_predict(self.nodes, row) for row in self.derived.to_dict(orient="records")]
        self.assertEqual(naive, list(self.stored["naive_double_compare"]))
        self.assertEqual(sum(a != b for a, b in zip(naive, py)), 2)
        vs = S.read_json(S.CPORT_RESULTS / "verification_summary.json")["boundary_tests"]
        self.assertEqual((vs["n"], vs["agree"], vs["naive_double_compare_disagreements"]), (26, 26, 2))

    # ------------------------------------------------------------ invalid input
    def test_non_finite_input_rejected(self):
        # "x nan 100": NaN at the root feature; "x 0.5 nan": NaN reached at node 4 (segment_1_mean)
        for line in ("x nan 100\n", "x 0.5 nan\n", "x inf 100\n"):
            with self.subTest(line=line.strip()):
                r = S.run_harness(self.build["exe"], "features", line)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("invalid input", r.stderr)


if __name__ == "__main__":
    unittest.main()
