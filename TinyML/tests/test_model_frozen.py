"""The selected depth-2 tree, its C port, and the source CSV are exactly the recorded artifacts."""
from __future__ import annotations

import unittest

from gripsense_features import LABELS

from . import _support as S


class TestModelFrozen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = S.read_json(S.TREE_METADATA)
        cls.cport_baseline = S.read_json(S.CPORT_RESULTS / "preservation_baseline.json")["files"]

    def test_model_hash_matches_records(self):
        h = S.sha(S.MODEL_PATH)
        self.assertEqual(h, self.meta["source_model_sha256"])
        self.assertEqual(h, self.cport_baseline["models/followup/DT_depth2.joblib"])

    def test_c_source_hash_matches_metadata(self):
        self.assertEqual(S.sha(S.C_SRC), self.meta["c_source_sha256"])

    def test_csv_hash_matches_phase1_metrics(self):
        metrics = S.read_json(S.ROOT / "reports" / "metrics.json")
        self.assertEqual(S.sha(S.CSV_PATH), metrics["input"]["sha256"])

    def test_tree_shape(self):
        m = S.bundle()["model"]
        self.assertEqual(type(m).__name__, "DecisionTreeClassifier")
        self.assertEqual(m.tree_.max_depth, 2)
        self.assertEqual(m.tree_.node_count, 7)
        self.assertEqual(S.tree()["used_names"], ["longest_active_run_frac", "segment_1_mean"])
        self.assertEqual(S.bundle()["name"], "DT_depth2")

    def test_nodes_match_metadata_exactly(self):
        # thresholds compared as exact float64 hex strings, not rounded decimals
        self.assertEqual(S.tree()["nodes"], self.meta["nodes"])
        self.assertEqual(self.meta["label_order_c_enum"], LABELS)

    def test_nodes_match_saved_rules_file(self):
        rules = (S.ROOT / "models" / "followup" / "DT_depth2_rules.txt").read_text()
        for nd in S.tree()["nodes"]:
            if nd["is_leaf"]:
                self.assertIn(f"class: {nd['leaf_class']}", rules)
            else:
                self.assertIn(f"{nd['feature_name']} <= {nd['threshold']:.3f}", rules)

    def test_regenerated_c_is_byte_identical(self):
        """Generating C from the saved model in memory reproduces the saved decision_tree.c."""
        regenerated = S.port.generate_c(S.tree(), S.segment_bounds())
        self.assertEqual(regenerated, S.C_SRC.read_text())


if __name__ == "__main__":
    unittest.main()
