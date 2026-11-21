"""
No pre-existing project file was changed, removed, or added by the test suite.

The baseline (tests._support.BASELINE) is taken when the `tests` package is first imported,
before any test runs. unittest discovery runs modules in sorted file-name order, so this module
(test_preservation) runs after test_c_boundary_26, test_c_parity_200 and test_model_frozen.
Scope: every file under the project root except `.venv/`, `tests/`, and `.DS_Store` files.
"""
from __future__ import annotations

import unittest

from . import _support as S


class TestPreservation(unittest.TestCase):
    def test_baseline_is_nonempty(self):
        self.assertIn("gripsense_clean_data.csv", S.BASELINE)
        self.assertIn("models/followup/DT_depth2.joblib", S.BASELINE)
        self.assertIn("models/c_port/decision_tree.c", S.BASELINE)

    def test_no_files_changed_removed_or_added(self):
        now = S.snapshot()
        changed = sorted(f for f in S.BASELINE if f in now and now[f] != S.BASELINE[f])
        removed = sorted(set(S.BASELINE) - set(now))
        added = sorted(set(now) - set(S.BASELINE))
        self.assertEqual((changed, removed, added), ([], [], []))


if __name__ == "__main__":
    unittest.main()
