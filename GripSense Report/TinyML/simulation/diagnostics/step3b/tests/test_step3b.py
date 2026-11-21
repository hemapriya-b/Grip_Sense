"""Tests for the Step 3B investigation code (splits, windows, candidates, strategies, scoring)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

from gripsense_features import extract_trial_features
from simulation import ROOT
from simulation.diagnostics.step3b import common as cm
from simulation.diagnostics.step3b import event_strategies as es
from simulation.diagnostics.step3b import run_step3b as rs
from simulation.diagnostics.step3b.candidates import build_candidates, make_gate, metrics, trial_level_evaluation
from simulation.diagnostics.step3b.transitions import mixed_window_probe, record_predictions, score_events
from simulation.inference import MODEL_PATH, CTreeBackend
from simulation.simulator import simulate

from . import BASELINE, snapshot

L, H, D, N = "LIGHT_PRESS", "HARD_PRESS", "DOUBLE_SQUEEZE", "NO_PRESS"


class TestSplitsAndSequences(unittest.TestCase):
    def test_manifest_split_sizes(self):
        tf = cm.trial_features()
        self.assertEqual(tf["split"].value_counts().to_dict(), {"train": 120, "validation": 40, "test": 40})

    def test_sequences_use_only_their_split(self):
        for split in ("validation", "test"):
            for seed in (0, 1, 2):
                o = cm.rest_separated_order(split, seed)
                self.assertTrue(all(cm.manifest()[t] == split for t in o), (split, seed))
                press = [t for t in o if cm.trials()[t].label != N]
                self.assertEqual(sorted(press), sorted(t for lab in cm.PRESS_LABELS for t in cm.trial_ids(split, lab)))
                labs = [cm.trials()[t].label for t in o]
                self.assertEqual(labs[0], N)
                self.assertEqual(labs[-1], N)
                self.assertTrue(all(not (labs[i] != N and labs[i + 1] != N) for i in range(len(labs) - 1)))

    def test_sequences_deterministic_and_seed_dependent(self):
        self.assertEqual(cm.rest_separated_order("test", 1), cm.rest_separated_order("test", 1))
        self.assertNotEqual(cm.rest_separated_order("test", 0), cm.rest_separated_order("test", 1))

    def test_episodes(self):
        eps = cm.episodes((1, 2, 3, 151, 152, 4))
        self.assertEqual([(e.label, e.first_segment, e.last_segment, e.start_index, e.end_index) for e in eps],
                         [(L, 0, 2, 0, 449), (L, 5, 5, 750, 899)])
        with self.assertRaises(ValueError):
            cm.episodes((151, 1, 51, 152))

    def test_window_table_matches_production_schedule_and_never_labels_mixed(self):
        order = (151, 1, 152)
        wt = cm.window_table(order, 10)
        with CTreeBackend() as c:
            res = simulate(cm.trials(), list(order), "continuous", c, hop=10)
        self.assertEqual(list(wt["start"]), [w.stream_start_index for w in res.windows])
        self.assertEqual(list(wt["mixed"]), [w.mixed for w in res.windows])
        self.assertTrue(wt.loc[wt["mixed"], "true_label"].isna().all())
        self.assertEqual(wt.loc[wt["aligned"], "true_label"].tolist(), [N, L, N])

    def test_window_features_use_existing_definitions(self):
        order = (151, 7)
        stream = cm.stream_of(order)
        X = cm.window_feature_frame(stream, [0, 37, 150])
        for i, s in enumerate([0, 37, 150]):
            self.assertEqual(X.iloc[i].to_dict(), {k: v for k, v in extract_trial_features(stream[s:s + 150]).items()})
        tf = cm.trial_features().set_index("trial")
        self.assertEqual(X.iloc[2].tolist(), tf.loc[7, cm.feature_columns()].tolist())


class TestCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = CTreeBackend()
        cls.cands = {x.name: x for x in build_candidates(cls.c)}
        cls.tf = cm.trial_features()
        cls.adc = [np.array(cm.trials()[t].adc) for t in cls.tf["trial"]]

    @classmethod
    def tearDownClass(cls):
        cls.c.close()

    def test_candidate_names(self):
        self.assertEqual(list(self.cands), ["current_dt2", "gate_ev2", "gate_ev3", "dt2_robust", "dt3_robust", "dt3_all"])

    def test_current_matches_stored_reference(self):
        stored = pd.read_csv(ROOT / "results" / "c_port" / "predictions_python_vs_c.csv").set_index("trial")
        pred = self.cands["current_dt2"].predict(self.tf, self.adc)
        self.assertEqual(list(pred), [stored.loc[t, "python_sklearn"] for t in self.tf["trial"]])

    def test_gate_with_zero_events_reproduces_current_tree(self):
        train = self.tf[self.tf["split"] == "train"]
        g0 = make_gate(train, 0)
        self.assertEqual(list(g0.predict(self.tf)), list(self.cands["current_dt2"].predict(self.tf, self.adc)))
        wt_order = (151, 1, 152, 101, 153)
        stream = cm.stream_of(wt_order)
        starts = cm.window_starts(len(stream), 5)
        X = cm.window_feature_frame(stream, starts)
        self.assertEqual(list(g0.predict(X)),
                         list(self.cands["current_dt2"].predict(X, [stream[s:s + 150] for s in starts])))

    def test_gate_never_ds_below_event_threshold(self):
        for n, k in (("gate_ev2", 2), ("gate_ev3", 3)):
            p = self.cands[n].predict(self.tf)
            self.assertFalse(((p == D) & (self.tf["press_events_100"].to_numpy() < k)).any())

    def test_trained_candidates_fit_on_train_only(self):
        for n in ("dt2_robust", "dt3_robust", "dt3_all"):
            self.assertEqual(int(self.cands[n].params["model"].tree_.n_node_samples[0]), 120)

    def test_saved_model_unchanged_by_candidates(self):
        before = (ROOT / "models" / "followup" / "DT_depth2.joblib").read_bytes()
        trial_level_evaluation(self.c)
        self.assertEqual((ROOT / "models" / "followup" / "DT_depth2.joblib").read_bytes(), before)
        self.assertEqual(joblib.load(MODEL_PATH)["model"].tree_.node_count, 7)

    def test_trial_level_current_matches_followup_metrics(self):
        ev = trial_level_evaluation(self.c)["current_dt2"]
        fm = json.loads((ROOT / "reports" / "followup" / "followup_metrics.json").read_text())["evaluations"]["DT_depth2"]
        self.assertAlmostEqual(ev["validation"]["accuracy"], fm["validation"]["accuracy"])
        self.assertAlmostEqual(ev["test"]["accuracy"], fm["test"]["accuracy"])
        self.assertEqual(ev["test"]["confusion_matrix"], fm["test"]["confusion_matrix"]["matrix"])

    def test_metrics_helper(self):
        m = metrics([L, L, H, D], [L, H, H, D])
        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(sum(map(sum, m["confusion_matrix"])), 4)
        self.assertEqual(m["per_class"][H]["precision"], 0.5)


class TestStrategies(unittest.TestCase):
    def test_latch_equals_production_simulator(self):
        order = list(cm.rest_separated_order("validation", 0)[:9])
        with CTreeBackend() as c:
            res = simulate(cm.trials(), order, "continuous", c, hop=10)
            wt = record_predictions(tuple(order), 10, [x for x in build_candidates(c) if x.name == "current_dt2"])
        self.assertEqual(wt["pred_current_dt2"].tolist(), [w.label for w in res.windows])
        self.assertEqual([(e.label, e.first_window, e.emit_window) for e in es.latch(wt["pred_current_dt2"].tolist())],
                         [(e.label, e.first_window, e.emit_window) for e in res.events])

    def test_generic_latch_uniform_k_equals_production(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            seq = list(rng.choice([L, H, D, N, N], size=60))
            for k in (1, 2, 3):
                for n in (1, 2):
                    self.assertEqual(es.generic_latch(seq, {L: k, H: k, D: k}, n), es.latch(seq, k, n))

    def test_ds_slower(self):
        seq = [N, D, D, D, L, L, L, N]
        self.assertEqual([e.label for e in es.generic_latch(seq, {D: 4})], [L])
        self.assertEqual([e.label for e in es.generic_latch(seq, {D: 3})], [D])

    def test_correctable(self):
        seq = [D, D, L, L, L, N, L, L]
        ev = es.generic_latch(seq, {}, 1, correction_windows=3)
        self.assertEqual([(e.label, e.kind) for e in ev], [(D, "event"), (L, "correction"), (L, "event")])
        self.assertEqual([e.label for e in es.generic_latch(seq, {}, 1, correction_windows=4)], [D, L])

    def test_on_release(self):
        self.assertEqual([(e.label, e.first_window, e.emit_window) for e in es.on_release([N, D, D, L, L, L, N, N])],
                         [(L, 1, 6)])
        self.assertEqual(es.on_release([N, D, N]), [])                      # streak 1 < 2
        self.assertEqual([e.label for e in es.on_release([N, L, L, D, D])], [D])   # tie -> later; flushed at end
        self.assertEqual(es.on_release([N, L, L, D, D])[0].emit_window, 4)

    def test_windows_for_ms(self):
        self.assertEqual(es.windows_for_ms(400, 10), 2)
        self.assertEqual(es.windows_for_ms(1200, 5), 12)
        self.assertEqual(es.windows_for_ms(100, 25), 2)


class TestScoring(unittest.TestCase):
    def setUp(self):
        # 6 windows of 150 samples each at hop 150 over order: rest, L-trial, rest, H-trial, rest, rest
        self.wt = pd.DataFrame({"start": [0, 150, 300, 450, 600, 750], "end": [149, 299, 449, 599, 749, 899]})
        self.eps = [cm.Episode(L, 1, 1, 150, 299), cm.Episode(H, 3, 3, 450, 599)]

    def ev(self, lab, a, b):
        return es.Ev(lab, a, b)

    def test_hits_misses_false_duplicates_recovery(self):
        s = score_events([self.ev(D, 1, 1), self.ev(L, 1, 1), self.ev(L, 1, 1), self.ev(D, 5, 5)], self.wt, self.eps)
        self.assertEqual((s["hits"], s["missed"], s["wrong_label_events"], s["false_ds_events"], s["rest_false_events"],
                          s["duplicate_events"], s["recovered_after_false"], s["first_event_correct"]),
                         (1, 1, 1, 1, 1, 1, 1, 0))
        self.assertEqual(s["false_events_total"], 2)
        self.assertEqual(s["latencies_ms"], [(299 - 150) * 20])

    def test_ambiguous(self):
        s = score_events([self.ev(L, 1, 3)], self.wt, self.eps)
        self.assertEqual((s["ambiguous_events"], s["hits"], s["missed"]), (1, 0, 2))

    def test_mixed_probe_sample_accounting(self):
        fake = SimpleNamespace(name="count_active",
                               predict=lambda X, adc: np.rint(X["above_100_fraction"].to_numpy() * 150).astype(int))
        self.assertGreater(min(cm.trials()[1].adc), 100)        # precondition: trial 1 is above 100 throughout
        pr = mixed_window_probe("train", [fake], rest_trial=151, only_trials=[1])
        self.assertEqual(len(pr), 2 * 149)
        self.assertTrue((pr["pred"] == pr["samples_from_trial"]).all())


class TestRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.R = rs.run(seeds=(0,), hops=(25,), probe=False)

    def test_structure_and_episode_counts(self):
        for split in ("validation", "test"):
            a = self.R["seq_candidates"][f"{split}|current_dt2"]
            self.assertEqual(a["episodes"], 30)
            self.assertEqual(a["hits"] + a["missed"], 30)
        self.assertEqual(set(self.R["trial_eval"]), {"current_dt2", "gate_ev2", "gate_ev3", "dt2_robust",
                                                     "dt3_robust", "dt3_all"})

    def test_selection_uses_validation_only(self):
        sel = self.R["strategy_selected_on_validation"]
        for key, s in sel.items():
            n, fam = key.split("|")
            v = self.R["strategy_agg"][f"validation|{n}"]
            opts = [x for x in v if es.family(x) == fam]
            best = min(v[x]["false_events_total"] + v[x]["missed"] for x in opts)
            self.assertEqual(v[s]["false_events_total"] + v[s]["missed"], best)

    def test_report_disclaimers(self):
        md = rs.report_markdown(self.R)
        for phrase in ("EXPLORATORY", "mixed windows have no", "not independent validation", "TRAINING trials"):
            self.assertIn(phrase, md)

    def test_write_results_to_temp_dir_only(self):
        with tempfile.TemporaryDirectory() as td:
            files = rs.write_results(self.R, Path(td))
            self.assertTrue(all(Path(td) in f.parents for f in files))
            self.assertTrue((Path(td) / "step3b_report.md").exists())


class TestPreservation(unittest.TestCase):
    """Runs last (see simulation/diagnostics/step3b/tests/__init__.py)."""

    def test_files_outside_step3b_unchanged(self):
        now = snapshot()
        for must in ("gripsense_clean_data.csv", "models/followup/DT_depth2.joblib", "simulation/decision.py",
                     "simulation/diagnostics/transition_report.py", "tests/test_preservation.py"):
            self.assertIn(must, BASELINE)
        changed = sorted(f for f in BASELINE if f in now and now[f] != BASELINE[f])
        self.assertEqual((changed, sorted(set(BASELINE) - set(now)), sorted(set(now) - set(BASELINE))),
                         ([], [], []))


if __name__ == "__main__":
    unittest.main()
