"""
Investigation 2: candidate classifiers using EXISTING features only, evaluated at trial level.

All candidates are fitted in memory on the 120 train trials only (nothing is saved). Candidates
were defined before any validation/test or window results were computed, from the reasoning that
fraction/position features are diluted by partial windows while count/level features are less so:

  current_dt2      the saved depth-2 tree (not retrained), predicted through the verified C code
  gate_ev2         current tree, but its DOUBLE_SQUEEZE leaf requires press_events_100 >= 2;
                   gated-out windows go to LIGHT/HARD by an active_mean stump fitted on train L/H trials
  gate_ev3         same with press_events_100 >= 3 (fits the recorded 3-7-press DS trials, but would
                   reject a genuine two-squeeze gesture by construction)
  dt2_robust       depth-2 tree on [press_events_100, press_events_300, active_mean, max, q90]
  dt3_robust       depth-3 tree on the same subset
  dt3_all          depth-3 tree on all 49 existing features (flexibility reference)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from gripsense_features import LABELS

from ...inference import MODEL_PATH, CTreeBackend
from .common import W, trial_features

ROBUST = ["press_events_100", "press_events_300", "active_mean", "max", "q90"]
SEED = 42


@dataclass
class Candidate:
    name: str
    description: str
    predict: Callable[[pd.DataFrame, list], np.ndarray]   # (feature frame, adc windows or None) -> labels
    params: dict = field(default_factory=dict)


def _tree_nodes():
    t = joblib.load(MODEL_PATH)["model"].tree_
    return float(t.threshold[0]), float(t.threshold[1]), float(t.threshold[4])


def _f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def fit_active_mean_stump(train: pd.DataFrame) -> float:
    lh = train[train["label"].isin(["LIGHT_PRESS", "HARD_PRESS"])]
    s = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(lh[["active_mean"]], lh["label"])
    return float(s.tree_.threshold[0])


def make_gate(train: pd.DataFrame, min_events: int) -> Candidate:
    root, node1, seg1_thr = _tree_nodes()
    am_thr = fit_active_mean_stump(train)

    def predict(X: pd.DataFrame, _adc=None) -> np.ndarray:
        larf, seg1 = _f32(X["longest_active_run_frac"]), _f32(X["segment_1_mean"])
        ev, am = X["press_events_100"].to_numpy(), _f32(X["active_mean"])
        out = np.where(larf <= root,
                       np.where(larf <= node1, "NO_PRESS",
                                np.where(ev >= min_events, "DOUBLE_SQUEEZE",
                                         np.where(am <= am_thr, "LIGHT_PRESS", "HARD_PRESS"))),
                       np.where(seg1 <= seg1_thr, "LIGHT_PRESS", "HARD_PRESS"))
        return out.astype(object)

    return Candidate(f"gate_ev{min_events}",
                     f"current tree; DS leaf only if press_events_100 >= {min_events}, else L/H by "
                     f"active_mean <= {am_thr:.3f}", predict, {"min_events": min_events, "active_mean_thr": am_thr})


def make_tree(train: pd.DataFrame, name: str, cols: list[str], depth: int) -> Candidate:
    m = DecisionTreeClassifier(max_depth=depth, random_state=SEED).fit(train[cols], train["label"])
    return Candidate(name, f"DecisionTree depth {depth} on {len(cols)} features (train trials only)",
                     lambda X, _adc=None: m.predict(X[cols]).astype(object), {"model": m, "cols": cols})


def make_current(c: CTreeBackend) -> Candidate:
    def predict(X, adc_windows):
        if adc_windows is None:     # trial-level: the stored whole-trial ADC is needed for the C path
            raise ValueError("current_dt2 needs raw ADC windows")
        return np.array([c.predict(a).label for a in adc_windows], dtype=object)
    return Candidate("current_dt2", "saved depth-2 tree via verified C code (unchanged)", predict)


def build_candidates(c: CTreeBackend, train: pd.DataFrame | None = None) -> list[Candidate]:
    tf = trial_features()
    train = tf[tf["split"] == "train"] if train is None else train
    cols = [x for x in tf.columns if x not in ("trial", "label", "split")]
    return [make_current(c), make_gate(train, 2), make_gate(train, 3),
            make_tree(train, "dt2_robust", ROBUST, 2), make_tree(train, "dt3_robust", ROBUST, 3),
            make_tree(train, "dt3_all", cols, 3)]


def metrics(y_true, y_pred) -> dict:
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {"accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))), "n": int(len(y_true)),
            "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "support": int(n[i])}
                          for i, lab in enumerate(LABELS)},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}


def trial_level_evaluation(c: CTreeBackend) -> dict:
    """Fit on train; report train 5-fold CV (refit per fold), validation, and test.
    The test split was already inspected in earlier project phases, so test numbers are NOT
    independent validation; nothing in Step 3B is selected on them."""
    from .common import trials as all_trials
    tf = trial_features()
    adc = {t: np.array(all_trials()[t].adc) for t in tf["trial"]}
    train = tf[tf["split"] == "train"].reset_index(drop=True)
    out = {}
    cands = build_candidates(c, train)
    for cand in cands:
        res = {}
        for split in ("train", "validation", "test"):
            d = tf[tf["split"] == split]
            pred = cand.predict(d, [adc[t] for t in d["trial"]])
            res[split] = metrics(d["label"].tolist(), list(pred))
        # trial-level CV inside train (each row is one whole trial, so folds never share a trial)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_all, p_all = [], []
        for tr_idx, te_idx in skf.split(train, train["label"]):
            fold_tr, fold_te = train.iloc[tr_idx], train.iloc[te_idx]
            if cand.name == "current_dt2":
                # the saved model saw all train trials, so CV refits an unfitted clone of it per fold
                saved = joblib.load(MODEL_PATH)
                m = clone(saved["model"]).fit(fold_tr[saved["feature_columns"]], fold_tr["label"])
                p = m.predict(fold_te[saved["feature_columns"]])
            else:
                p = {cc.name: cc for cc in build_candidates(c, fold_tr)}[cand.name].predict(fold_te, None)
            y_all += fold_te["label"].tolist()
            p_all += list(p)
        res["train_cv5"] = metrics(y_all, p_all)
        out[cand.name] = {"description": cand.description, **res}
    return out
