"""
Window classification. Knows nothing about events; returns one label per 150-sample window.

CTreeBackend (primary)
    Compiles the unchanged, verified models/c_port/decision_tree.c into a shared library in a
    temporary directory (same warning flags as the verified build, plus -DGS_NO_MAIN -shared -fPIC)
    and calls gs_predict_from_adc() through ctypes. Features are computed by the C code itself
    (uint16 input, double features, sklearn's float32-cast comparison), exactly as verified.

SklearnReferenceBackend (reference)
    gripsense_features.extract_trial_features + models/followup/DT_depth2.joblib (read-only).

Neither backend produces a confidence score. Every leaf of the depth-2 tree is pure (one class
in training), so sklearn's predict_proba is always 1.0 and carries no information.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

import export_and_verify_c_port as _port   # CFLAGS only; main() is guarded
from gripsense_features import LABELS, SAMPLES_PER_TRIAL, extract_trial_features

from . import ROOT
from .window_buffer import validate_adc

C_SRC = ROOT / "models" / "c_port" / "decision_tree.c"
TREE_METADATA = ROOT / "models" / "c_port" / "tree_metadata.json"
MODEL_PATH = ROOT / "models" / "followup" / "DT_depth2.joblib"
CFLAGS = list(_port.CFLAGS)
C_FEATURES = ("longest_active_run_frac", "segment_1_mean")


@dataclass(frozen=True)
class Prediction:
    label: Optional[str]      # one of LABELS, or None if the C code rejected the input (-1)
    class_index: int          # C enum index (LIGHT_PRESS=0 ... NO_PRESS=3) or -1


def as_window(adc) -> np.ndarray:
    """Validate a window: exactly 150 integer ADC counts in 0..4095. No clipping, no rounding."""
    arr = np.asarray(adc)
    if arr.ndim != 1 or arr.shape[0] != SAMPLES_PER_TRIAL:
        raise ValueError(f"window must be 1-D with exactly {SAMPLES_PER_TRIAL} samples, got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"window must contain integers, got dtype {arr.dtype}")
    for v in arr.tolist():
        validate_adc(v)
    return np.ascontiguousarray(arr, dtype=np.uint16)


class CTreeBackend:
    name = "c_tree"

    def __init__(self, c_src: Path = C_SRC, check_hash: bool = True):
        if check_hash:
            expected = json.loads(TREE_METADATA.read_text())["c_source_sha256"]
            actual = hashlib.sha256(Path(c_src).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"{c_src} does not match the verified source hash in tree_metadata.json")
        self._tmp = tempfile.TemporaryDirectory(prefix="gripsense_sim_")
        self.lib_path = Path(self._tmp.name) / "libgripsense_tree.so"
        cmd = ["cc", *CFLAGS, "-DGS_NO_MAIN", "-shared", "-fPIC", str(c_src), "-o", str(self.lib_path), "-lm"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.compile_command = cmd
        self.compile_diagnostics = (r.stdout + r.stderr).strip()
        self.n_warnings = len(re.findall(r"warning:", self.compile_diagnostics))
        if r.returncode != 0:
            self._tmp.cleanup()
            raise RuntimeError(f"C compilation failed:\n{self.compile_diagnostics}")
        lib = ctypes.CDLL(str(self.lib_path))
        u16p, dblp = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_double)
        lib.gs_predict_from_adc.argtypes, lib.gs_predict_from_adc.restype = [u16p], ctypes.c_int
        lib.gs_extract_features.argtypes, lib.gs_extract_features.restype = [u16p, dblp], ctypes.c_int
        lib.gs_predict.argtypes, lib.gs_predict.restype = [dblp], ctypes.c_int
        lib.gs_label_name.argtypes, lib.gs_label_name.restype = [ctypes.c_int], ctypes.c_char_p
        lib.gs_feature_name.argtypes, lib.gs_feature_name.restype = [ctypes.c_int], ctypes.c_char_p
        c_labels = [lib.gs_label_name(i).decode() for i in range(len(LABELS))]
        c_feats = tuple(lib.gs_feature_name(i).decode() for i in range(len(C_FEATURES)))
        if c_labels != LABELS or c_feats != C_FEATURES:
            raise RuntimeError(f"C label/feature order mismatch: {c_labels}, {c_feats}")
        self._lib, self._u16p, self._dblp = lib, u16p, dblp

    def _to_prediction(self, idx: int) -> Prediction:
        if idx < 0:
            return Prediction(None, -1)
        if idx >= len(LABELS):
            raise RuntimeError(f"C returned out-of-range class index {idx}")
        return Prediction(LABELS[idx], idx)

    def predict(self, adc) -> Prediction:
        w = as_window(adc)
        return self._to_prediction(self._lib.gs_predict_from_adc(w.ctypes.data_as(self._u16p)))

    def features(self, adc) -> tuple[float, float]:
        w = as_window(adc)
        out = (ctypes.c_double * len(C_FEATURES))()
        self._lib.gs_extract_features(w.ctypes.data_as(self._u16p), out)
        return tuple(float(v) for v in out)

    def predict_features(self, values) -> Prediction:
        """Classify a ready-made (longest_active_run_frac, segment_1_mean) vector; NaN/inf -> label None."""
        arr = (ctypes.c_double * len(C_FEATURES))(*[float(v) for v in values])
        return self._to_prediction(self._lib.gs_predict(arr))

    def close(self) -> None:
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SklearnReferenceBackend:
    name = "sklearn_reference"

    def __init__(self, model_path: Path = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model, self.columns = bundle["model"], bundle["feature_columns"]

    def predict(self, adc) -> Prediction:
        w = as_window(adc)
        feats = pd.DataFrame([extract_trial_features(w.astype(np.int64))])[self.columns]
        label = str(self.model.predict(feats)[0])
        return Prediction(label, LABELS.index(label))

    def close(self) -> None:
        pass
