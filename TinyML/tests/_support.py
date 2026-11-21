"""
Shared helpers for the GripSense test suite.

Everything here is read-only with respect to the project: C code is compiled into a
temporary directory, and no project file is written. The preservation baseline is taken
the first time this module is imported (i.e. before any test runs).
"""
from __future__ import annotations

import functools
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import export_and_verify_c_port as port  # noqa: E402  (main() is guarded; import has no side effects)
from gripsense_features import N_SEGMENTS, SAMPLES_PER_TRIAL, build_feature_table  # noqa: E402

MODEL_PATH = ROOT / "models" / "followup" / "DT_depth2.joblib"
C_SRC = ROOT / "models" / "c_port" / "decision_tree.c"
CSV_PATH = ROOT / "gripsense_clean_data.csv"
CPORT_RESULTS = ROOT / "results" / "c_port"
TREE_METADATA = ROOT / "models" / "c_port" / "tree_metadata.json"
CFLAGS = list(port.CFLAGS)

# Preservation scope: everything under ROOT except these top-level folders and Finder metadata.
EXCLUDED_TOP_DIRS = {".venv", "tests"}
EXCLUDED_NAMES = {".DS_Store"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot() -> dict[str, str]:
    out = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in EXCLUDED_TOP_DIRS or p.name in EXCLUDED_NAMES or not p.is_file():
            continue
        out[rel.as_posix()] = sha(p)
    return out


BASELINE = snapshot()


# ---------------------------------------------------------------- C build (temp dir only)
def compile_c(outdir: Path) -> dict:
    """Compile the saved decision_tree.c (read-only) into `outdir`, exactly as the export script does."""
    exe, obj = outdir / "decision_tree", outdir / "decision_tree.o"
    r_exe = subprocess.run(["cc", *CFLAGS, str(C_SRC), "-o", str(exe), "-lm"], capture_output=True, text=True)
    r_obj = subprocess.run(["cc", *CFLAGS, "-DGS_NO_MAIN", "-c", str(C_SRC), "-o", str(obj)],
                           capture_output=True, text=True)
    diag = (r_exe.stdout + r_exe.stderr + r_obj.stdout + r_obj.stderr).strip()
    return {"exe": exe, "obj": obj, "exe_rc": r_exe.returncode, "obj_rc": r_obj.returncode,
            "diagnostics": diag, "n_warnings": len(re.findall(r"warning:", diag))}


def run_harness(exe: Path, mode: str, stdin_text: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(exe), mode], input=stdin_text, capture_output=True, text=True)


# ---------------------------------------------------------------- shared read-only data
@functools.lru_cache(maxsize=None)
def bundle() -> dict:
    return joblib.load(MODEL_PATH)


@functools.lru_cache(maxsize=None)
def tree() -> dict:
    return port.read_tree(bundle())


@functools.lru_cache(maxsize=None)
def raw_df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


@functools.lru_cache(maxsize=None)
def features() -> pd.DataFrame:
    return build_feature_table(raw_df()).sort_values("trial").reset_index(drop=True)


def segment_bounds() -> dict:
    # identical derivation to export_and_verify_c_port.main()
    split_idx = np.array_split(np.arange(SAMPLES_PER_TRIAL), N_SEGMENTS)
    return {"segment_1_mean": (int(split_idx[1][0]), int(split_idx[1][-1]) + 1)}


def read_json(p: Path) -> dict:
    return json.loads(p.read_text())
