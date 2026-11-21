#!/usr/bin/env python3
"""
GripSense TinyML - export the saved depth-2 decision tree to plain C and verify it.

Software-only: no hardware, no retraining, original CSV and split are only read.

Steps
  1. Read the fitted tree arrays from models/followup/DT_depth2.joblib.
  2. Generate models/c_port/decision_tree.c (tables + exact float32 comparison semantics
     + the two features it needs, computed from raw ADC samples).
  3. Compile with `cc` and strict warnings; record the compiler output.
  4. Verify, for all 200 trials:
       a) C(feature vector from Python)  == sklearn predict
       b) C(raw 150 ADC samples)         == sklearn predict, and C features == Python features
     plus synthetic threshold-boundary vectors, and consistency with the follow-up predictions file.
  5. Measure binary/object sizes (host macOS build - NOT ESP32 numbers).
  6. Confirm every pre-existing project file is unchanged.

    .venv/bin/python export_and_verify_c_port.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from gripsense_features import (ACTIVE_THRESHOLD, LABELS, N_SEGMENTS, SAMPLES_PER_TRIAL,
                                build_feature_table)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "followup" / "DT_depth2.joblib"
CDIR, RDIR = ROOT / "models" / "c_port", ROOT / "results" / "c_port"
BUILD = CDIR / "build"
C_SRC, EXE, OBJ = CDIR / "decision_tree.c", BUILD / "decision_tree", BUILD / "decision_tree.o"
CFLAGS = ["-std=c99", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wsign-conversion",
          "-Wshadow", "-Wdouble-promotion", "-Wstrict-prototypes", "-Wmissing-prototypes",
          "-Wfloat-equal", "-Wcast-qual", "-Wformat=2", "-Wundef", "-Wvla"]
SUPPORTED_FEATURES = {"longest_active_run_frac", "segment_1_mean"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd, stdin_text=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, input=stdin_text, capture_output=True, text=True)


# ====================================================================== 1. read tree
def read_tree(bundle: dict) -> dict:
    model = bundle["model"]
    t = model.tree_
    cols = bundle["feature_columns"]
    assert type(model).__name__ == "DecisionTreeClassifier" and t.n_outputs == 1
    used_orig = sorted({int(f) for f in t.feature if f >= 0})
    used_names = [cols[i] for i in used_orig]
    unsupported = set(used_names) - SUPPORTED_FEATURES
    if unsupported:
        raise SystemExit(f"tree uses features without a C implementation: {unsupported}")
    compact = {orig: k for k, orig in enumerate(used_orig)}
    nodes = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        cls_name = str(model.classes_[int(np.argmax(t.value[i].ravel()))]) if leaf else None
        nodes.append({
            "node": i, "is_leaf": bool(leaf),
            "feature_orig_index": int(t.feature[i]) if not leaf else -1,
            "feature_c_index": compact[int(t.feature[i])] if not leaf else -1,
            "feature_name": cols[int(t.feature[i])] if not leaf else None,
            "threshold": float(t.threshold[i]) if not leaf else None,
            "threshold_hex": float(t.threshold[i]).hex() if not leaf else None,
            "threshold_is_float32_exact": (bool(float(np.float32(t.threshold[i])) == float(t.threshold[i]))
                                           if not leaf else None),
            "left": int(t.children_left[i]), "right": int(t.children_right[i]),
            "leaf_class": cls_name,
            "leaf_class_c_index": LABELS.index(cls_name) if leaf else -1,
            "n_train_samples": int(t.n_node_samples[i]),
        })
    return {"model": model, "nodes": nodes, "used_orig": used_orig, "used_names": used_names,
            "sklearn_classes_order": [str(c) for c in model.classes_]}


def rules_text(nodes, i=0, depth=0) -> list[str]:
    n = nodes[i]
    pad = "    " * depth
    if n["is_leaf"]:
        return [f"{pad}-> {n['leaf_class']}"]
    out = [f"{pad}if ((float){n['feature_name']} <= {n['threshold']!r}):"]
    out += rules_text(nodes, n["left"], depth + 1)
    out += [f"{pad}else:"]
    out += rules_text(nodes, n["right"], depth + 1)
    return out


# ====================================================================== 2. generate C
def generate_c(tree: dict, segment_bounds: dict) -> str:
    nodes = tree["nodes"]
    n = len(nodes)
    feat_names = tree["used_names"]
    rules = "\n".join(" *   " + line for line in rules_text(nodes))
    feat_tbl = ", ".join(str(nd["feature_c_index"]) for nd in nodes)
    thr_tbl = ",\n    ".join(
        (f"{nd['threshold_hex']} /* {nd['threshold']!r} */" if not nd["is_leaf"] else "0.0 /* leaf */")
        for nd in nodes)
    left_tbl = ", ".join(str(nd["left"]) for nd in nodes)
    right_tbl = ", ".join(str(nd["right"]) for nd in nodes)
    cls_tbl = ", ".join(str(nd["leaf_class_c_index"]) for nd in nodes)
    label_enum = ",\n    ".join(f"GS_{lab} = {i}" for i, lab in enumerate(LABELS))
    label_str = ", ".join(f'"{lab}"' for lab in LABELS)
    feat_str = ", ".join(f'"{f}"' for f in feat_names)
    seg_start, seg_end = segment_bounds["segment_1_mean"]
    feat_code = []
    for k, name in enumerate(feat_names):
        if name == "longest_active_run_frac":
            feat_code.append(f"    out[{k}] = (double)longest_run_above(adc, GS_ACTIVE_THRESHOLD) / (double)GS_SAMPLES;")
        elif name == "segment_1_mean":
            feat_code.append(f"    out[{k}] = segment_mean(adc, {seg_start}, {seg_end});")
    feat_code = "\n".join(feat_code)
    max_depth = int(tree["model"].tree_.max_depth)

    return f"""/*
 * GripSense TinyML - depth-2 decision tree, exported from
 *   models/followup/DT_depth2.joblib  (scikit-learn {sklearn.__version__})
 * GENERATED by export_and_verify_c_port.py - do not edit by hand.
 *
 * Exact rules read from the fitted tree (sklearn semantics: the feature value is
 * cast to float32, then compared `<=` against the float64 threshold; true -> left):
{rules}
 *
 * Features (units):
 *   longest_active_run_frac : longest run of consecutive samples with ADC > {ACTIVE_THRESHOLD},
 *                             divided by {SAMPLES_PER_TRIAL} (dimensionless, 0..1)
 *   segment_1_mean          : mean ADC count of samples [{seg_start}, {seg_end})  (= {seg_start * 20}-{seg_end * 20 - 20} ms
 *                             at 20 ms sampling; segment 1 of {N_SEGMENTS} equal segments)
 *
 * Scope: classifies ONE complete, pre-segmented 3-second window of {SAMPLES_PER_TRIAL} samples.
 * It is NOT a real-time gesture detector (no windowing, onset detection or press counting).
 *
 * Build:  cc -std=c99 -O2 -Wall -Wextra -Wpedantic decision_tree.c -o decision_tree
 * Library use: compile with -DGS_NO_MAIN and call gs_predict_from_adc().
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GS_SAMPLES {SAMPLES_PER_TRIAL}
#define GS_ACTIVE_THRESHOLD {ACTIVE_THRESHOLD}
#define GS_N_FEATURES {len(feat_names)}
#define GS_N_NODES {n}
#define GS_N_CLASSES {len(LABELS)}
#define GS_MAX_DEPTH {max_depth}

/* Class labels in GripSense project order (NOT sklearn's alphabetical classes_ order). */
typedef enum {{
    {label_enum}
}} gs_label_t;

static const char *const GS_LABEL_NAMES[GS_N_CLASSES] = {{{label_str}}};
static const char *const GS_FEATURE_NAMES[GS_N_FEATURES] = {{{feat_str}}};

/* Flattened tree. Leaves have feature -1. Thresholds are exact hex-float copies of the
 * sklearn float64 thresholds. */
static const int8_t GS_FEATURE[GS_N_NODES] = {{{feat_tbl}}};
static const double GS_THRESHOLD[GS_N_NODES] = {{
    {thr_tbl}
}};
static const int8_t GS_LEFT[GS_N_NODES] = {{{left_tbl}}};
static const int8_t GS_RIGHT[GS_N_NODES] = {{{right_tbl}}};
static const int8_t GS_LEAF_CLASS[GS_N_NODES] = {{{cls_tbl}}};

int gs_predict(const double features[GS_N_FEATURES]);
const char *gs_label_name(int label);
const char *gs_feature_name(int feature);
int gs_extract_features(const uint16_t adc[GS_SAMPLES], double out[GS_N_FEATURES]);
int gs_predict_from_adc(const uint16_t adc[GS_SAMPLES]);

/* Returns class index (gs_label_t) or -1 for invalid (non-finite) input. */
int gs_predict(const double features[GS_N_FEATURES])
{{
    int node = 0;
    int steps = 0;
    while (GS_FEATURE[node] >= 0) {{
        const double v = features[GS_FEATURE[node]];
        if (!isfinite(v) || steps++ > GS_MAX_DEPTH) {{
            return -1;
        }}
        /* sklearn casts X to float32 before comparing with the float64 threshold. */
        node = ((double)(float)v <= GS_THRESHOLD[node]) ? GS_LEFT[node] : GS_RIGHT[node];
    }}
    return GS_LEAF_CLASS[node];
}}

/* Name lookups; NULL when out of range. */
const char *gs_label_name(int label)
{{
    return (label >= 0 && label < GS_N_CLASSES) ? GS_LABEL_NAMES[label] : NULL;
}}

const char *gs_feature_name(int feature)
{{
    return (feature >= 0 && feature < GS_N_FEATURES) ? GS_FEATURE_NAMES[feature] : NULL;
}}

static int longest_run_above(const uint16_t adc[GS_SAMPLES], int threshold)
{{
    int best = 0, run = 0, i;
    for (i = 0; i < GS_SAMPLES; i++) {{
        run = ((int)adc[i] > threshold) ? run + 1 : 0;
        if (run > best) {{
            best = run;
        }}
    }}
    return best;
}}

static double segment_mean(const uint16_t adc[GS_SAMPLES], int start, int end)
{{
    double sum = 0.0;  /* integer-valued sum: exact in double */
    int i;
    for (i = start; i < end; i++) {{
        sum += (double)adc[i];
    }}
    return sum / (double)(end - start);
}}

int gs_extract_features(const uint16_t adc[GS_SAMPLES], double out[GS_N_FEATURES])
{{
{feat_code}
    return 0;
}}

int gs_predict_from_adc(const uint16_t adc[GS_SAMPLES])
{{
    double f[GS_N_FEATURES];
    gs_extract_features(adc, f);
    return gs_predict(f);
}}

#ifndef GS_NO_MAIN
/* Test harness (host only).
 *   decision_tree --info
 *   decision_tree features   stdin: "<id> <f0> <f1>"       stdout: "<id> <class_idx> <class_name>"
 *   decision_tree raw        stdin: "<id> <a0> ... <a149>"  stdout: "<id> <f0> <f1> <class_idx> <class_name>"
 */
static int print_info(void)
{{
    int k;
    printf("n_nodes %d\\nn_features %d\\nn_classes %d\\nmax_depth %d\\n", GS_N_NODES, GS_N_FEATURES,
           GS_N_CLASSES, GS_MAX_DEPTH);
    for (k = 0; k < GS_N_FEATURES; k++) {{
        printf("feature %d %s\\n", k, gs_feature_name(k));
    }}
    for (k = 0; k < GS_N_CLASSES; k++) {{
        printf("label %d %s\\n", k, gs_label_name(k));
    }}
    printf("sizeof_tables_bytes %lu\\n", (unsigned long)(sizeof GS_FEATURE + sizeof GS_THRESHOLD +
           sizeof GS_LEFT + sizeof GS_RIGHT + sizeof GS_LEAF_CLASS));
    printf("sizeof_threshold_table_bytes %lu\\n", (unsigned long)sizeof GS_THRESHOLD);
    printf("sizeof_adc_window_bytes %lu\\n", (unsigned long)(sizeof(uint16_t) * GS_SAMPLES));
    return 0;
}}

static int emit(const char *id, int cls)
{{
    if (cls < 0 || cls >= GS_N_CLASSES) {{
        fprintf(stderr, "invalid input for id %s\\n", id);
        return 1;
    }}
    printf("%d %s\\n", cls, gs_label_name(cls));
    return 0;
}}

int main(int argc, char **argv)
{{
    char line[8192];
    int errors = 0;
    if (argc == 2 && strcmp(argv[1], "--info") == 0) {{
        return print_info();
    }}
    if (argc != 2 || (strcmp(argv[1], "features") != 0 && strcmp(argv[1], "raw") != 0)) {{
        fprintf(stderr, "usage: %s --info | features | raw  (input on stdin)\\n", argv[0]);
        return 2;
    }}
    while (fgets(line, (int)sizeof line, stdin) != NULL) {{
        char *p = line, *end = NULL;
        char id[64];
        int k;
        size_t idlen = strcspn(p, " \\t\\n");
        if (idlen == 0 || idlen >= sizeof id) {{
            continue;
        }}
        memcpy(id, p, idlen);
        id[idlen] = '\\0';
        p += idlen;
        if (strcmp(argv[1], "features") == 0) {{
            double f[GS_N_FEATURES];
            for (k = 0; k < GS_N_FEATURES; k++) {{
                f[k] = strtod(p, &end);
                if (end == p) {{
                    fprintf(stderr, "parse error (feature %d) for id %s\\n", k, id);
                    return 3;
                }}
                p = end;
            }}
            printf("%s ", id);
            errors += emit(id, gs_predict(f));
        }} else {{
            uint16_t adc[GS_SAMPLES];
            double f[GS_N_FEATURES];
            for (k = 0; k < GS_SAMPLES; k++) {{
                long v = strtol(p, &end, 10);
                if (end == p || v < 0 || v > 65535) {{
                    fprintf(stderr, "parse error (sample %d) for id %s\\n", k, id);
                    return 3;
                }}
                adc[k] = (uint16_t)v;
                p = end;
            }}
            gs_extract_features(adc, f);
            printf("%s", id);
            for (k = 0; k < GS_N_FEATURES; k++) {{
                printf(" %.17g", f[k]);
            }}
            printf(" ");
            errors += emit(id, gs_predict(f));
        }}
    }}
    return errors ? 4 : 0;
}}
#endif
"""


# ====================================================================== helpers
def parse_size(text: str) -> dict:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) >= 2:
        head, vals = lines[0].split(), lines[1].split()
        return {h: v for h, v in zip(head, vals)}
    return {"raw": text}


def main() -> None:
    for d in (CDIR, RDIR, BUILD):
        d.mkdir(parents=True, exist_ok=True)
    log = []

    def stage(s):
        log.append({"stage": s, "utc": now()})
        print(f"[{log[-1]['utc']}] {s}")

    # ---------------------------------------------------------------- 1
    stage("1. read fitted tree from saved model")
    model_hash_before = sha(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    tree = read_tree(bundle)
    model = tree["model"]
    cols = bundle["feature_columns"]
    # segment bounds exactly as gripsense_features computes them (np.array_split)
    split_idx = np.array_split(np.arange(SAMPLES_PER_TRIAL), N_SEGMENTS)
    seg_bounds = {"segment_1_mean": (int(split_idx[1][0]), int(split_idx[1][-1]) + 1)}
    # sanity: the exported rule text reproduces the saved rules file's structure
    saved_rules = (ROOT / "models" / "followup" / "DT_depth2_rules.txt").read_text()
    for nd in tree["nodes"]:
        if not nd["is_leaf"]:
            assert f"{nd['feature_name']} <= {nd['threshold']:.3f}" in saved_rules
        else:
            assert f"class: {nd['leaf_class']}" in saved_rules

    # ---------------------------------------------------------------- 2
    stage("2. generate C source + metadata")
    C_SRC.write_text(generate_c(tree, seg_bounds))
    meta = {
        "generated_utc": now(), "source_model": str(MODEL_PATH.relative_to(ROOT)),
        "source_model_sha256": model_hash_before, "sklearn_version": sklearn.__version__,
        "label_order_c_enum": LABELS, "sklearn_classes_order": tree["sklearn_classes_order"],
        "features_c_order": tree["used_names"],
        "features_original_column_index": tree["used_orig"],
        "feature_units": {"longest_active_run_frac": f"fraction of {SAMPLES_PER_TRIAL} samples (0..1); "
                          f"longest consecutive run with ADC > {ACTIVE_THRESHOLD}",
                          "segment_1_mean": f"ADC counts; mean of samples {seg_bounds['segment_1_mean'][0]}.."
                          f"{seg_bounds['segment_1_mean'][1] - 1}"},
        "comparison_semantics": "go left if (double)(float)x <= threshold (sklearn casts X to float32)",
        "nodes": tree["nodes"], "rules": rules_text(tree["nodes"]),
        "samples_per_window": SAMPLES_PER_TRIAL, "active_threshold_adc": ACTIVE_THRESHOLD,
        "c_source": str(C_SRC.relative_to(ROOT)), "c_source_sha256": sha(C_SRC),
    }

    # ---------------------------------------------------------------- 3
    stage("3. compile with cc (strict warnings)")
    cc_ver = run(["cc", "--version"]).stdout.splitlines()[0]
    comp_exe = run(["cc", *CFLAGS, str(C_SRC), "-o", str(EXE), "-lm"])
    comp_obj = run(["cc", *CFLAGS, "-DGS_NO_MAIN", "-c", str(C_SRC), "-o", str(OBJ)])
    compile_log = (f"$ cc {' '.join(CFLAGS)} {C_SRC.name} -o build/{EXE.name} -lm\n"
                   f"exit={comp_exe.returncode}\n{comp_exe.stdout}{comp_exe.stderr}\n"
                   f"$ cc {' '.join(CFLAGS)} -DGS_NO_MAIN -c {C_SRC.name} -o build/{OBJ.name}\n"
                   f"exit={comp_obj.returncode}\n{comp_obj.stdout}{comp_obj.stderr}")
    (RDIR / "compile_log.txt").write_text(compile_log)
    warnings_text = (comp_exe.stderr + comp_obj.stderr).strip()
    n_warnings = len(re.findall(r"warning:", warnings_text))
    if comp_exe.returncode or comp_obj.returncode:
        print(compile_log)
        raise SystemExit("compilation failed")

    info = run([str(EXE), "--info"])
    info_lines = info.stdout.splitlines()
    c_feats = [l.split()[2] for l in info_lines if l.startswith("feature ")]
    c_labels = [l.split()[2] for l in info_lines if l.startswith("label ")]
    sizeof_tables = int(next(l.split()[1] for l in info_lines if l.startswith("sizeof_tables_bytes")))
    assert c_feats == tree["used_names"], (c_feats, tree["used_names"])
    assert c_labels == LABELS

    # ---------------------------------------------------------------- 4
    stage("4. verify Python vs C on all 200 trials")
    df = pd.read_csv(ROOT / "gripsense_clean_data.csv")
    feats = build_feature_table(df).sort_values("trial").reset_index(drop=True)
    saved_feats = pd.read_csv(ROOT / "reports" / "trial_features.csv",
                              float_precision="round_trip").sort_values("trial").reset_index(drop=True)
    features_match_saved = bool(np.array_equal(feats[cols].to_numpy(), saved_feats[cols].to_numpy()))
    manifest = pd.read_csv(ROOT / "reports" / "split_manifest.csv").set_index("trial")
    py_pred = model.predict(feats[cols])

    # a) C on the Python feature vectors (only the columns the tree reads, in C order)
    fv_lines = "".join(f"{int(t)} " + " ".join(repr(float(feats.loc[i, c])) for c in tree["used_names"]) + "\n"
                       for i, t in enumerate(feats["trial"]))
    r_feat = run([str(EXE), "features"], fv_lines)
    assert r_feat.returncode == 0, r_feat.stderr
    c_feat_out = [l.split() for l in r_feat.stdout.splitlines()]
    # b) C on raw ADC samples (C computes its own features)
    waves = {int(t): g.sort_values("elapsed_ms")["adc"].to_numpy() for t, g in df.groupby("trial")}
    raw_lines = "".join(f"{t} " + " ".join(map(str, waves[t])) + "\n" for t in feats["trial"])
    r_raw = run([str(EXE), "raw"], raw_lines)
    assert r_raw.returncode == 0, r_raw.stderr
    c_raw_out = [l.split() for l in r_raw.stdout.splitlines()]
    nf = len(tree["used_names"])

    rows = []
    for i, t in enumerate(feats["trial"]):
        cf, cr = c_feat_out[i], c_raw_out[i]
        assert int(cf[0]) == int(t) and int(cr[0]) == int(t)
        row = {"trial": int(t), "split": manifest.loc[t, "split"], "actual": feats.loc[i, "label"],
               "python_sklearn": py_pred[i], "c_from_features": cf[2], "c_from_raw_adc": cr[2 + nf]}
        for k, name in enumerate(tree["used_names"]):
            py_v = float(feats.loc[i, name])
            c_v = float(cr[1 + k])
            row[f"py_{name}"] = py_v
            row[f"c_{name}"] = c_v
            row[f"{name}_bit_exact"] = py_v == c_v
        row["agree_features_mode"] = row["c_from_features"] == row["python_sklearn"]
        row["agree_raw_mode"] = row["c_from_raw_adc"] == row["python_sklearn"]
        row["correct_vs_label"] = row["python_sklearn"] == row["actual"]
        rows.append(row)
    comp = pd.DataFrame(rows)
    comp.to_csv(RDIR / "predictions_python_vs_c.csv", index=False)

    # consistency with phase-2 follow-up predictions (validation + test trials)
    fu = pd.read_csv(ROOT / "results" / "followup" / "predictions_all_models.csv")
    fu = fu[fu["model"] == "DT_depth2"].set_index("trial")["predicted"]
    fu_match = bool(all(comp.set_index("trial").loc[fu.index, "python_sklearn"] == fu))

    # c) synthetic threshold-boundary vectors (not data; tests comparison semantics)
    stage("4c. threshold boundary tests")
    bvecs = []
    base_by_leaf = {}
    # a representative in-range feature vector that reaches each internal node
    for _, r in feats.iterrows():
        x = pd.DataFrame([r[cols].to_numpy(dtype=float)], columns=cols)
        path = model.decision_path(x).indices
        for nd in path:
            base_by_leaf.setdefault(int(nd), r[cols].to_numpy(dtype=float))
    for nd in tree["nodes"]:
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
            vec = base_by_leaf[nd["node"]].copy()
            vec[nd["feature_orig_index"]] = v
            bvecs.append(vec)
    B = pd.DataFrame(bvecs, columns=cols)
    b_py = model.predict(B)
    b_lines = "".join(f"b{i} " + " ".join(repr(float(B.loc[i, c])) for c in tree["used_names"]) + "\n"
                      for i in range(len(B)))
    r_b = run([str(EXE), "features"], b_lines)
    assert r_b.returncode == 0, r_b.stderr
    b_c = [l.split()[2] for l in r_b.stdout.splitlines()]
    bdf = B[tree["used_names"]].copy()
    bdf["python_sklearn"], bdf["c"] = b_py, b_c
    bdf["agree"] = bdf["python_sklearn"] == bdf["c"]
    # does a naive double comparison (no float32 cast) disagree anywhere? (diagnostic only)
    def naive(vec):
        node = 0
        while not tree["nodes"][node]["is_leaf"]:
            n_ = tree["nodes"][node]
            node = n_["left"] if vec[n_["feature_orig_index"]] <= n_["threshold"] else n_["right"]
        return tree["nodes"][node]["leaf_class"]
    bdf["naive_double_compare"] = [naive(v) for v in B.to_numpy()]
    bdf.to_csv(RDIR / "boundary_tests.csv", index=False)
    naive_disagree = int((bdf["naive_double_compare"] != bdf["python_sklearn"]).sum())
    naive_real_disagree = int(sum(naive(v) != p for v, p in zip(feats[cols].to_numpy(), py_pred)))

    # d) invalid input handling
    r_nan = run([str(EXE), "features"], "x nan 100\n")

    # ---------------------------------------------------------------- 5
    stage("5. sizes (host build; not ESP32)")
    size_exe = run(["size", str(EXE)])
    size_obj = run(["size", str(OBJ)])
    size_obj_m = run(["size", "-m", str(OBJ)])
    arch = run(["file", str(EXE)]).stdout.strip()
    sizes = {
        "estimate_phase2_12_bytes_per_node": 12 * len(tree["nodes"]),
        "c_tables_sizeof_bytes_host": sizeof_tables,
        "adc_window_buffer_bytes": 2 * SAMPLES_PER_TRIAL,
        "executable_file_bytes": EXE.stat().st_size,
        "executable_size_cmd": parse_size(size_exe.stdout),
        "object_file_bytes": OBJ.stat().st_size,
        "object_size_cmd": parse_size(size_obj.stdout),
        "object_size_m_output": size_obj_m.stdout.strip(),
        "binary_arch": arch,
        "note": "Host macOS build (Apple clang). Does not represent ESP32 flash/RAM usage; the "
                "ESP32 toolchain (xtensa/riscv gcc), libc, and FreeRTOS change all sizes.",
    }

    # ---------------------------------------------------------------- 6
    stage("6. preservation checks")
    base = json.loads((RDIR / "preservation_baseline.json").read_text())
    changed = [f for f, h in base["files"].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
    model_unchanged = sha(MODEL_PATH) == model_hash_before == base["files"]["models/followup/DT_depth2.joblib"]

    n = len(comp)
    summary = {
        "generated_utc": now(), "hardware_used": False, "new_data_collected": False,
        "model_retrained": False,
        "environment": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "cc": cc_ver,
                        "platform": platform.platform()},
        "compiler": {"flags": CFLAGS, "exe_exit": comp_exe.returncode, "obj_exit": comp_obj.returncode,
                     "n_warnings": n_warnings, "warnings_text": warnings_text},
        "trials": n,
        "agreement_features_mode": int(comp["agree_features_mode"].sum()),
        "agreement_raw_adc_mode": int(comp["agree_raw_mode"].sum()),
        "features_bit_exact_raw_mode": {name: int(comp[f"{name}_bit_exact"].sum())
                                        for name in tree["used_names"]},
        "python_features_match_saved_trial_features": features_match_saved,
        "python_matches_followup_predictions_val_test": fu_match,
        "followup_trials_compared": int(len(fu)),
        "boundary_tests": {"n": int(len(bdf)), "agree": int(bdf["agree"].sum()),
                           "naive_double_compare_disagreements": naive_disagree,
                           "naive_double_compare_disagreements_on_200_real_trials": naive_real_disagree},
        "invalid_input_rejected": r_nan.returncode != 0,
        "accuracy_vs_labels_by_split": {s: float(g["correct_vs_label"].mean())
                                        for s, g in comp.groupby("split")},
        "misclassified_vs_labels": comp.loc[~comp["correct_vs_label"], ["trial", "split", "actual",
                                                                        "python_sklearn"]].to_dict(orient="records"),
        "sizes": sizes,
        "preservation": {"files_checked": len(base["files"]), "changed": changed,
                         "saved_model_unchanged": model_unchanged},
        "stage_log": log,
    }
    summary["all_checks_pass"] = bool(
        summary["agreement_features_mode"] == n and summary["agreement_raw_adc_mode"] == n
        and all(v == n for v in summary["features_bit_exact_raw_mode"].values())
        and features_match_saved and fu_match and summary["boundary_tests"]["agree"] == len(bdf)
        and summary["invalid_input_rejected"] and n_warnings == 0 and not changed and model_unchanged
        and n == 200)
    (RDIR / "verification_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (CDIR / "tree_metadata.json").write_text(json.dumps(meta, indent=2))
    write_report(summary, meta, tree)

    print(f"features-mode agreement: {summary['agreement_features_mode']}/{n}")
    print(f"raw-ADC-mode agreement:  {summary['agreement_raw_adc_mode']}/{n}")
    print(f"boundary tests:          {summary['boundary_tests']['agree']}/{summary['boundary_tests']['n']}")
    print(f"compiler warnings:       {n_warnings}")
    print(f"pre-existing files changed: {len(changed)}")
    print(f"ALL CHECKS PASS: {summary['all_checks_pass']}")
    sys.exit(0 if summary["all_checks_pass"] else 1)


# ====================================================================== report
def write_report(s, meta, tree) -> None:
    z = s["sizes"]
    nodes_md = ["| Node | Type | Feature | Threshold (float64, exact hex) | Left / Right | Leaf class | Train samples |",
                "|---|---|---|---|---|---|---|"]
    for nd in meta["nodes"]:
        if nd["is_leaf"]:
            nodes_md.append(f"| {nd['node']} | leaf | | | | {nd['leaf_class']} | {nd['n_train_samples']} |")
        else:
            nodes_md.append(f"| {nd['node']} | split | {nd['feature_name']} | {nd['threshold']!r} (`{nd['threshold_hex']}`) "
                            f"| {nd['left']} / {nd['right']} | | {nd['n_train_samples']} |")
    acc = ", ".join(f"{k} {v:.3f}" for k, v in s["accuracy_vs_labels_by_split"].items())
    mis = ", ".join(f"#{r['trial']} ({r['split']}: {r['actual']} -> {r['python_sklearn']})"
                    for r in s["misclassified_vs_labels"])
    exe_sz = " ".join(f"{k}={v}" for k, v in z["executable_size_cmd"].items())
    obj_sz = " ".join(f"{k}={v}" for k, v in z["object_size_cmd"].items())
    rules = "\n".join(meta["rules"])
    warn = s["compiler"]["warnings_text"] or "(none)"
    rep = f"""# GripSense TinyML - C port of the depth-2 decision tree

_Generated {s['generated_utc']} by `export_and_verify_c_port.py`. Every number below is read from
`results/c_port/verification_summary.json` and `models/c_port/tree_metadata.json`._

> **Software-only.** No new data was collected, the ESP32 was not connected, and no hardware testing was
> performed. The saved model was **not** retrained or modified; the original CSV and split manifest were only read.

## Result

| Check | Result |
|---|---|
| C (given Python feature vectors) vs sklearn `predict`, all trials | **{s['agreement_features_mode']}/{s['trials']} agree** |
| C (computing its own features from raw ADC) vs sklearn `predict` | **{s['agreement_raw_adc_mode']}/{s['trials']} agree** |
| C features bit-identical to Python features (raw mode) | {', '.join(f'{k}: {v}/{s["trials"]}' for k, v in s['features_bit_exact_raw_mode'].items())} |
| Threshold-boundary test vectors (synthetic) | {s['boundary_tests']['agree']}/{s['boundary_tests']['n']} agree |
| Python features == saved `reports/trial_features.csv` | {s['python_features_match_saved_trial_features']} |
| sklearn predictions == `results/followup/predictions_all_models.csv` (DT_depth2, {s['followup_trials_compared']} val+test trials) | {s['python_matches_followup_predictions_val_test']} |
| Non-finite input rejected by C | {s['invalid_input_rejected']} |
| Compiler warnings (`cc` {' '.join(s['compiler']['flags'])}) | **{s['compiler']['n_warnings']}** |
| Pre-existing project files changed | {len(s['preservation']['changed'])} of {s['preservation']['files_checked']} |
| Saved model unchanged | {s['preservation']['saved_model_unchanged']} |
| **All checks pass** | **{s['all_checks_pass']}** |

The tree's own accuracy against the recorded labels (unchanged from phase 2, shown for context only): {acc}.
Misclassified: {mis}.

## Exported tree (read programmatically from `models/followup/DT_depth2.joblib`)

{chr(10).join(nodes_md)}

Rules (go left when the float32-cast feature is `<=` the threshold):

```
{rules}
```

**Class mapping.** sklearn stores classes alphabetically ({', '.join(meta['sklearn_classes_order'])}). Each leaf is mapped
**by name** to the C enum in project order: {', '.join(f'{i} = {l}' for i, l in enumerate(meta['label_order_c_enum']))}.

**Feature units.**
- `longest_active_run_frac`: {meta['feature_units']['longest_active_run_frac']}.
- `segment_1_mean`: {meta['feature_units']['segment_1_mean']} (300-580 ms into the window).

**Comparison semantics.** scikit-learn casts inputs to float32 before comparing them with the float64 thresholds
(which are themselves float32 midpoints). The C code does `(double)(float)x <= threshold`, with thresholds stored as exact
hex-float literals. The boundary test set probes each threshold with its float64/float32 neighbours and the nearest
achievable feature values. There, a naive double-precision comparison without the float32 cast disagreed with sklearn on
**{s['boundary_tests']['naive_double_compare_disagreements']}** vector(s), while the C port agreed on all of them. On the
200 real trials, the naive comparison disagreed on {s['boundary_tests']['naive_double_compare_disagreements_on_200_real_trials']}. The cast matters only in the narrow gap
between a float64 value and its float32 rounding, but it is required for an exact port.

## Size: estimates vs measurements

| Quantity | Value | Kind |
|---|---|---|
| Phase-2 rough estimate (12 bytes x 7 nodes) | {z['estimate_phase2_12_bytes_per_node']} bytes | estimate |
| `sizeof` of the five C tree tables (host) | {z['c_tables_sizeof_bytes_host']} bytes | measured on host (compile-time `sizeof`) |
| ADC window buffer (150 x uint16) | {z['adc_window_buffer_bytes']} bytes | by construction |
| Object file `decision_tree.o` (library build, `-DGS_NO_MAIN`), `size` | {obj_sz} | measured, host |
| Object file on disk | {z['object_file_bytes']:,} bytes | measured, host (includes symbols/metadata) |
| Test executable, `size` | {exe_sz} | measured, host (includes harness, stdio, Mach-O padding) |
| Test executable on disk | {z['executable_file_bytes']:,} bytes | measured, host |

Notes: the executable's `others` column is dominated by macOS's 4 GB `__PAGEZERO` virtual reservation, which is not
code or data. The `sizeof` value equalling the 84-byte estimate is a coincidence of the chosen types (7 x 8-byte double
thresholds + 4 x 7 one-byte tables), not a confirmation of the estimate method. Storing thresholds as float32 would halve
that table, but would need its own exactness argument.

Binary: `{z['binary_arch']}`. **These are macOS host measurements, not ESP32 flash or RAM figures.** An ESP32 build uses
a different compiler (Xtensa/RISC-V GCC), libc, and runtime, and the stdio test harness would not ship. The model tables
themselves are tiny, so on-device size will be dominated by sampling, the window buffer, and application code.

`size -m` output for the object file:

```
{z['object_size_m_output']}
```

## Compiler output

```
{warn}
```

Full log: `results/c_port/compile_log.txt`.

## Limitations (read before using the port)

1. **This is not a real-time gesture detector.** The C code classifies one complete, pre-segmented 3-second window of 150
   samples. There is no live windowing, onset detection, debouncing, or press counting.
2. **Trial-level training.** The tree was trained on features of whole recorded trials, not on a stream. Live window
   alignment has not been validated. Phase-2 probe P3 showed that adding rest at the window edges changes predictions,
   and `segment_1_mean` depends on a fixed 300-580 ms slice of the window.
3. **Hardware not validated.** ADC configuration, sampling jitter, the sensor's resting floor, and timing on the ESP32 are
   untested. The verification proves equivalence to the Python model on the recorded data only.
4. **Model limitations carry over unchanged.** The depth-2 tree scored {s['accuracy_vs_labels_by_split']['test']:.3f} on the held-out test set, with wide
   uncertainty (10 trials per class). The DOUBLE_SQUEEZE class reflects repeated squeezing, not a true double squeeze.
   See `reports/followup/GripSense_TinyML_Final_Report.md`.
5. **Floating point.** The port relies on IEEE-754 float/double, which the ESP32 toolchains provide (the classic ESP32 has
   a single-precision FPU; double is done in software, and that is fine for 2-3 comparisons per window). Re-run the parity test
   with the target toolchain or an emulator before relying on bit-exactness there.

## Reproduce

```bash
cd "/Users/zenith/Desktop/GripSense-TinyML "
.venv/bin/python export_and_verify_c_port.py      # generates, compiles, verifies; exit 0 only if all checks pass
# manual use of the compiled harness:
./models/c_port/build/decision_tree --info
```

## Artifacts

| Path | Content |
|---|---|
| `models/c_port/decision_tree.c` | generated standalone C (tree + 2 features + test harness) |
| `models/c_port/tree_metadata.json` | nodes, thresholds (decimal + hex), class mapping, units, source hashes |
| `models/c_port/build/` | compiled test executable and object file (host build) |
| `results/c_port/predictions_python_vs_c.csv` | per-trial sklearn vs C (both modes) + feature values |
| `results/c_port/boundary_tests.csv` | synthetic threshold-boundary vectors |
| `results/c_port/verification_summary.json` | all measured results |
| `results/c_port/compile_log.txt` | compiler commands and output |
| `results/c_port/preservation_baseline.json` | hashes of all pre-existing files, taken before this work |
"""
    (ROOT / "reports" / "c_port_report.md").write_text(rep)


if __name__ == "__main__":
    main()
