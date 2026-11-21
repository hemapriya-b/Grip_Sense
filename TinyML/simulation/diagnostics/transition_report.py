#!/usr/bin/env python3
"""
Diagnose simulator predictions around trial boundaries in continuous replay (Step 3A).

Runs the UNCHANGED production simulator (simulation.simulator.simulate, default K/N/latch) with the
C backend and the sklearn reference, then, for windows around each selected boundary, reconstructs
the exact window from the concatenated stream and reports:
  source trials and per-trial sample proportions, start/end stream index and timestamp,
  C and reference predictions, window kind (aligned / single-trial / mixed), decider state, events,
  the two tree features with their location in the window, and the tree decision path.
Mixed windows are never given a ground-truth label.

    .venv/bin/python -B -m simulation.diagnostics.transition_report
    .venv/bin/python -B -m simulation.diagnostics.transition_report --trials 1 2 3 151 152 4 --hop 10 --context 10

Prints to stdout only; --csv writes a table to a path outside the project folder.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from gripsense_features import ACTIVE_THRESHOLD, N_SEGMENTS, SAMPLE_INTERVAL_MS, SAMPLES_PER_TRIAL

from .. import ROOT
from ..decision import DeciderConfig
from ..inference import TREE_METADATA, CTreeBackend, SklearnReferenceBackend
from ..run_simulation import resolve_out_dir
from ..simulator import DEFAULT_HOP, SimulationResult, simulate
from ..stream import TrialRecord, load_trials

W = SAMPLES_PER_TRIAL
SEG1 = tuple(int(i) for i in np.array_split(np.arange(W), N_SEGMENTS)[1][[0, -1]])  # (15, 29)
STORED_C = ROOT / "results" / "c_port" / "predictions_python_vs_c.csv"
DEFAULT_ORDER = (1, 2, 3, 151, 152, 4)


# ---------------------------------------------------------------- tree path (read from metadata)
def load_nodes() -> list[dict]:
    return json.loads(TREE_METADATA.read_text())["nodes"]


def decision_path(nodes: list[dict], feats: dict) -> tuple[str, list[str]]:
    """Walk the exported tree with the verified semantics: go left if (double)(float)x <= threshold."""
    steps, node = [], 0
    while not nodes[node]["is_leaf"]:
        n = nodes[node]
        x = feats[n["feature_name"]]
        left = float(np.float32(x)) <= n["threshold"]
        steps.append(f"n{node}:{n['feature_name']}={x:.6g}{'<=' if left else '>'}{n['threshold']:.6g}")
        node = n["left"] if left else n["right"]
    return nodes[node]["leaf_class"], steps + [f"leaf{node}"]


# ---------------------------------------------------------------- window accounting
@dataclass(frozen=True)
class Boundary:
    segment: int           # first segment after the boundary
    stream_index: int      # first stream sample of that segment
    before_trial: int
    after_trial: int
    before_label: str
    after_label: str


def boundaries(trials: dict[int, TrialRecord], order: Sequence[int], only_label_changes: bool = True) -> list[Boundary]:
    out = []
    for seg in range(1, len(order)):
        a, b = trials[order[seg - 1]], trials[order[seg]]
        if only_label_changes and a.label == b.label:
            continue
        out.append(Boundary(seg, seg * W, a.trial, b.trial, a.label, b.label))
    return out


def stream_array(trials, order) -> np.ndarray:
    return np.concatenate([np.array(trials[t].adc, dtype=np.int64) for t in order])


def locate(order, stream_index: int) -> tuple[int, int, int]:
    """(segment, trial, index within trial) of a stream position."""
    seg = stream_index // W
    return seg, int(order[seg]), stream_index % W


def proportions(order, start: int, end: int) -> list[tuple[int, int, int]]:
    """[(segment, trial, n_samples)] for stream positions start..end inclusive."""
    out = []
    for seg in range(start // W, end // W + 1):
        lo, hi = max(start, seg * W), min(end, seg * W + W - 1)
        out.append((seg, int(order[seg]), hi - lo + 1))
    return out


def longest_run(adc: np.ndarray, threshold: int = ACTIVE_THRESHOLD) -> tuple[int, int]:
    """(length, start offset) of the first longest run of samples > threshold (same rule as the C code)."""
    best = run = 0
    best_start = start = -1
    for i, v in enumerate(adc):
        if v > threshold:
            if run == 0:
                start = i
            run += 1
            if run > best:
                best, best_start = run, start
        else:
            run = 0
    return best, best_start


def window_kind(w) -> str:
    if w.mixed:
        return "MIXED"
    return "ALIGNED" if w.aligned else "single-trial"


def select_windows(res: SimulationResult, b: Boundary, context: int) -> dict[str, list[int]]:
    """Window indices: `context` windows ending before the boundary, all windows straddling it,
    and `context` windows starting at/after it (fewer if the data ends)."""
    before = [w.window_index for w in res.windows if w.stream_end_index < b.stream_index][-context:]
    straddle = [w.window_index for w in res.windows
                if w.stream_start_index < b.stream_index <= w.stream_end_index]
    after = [w.window_index for w in res.windows if w.stream_start_index >= b.stream_index][:context]
    return {"before": before, "straddling": straddle, "after": after}


def build_rows(trials, order, hop: int = DEFAULT_HOP, context: int = 10, c=None, ref=None,
               decider_config: Optional[DeciderConfig] = None) -> tuple[SimulationResult, list[Boundary], pd.DataFrame]:
    own_c = c is None
    c = c or CTreeBackend()
    ref = ref or SklearnReferenceBackend()
    try:
        res = simulate(trials, order, "continuous", c, reference=ref, hop=hop, decider_config=decider_config)
        stream = stream_array(trials, order)
        nodes = load_nodes()
        stored = pd.read_csv(STORED_C, float_precision="round_trip").set_index("trial")
        events = {e.emit_window: e for e in res.events}
        rows = []
        for b in boundaries(trials, order):
            sel = select_windows(res, b, context)
            for position, idxs in sel.items():
                for wi in idxs:
                    w = res.windows[wi]
                    adc = stream[w.stream_start_index:w.stream_end_index + 1]
                    larf, seg1 = c.features(adc)
                    path_leaf, path = decision_path(nodes, {"longest_active_run_frac": larf, "segment_1_mean": seg1})
                    run_len, run_off = longest_run(adc)
                    run_loc = (locate(order, w.stream_start_index + run_off), locate(order, w.stream_start_index + run_off + run_len - 1)) if run_len else None
                    s1 = (locate(order, w.stream_start_index + SEG1[0]), locate(order, w.stream_start_index + SEG1[1]))
                    ev = events.get(wi)
                    rows.append({
                        "boundary": f"{b.before_trial}->{b.after_trial}",
                        "boundary_stream_index": b.stream_index,
                        "position": position,
                        "window": wi,
                        "kind": window_kind(w),
                        "source_trials": "+".join(str(t) for t in w.source_trials),
                        "proportions": " ".join(f"{t}:{n}/{W}" for _, t, n in proportions(order, w.stream_start_index, w.stream_end_index)),
                        "start_index": w.stream_start_index,
                        "end_index": w.stream_end_index,
                        "start_ms": w.stream_start_index * SAMPLE_INTERVAL_MS,
                        "end_ms": w.stream_end_ms,
                        "start_trial_idx": f"{w.start[0]}[{w.start[1]}]",
                        "end_trial_idx": f"{w.end[0]}[{w.end[1]}]",
                        "longest_active_run_frac": larf,
                        "longest_run_samples": run_len,
                        "longest_run_location": (f"{run_loc[0][1]}[{run_loc[0][2]}]..{run_loc[1][1]}[{run_loc[1][2]}]" if run_loc else "-"),
                        "segment_1_mean": seg1,
                        "segment_1_source": f"{s1[0][1]}[{s1[0][2]}]..{s1[1][1]}[{s1[1][2]}]",
                        "tree_path": " -> ".join(path),
                        "path_leaf": path_leaf,
                        "c_label": w.label,
                        "c_label_recomputed": c.predict(adc).label,
                        "reference_label": w.reference_label,
                        "true_label": w.true_label if not w.mixed else None,
                        "state": w.state,
                        "event": ev.label if ev else None,
                        "event_streak_first_window": ev.first_window if ev else None,
                        "event_streak_start_index": res.windows[ev.first_window].stream_start_index if ev else None,
                    })
        df = pd.DataFrame(rows)
        df.attrs["complete_trial_features"] = {
            int(t): {"longest_active_run_frac": float(stored.loc[t, "c_longest_active_run_frac"]),
                     "segment_1_mean": float(stored.loc[t, "c_segment_1_mean"]),
                     "prediction": str(stored.loc[t, "c_from_raw_adc"]), "label": str(stored.loc[t, "actual"])}
            for t in sorted(set(order))}
        return res, boundaries(trials, order), df
    finally:
        if own_c:
            c.close()


# ---------------------------------------------------------------- printing
def print_report(res, bounds, df, context) -> None:
    nodes = load_nodes()
    print("GripSense Step 3A transition diagnostics (SIMULATION on concatenated recordings; software only)")
    print(f"trials={res.trial_order} hop={res.hop} window={res.window_size} decider={res.decider} "
          f"(unchanged production simulator)")
    print("Tree (from models/c_port/tree_metadata.json; compare (double)(float)x <= threshold):")
    for n in nodes:
        if n["is_leaf"]:
            print(f"  node {n['node']}: leaf {n['leaf_class']}")
        else:
            print(f"  node {n['node']}: {n['feature_name']} <= {n['threshold']!r} "
                  f"(= {n['threshold'] * W:.4f}/150 if a run fraction) -> left {n['left']} else {n['right']}")
    print(f"segment_1_mean = mean of window samples {SEG1[0]}..{SEG1[1]}; active = ADC > {ACTIVE_THRESHOLD}")
    print("\nComplete-trial reference features (results/c_port/predictions_python_vs_c.csv):")
    for t, f in df.attrs["complete_trial_features"].items():
        print(f"  trial {t:>3} ({f['label']}): longest_active_run_frac={f['longest_active_run_frac']:.6f} "
              f"({f['longest_active_run_frac'] * W:.0f}/150)  segment_1_mean={f['segment_1_mean']:.4f}  "
              f"prediction={f['prediction']}")
    for b in bounds:
        sub = df[df["boundary_stream_index"] == b.stream_index]
        print(f"\n=== Boundary {b.before_trial} ({b.before_label}) -> {b.after_trial} ({b.after_label}) "
              f"at stream index {b.stream_index} ({b.stream_index * SAMPLE_INTERVAL_MS} ms) ===")
        counts = sub["position"].value_counts().to_dict()
        print(f"windows: before={counts.get('before', 0)} straddling={counts.get('straddling', 0)} "
              f"after={counts.get('after', 0)} (requested context {context} each side; fewer means the data ends)")
        for _, r in sub.iterrows():
            truth = "(none: mixed)" if r["kind"] == "MIXED" else (r["true_label"] if isinstance(r["true_label"], str) else "-")
            has_event = isinstance(r["event"], str)
            print(f"[{r['position']:<10}] w{r['window']:>3} {r['kind']:<12} trials {r['source_trials']:<8} "
                  f"{r['proportions']:<16} idx {r['start_index']}..{r['end_index']} "
                  f"({r['start_ms']}-{r['end_ms']} ms) {r['start_trial_idx']}..{r['end_trial_idx']}")
            print(f"      run={r['longest_run_samples']:>3}/150 ({r['longest_active_run_frac']:.6f}) at "
                  f"{r['longest_run_location']:<18} seg1_mean={r['segment_1_mean']:.4f} from {r['segment_1_source']}")
            print(f"      path: {r['tree_path']}")
            print(f"      C={r['c_label']} ref={r['reference_label']} truth={truth} state={r['state']}"
                  + (f"  EVENT EMITTED {r['event']} (streak from window {int(r['event_streak_first_window'])}, "
                     f"streak start stream index {int(r['event_streak_start_index'])})" if has_event else "  no event"))
    print("\nAll events in this run: " + (", ".join(
        f"{e.label}@w{e.emit_window}(streak from w{e.first_window}, end idx {e.context['stream_end_index']})"
        for e in res.events) or "none"))
    checks = {
        "C label recomputed from reconstructed window == simulator label": bool((df["c_label"] == df["c_label_recomputed"]).all()),
        "tree path leaf == C label": bool((df["path_leaf"] == df["c_label"]).all()),
        "C == sklearn reference": bool((df["c_label"] == df["reference_label"]).all()),
        "no ground truth on mixed windows": bool(df.loc[df["kind"] == "MIXED", "true_label"].isna().all()),
    }
    print("Consistency checks: " + "; ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, nargs="*", default=list(DEFAULT_ORDER))
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP)
    ap.add_argument("--context", type=int, default=10, help="windows before and after each boundary")
    ap.add_argument("--csv", help="optional CSV output path (must be outside the project folder)")
    a = ap.parse_args(argv)
    out = resolve_out_dir(a.csv) if a.csv else None
    trials = load_trials()
    res, bounds, df = build_rows(trials, a.trials, hop=a.hop, context=a.context)
    print_report(res, bounds, df, a.context)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"wrote {out}")
    ok = bool((df["c_label"] == df["c_label_recomputed"]).all() and (df["path_leaf"] == df["c_label"]).all()
              and (df["c_label"] == df["reference_label"]).all())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
