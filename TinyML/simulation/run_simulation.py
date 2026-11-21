#!/usr/bin/env python3
"""
Command-line runner for the GripSense live-input simulator (software only).

    .venv/bin/python -B -m simulation.run_simulation --mode isolated
    .venv/bin/python -B -m simulation.run_simulation --mode continuous --trials 1 2 3 151 152 4 --hop 10

Prints to stdout. Files are written only with --out-dir, which must be outside the project folder.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from . import ROOT
from .decision import DeciderConfig
from .inference import CTreeBackend, SklearnReferenceBackend
from .simulator import DEFAULT_HOP, compare_aligned_to_reference, load_reference_predictions, simulate
from .stream import DEFAULT_CSV, load_trials


def resolve_out_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if p == ROOT or ROOT in p.parents:
        raise SystemExit(f"refusing to write inside the project folder ({p}); use e.g. /tmp/gripsense_sim")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["isolated", "continuous"], required=True)
    ap.add_argument("--trials", type=int, nargs="*", help="trial IDs in replay order (default: all 1..200)")
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP, help="samples between inferences (default 10 = 200 ms)")
    ap.add_argument("--k", type=int, default=2, help="consecutive agreeing non-idle windows to emit (default 2)")
    ap.add_argument("--n-rearm", type=int, default=1, help="consecutive NO_PRESS windows to re-arm (default 1)")
    ap.add_argument("--no-reference", action="store_true", help="skip the sklearn reference backend")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out-dir", help="optional output folder (must be outside the project)")
    ap.add_argument("--show-windows", action="store_true", help="print every window")
    a = ap.parse_args(argv)

    out_dir = resolve_out_dir(a.out_dir) if a.out_dir else None
    trials = load_trials(a.csv)
    order = a.trials or sorted(trials)
    unknown = [t for t in order if t not in trials]
    if unknown:
        raise SystemExit(f"unknown trial IDs: {unknown}")

    ref = None if a.no_reference else SklearnReferenceBackend()
    with CTreeBackend() as c:
        res = simulate(trials, order, a.mode, c, reference=ref, hop=a.hop,
                       decider_config=DeciderConfig(a.k, a.n_rearm))
        c_warnings = c.n_warnings

    stored = load_reference_predictions()
    match, n_aligned, mism = compare_aligned_to_reference(res, stored)
    win = res.windows_frame()
    print("GripSense live-input SIMULATION (replayed recordings; software only; no hardware)")
    print(f"mode={res.mode}  trials={len(order)}  window={res.window_size} samples (3 s @ 20 ms)  hop={res.hop}"
          + (f"  K={a.k}  N_rearm={a.n_rearm}" if res.mode == "continuous" else ""))
    print(f"C shared library compiler warnings: {c_warnings}")
    print(f"windows classified: {len(win)}  (mixed-trial windows: {int(win['mixed'].sum())}, "
          f"aligned whole-trial windows: {int(win['aligned'].sum())})")
    print(f"aligned windows matching stored reference predictions: {match}/{n_aligned}")
    if mism:
        print(f"  MISMATCHES (window, trial, simulator, stored): {mism}")
    if ref is not None:
        print(f"C vs sklearn reference, all windows: {int(win['backends_agree'].sum())}/{len(win)} agree")
    single = win[~win["mixed"]]
    if len(single):
        print(f"single-trial windows whose prediction equals the recorded label: "
              f"{int((single['label'] == single['true_label']).sum())}/{len(single)} (context only, not accuracy)")
    print("window label counts: " + json.dumps(win["label"].value_counts(dropna=False).to_dict()))
    if res.mode == "isolated":
        print("events: not computed in isolated mode (one window per trial; inference parity only)")
    else:
        print(f"events emitted: {len(res.events)}   suppressed label changes while latched: "
              f"{res.suppressed_label_changes}")
        for e in res.events:
            ctx = e.context
            print(f"  [{ctx['stream_end_ms']:>6} ms] window {e.emit_window:>3} (streak from {e.first_window}) "
                  f"{e.label} ({e.meaning}) source trials {list(ctx['source_trials'])}"
                  f"{' MIXED' if ctx['mixed'] else ''}")
            for note in e.notes[1:]:
                print(f"      note: {note}")
        print("NOTE: concatenated trials are not real continuous recordings; mixed windows have no ground truth.")
        print("NOTE: K, N and hop are initial simulation settings, not validated final settings.")
    if a.show_windows:
        cols = ["window_index", "stream_end_ms", "source_trials", "mixed", "aligned", "true_label", "label",
                "reference_label", "state", "event_label"]
        print(win[[c for c in cols if c in win]].to_string(index=False))

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        win.to_csv(out_dir / f"windows_{res.mode}.csv", index=False)
        (out_dir / f"events_{res.mode}.json").write_text(json.dumps([asdict(e) for e in res.events], indent=2,
                                                                    default=str))
        print(f"wrote {out_dir}")
    ok = not mism and (ref is None or bool(win["backends_agree"].all())) and c_warnings == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
