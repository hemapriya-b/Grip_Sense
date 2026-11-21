#!/usr/bin/env python3
"""
Step 3B - offline, EXPLORATORY evaluation of transition robustness. Nothing here is deployed.

    .venv/bin/python -B -m simulation.diagnostics.step3b.run_step3b             # print only
    .venv/bin/python -B -m simulation.diagnostics.step3b.run_step3b --write     # + results/ files

--write saves CSV/JSON/Markdown into simulation/diagnostics/step3b/results/ (the only location
this step may write to). No model is saved; candidates exist only in memory.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gripsense_features import LABELS

from ...inference import CTreeBackend
from . import characterization as ch
from .candidates import build_candidates, trial_level_evaluation
from .common import episodes, rest_separated_order
from .event_strategies import family, latch, strategy_grid
from .transitions import (mixed_window_probe, probe_summary, record_predictions, score_events,
                          single_trial_window_accuracy)

OUT = Path(__file__).resolve().parent / "results"
SEEDS = (0, 1, 2)
HOPS = (5, 10, 25)
DIAG_ORDER = (1, 2, 3, 151, 152, 4)
COUNT_KEYS = ["episodes", "hits", "missed", "first_event_correct", "wrong_label_events", "false_ds_events",
              "rest_false_events", "ambiguous_events", "false_events_total", "duplicate_events",
              "recovered_after_false"]


def aggregate(scores: list[dict]) -> dict:
    out = {k: int(sum(s[k] for s in scores)) for k in COUNT_KEYS}
    lat = [x for s in scores for x in s["latencies_ms"]]
    out["median_latency_ms"] = float(np.median(lat)) if lat else None
    return out


def run(seeds=SEEDS, hops=HOPS, probe=True) -> dict:
    R: dict = {"config": {"seeds": list(seeds), "hops": list(hops), "diagnostic_order": list(DIAG_ORDER)}}
    with CTreeBackend() as c:
        # ---- 1. characterization
        R["char_quantiles"] = ch.feature_quantiles("train")
        R["char_overlap"] = ch.range_overlap("train")
        R["char_run_bins"] = ch.run_length_bins()
        R["char_events100"] = ch.press_event_counts("press_events_100")
        R["char_events300"] = ch.press_event_counts("press_events_300")
        R["char_press_segments"] = ch.press_segment_summary("train")
        R["char_ds_vs_partial"] = ch.ds_runs_vs_partial_press_runs()

        # ---- 2. trial-level evaluation
        R["trial_eval"] = trial_level_evaluation(c)

        # ---- 3 + 4. sequences: record window predictions once, then score
        cands = build_candidates(c)
        names = [x.name for x in cands]
        seqs = [(split, seed, rest_separated_order(split, seed)) for split in ("validation", "test") for seed in seeds]
        recorded = {}
        for split, seed, order in seqs:
            for hop in hops:
                recorded[(split, seed, hop)] = (order, record_predictions(order, hop, cands))
        diag_wt = record_predictions(DIAG_ORDER, 10, cands)

        # 3. candidates x production latch (K=2, N=1)
        rows, ctx = [], []
        for (split, seed, hop), (order, wt) in recorded.items():
            eps = episodes(order)
            for n in names:
                s = score_events(latch(wt[f"pred_{n}"].tolist(), 2, 1), wt, eps)
                rows.append({"split": split, "seed": seed, "hop": hop, "candidate": n, **s})
                a = single_trial_window_accuracy(wt, n)
                ctx.append({"split": split, "seed": seed, "hop": hop, "candidate": n, **a})
        R["seq_candidates_raw"] = rows
        R["seq_candidates"] = {f"{split}|{n}": aggregate([r for r in rows if r["split"] == split and r["candidate"] == n])
                               for split in ("validation", "test") for n in names}
        R["seq_candidates_by_hop"] = {f"{split}|{hop}|{n}": aggregate([r for r in rows if r["split"] == split
                                                                       and r["candidate"] == n and r["hop"] == hop])
                                      for split in ("validation", "test") for hop in hops for n in names}
        R["seq_single_trial_windows"] = pd.DataFrame(ctx).groupby(["split", "candidate"])[["n", "correct"]].sum()
        R["diag_sequence"] = {n: {"events": [(e.label, e.emit_window) for e in latch(diag_wt[f"pred_{n}"].tolist())],
                                  **{k: v for k, v in score_events(latch(diag_wt[f"pred_{n}"].tolist()), diag_wt,
                                                                   episodes(DIAG_ORDER)).items()
                                     if k in COUNT_KEYS}} for n in names}
        R["diag_windows"] = diag_wt

        # 4. event strategies on recorded predictions; parameters chosen on validation only
        srows = []
        for (split, seed, hop), (order, wt) in recorded.items():
            eps = episodes(order)
            for n in names:
                labels = wt[f"pred_{n}"].tolist()
                for sname, params, fn in strategy_grid(hop):
                    s = score_events(fn(labels), wt, eps)
                    srows.append({"split": split, "seed": seed, "hop": hop, "candidate": n, "strategy": sname,
                                  "family": family(sname), **s})
        R["strategy_raw"] = srows
        agg = defaultdict(dict)
        for split in ("validation", "test"):
            for n in names:
                for sname in {r["strategy"] for r in srows}:
                    agg[f"{split}|{n}"][sname] = aggregate([r for r in srows if r["split"] == split
                                                            and r["candidate"] == n and r["strategy"] == sname])
        R["strategy_agg"] = dict(agg)
        selected = {}
        for n in names:
            v = agg[f"validation|{n}"]
            for fam in sorted({family(s) for s in v}):
                opts = [s for s in v if family(s) == fam]
                best = min(opts, key=lambda s: (v[s]["false_events_total"] + v[s]["missed"], v[s]["duplicate_events"],
                                                v[s]["median_latency_ms"] or 1e9, s))
                selected[f"{n}|{fam}"] = best
        R["strategy_selected_on_validation"] = selected

        # mixed-window probe (descriptive)
        if probe:
            pr = pd.concat([mixed_window_probe(sp, cands) for sp in ("validation", "test")], ignore_index=True)
            R["probe"] = pr
            R["probe_summary"] = probe_summary(pr)
    return R


# ---------------------------------------------------------------- reporting
def _md(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal Markdown table (avoids the optional `tabulate` dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def _pct(x):
    return f"{100 * x:.1f}%"


def report_markdown(R: dict) -> str:
    L = ["# Step 3B results (generated by run_step3b.py) - EXPLORATORY", "",
         "All numbers come from replayed recordings. Concatenated sequences are synthetic; mixed windows have no",
         "ground truth. The test split was inspected in earlier phases, so test figures are not independent validation;",
         "nothing in Step 3B was selected on test data.", ""]
    L += ["## 1. Training data (train split, 120 trials)", "", "Longest-active-run bins (all splits):", "",
          _md(R["char_run_bins"]), "", "press_events_100 counts (all splits):", "",
          _md(R["char_events100"]), "", "Existing press-segment table (train):", "",
          _md(R["char_press_segments"], index=False), "", "DS runs vs partial presses: `"
          + json.dumps(R["char_ds_vs_partial"]) + "`", "", "Feature quantiles (train):", "",
          _md(R["char_quantiles"].round(3), index=False), ""]
    L += ["## 2. Trial-level evaluation (fit on train trials only)", "",
          "| candidate | train 5-fold CV acc | validation acc | test acc (not independent) | description |",
          "|---|---|---|---|---|"]
    for n, r in R["trial_eval"].items():
        L.append(f"| {n} | {r['train_cv5']['accuracy']:.3f} | {r['validation']['accuracy']:.3f} | "
                 f"{r['test']['accuracy']:.3f} | {r['description']} |")
    L += ["", "Per-class precision / recall (validation):", "",
          "| candidate | " + " | ".join(LABELS) + " |", "|---|" + "---|" * len(LABELS)]
    for n, r in R["trial_eval"].items():
        pc = r["validation"]["per_class"]
        L.append(f"| {n} | " + " | ".join(f"{pc[l]['precision']:.2f} / {pc[l]['recall']:.2f}" for l in LABELS) + " |")
    L += ["", "Confusion matrices (rows = true, cols = predicted, order " + ", ".join(LABELS) + "):", ""]
    for n, r in R["trial_eval"].items():
        L.append(f"- {n}: validation {r['validation']['confusion_matrix']}; test {r['test']['confusion_matrix']}")
    L += ["", "## 3. Transition behaviour (production latch K=2, N=1)", "",
          f"Sequences: rest-separated, seeds {R['config']['seeds']}, hops {R['config']['hops']}; each press trial = one episode.", "",
          "| split | candidate | episodes | hits | missed | false events (wrong label / during rest / ambiguous) | false DS | duplicates | recovered | median latency ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, a in R["seq_candidates"].items():
        split, n = key.split("|")
        L.append(f"| {split} | {n} | {a['episodes']} | {a['hits']} | {a['missed']} | {a['false_events_total']} "
                 f"({a['wrong_label_events']} / {a['rest_false_events']} / {a['ambiguous_events']}) | {a['false_ds_events']} | "
                 f"{a['duplicate_events']} | {a['recovered_after_false']} | {a['median_latency_ms']} |")
    L += ["", "Diagnostic sequence 1 2 3 151 152 4 (TRAINING trials; hop 10):", ""]
    for n, d in R["diag_sequence"].items():
        L.append(f"- {n}: events {d['events']}; missed {d['missed']}, false {d['false_events_total']}")
    if "probe_summary" in R:
        ps = R["probe_summary"]
        L += ["", "Mixed-window probe (every offset, hop 1; descriptive, no ground truth) - share of mixed windows "
              "predicted DOUBLE_SQUEEZE:", "",
              "| split | candidate | LIGHT onset | LIGHT release | HARD onset | HARD release | DS onset | DS release |",
              "|---|---|---|---|---|---|---|---|"]
        for (split, cand), g in ps.groupby(["split", "candidate"]):
            def val(lab, d):
                r = g[(g["trial_label"] == lab) & (g["direction"].str.startswith(d))]
                return _pct(float(r["DOUBLE_SQUEEZE"].iloc[0])) if len(r) and "DOUBLE_SQUEEZE" in r else "0.0%"
            L.append(f"| {split} | {cand} | {val('LIGHT_PRESS', 'onset')} | {val('LIGHT_PRESS', 'release')} | "
                     f"{val('HARD_PRESS', 'onset')} | {val('HARD_PRESS', 'release')} | "
                     f"{val('DOUBLE_SQUEEZE', 'onset')} | {val('DOUBLE_SQUEEZE', 'release')} |")
    L += ["", "## 4. Event strategies (recorded window predictions; parameters chosen on validation sequences)", ""]
    for n in R["trial_eval"]:
        L += [f"### {n}", "", "| strategy | chosen on validation | val missed | val false | val dup | test missed | test false | test false DS | test dup | test recovered | test median latency ms |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        v, t = R["strategy_agg"][f"validation|{n}"], R["strategy_agg"][f"test|{n}"]
        for s in sorted(v):
            chosen = R["strategy_selected_on_validation"].get(f"{n}|{family(s)}") == s
            L.append(f"| {s} | {'yes' if chosen else ''} | {v[s]['missed']} | {v[s]['false_events_total']} | {v[s]['duplicate_events']} | "
                     f"{t[s]['missed']} | {t[s]['false_events_total']} | {t[s]['false_ds_events']} | {t[s]['duplicate_events']} | "
                     f"{t[s]['recovered_after_false']} | {t[s]['median_latency_ms']} |")
        L.append("")
    return "\n".join(L)


def print_summary(R: dict) -> None:
    print("GripSense Step 3B - offline EXPLORATORY transition-robustness evaluation (no deployment)")
    print("DS runs vs partial presses:", json.dumps(R["char_ds_vs_partial"]))
    print("\nTrial level (fit on train only):  name | CV5 | validation | test(not independent)")
    for n, r in R["trial_eval"].items():
        print(f"  {n:<12} {r['train_cv5']['accuracy']:.3f}  {r['validation']['accuracy']:.3f}  {r['test']['accuracy']:.3f}")
    print("\nSequences, production latch K=2 N=1:  split | candidate | episodes hits missed false(falseDS) dup recovered")
    for key, a in R["seq_candidates"].items():
        split, n = key.split("|")
        print(f"  {split:<10} {n:<12} {a['episodes']:>4} {a['hits']:>4} {a['missed']:>4} "
              f"{a['false_events_total']:>4}({a['false_ds_events']}) {a['duplicate_events']:>3} {a['recovered_after_false']:>3}")
    print("\nDiagnostic sequence 1 2 3 151 152 4 (training trials, hop 10):")
    for n, d in R["diag_sequence"].items():
        print(f"  {n:<12} events={d['events']}")
    print("\nEvent strategies on current_dt2 predictions (chosen on validation -> test):")
    v, t = R["strategy_agg"]["validation|current_dt2"], R["strategy_agg"]["test|current_dt2"]
    for fam_key, s in sorted(R["strategy_selected_on_validation"].items()):
        if fam_key.startswith("current_dt2|"):
            print(f"  {s:<20} val missed/false {v[s]['missed']}/{v[s]['false_events_total']}  "
                  f"test missed/false/falseDS/dup/recovered {t[s]['missed']}/{t[s]['false_events_total']}/"
                  f"{t[s]['false_ds_events']}/{t[s]['duplicate_events']}/{t[s]['recovered_after_false']}  "
                  f"latency {t[s]['median_latency_ms']} ms")


def write_results(R: dict, out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = []

    def w(name, obj):
        p = out / name
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(p, index=not isinstance(obj.index, pd.RangeIndex))
        else:
            p.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, default=str))
        files.append(p)

    w("char_feature_quantiles_train.csv", R["char_quantiles"])
    w("char_range_overlap_train.csv", R["char_overlap"])
    w("char_run_bins.csv", R["char_run_bins"])
    w("char_press_events_100.csv", R["char_events100"])
    w("char_press_events_300.csv", R["char_events300"])
    w("trial_level_evaluation.json", R["trial_eval"])
    w("sequence_candidates_latch_raw.csv", pd.DataFrame(R["seq_candidates_raw"]).drop(columns=["by_class", "latencies_ms"]))
    w("sequence_candidates_latch_summary.json", {"overall": R["seq_candidates"], "by_hop": R["seq_candidates_by_hop"]})
    w("diagnostic_sequence.json", R["diag_sequence"])
    w("diagnostic_sequence_windows.csv", R["diag_windows"])
    w("event_strategies_raw.csv", pd.DataFrame(R["strategy_raw"]).drop(columns=["by_class", "latencies_ms"]))
    w("event_strategies_summary.json", {"aggregated": R["strategy_agg"],
                                        "selected_on_validation": R["strategy_selected_on_validation"]})
    if "probe_summary" in R:
        w("mixed_window_probe_summary.csv", R["probe_summary"])
    w("step3b_report.md", report_markdown(R))
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help=f"write results to {OUT}")
    a = ap.parse_args(argv)
    R = run()
    print_summary(R)
    if a.write:
        for p in write_results(R):
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
