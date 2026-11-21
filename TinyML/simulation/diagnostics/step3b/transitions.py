"""
Investigation 3: transition behaviour on replayed, concatenated trials (synthetic streams).

* Windows spanning two trials are `mixed` and never receive a ground-truth label.
* Labelled window accuracy is NOT computed on mixed windows.
* Events are scored per EPISODE (a press trial, or a run of same-label press trials, between rest
  trials): exactly one event with the episode's label is expected.

Event attribution: an event covers windows first_window..emit_window. It is attributed to the
episode whose samples overlap that span; overlapping no episode = false event during rest;
overlapping several = ambiguous (counted as false).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gripsense_features import SAMPLE_INTERVAL_MS

from .common import IDLE, PRESS_LABELS, W, Episode, episodes, stream_of, trial_ids, trials, window_feature_frame, window_table


def record_predictions(order, hop: int, cands) -> pd.DataFrame:
    """Window table + one prediction column per candidate (window labels are 'recorded' here and
    reused by the event-strategy comparison)."""
    stream = stream_of(order)
    wt = window_table(order, hop)
    X = window_feature_frame(stream, wt["start"])
    adc = [stream[s:s + W] for s in wt["start"]]
    for c in cands:
        wt[f"pred_{c.name}"] = c.predict(X, adc)
    wt["press_events_100"] = X["press_events_100"].to_numpy()
    wt["longest_run_samples"] = np.rint(X["longest_active_run_frac"].to_numpy() * W).astype(int)
    return wt


def score_events(events, wt: pd.DataFrame, eps: list[Episode]) -> dict:
    per_ep = {i: [] for i in range(len(eps))}
    rest_false = ambiguous = 0
    for e in events:
        lo, hi = wt.loc[e.first_window, "start"], wt.loc[e.emit_window, "end"]
        hit = [i for i, ep in enumerate(eps) if ep.start_index <= hi and lo <= ep.end_index]
        if not hit:
            rest_false += 1
        elif len(hit) > 1:
            ambiguous += 1
        else:
            per_ep[hit[0]].append(e)
    hits = missed = wrong = dup = recovered = first_correct = false_ds = 0
    latency = []
    by_class = {lab: {"episodes": 0, "hit": 0, "missed": 0, "wrong_events": 0} for lab in PRESS_LABELS}
    for i, ep in enumerate(eps):
        evs = per_ep[i]
        good = [e for e in evs if e.label == ep.label]
        bad = [e for e in evs if e.label != ep.label]
        by_class[ep.label]["episodes"] += 1
        by_class[ep.label]["wrong_events"] += len(bad)
        wrong += len(bad)
        false_ds += sum(e.label == "DOUBLE_SQUEEZE" for e in bad)
        if good:
            hits += 1
            by_class[ep.label]["hit"] += 1
            dup += len(good) - 1
            latency.append((wt.loc[good[0].emit_window, "end"] - ep.start_index) * SAMPLE_INTERVAL_MS)
            if evs[0].label == ep.label:
                first_correct += 1
            else:
                recovered += 1
        else:
            missed += 1
            by_class[ep.label]["missed"] += 1
    return {"episodes": len(eps), "hits": hits, "missed": missed, "first_event_correct": first_correct,
            "wrong_label_events": wrong, "false_ds_events": false_ds, "rest_false_events": rest_false,
            "ambiguous_events": ambiguous, "false_events_total": wrong + rest_false + ambiguous,
            "duplicate_events": dup, "recovered_after_false": recovered,
            "median_latency_ms": float(np.median(latency)) if latency else None, "latencies_ms": latency,
            "by_class": by_class}


def single_trial_window_accuracy(wt: pd.DataFrame, cand: str) -> dict:
    """Context only: windows lying inside one trial (here: aligned whole trials). Mixed excluded."""
    s = wt[~wt["mixed"]]
    return {"n": int(len(s)), "correct": int((s[f"pred_{cand}"] == s["true_label"]).sum())}


def mixed_window_probe(split: str, cands, rest_trial: int | None = None, only_trials=None) -> pd.DataFrame:
    """For every non-rest trial X of the split, build rest+X and X+rest and classify every mixed
    window offset (hop 1). Descriptive only: mixed windows have no ground truth. Reports the
    predicted-label distribution by direction and by how many samples come from X."""
    rest = rest_trial if rest_trial is not None else _quiet_rest(split)
    rows = []
    for lab in PRESS_LABELS:
        for t in trial_ids(split, lab):
            if only_trials is not None and t not in only_trials:
                continue
            for direction, order in (("onset rest->X", (rest, t)), ("release X->rest", (t, rest))):
                stream = stream_of(order)
                starts = list(range(1, W))                          # all strictly mixed offsets
                X = window_feature_frame(stream, starts)
                adc = [stream[s:s + W] for s in starts]
                # window [s, s+149]: onset has s samples from X (X starts at 150); release has 150-s
                n_from_x = [(W - s) if order[0] == t else s for s in starts]
                for c in cands:
                    pred = c.predict(X, adc)
                    for nx, p in zip(n_from_x, pred):
                        rows.append({"split": split, "trial": t, "trial_label": lab, "direction": direction,
                                     "samples_from_trial": nx, "candidate": c.name, "pred": p})
    return pd.DataFrame(rows)


def _quiet_rest(split: str) -> int:
    """First NO_PRESS trial of the split with no sample above 0 (a clean rest segment)."""
    for t in trial_ids(split, IDLE):
        if max(trials()[t].adc) == 0:
            return t
    raise ValueError(f"no all-zero NO_PRESS trial in {split}")


def probe_summary(probe: pd.DataFrame) -> pd.DataFrame:
    g = probe.groupby(["split", "candidate", "trial_label", "direction"])
    return g["pred"].value_counts(normalize=True).unstack(fill_value=0.0).reset_index()
