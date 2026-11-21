# Step 3B - offline evaluation of transition robustness (EXPLORATORY)

Nothing in this folder changes the production simulator, the saved model, the C code, the dataset or any
earlier result. Candidate models exist only in memory. **No result here is validated performance**: the
sequences are concatenations of recorded 3-second trials (not continuous recordings), mixed windows have no
ground truth, and the test split had already been inspected in earlier phases.

All numbers below are copied from `results/step3b_report.md` / `results/*.json`, which `run_step3b.py --write`
generates.

## Files

| File | Purpose |
|---|---|
| `common.py` | data access via the existing trial-level split manifest, sequence builder, window schedule (same as the production simulator), episode bookkeeping |
| `characterization.py` | Investigation 1: training-data feature distributions, class overlap, run-length bins, press counts |
| `candidates.py` | Investigation 2: candidate classifiers on existing features; trial-level metrics |
| `transitions.py` | Investigation 3: window prediction recording, episode-based event scoring, mixed-window probe |
| `event_strategies.py` | Investigation 4: alternative event/state strategies on recorded window labels |
| `run_step3b.py` | Runs everything; prints a summary; `--write` writes `results/` |
| `tests/` | 28 tests (incl. a preservation check that runs last) |
| `results/` | generated CSV/JSON + `step3b_report.md` |

## Methodology

- **Splits.** The existing trial-level manifest (`reports/split_manifest.csv`): 120 train / 40 validation /
  40 test trials, 30/10/10 per class. Rows are whole trials, so no window or trial is shared across splits.
  Candidates are fitted on the 120 train trials only. Train 5-fold CV is stratified over whole trials; the saved
  tree is CV'd by refitting an unfitted clone per fold.
- **Candidates** (defined before any validation/window results were computed):
  `current_dt2` (saved tree via the verified C code), `gate_ev2` / `gate_ev3` (current tree, but its DS leaf
  requires `press_events_100 >= 2` / `>= 3`; gated-out windows go to LIGHT/HARD via an `active_mean` stump fitted on
  train trials), `dt2_robust` / `dt3_robust` (depth-2/3 trees on press_events_100, press_events_300, active_mean,
  max and q90), and `dt3_all` (depth 3, all 49 features).
- **Sequences.** For each held-out split and seeds 0, 1, 2: rest, P1, rest, P2, ..., rest, where every non-rest
  trial of that split appears once and the split's NO_PRESS trials are reused as separators. Hops 5, 10 and 25. Each
  press trial is one **episode** and expects exactly one event with its label. The diagnostic sequence
  `1 2 3 151 152 4` uses TRAINING trials and is reported separately.
- **Event scoring.** An event spans windows first_window..emit_window and is attributed to the episode it overlaps.
  The scoring counts: **hits** (an event with the correct label), **missed** (none), **false events** (wrong label
  within an episode, or during rest, or overlapping two episodes), **duplicates**, **recovered** (a wrong event
  followed by the correct one), and latency (episode start to the end of the emitting window).
- **Mixed-window probe.** Every offset of rest+X and X+rest (hop 1). Only the predicted-label distribution is
  reported; it is not accuracy.
- **Event strategies.** These run on the recorded window labels. Parameters (K; DS agreement time; correction
  time) are chosen on the **validation** sequences by minimising false + missed. Test sequences are reported for the
  chosen settings and for all others, for transparency.

## Findings

1. **The DS class is defined by short runs, which partial windows reproduce.** All 50 DOUBLE_SQUEEZE trials
   have a longest active run of 8-46 samples, inside the tree's DS leaf (8-53). A sustained press that only
   partly overlaps a window produces any run length, so its 8-53-sample offsets land in the DS leaf.
   Mixed-window probe, `current_dt2`: 30.9-37.3% of mixed LIGHT/HARD windows are predicted DOUBLE_SQUEEZE.
2. **Press count separates the recorded classes at trial level.** Every DS trial has `press_events_100 >= 3`.
   Across all 200 trials, 10 non-DS trials have >= 2 and 5 have >= 3. The training DS trials contain 3-6 presses,
   and **0 have exactly two** (existing press-segment table). Gating DS on press count kept trial-level validation
   accuracy at 0.975 (same as `current_dt2`), and cut DS predictions on mixed LIGHT/HARD windows to 0-5.9%.
3. **The production latch turns the false DS windows into missed events.** With K=2/N=1, `current_dt2` got a
   false DS event on every LIGHT/HARD episode in the validation sequences (180 false, 90/270 hits, 0 recovered). On
   test it had 162 false and 81/270 hits. `dt3_all` behaves identically. `gate_ev2`: validation 204 hits / 66 missed
   / 66 false (9 DS); test 166 / 104 / 77 (9 DS).
4. **Gating causes a second failure.** `segment_1_mean` is positional (window samples 15-29). On onset windows it
   reads rest, so HARD onsets are often labelled LIGHT (gate_ev2 test: 53.2% of mixed HARD-onset windows). Also,
   partial DS windows contain fewer presses, so DS episodes get locked to LIGHT/HARD (gate_ev2 DS hits: validation
   42/90, test 31/90). `dt3_robust` and `gate_ev3` almost never fire DS in sequences (DS hits: dt3_robust validation
   3/90, test 0/90).
5. **Event strategies trade false events against latency.**
   - `on_release` (emit the longest-streak label when the press ends) had the lowest missed + false total for all
     six candidates, on both validation and test sequences:
     current_dt2 test 63 missed / 36 false; gate_ev2 test 63 / 36 (validation 12 / 12).
   - Its latency is about 6 s (median 5980 ms), because the press must leave the 3-s window before the decision.
   - `correctable_1200ms` recovers almost every episode (current_dt2 test: 9 missed) but emits a false event first
     (288 false events, 168 recovered).
   - Higher K / slower DS reduce false events but miss more episodes.
6. **Results depend on hop.** gate_ev2 + latch K=2: test hits 47/90 at hop 5 and 10, but 72/90 at hop 25. A larger
   hop samples fewer partial offsets.
7. **Recorded rest is not always quiet.** Test NO_PRESS trial 157 contains a 12-sample press, so its windows never
   read NO_PRESS. The latch then never re-arms, and the following episode gets no event.
8. **Diagnostic sequence (training trials).** `gate_ev3` and `dt3_robust` emit LIGHT_PRESS for trial 4.
   `current_dt2`, `dt3_all` and `gate_ev2` still emit DOUBLE_SQUEEZE: trial 4's initial decaying tail plus its
   press count as 2 events.

## What can and cannot be concluded

- **Can (on recorded data):** the false DS events come from the tree's run-length feature meeting partial windows.
  Press count and level features are less affected by window position. The latch converts single misclassified
  onset windows into missed events. Waiting for the press to end avoids most of this, at a large latency cost.
- **Cannot:**
  - Real transition behaviour: the concatenations have abrupt joins, and 58/60 train LIGHT/HARD trials were
    captured already pressed (active at sample 0).
  - Real event rates or latency on a live sensor.
  - Any statement about **exact two-squeeze recognition**: no labelled two-squeeze trials exist. `gate_ev3` fits
    the data only by excluding a true double squeeze, and `gate_ev2` was never tested on two-press examples.
  - Independent performance: candidate design and strategy parameters were chosen by looking at
    validation-derived sequences, and the test split had been seen before.

## Recommended next experiment

Collect fresh, **genuinely continuous** FSR recordings on the ESP32 with event-level labels:
- rest -> press -> release -> rest, including onsets and releases
- LIGHT/HARD holds of varied length
- **deliberate exactly-two-squeeze gestures**, plus 3+-squeeze and single short-tap negatives
- noisy rest

Label the gesture start/end times rather than fixed 3-s windows. Then:
1. Re-evaluate `current_dt2` with `on_release`, and a press-count-gated DS rule, on the new recordings.
2. Train or check a window-level classifier on windows cut from continuous data. The first candidate would be one
   that avoids positional features (e.g. replacing `segment_1_mean` with a level feature).
3. Report results on a fresh held-out recording session.

Until then, keep production unchanged.
