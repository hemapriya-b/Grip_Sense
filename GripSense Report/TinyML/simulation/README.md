# GripSense live-input simulator (Step 3, software only)

Replays recorded trials from `gripsense_clean_data.csv` as a simulated ADC stream, classifies 150-sample
windows with the **verified C decision tree**, and turns per-window labels into deduplicated events.

> Simulation only. No hardware, no firmware, no new data. The model, dataset, C source and all earlier
> results are only read. Replayed and concatenated trials are **not** real continuous recordings.

## Pipeline

```
CSV (read-only) -> stream.replay -> WindowBuffer(150) -> inference backend -> label -> EventDecider -> events
                   (trial order,     (ring buffer,       (C tree via ctypes;    (pure state machine;
                    sample order,     reset per trial     sklearn reference)     no model knowledge)
                    trial tags)       in isolated mode)
```

| File | Role |
|---|---|
| `stream.py` | Loads and validates trials (150 samples, `elapsed_ms` 0..2980 in 20 ms steps), replays them in a given order; every sample carries segment, trial, index, stream time |
| `window_buffer.py` | Configurable ring buffer (default 150 samples = 3 s @ 20 ms); integer ADC 0..4095 only, no clipping |
| `inference.py` | `CTreeBackend`: compiles the unchanged `models/c_port/decision_tree.c` (hash-checked against `tree_metadata.json`) into a temp-dir shared library with the verified warning flags + `-DGS_NO_MAIN -shared -fPIC`, calls `gs_predict_from_adc` via `ctypes`. `SklearnReferenceBackend`: `gripsense_features` + `DT_depth2.joblib` |
| `decision.py` | `EventDecider` state machine (IDLE / CANDIDATE / UNCERTAIN / ACTIVE) |
| `simulator.py` | Wires the above; `isolated` and `continuous` modes |
| `run_simulation.py` | CLI |

## Modes

- **isolated** - buffer reset at every trial boundary, so each trial yields exactly one window, equal to the
  recorded trial ("aligned"). Used for parity with `results/c_port/predictions_python_vs_c.csv`. The event
  decider is not run (one window per trial cannot meet K>=2, and chaining independent trials would invent a stream).
- **continuous** - requested trials are concatenated; a window is classified every `hop` samples. Windows that
  span two trials are flagged `mixed` and have no ground truth. Windows that coincide exactly with one trial are
  `aligned` and are checked against the stored reference.

## Event logic (initial simulation settings - NOT validated final settings)

| Setting | Default | Meaning |
|---|---|---|
| `hop` | 10 samples (200 ms) | samples between inferences |
| `K` (`--k`) | 2 | consecutive windows with the same non-idle label needed to emit an event |
| `N` (`--n-rearm`) | 1 | consecutive NO_PRESS windows needed to re-arm after an event |

- **IDLE**: last window NO_PRESS (or start). **CANDIDATE**: non-idle label seen in fewer than K consecutive windows.
  **UNCERTAIN**: latest non-idle window disagrees with the previous one, or the C code rejected the input.
  **ACTIVE**: event emitted, latched.
- **Deduplication**: while ACTIVE, nothing is emitted for overlapping windows, *including a different non-idle
  label* (counted as `suppressed_label_changes`). Latching on a label change is a **design choice to revisit after
  testing** (see the observed limitation below).
- **No confidence score.** Every leaf of the depth-2 tree is pure, so `predict_proba` is always 1.0. The simulator
  outputs labels only; UNCERTAIN is based on window-to-window disagreement, not probability.
- **DOUBLE_SQUEEZE** events carry a note: the training trials contain 3-7 presses and none with exactly two, so
  exact two-squeeze recognition is **not validated**.

## Run

```bash
cd "/Users/zenith/Desktop/GripSense-TinyML "
.venv/bin/python -B -m simulation.run_simulation --mode isolated
.venv/bin/python -B -m simulation.run_simulation --mode continuous --trials 1 2 3 151 152 4 --hop 10 --show-windows
.venv/bin/python -B -m unittest discover -s simulation/tests -t . -v      # simulator tests
.venv/bin/python -B -m unittest discover -s tests -t . -v                 # Step-2 regression suite
```

Output goes to stdout. `--out-dir` writes `windows_<mode>.csv` / `events_<mode>.json`, and refuses any path inside
the project folder (use e.g. `/tmp/gripsense_sim`). The C library is built in a system temp directory and deleted
afterwards.

## Observed behaviour and limitations

1. **Transition windows mislabel.** In the continuous example (`1 2 3 151 152 4`), windows mixing rest with the start of
   LIGHT_PRESS trial 4 were classified DOUBLE_SQUEEZE for 2+ consecutive windows, so a DOUBLE_SQUEEZE event was emitted
   and the latch then suppressed the correct LIGHT_PRESS. Windows mixing LIGHT_PRESS with rest also produced
   DOUBLE_SQUEEZE. The tree was trained on whole, aligned trials; it is not robust to partial presses inside a window
   (consistent with follow-up probe P3). `simulation/tests/test_simulator_e2e.py` pins this as a characterization test.
2. **Concatenated trials are not continuous recordings.** Boundaries between trials are artificial joins; real
   transitions (onset, release, rest floor) may look different.
3. Only aligned windows have reference predictions and ground truth. Mixed-window predictions have no correct answer to
   score against and must not be read as accuracy.
4. K, N, hop and the latch are untuned initial settings. No onset detection, press counting, or timing on hardware.
5. Host build only (macOS clang, shared library). Not ESP32 firmware; not dashboard-integrated.
