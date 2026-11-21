# GripSense TinyML - C port of the depth-2 decision tree

_Generated 2026-10-10T15:22:42+00:00 by `export_and_verify_c_port.py`. Every number below is read from
`results/c_port/verification_summary.json` and `models/c_port/tree_metadata.json`._

> **Software-only.** No new data was collected, the ESP32 was not connected, and no hardware testing was
> performed. The saved model was **not** retrained or modified; the original CSV and split manifest were only read.

## Result

| Check | Result |
|---|---|
| C (given Python feature vectors) vs sklearn `predict`, all trials | **200/200 agree** |
| C (computing its own features from raw ADC) vs sklearn `predict` | **200/200 agree** |
| C features bit-identical to Python features (raw mode) | longest_active_run_frac: 200/200, segment_1_mean: 200/200 |
| Threshold-boundary test vectors (synthetic) | 26/26 agree |
| Python features == saved `reports/trial_features.csv` | True |
| sklearn predictions == `results/followup/predictions_all_models.csv` (DT_depth2, 80 val+test trials) | True |
| Non-finite input rejected by C | True |
| Compiler warnings (`cc` -std=c99 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wdouble-promotion -Wstrict-prototypes -Wmissing-prototypes -Wfloat-equal -Wcast-qual -Wformat=2 -Wundef -Wvla) | **0** |
| Pre-existing project files changed | 0 of 70 |
| Saved model unchanged | True |
| **All checks pass** | **True** |

The tree's own accuracy against the recorded labels (unchanged from phase 2, shown for context only): test 0.950, train 1.000, validation 0.975.
Misclassified: #6 (validation: LIGHT_PRESS -> DOUBLE_SQUEEZE), #39 (test: LIGHT_PRESS -> HARD_PRESS), #157 (test: NO_PRESS -> DOUBLE_SQUEEZE).

## Exported tree (read programmatically from `models/followup/DT_depth2.joblib`)

| Node | Type | Feature | Threshold (float64, exact hex) | Left / Right | Leaf class | Train samples |
|---|---|---|---|---|---|---|
| 0 | split | longest_active_run_frac | 0.35333333909511566 (`0x1.69d0370000000p-2`) | 1 / 4 | | 120 |
| 1 | split | longest_active_run_frac | 0.046666666865348816 (`0x1.7e4b180000000p-5`) | 2 / 3 | | 60 |
| 2 | leaf | | | | NO_PRESS | 30 |
| 3 | leaf | | | | DOUBLE_SQUEEZE | 30 |
| 4 | split | segment_1_mean | 840.6333312988281 (`0x1.a451110000000p+9`) | 5 / 6 | | 60 |
| 5 | leaf | | | | LIGHT_PRESS | 30 |
| 6 | leaf | | | | HARD_PRESS | 30 |

Rules (go left when the float32-cast feature is `<=` the threshold):

```
if ((float)longest_active_run_frac <= 0.35333333909511566):
    if ((float)longest_active_run_frac <= 0.046666666865348816):
        -> NO_PRESS
    else:
        -> DOUBLE_SQUEEZE
else:
    if ((float)segment_1_mean <= 840.6333312988281):
        -> LIGHT_PRESS
    else:
        -> HARD_PRESS
```

**Class mapping.** sklearn stores classes alphabetically (DOUBLE_SQUEEZE, HARD_PRESS, LIGHT_PRESS, NO_PRESS). Each leaf is mapped
**by name** to the C enum in project order: 0 = LIGHT_PRESS, 1 = HARD_PRESS, 2 = DOUBLE_SQUEEZE, 3 = NO_PRESS.

**Feature units.**
- `longest_active_run_frac`: fraction of 150 samples (0..1); longest consecutive run with ADC > 100.
- `segment_1_mean`: ADC counts; mean of samples 15..29 (300-580 ms into the window).

**Comparison semantics.** scikit-learn casts inputs to float32 before comparing them with the float64 thresholds
(which are themselves float32 midpoints). The C code does `(double)(float)x <= threshold`, with thresholds stored as exact
hex-float literals. The boundary test set probes each threshold with its float64/float32 neighbours and the nearest
achievable feature values. There, a naive double-precision comparison without the float32 cast disagreed with sklearn on
**2** vector(s), while the C port agreed on all of them. On the
200 real trials, the naive comparison disagreed on 0. The cast matters only in the narrow gap
between a float64 value and its float32 rounding, but it is required for an exact port.

## Size: estimates vs measurements

| Quantity | Value | Kind |
|---|---|---|
| Phase-2 rough estimate (12 bytes x 7 nodes) | 84 bytes | estimate |
| `sizeof` of the five C tree tables (host) | 84 bytes | measured on host (compile-time `sizeof`) |
| ADC window buffer (150 x uint16) | 300 bytes | by construction |
| Object file `decision_tree.o` (library build, `-DGS_NO_MAIN`), `size` | __TEXT=1159 __DATA=48 __OBJC=0 others=160 dec=1367 hex=557 | measured, host |
| Object file on disk | 3,296 bytes | measured, host (includes symbols/metadata) |
| Test executable, `size` | __TEXT=16384 __DATA=0 __OBJC=0 others=4295000064 dec=4295016448 hex=10000c000 | measured, host (includes harness, stdio, Mach-O padding) |
| Test executable on disk | 34,560 bytes | measured, host |

Notes: the executable's `others` column is dominated by macOS's 4 GB `__PAGEZERO` virtual reservation, which is not
code or data. The `sizeof` value equalling the 84-byte estimate is a coincidence of the chosen types (7 x 8-byte double
thresholds + 4 x 7 one-byte tables), not a confirmation of the estimate method. Storing thresholds as float32 would halve
that table, but would need its own exactness argument.

Binary: `/Users/zenith/Desktop/GripSense-TinyML /models/c_port/build/decision_tree: Mach-O 64-bit executable arm64`. **These are macOS host measurements, not ESP32 flash or RAM figures.** An ESP32 build uses
a different compiler (Xtensa/RISC-V GCC), libc, and runtime, and the stdio test harness would not ship. The model tables
themselves are tiny, so on-device size will be dominated by sampling, the window buffer, and application code.

`size -m` output for the object file:

```
Segment : 1376
	Section (__TEXT, __text): 972
	Section (__TEXT, __literal16): 16
	Section (__TEXT, __const): 85
	Section (__DATA, __const): 48
	Section (__TEXT, __cstring): 86
	Section (__LD, __compact_unwind): 160
	total 1367
total 1376
```

## Compiler output

```
(none)
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
4. **Model limitations carry over unchanged.** The depth-2 tree scored 0.950 on the held-out test set, with wide
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
