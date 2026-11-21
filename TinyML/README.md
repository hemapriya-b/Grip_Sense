# GripSense TinyML - offline FSR gesture classifier

Classifies a 3-second force-sensitive-resistor (FSR) recording from an ESP32 into four intents:

| Label | Meaning |
|---|---|
| LIGHT_PRESS | Hungry |
| HARD_PRESS | Pain |
| DOUBLE_SQUEEZE | Washroom |
| NO_PRESS | Idle/resting |

This phase runs **entirely offline** on `gripsense_clean_data.csv`. No ESP32, serial port or live sensor is needed.
The original CSV is only read, never modified; the experiment checks its SHA-256 hash before and after.

## Install

Requires Python 3.10+ (tested with 3.12). Homebrew Python blocks global `pip install`, so use a virtual environment:

```bash
cd "/Users/zenith/Desktop/GripSense-TinyML "     # note: the folder name ends with a space
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run the full experiment

```bash
.venv/bin/python run_experiment.py --csv gripsense_clean_data.csv --seed 42
.venv/bin/python validate_outputs.py        # independent checks; exits non-zero on any failure
```

`run_experiment.py` runs, in order: dataset audit -> trial-level 60/20/20 stratified split -> feature
extraction -> baselines -> pre-training plan -> model selection (train/validation only) -> one final
test evaluation -> artifacts and report. It takes about 15 seconds on a laptop.

## Predict with the saved model

```bash
.venv/bin/python predict.py --csv gripsense_clean_data.csv --trials 1 60 120 180
```

Any CSV with `trial, elapsed_ms, adc` columns works (150 readings at 20 ms per trial).

## Files

| Path | Purpose |
|---|---|
| `gripsense_features.py` | Feature extraction shared by training and inference (stateless, no fitted params) |
| `run_experiment.py` | Complete reproducible experiment |
| `validate_outputs.py` | Re-derives counts/metrics from raw files, reloads the model, cross-checks the report |
| `predict.py` | Offline inference with the saved model |
| `train_gripsense.py` | Original 80/20 script (kept unchanged for reference; superseded) |
| `reports/GripSense_TinyML_Report.md` | Full report |
| `reports/dataset_audit.csv`, `reports/trial_quality_flags.csv` | Data-quality audit |
| `reports/trial_features.csv` | One row per trial: features, label, split |
| `reports/split_manifest.csv` | Trial -> label -> split |
| `reports/metrics.json` | All measured metrics, versions, seed, stage log |
| `reports/figures/` | Plots |
| `models/` | `gripsense_model.joblib` (model + feature list + labels) and `model_metadata.json` |
| `results/` | Predictions, model-selection table, baselines, classification reports, errors |

## Reproducibility

- Random seed: 42 (split, CV folds, models, figure sampling)
- Split: 120 / 40 / 40 trials (30 / 10 / 10 per class) = 18,000 / 6,000 / 6,000 readings, split by whole trial
- Library versions are pinned in `requirements.txt` and recorded in `reports/metrics.json`
