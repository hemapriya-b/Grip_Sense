## Pre-training plan (written 2026-10-10T14:34:36+00:00, before any classifier was fitted)

**Dataset.** `gripsense_clean_data.csv` (SHA-256 `05e0fc031e28a240...`): 30,000 ADC readings from
200 trials x 150 readings at 20 ms (3.0 s per trial), 4 classes x 50 trials.

**Quality.** Audit outcome: {'PASS': 28, 'INFO': 12, 'WARN': 3}. No missing values, duplicates, inconsistent labels, irregular
sampling or saturation. 20 trials flagged for review and **kept**. Main concern: classes were
recorded in contiguous blocks (LIGHT_PRESS: 1-50; HARD_PRESS: 51-100; DOUBLE_SQUEEZE: 101-150; NO_PRESS: 151-200), so class is confounded with recording session.

**Class distribution.** Perfectly balanced: 50 trials (7,500 readings) per class.

**Planned features.** 49 per-trial features from the trial's own readings only: summary statistics
(mean, median, std, min, max, range, percentiles, IQR, RMS), zero-reading fraction, fixed-threshold
fractions (50, 100, 500, 1000, 1500, 2000 ADC), active-segment mean (> 100), longest zero/active runs,
press-event counts (rising crossings of [100, 300] ADC), peak position, consecutive-difference
statistics, and mean/max over 10 equal time segments. Thresholds are fixed constants, not tuned.
Trial ID and `elapsed_ms` are not used as features.

**Split.** Stratified by label at trial level, seed 42: 120/40/40 trials (30/10/10 per class),
18,000/6,000/6,000 readings. No trial appears in more than one split (verified).

**Baseline (train -> validation).** B0_majority_class: accuracy 0.250, macro-F1 0.100.
B1_nearest_centroid_mean_adc: accuracy 0.725, macro-F1 0.714. These are baseline results, not model results.

**Evaluation plan.** Candidates (decision tree, logistic regression, Random Forest; small fixed grid) are fitted
on the training set only. Selection rule, declared in advance: highest validation macro-F1, then highest
5-fold CV macro-F1 on the training set, then lowest validation log-loss, then the simpler model family.
The selected model (trained on the training set only) is then evaluated **once** on the 40 test trials.
Test results are not used for any decision.
