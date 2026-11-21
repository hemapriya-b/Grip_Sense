# GripSense - Executive Summary

_2026-10-10 | offline, software-only work on the existing dataset; no hardware used._

**Goal.** Recognise four grip gestures on a force sensor (FSR) read by an ESP32, so that a person can signal a
need to a caretaker: LIGHT_PRESS = Hungry, HARD_PRESS = Pain, DOUBLE_SQUEEZE = Washroom, NO_PRESS = resting. These
meanings are design intentions, not evidence that the system can infer real needs.

**Data.** 30,000 sensor readings: 200 recorded trials of 3 seconds (150 samples each), 50 per gesture. The data is clean
(nothing missing or duplicated). Trials are split by whole trial into 120 training, 40 validation and 40 test trials.

**Methods.** Two baselines (always-the-same-answer; average force only), a Random Forest, logistic regression, and
small decision trees, all using 49 summary features per trial.

**Results (40 test trials).** Random Forest 0.975, logistic regression (C=1) 0.975, depth-2 decision tree
0.950; baselines 0.250 and 0.750. The strong models are statistically indistinguishable.
With 10 test trials per gesture, these numbers are uncertain, and the test set had already been looked at.

**C port.** The depth-2 tree was chosen for its small size (7 decisions, 2 features, ~84 bytes), not for higher
accuracy. It was exported to plain C. The C version matches Python on **200/200** trials (given features) and
**200/200** (from raw sensor data), plus 26/26 edge-case tests, with 0 compiler warnings. This proves the C code
equals the Python model on recorded data. It does not prove real-world accuracy.

**Major limitations.** The "double squeeze" recordings actually contain 3-7 squeezes. Each gesture was recorded in
one block, so changes over time are mixed up with the gesture. Most presses are recorded mid-hold. Nothing has run on
the ESP32 or on a live signal. The caretaker dashboard exists as a separate project and is not connected to this
model.

**Status.** Offline pipeline, model comparison, C export and parity tests: done. ESP32 build, live inference,
windowing and dashboard integration: not verified. This is a prototype, not a medical or emergency device.
