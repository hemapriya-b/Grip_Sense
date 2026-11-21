# Quality checks - GripSense documentation package

_Run 2026-10-10T15:57:32+00:00 by `scripts/check_documentation.py`._

**Result: 109/109 checks passed, 0 failed.**

Project scripts that rewrite their own outputs (`export_and_verify_c_port.py`, `validate_followup.py`) were executed in a temporary copy of the project, so no existing file was modified. `decision_tree --info` was executed in place (read-only).

## files

| Status | Check | Detail |
|---|---|---|
| PASS | exists and non-empty: GripSense_Project_Report.md |  |
| PASS | exists and non-empty: GripSense_Project_Report.pdf |  |
| PASS | exists and non-empty: GripSense_Executive_Summary.md |  |
| PASS | exists and non-empty: README.md |  |
| PASS | exists and non-empty: report_facts.json |  |
| PASS | exists and non-empty: preservation_baseline.json |  |
| PASS | exists and non-empty: figures/GripSense_System_Architecture.png |  |
| PASS | exists and non-empty: figures/GripSense_Model_Comparison.png |  |
| PASS | exists and non-empty: figures/GripSense_Model_Size.png |  |
| PASS | exists and non-empty: figures/GripSense_C_Port_Verification.png |  |
| PASS | exists and non-empty: figures/GripSense_Confusion_Matrix.png |  |
| PASS | exists and non-empty: figures/GripSense_Test_Recall_Uncertainty.png |  |
| PASS | exists and non-empty: figures/GripSense_DoubleSqueeze_Presses.png |  |

## figures

| Status | Check | Detail |
|---|---|---|
| PASS | GripSense_System_Architecture.png: not blank | ink fraction 0.067 |
| PASS | GripSense_System_Architecture.png: resolution >= 200 dpi and >= 1500 px wide | 2772x1873 px @ 220 dpi |
| PASS | GripSense_Model_Comparison.png: not blank | ink fraction 0.350 |
| PASS | GripSense_Model_Comparison.png: resolution >= 200 dpi and >= 1500 px wide | 2406x1407 px @ 220 dpi |
| PASS | GripSense_Model_Size.png: not blank | ink fraction 0.123 |
| PASS | GripSense_Model_Size.png: resolution >= 200 dpi and >= 1500 px wide | 2202x1091 px @ 220 dpi |
| PASS | GripSense_C_Port_Verification.png: not blank | ink fraction 0.060 |
| PASS | GripSense_C_Port_Verification.png: resolution >= 200 dpi and >= 1500 px wide | 2601x1704 px @ 220 dpi |
| PASS | GripSense_Confusion_Matrix.png: not blank | ink fraction 0.144 |
| PASS | GripSense_Confusion_Matrix.png: resolution >= 200 dpi and >= 1500 px wide | 2551x1173 px @ 220 dpi |
| PASS | GripSense_Test_Recall_Uncertainty.png: not blank | ink fraction 0.408 |
| PASS | GripSense_Test_Recall_Uncertainty.png: resolution >= 200 dpi and >= 1500 px wide | 2179x942 px @ 220 dpi |
| PASS | GripSense_DoubleSqueeze_Presses.png: not blank | ink fraction 0.187 |
| PASS | GripSense_DoubleSqueeze_Presses.png: resolution >= 200 dpi and >= 1500 px wide | 1849x901 px @ 220 dpi |
| PASS | GripSense_System_Architecture.png: visually inspected (no clipping/overlap) | after fix: subtitle/lane-label overlap and arrows crossing labels removed (elbow connectors); parity box text wrapped |
| PASS | GripSense_Model_Comparison.png: visually inspected (no clipping/overlap) | legend above plot, value labels and Wilson error bars legible |
| PASS | GripSense_Model_Size.png: visually inspected (no clipping/overlap) | after fix: '1 feature' singular; estimate vs measured legend legible |
| PASS | GripSense_C_Port_Verification.png: visually inspected (no clipping/overlap) | after fix: red subtitle no longer collides with 'Comparison results'; warning box text inside box |
| PASS | GripSense_Confusion_Matrix.png: visually inspected (no clipping/overlap) | both matrices, labels and titles legible; marked TEST split |
| PASS | GripSense_Test_Recall_Uncertainty.png: visually inspected (no clipping/overlap) | after fix: legend moved above plot (previously covered NO_PRESS bars) |
| PASS | GripSense_DoubleSqueeze_Presses.png: visually inspected (no clipping/overlap) | after fix: annotation arrow no longer covers the zero label |

## pdf

| Status | Check | Detail |
|---|---|---|
| PASS | page count >= 10 | 15 pages |
| PASS | every page has selectable text | min chars/page 387 |
| PASS | embedded figures >= 7 | 7 images |
| PASS | cover page with title and date |  |
| PASS | table of contents present |  |
| PASS | page numbers in footer |  |
| PASS | PDF bookmarks/outline present | 19 top-level entries |
| PASS | section heading present: 1. Title and project overview |  |
| PASS | section heading present: 9. Model evaluation and comparison |  |
| PASS | section heading present: 11. C-port verification results |  |
| PASS | section heading present: 13. Data quality and methodological limitations |  |
| PASS | section heading present: 19. Conclusion |  |

## links

| Status | Check | Detail |
|---|---|---|
| PASS | all 19 README links resolve | [] |
| PASS | all 7 report image links resolve |  |

## metrics

| Status | Check | Detail |
|---|---|---|
| PASS | dataset: 30,000 rows / 200 trials / 150 per trial / 50 per label |  |
| PASS | split 120/40/40 |  |
| PASS | B0_majority validation accuracy 0.250 matches saved metrics and report facts |  |
| PASS | B0_majority test accuracy 0.250 matches saved metrics and report facts |  |
| PASS | B1_mean_adc_centroid validation accuracy 0.725 matches saved metrics and report facts |  |
| PASS | B1_mean_adc_centroid test accuracy 0.750 matches saved metrics and report facts |  |
| PASS | RF_existing validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | RF_existing test accuracy 0.975 matches saved metrics and report facts |  |
| PASS | LR_C0.1 validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | LR_C0.1 test accuracy 0.950 matches saved metrics and report facts |  |
| PASS | LR_C1 validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | LR_C1 test accuracy 0.975 matches saved metrics and report facts |  |
| PASS | LR_C10 validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | LR_C10 test accuracy 0.975 matches saved metrics and report facts |  |
| PASS | DT_depth2 validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | DT_depth2 test accuracy 0.950 matches saved metrics and report facts |  |
| PASS | DT_depth3 validation accuracy 0.975 matches saved metrics and report facts |  |
| PASS | DT_depth3 test accuracy 0.950 matches saved metrics and report facts |  |
| PASS | C-port agreement re-derived = summary json | 200/200, 200/200 |
| PASS | boundary tests re-derived = summary json | 26/26 |
| PASS | compiler warnings = 0 in compile log |  |

## consistency

| Status | Check | Detail |
|---|---|---|
| PASS | '30,000' in Markdown and PDF (rows 30,000) |  |
| PASS | '200/200' in Markdown and PDF (200/200 agreement) |  |
| PASS | '26/26' in Markdown and PDF (boundary 26/26) |  |
| PASS | '0 of 70' in Markdown and PDF (0 of 70 files changed) |  |
| PASS | '0.975' in Markdown and PDF (RF test acc) |  |
| PASS | '0.950' in Markdown and PDF (DT test acc) |  |
| PASS | '0.750' in Markdown and PDF (mean-ADC test acc) |  |
| PASS | '0.250' in Markdown and PDF (majority acc) |  |
| PASS | '120' in Markdown and PDF (split 120 / 40 / 40) |  |
| PASS | '0.72' in Markdown and PDF (Wilson lower bound) |  |
| PASS | '30,672' in Markdown and PDF (RF size estimate) |  |
| PASS | '1,192' in Markdown and PDF (LR size) |  |
| PASS | '84 bytes' in Markdown and PDF (DT tables bytes) |  |
| PASS | '{3: 13, 4: 20, 5: 11, 6: 5, 7: 1}' in Markdown and PDF (DS press distribution) |  |
| PASS | 'trial 39' in Markdown and PDF (misclassified trial 39) |  |
| PASS | 'trial 156' in Markdown and PDF (flagged NO_PRESS 156) |  |
| PASS | '157' in Markdown and PDF (flagged NO_PRESS 157) |  |
| PASS | 'LIGHT_PRESS = 0' in Markdown and PDF (class map) |  |
| PASS | '(double)(float)v' in Markdown and PDF (float cast) |  |
| PASS | executive summary contains 200/200 |  |
| PASS | executive summary contains 0.975 |  |
| PASS | executive summary contains 0.950 |  |
| PASS | executive summary contains 26/26 |  |

## claims

| Status | Check | Detail |
|---|---|---|
| PASS | report states: no hardware statement |  |
| PASS | report states: not a medical device |  |
| PASS | report states: parity is not real-world accuracy |  |
| PASS | report states: test previously inspected |  |
| PASS | report states: simulated edits not real |  |
| PASS | report states: C port not live detector |  |
| PASS | PDF repeats offline/not-medical cover statement |  |
| PASS | no over-claiming phrases (real-world proof, ESP32-validated, clinical) | [] |
| PASS | every 'medical device' mention is negated | 2 mentions |
| PASS | status table marks 'ESP32 target compilation' as not verified | / ESP32 target compilation / Not verified / No target build artifacts / |
| PASS | status table marks 'Live sensor inference' as not verified | / Live sensor inference / Not verified / No hardware tests were run / |

## commands

| Status | Check | Detail |
|---|---|---|
| PASS | export_and_verify_c_port.py (temp copy) exit 0 and ALL CHECKS PASS | features-mode agreement: 200/200 / raw-ADC-mode agreement:  200/200 / ALL CHECKS PASS: True |
| PASS | validate_followup.py (temp copy) exit 0 | 94/94 checks passed |
| PASS | phase-1 validate_outputs.py (run by validate_followup) passes | 91/91 checks passed |
| PASS | ./models/c_port/build/decision_tree --info (in place, read-only) exit 0 | n_nodes 7 |

## preservation

| Status | Check | Detail |
|---|---|---|
| PASS | all 81 pre-existing files unchanged | [] |
| PASS | no new files outside results/project_documentation/ (macOS Finder .DS_Store excluded) | other: []; Finder metadata ignored: ['.DS_Store', 'reports/.DS_Store'] |

## Figure properties

| Figure | Pixels | DPI | Ink fraction |
|---|---|---|---|
| GripSense_System_Architecture.png | 2772x1873 | 220 | 0.0674 |
| GripSense_Model_Comparison.png | 2406x1407 | 220 | 0.3504 |
| GripSense_Model_Size.png | 2202x1091 | 220 | 0.1226 |
| GripSense_C_Port_Verification.png | 2601x1704 | 220 | 0.0596 |
| GripSense_Confusion_Matrix.png | 2551x1173 | 220 | 0.1436 |
| GripSense_Test_Recall_Uncertainty.png | 2179x942 | 220 | 0.4081 |
| GripSense_DoubleSqueeze_Presses.png | 1849x901 | 220 | 0.1873 |

## Limitations of these checks

- Visual quality (clipping, overlap, legibility) was judged by a manual look at each PNG and at rendered PDF pages; automated checks only detect blank or low-resolution images.
- Claim checks are pattern-based (required statements present, risky phrases absent); they complement but do not replace reading the report.
- The PDF's internal layout was checked via text extraction and image counts, not pixel comparison.
- Nothing here validates GripSense on hardware or in real-world use.
