# GripSense project documentation

Documentation package for **GripSense**, an ESP32 + FSR pressure-gesture recognition prototype with a TinyML
decision tree. It was generated on 2026-10-10 from the project's saved artifacts. No models, data or earlier results were changed.

Start with the [Executive Summary](GripSense_Executive_Summary.md), then the full report
([Markdown](GripSense_Project_Report.md) / [PDF](GripSense_Project_Report.pdf)).

| File | Purpose |
|---|---|
| [GripSense_Project_Report.md](GripSense_Project_Report.md) | Full technical report (Markdown) |
| [GripSense_Project_Report.pdf](GripSense_Project_Report.pdf) | Full technical report (PDF, with cover, contents and page numbers) |
| [GripSense_Executive_Summary.md](GripSense_Executive_Summary.md) | Two-to-three-minute summary |
| [manifest.json](manifest.json) | Every deliverable, its type, purpose and check status |
| [QUALITY_CHECKS.md](QUALITY_CHECKS.md) | Quality-control results for this package |
| [report_facts.json](report_facts.json) | All values used by the report, re-derived from project artifacts |
| [preservation_baseline.json](preservation_baseline.json) | Hashes of pre-existing project files (taken before this work) |
| [figures/GripSense_System_Architecture.png](figures/GripSense_System_Architecture.png) | System architecture and methodology |
| [figures/GripSense_Model_Comparison.png](figures/GripSense_Model_Comparison.png) | Validation vs test accuracy for all evaluated models |
| [figures/GripSense_Model_Size.png](figures/GripSense_Model_Size.png) | Approximate model size and how it was measured |
| [figures/GripSense_C_Port_Verification.png](figures/GripSense_C_Port_Verification.png) | Offline Python-vs-C parity verification |
| [figures/GripSense_Confusion_Matrix.png](figures/GripSense_Confusion_Matrix.png) | Test-split confusion matrices (RF and depth-2 tree) |
| [figures/GripSense_Test_Recall_Uncertainty.png](figures/GripSense_Test_Recall_Uncertainty.png) | Per-class test recall with uncertainty |
| [figures/GripSense_DoubleSqueeze_Presses.png](figures/GripSense_DoubleSqueeze_Presses.png) | Press counts in DOUBLE_SQUEEZE trials |
| [scripts/build_documentation.py](scripts/build_documentation.py) | Generates figures, Markdown, PDF |
| [scripts/check_documentation.py](scripts/check_documentation.py) | Runs the quality checks and writes the manifest |

Rebuild and re-check (from the project folder):

```bash
.venv/bin/python results/project_documentation/scripts/build_documentation.py
.venv/bin/python results/project_documentation/scripts/check_documentation.py
```

Important: all results are offline results on recorded data. Nothing in this package has been validated on an ESP32
or in real-world use.
