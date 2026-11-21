#!/usr/bin/env python3
"""
Quality checks for the GripSense documentation package. Writes manifest.json and QUALITY_CHECKS.md.

Non-destructive: project scripts that rewrite their own outputs are executed in a TEMPORARY COPY of the
project, never in place. Only results/project_documentation/ is written.

    .venv/bin/python results/project_documentation/scripts/check_documentation.py
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "results" / "project_documentation"
FIG = DOC / "figures"
PY = sys.executable
checks: list[dict] = []


def check(group, name, ok, detail=""):
    checks.append({"group": group, "check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {group}: {name}" + (f" ({detail})" if detail else ""))


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def main() -> None:
    F = json.loads((DOC / "report_facts.json").read_text())
    figures = ["GripSense_System_Architecture.png", "GripSense_Model_Comparison.png", "GripSense_Model_Size.png",
               "GripSense_C_Port_Verification.png", "GripSense_Confusion_Matrix.png",
               "GripSense_Test_Recall_Uncertainty.png", "GripSense_DoubleSqueeze_Presses.png"]
    core = ["GripSense_Project_Report.md", "GripSense_Project_Report.pdf", "GripSense_Executive_Summary.md",
            "README.md", "report_facts.json", "preservation_baseline.json"]
    # placeholder so README links to QC/manifest resolve while checking
    for name in ("QUALITY_CHECKS.md", "manifest.json"):
        if not (DOC / name).exists():
            (DOC / name).write_text("(pending)\n")

    # ---- 1. files exist
    for f in core + [f"figures/{x}" for x in figures]:
        p = DOC / f
        check("files", f"exists and non-empty: {f}", p.exists() and p.stat().st_size > 0)

    # ---- 2. figures not blank / adequate resolution
    fig_info = {}
    for f in figures:
        im = Image.open(FIG / f)
        arr = np.asarray(im.convert("L"), dtype=float)
        bg = np.median(arr)
        ink = float(np.mean(np.abs(arr - bg) > 30))
        dpi = im.info.get("dpi", (0, 0))[0]
        fig_info[f] = {"px": im.size, "dpi": round(dpi), "ink_fraction": round(ink, 4)}
        check("figures", f"{f}: not blank", ink > 0.01, f"ink fraction {ink:.3f}")
        check("figures", f"{f}: resolution >= 200 dpi and >= 1500 px wide",
              dpi >= 199 and im.size[0] >= 1500, f"{im.size[0]}x{im.size[1]} px @ {dpi:.0f} dpi")
    manual = DOC / "scripts" / "manual_figure_inspection.json"
    if manual.exists():
        mi = json.loads(manual.read_text())
        for f in figures:
            r = mi.get(f, {})
            check("figures", f"{f}: visually inspected (no clipping/overlap)", r.get("ok") is True, r.get("note", "not inspected"))
    else:
        check("figures", "manual visual inspection recorded", False, "scripts/manual_figure_inspection.json missing")

    # ---- 3. PDF
    rd = PdfReader(str(DOC / "GripSense_Project_Report.pdf"))
    pages = [p.extract_text() or "" for p in rd.pages]
    pdf_text = norm(" ".join(pages))
    n_img = sum(len(p.images) for p in rd.pages)
    check("pdf", "page count >= 10", len(rd.pages) >= 10, f"{len(rd.pages)} pages")
    check("pdf", "every page has selectable text", all(len(t.strip()) > 20 for t in pages[1:]) and len(pages[0].strip()) > 20,
          f"min chars/page {min(len(t) for t in pages)}")
    check("pdf", "embedded figures >= 7", n_img >= 7, f"{n_img} images")
    check("pdf", "cover page with title and date", "GripSense" in pages[0] and F["report_date"] in pages[0])
    check("pdf", "table of contents present", "Contents" in pages[1] and "Executive summary" in norm(pages[1]))
    check("pdf", "page numbers in footer", all(f"Page {i + 1}" in pages[i] for i in range(1, len(pages))))
    outline = rd.outline
    check("pdf", "PDF bookmarks/outline present", len(outline) >= 19, f"{len(outline)} top-level entries")
    for sec in ["1. Title and project overview", "9. Model evaluation and comparison", "11. C-port verification results",
                "13. Data quality and methodological limitations", "19. Conclusion"]:
        check("pdf", f"section heading present: {sec}", sec in pdf_text)

    # ---- 4. README links
    readme = (DOC / "README.md").read_text()
    links = re.findall(r"\]\(([^)]+)\)", readme)
    bad = [l for l in links if not (DOC / l).exists()]
    check("links", f"all {len(links)} README links resolve", not bad, bad)
    md = (DOC / "GripSense_Project_Report.md").read_text()
    img_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", md)
    check("links", f"all {len(img_links)} report image links resolve", all((DOC / l).exists() for l in img_links))

    # ---- 5. metrics agree with original artifacts (independent re-derivation)
    md_n = norm(md)
    ex = norm((DOC / "GripSense_Executive_Summary.md").read_text())
    df = pd.read_csv(ROOT / "gripsense_clean_data.csv")
    check("metrics", "dataset: 30,000 rows / 200 trials / 150 per trial / 50 per label",
          len(df) == 30000 and df.trial.nunique() == 200 and set(df.groupby("trial").size()) == {150}
          and set(df.groupby("trial").label.first().value_counts()) == {50})
    man = pd.read_csv(ROOT / "reports" / "split_manifest.csv")
    check("metrics", "split 120/40/40", man.split.value_counts().to_dict() == {"train": 120, "validation": 40, "test": 40})
    preds = pd.read_csv(ROOT / "results" / "followup" / "predictions_all_models.csv")
    m2 = json.loads((ROOT / "reports" / "followup" / "followup_metrics.json").read_text())
    for model in preds.model.unique():
        for sp in ("validation", "test"):
            acc = preds[(preds.model == model) & (preds.split == sp)].correct.mean()
            fm = next(m for m in F["models"] if m["id"] == model)
            check("metrics", f"{model} {sp} accuracy {acc:.3f} matches saved metrics and report facts",
                  abs(acc - m2["evaluations"][model][sp]["accuracy"]) < 1e-12 and abs(acc - fm[f"{sp}_acc"]) < 1e-12)
    cp = pd.read_csv(ROOT / "results" / "c_port" / "predictions_python_vs_c.csv")
    vs = json.loads((ROOT / "results" / "c_port" / "verification_summary.json").read_text())
    nf = int((cp.c_from_features == cp.python_sklearn).sum())
    nr = int((cp.c_from_raw_adc == cp.python_sklearn).sum())
    check("metrics", "C-port agreement re-derived = summary json",
          nf == vs["agreement_features_mode"] == 200 and nr == vs["agreement_raw_adc_mode"] == 200, f"{nf}/200, {nr}/200")
    bt = pd.read_csv(ROOT / "results" / "c_port" / "boundary_tests.csv")
    check("metrics", "boundary tests re-derived = summary json", int(bt.agree.sum()) == vs["boundary_tests"]["agree"] == len(bt),
          f"{int(bt.agree.sum())}/{len(bt)}")
    check("metrics", "compiler warnings = 0 in compile log",
          "warning:" not in (ROOT / "results" / "c_port" / "compile_log.txt").read_text())

    # key values must appear in BOTH the Markdown and the PDF text
    rf = next(m for m in F["models"] if m["id"] == "RF_existing")
    dt = next(m for m in F["models"] if m["id"] == "DT_depth2")
    tokens = {
        "rows 30,000": "30,000", "200/200 agreement": "200/200", "boundary 26/26": "26/26",
        "0 of 70 files changed": "0 of 70", "RF test acc": f"{rf['test_acc']:.3f}", "DT test acc": f"{dt['test_acc']:.3f}",
        "mean-ADC test acc": "0.750", "majority acc": "0.250", "split 120 / 40 / 40": "120",
        "Wilson lower bound": f"{F['wilson_10_10']:.2f}", "RF size estimate": "30,672", "LR size": "1,192",
        "DT tables bytes": "84 bytes", "DS press distribution": "{3: 13, 4: 20, 5: 11, 6: 5, 7: 1}",
        "misclassified trial 39": "trial 39", "flagged NO_PRESS 156": "trial 156", "flagged NO_PRESS 157": "157",
        "class map": "LIGHT_PRESS = 0", "float cast": "(double)(float)v",
    }
    for name, tok in tokens.items():
        check("consistency", f"'{tok}' in Markdown and PDF ({name})", tok in md_n and tok in pdf_text)
    for name, tok in {"exec 200/200": "200/200", "exec RF": f"{rf['test_acc']:.3f}", "exec DT": f"{dt['test_acc']:.3f}",
                      "exec 26/26": "26/26"}.items():
        check("consistency", f"executive summary contains {tok}", tok in ex)

    # ---- 6. claims: offline parity vs real-world accuracy
    required = {
        "no hardware statement": "no ESP32 was connected",
        "not a medical device": "not** a validated medical device",
        "parity is not real-world accuracy": "It does not show real-world accuracy",
        "test previously inspected": "not a completely untouched final evaluation",
        "simulated edits not real": "These are edits, not real experiments",
        "C port not live detector": "not a live gesture detector",
    }
    for name, tok in required.items():
        check("claims", f"report states: {name}", tok in md)
    check("claims", "PDF repeats offline/not-medical cover statement",
          "no ESP32 hardware testing was performed" in pdf_text and "not a validated medical device" in pdf_text)
    risky = [r"(?<!not )(?<!never )\bproves? (?:real-world|clinical)", r"\bvalidated on (?:the )?ESP32\b", r"\bclinically (?:validated|proven)\b",
             r"\breal-world accuracy (?:is|of) \d", r"\bguarantee"]
    hits = [(p, m.group(0)) for p in risky for m in re.finditer(p, md + "\n" + ex, flags=re.I)]
    check("claims", "no over-claiming phrases (real-world proof, ESP32-validated, clinical)", not hits, hits)
    sent = [s for s in re.split(r"(?<=[.!?])\s+", md_n + " " + ex) if "medical device" in s]
    check("claims", "every 'medical device' mention is negated", all(re.search(r"\bnot\b", s) for s in sent),
          f"{len(sent)} mentions")
    status_tbl = md.split("## 15. Current status")[1].split("## 16")[0]
    for comp in ("ESP32 target compilation", "Live sensor inference"):
        line = next(l for l in status_tbl.splitlines() if l.startswith(f"| {comp}"))
        check("claims", f"status table marks '{comp}' as not verified", "Not verified" in line, line.strip())

    # ---- 7. commands (in a temporary copy; non-destructive)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "GripSense-TinyML copy"
        shutil.copytree(ROOT, tmp, ignore=shutil.ignore_patterns(".venv", "__pycache__", "project_documentation"))
        r1 = subprocess.run([PY, "export_and_verify_c_port.py"], cwd=tmp, capture_output=True, text=True)
        check("commands", "export_and_verify_c_port.py (temp copy) exit 0 and ALL CHECKS PASS",
              r1.returncode == 0 and "ALL CHECKS PASS: True" in r1.stdout,
              " | ".join(l for l in r1.stdout.splitlines() if "agreement" in l or "ALL" in l))
        r2 = subprocess.run([PY, "validate_followup.py"], cwd=tmp, capture_output=True, text=True)
        last = (r2.stdout.strip().splitlines() or [""])[-1]
        check("commands", "validate_followup.py (temp copy) exit 0", r2.returncode == 0, last)
        p1 = (tmp / "reports" / "followup" / "phase1_validator_output.txt").read_text().strip().splitlines()[-1]
        check("commands", "phase-1 validate_outputs.py (run by validate_followup) passes", "91/91" in p1, p1)
    r3 = subprocess.run([str(ROOT / "models" / "c_port" / "build" / "decision_tree"), "--info"], cwd=ROOT,
                        capture_output=True, text=True)
    check("commands", "./models/c_port/build/decision_tree --info (in place, read-only) exit 0",
          r3.returncode == 0 and "label 0 LIGHT_PRESS" in r3.stdout, r3.stdout.splitlines()[0] if r3.stdout else "")

    # ---- 8. preservation
    base = json.loads((DOC / "preservation_baseline.json").read_text())
    changed = [f for f, h in base["files"].items()
               if not (ROOT / f).exists() or hashlib.sha256((ROOT / f).read_bytes()).hexdigest() != h]
    check("preservation", f"all {len(base['files'])} pre-existing files unchanged", not changed, changed)
    outside = [p for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and "__pycache__" not in p.parts
               and "project_documentation" not in p.parts and str(p.relative_to(ROOT)) not in base["files"]]
    finder = [str(p.relative_to(ROOT)) for p in outside if p.name == ".DS_Store"]
    other = [str(p.relative_to(ROOT)) for p in outside if p.name != ".DS_Store"]
    check("preservation", "no new files outside results/project_documentation/ (macOS Finder .DS_Store excluded)",
          not other, f"other: {other}; Finder metadata ignored: {finder}")

    # ---- manifest + QC report
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    desc = {
        "GripSense_Project_Report.md": ("markdown", "Complete technical report"),
        "GripSense_Project_Report.pdf": ("pdf", "Complete technical report with cover, contents, page numbers, figures"),
        "GripSense_Executive_Summary.md": ("markdown", "Short executive summary"),
        "README.md": ("markdown", "Entry point linking all deliverables"),
        "QUALITY_CHECKS.md": ("markdown", "Quality-control results"),
        "manifest.json": ("json", "This manifest"),
        "report_facts.json": ("json", "All values used in the report, re-derived from artifacts"),
        "preservation_baseline.json": ("json", "SHA-256 of pre-existing project files before documentation work"),
        "figures/GripSense_System_Architecture.png": ("png", "System architecture and methodology diagram"),
        "figures/GripSense_Model_Comparison.png": ("png", "Validation vs test accuracy, 8 models, Wilson intervals"),
        "figures/GripSense_Model_Size.png": ("png", "Approximate model size with measurement method"),
        "figures/GripSense_C_Port_Verification.png": ("png", "Offline C-port parity verification diagram"),
        "figures/GripSense_Confusion_Matrix.png": ("png", "Test-split confusion matrices (RF, depth-2 tree)"),
        "figures/GripSense_Test_Recall_Uncertainty.png": ("png", "Per-class test recall with uncertainty"),
        "figures/GripSense_DoubleSqueeze_Presses.png": ("png", "Press counts in DOUBLE_SQUEEZE trials"),
        "scripts/build_documentation.py": ("python", "Builds figures, Markdown, README and PDF"),
        "scripts/check_documentation.py": ("python", "Runs these checks; writes manifest and QC report"),
        "scripts/manual_figure_inspection.json": ("json", "Record of manual visual inspection of figures"),
    }
    failed_files = {c["check"].split(": ", 1)[-1].split(":")[0] for c in checks if c["status"] == "FAIL"}
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root": "results/project_documentation", "checks_passed": len(checks) - n_fail,
                "checks_failed": n_fail, "files": []}
    for rel, (typ, d) in desc.items():
        p = DOC / rel
        exists = p.exists() and p.stat().st_size > 0
        status = ("generated_and_checked" if exists and not any(rel.endswith(f) or f.endswith(rel) for f in failed_files)
                  else ("generated_with_failed_checks" if exists else "missing"))
        manifest["files"].append({"filename": p.name, "relative_path": rel, "type": typ, "description": d,
                                  "status": status, "bytes": p.stat().st_size if p.exists() else None})
    (DOC / "manifest.json").write_text(json.dumps(manifest, indent=2))

    groups = {}
    for c in checks:
        groups.setdefault(c["group"], []).append(c)
    lines = [f"# Quality checks - GripSense documentation package",
             "", f"_Run {manifest['generated_utc']} by `scripts/check_documentation.py`._", "",
             f"**Result: {len(checks) - n_fail}/{len(checks)} checks passed, {n_fail} failed.**", "",
             "Project scripts that rewrite their own outputs (`export_and_verify_c_port.py`, `validate_followup.py`) "
             "were executed in a temporary copy of the project, so no existing file was modified. "
             "`decision_tree --info` was executed in place (read-only).", ""]
    for g, cs in groups.items():
        lines += [f"## {g}", "", "| Status | Check | Detail |", "|---|---|---|"]
        lines += [f"| {c['status']} | {c['check']} | {c['detail'].replace('|', '/')[:160]} |" for c in cs]
        lines.append("")
    lines += ["## Figure properties", "", "| Figure | Pixels | DPI | Ink fraction |", "|---|---|---|---|"]
    lines += [f"| {f} | {v['px'][0]}x{v['px'][1]} | {v['dpi']} | {v['ink_fraction']} |" for f, v in fig_info.items()]
    lines += ["", "## Limitations of these checks", "",
              "- Visual quality (clipping, overlap, legibility) was judged by a manual look at each PNG and at "
              "rendered PDF pages; automated checks only detect blank or low-resolution images.",
              "- Claim checks are pattern-based (required statements present, risky phrases absent); they "
              "complement but do not replace reading the report.",
              "- The PDF's internal layout was checked via text extraction and image counts, not pixel comparison.",
              "- Nothing here validates GripSense on hardware or in real-world use."]
    (DOC / "QUALITY_CHECKS.md").write_text("\n".join(lines) + "\n")
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
