"""
Simulator tests (Step 3).

    .venv/bin/python -B -m unittest discover -s simulation/tests -t . -v

Importing this package takes a hash baseline of every project file outside `.venv/`, `simulation/`
and `.DS_Store` files (so the Step-2 `tests/` folder is covered too). `load_tests` runs the
preservation check (test_simulator_e2e.TestPreservation) after every other simulator test.
"""
from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_TOP_DIRS = {".venv", "simulation"}


def snapshot() -> dict[str, str]:
    out = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in EXCLUDED_TOP_DIRS or p.name == ".DS_Store" or not p.is_file():
            continue
        out[rel.as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


BASELINE = snapshot()


def _flatten(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _flatten(t)
        else:
            yield t


def load_tests(loader, standard_tests, pattern):
    found = loader.discover(start_dir=os.path.dirname(__file__), pattern=pattern or "test*.py",
                            top_level_dir=str(ROOT))
    tests = list(_flatten(standard_tests)) + list(_flatten(found))
    last = [t for t in tests if type(t).__name__ == "TestPreservation"]
    first = [t for t in tests if type(t).__name__ != "TestPreservation"]
    return unittest.TestSuite(first + last)
