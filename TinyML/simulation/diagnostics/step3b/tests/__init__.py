"""
Step 3B tests.

    .venv/bin/python -B -m unittest discover -s simulation/diagnostics/step3b/tests -t . -v

Importing this package hashes every project file outside `.venv/`, `simulation/diagnostics/step3b/`
and `.DS_Store` files. `load_tests` runs TestPreservation after all other Step 3B tests.
"""
from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXCLUDED_PREFIXES = (".venv/", "simulation/diagnostics/step3b/")


def snapshot() -> dict[str, str]:
    out = {}
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(EXCLUDED_PREFIXES) or rel == ".venv" or p.name == ".DS_Store" or not p.is_file():
            continue
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
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
    return unittest.TestSuite([t for t in tests if t not in last] + last)
