"""
GripSense offline test suite (Python vs C parity for the depth-2 decision tree).

    .venv/bin/python -B -m unittest discover -s tests -t . -v

Importing the package imports `_support`, which takes the file-preservation baseline
before any test module is loaded or any test runs.
"""
from . import _support  # noqa: F401  (side effect: preservation baseline)
