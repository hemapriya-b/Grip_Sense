"""
GripSense software-only live-input simulator (Step 3).

Replays recorded CSV trials as a simulated ADC stream through a 150-sample window buffer,
classifies each window with the verified C decision tree (via ctypes, built in a temporary
directory), and turns per-window labels into deduplicated events.

Simulation only: no hardware, no firmware, no new data. See simulation/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
