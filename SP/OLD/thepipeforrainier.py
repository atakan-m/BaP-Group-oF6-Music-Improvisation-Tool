"""DEPRECATED — renamed to `signalprocessing.py`.

Kept around only because Python may have cached imports of the old name.
The actual module is now at SP/signalprocessing.py — re-export from there.
"""
from signalprocessing import *      # noqa: F401, F403
from signalprocessing import PitchDetector, main, SAMPLE_RATE, HOP_SIZE   # noqa: F401
