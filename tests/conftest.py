"""
Shared unittest/pytest configuration and fixtures for camera_characterization tests.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure both repo root and src/ are importable regardless of test runner invocation
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class BaseTestCase(unittest.TestCase):
    """Base test case providing shared synthetic image and calibration fixtures."""

    def setUp(self):
        self.img_shape = (4080, 3072)
        self.f_mm = 6.9
        self.f_35mm = 25.0
        self.f_aperture = 1.68
        self.black_level = 64.0
        self.white_level = 1023.0

        # Synthetic flat patch (128x128)
        self.synthetic_patch = np.full((128, 128), 500.0, dtype=np.float64)