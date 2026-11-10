# tests/conftest.py
"""
Shared unittest/pytest configuration and fixtures for camera_characterization tests.
"""
import unittest
import numpy as np


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
