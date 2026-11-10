# tests/test_resolution.py
"""
Unit tests for Stage 2: Spatial resolution, MTF50 interpolation, and diffraction cutoff limits.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/artifacts")

from resolution import (
    extract_linear_green_plane,
    compute_mtf50,
    calculate_diffraction_limits
)


class TestResolution(unittest.TestCase):

    def test_extract_linear_green_plane_normalization(self):
        """Checks green-plane slicing and dynamic range normalization."""
        raw_visible = np.full((4, 4), 64.0, dtype=np.float64)
        raw_visible[0::2, 0::2] = 543.5  # Green channel values halfway up
        black_level, white_level = 64.0, 1023.0

        green_linear = extract_linear_green_plane(raw_visible, black_level, white_level)
        self.assertEqual(green_linear.shape, (2, 2))
        self.assertAlmostEqual(green_linear[0, 0], 0.5, places=3)

    def test_compute_mtf50_synthetic_crossing(self):
        """Checks linear interpolation solver for the 50% modulation threshold (MTF50)."""
        freq = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        mtf = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
        
        # MTF = 0.5 occurs halfway between 0.2 (0.6) and 0.3 (0.4) -> 0.25
        mtf50 = compute_mtf50(freq, mtf)
        self.assertAlmostEqual(mtf50, 0.25, places=4)

    def test_calculate_diffraction_limits(self):
        """Checks optical diffraction cutoff frequency and crossover F-number calculations."""
        res = calculate_diffraction_limits(pixel_pitch_um=1.72, f_number=1.68, wavelength_um=0.55)
        
        self.assertIn("v_c_um", res)
        self.assertIn("v_nyquist_um", res)
        self.assertIn("crossover_f_number", res)
        self.assertGreater(res["v_c_um"], 0.0)
        self.assertGreater(res["crossover_f_number"], 0.0)


if __name__ == "__main__":
    unittest.main()
