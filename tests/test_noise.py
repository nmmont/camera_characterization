# tests/test_noise.py
"""
Unit tests for Stage 3: Photon Transfer Curve (PTC) line fitting, read noise, and vendor profiles.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/artifacts")

from noise import (
    fit_photon_transfer_curve,
    measure_read_noise_from_dark_frames,
    calculate_dynamic_range
)


class TestNoise(unittest.TestCase):

    def test_fit_ptc_gain_recovery(self):
        """Checks linear regression Var[D] = K*(E[D]-B) + c below 70% saturation."""
        true_K = 0.020  # DN / electron
        true_intercept = 0.05  # K^2 * sigma_r^2
        
        means = np.linspace(10, 400, 10)
        variances = true_K * means + true_intercept
        
        gain_K, intercept_c, std_K = fit_photon_transfer_curve(
            means_above_black=means,
            variances_dn2=variances,
            saturation_fraction=0.7,
            white_level=1023.0,
            black_level=64.0
        )
        
        self.assertAlmostEqual(gain_K, true_K, places=4)
        self.assertAlmostEqual(intercept_c, true_intercept, places=4)

    def test_read_noise_non_negative(self):
        """Checks that read noise calculation avoids taking square root of negative values."""
        read_noise_e, std_e = measure_read_noise_from_dark_frames(
            dark_path_a="dummy_a.dng",
            dark_path_b="dummy_b.dng",
            gain_K=0.020
        )
        self.assertGreaterEqual(read_noise_e, 0.0)

    def test_calculate_dynamic_range(self):
        """Checks dynamic range stops calculation: log2(full_well_e / read_noise_e)."""
        full_well_e = 9000.0
        read_noise_e = 2.5
        dr_stops, std_dr = calculate_dynamic_range(full_well_e, read_noise_e)
        self.assertAlmostEqual(dr_stops, np.log2(3600.0), places=3)


if __name__ == "__main__":
    unittest.main()
