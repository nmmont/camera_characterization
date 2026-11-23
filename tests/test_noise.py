"""
Unit tests for Stage 3: Photon Transfer Curve (PTC), read noise, and dark current fitting.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import noise, simulator


class TestNoise(unittest.TestCase):

    def test_compute_exif_black_level_fallback(self):
        """Checks default black level tuple when no valid DNG paths are given."""
        bl_mean, bl_std = noise.compute_exif_black_level([])
        self.assertEqual(bl_mean, 64.0)
        self.assertEqual(bl_std, 0.0)

    def test_extract_green_plane_cfa_selection(self):
        """Checks CFA-aware green subplane extraction for GBRG layout."""
        bayer = np.zeros((8, 8), dtype=np.float64)
        # In GBRG: (0,0) is G0, (0,1) is B, (1,0) is R, (1,1) is G1
        bayer[0::2, 0::2] = 100.0
        bayer[1::2, 1::2] = 200.0

        g0 = noise.extract_green_plane(bayer, patch_size=8, cfa_pattern="GBRG", green_index=0)
        g1 = noise.extract_green_plane(bayer, patch_size=8, cfa_pattern="GBRG", green_index=1)

        self.assertEqual(g0.shape, (4, 4))
        np.testing.assert_array_equal(g0, 100.0)
        np.testing.assert_array_equal(g1, 200.0)

    def test_fit_ptc_and_read_noise_recovery(self):
        """Checks end-to-end synthetic PTC gain and read noise recovery."""
        true_K = 0.0213
        true_rn = 2.50
        black_lvl = 64.0
        white_lvl = 1023.0

        flat_pairs = simulator.generate_flat_field_dataset(
            true_gain=true_K, true_read_noise=true_rn, black_level=black_lvl
        )
        means, variances = [], []
        for f_a, f_b in flat_pairs:
            m, v, _ = noise.compute_pair_variance_and_mean(f_a, f_b, black_level=black_lvl)
            means.append(m)
            variances.append(v)

        K_fit, _, std_K, valid_mask = noise.fit_photon_transfer_curve(
            means, variances, saturation_fraction=0.70, white_level=white_lvl, black_level=black_lvl
        )
        self.assertAlmostEqual(K_fit, true_K, delta=0.001)
        self.assertGreater(std_K, 0.0)
        self.assertTrue(np.any(valid_mask))

        dark_series = simulator.generate_synthetic_dark_series(
            dark_current_e_per_sec=0.45,
            gain=true_K,
            read_noise_e=true_rn,
            black_level=black_lvl,
        )
        rn_est, std_rn = noise.measure_read_noise_from_dark_frames(
            dark_series, gain_K=K_fit, black_level=black_lvl
        )
        self.assertAlmostEqual(rn_est, true_rn, delta=0.25)
        self.assertGreater(std_rn, 0.0)

    def test_fit_dark_current_recovery(self):
        """Checks thermal dark current slope recovery from multi-exposure dark frames."""
        true_K = 0.0213
        true_dc = 0.50
        black_lvl = 64.0

        dark_series = simulator.generate_synthetic_dark_series(
            dark_current_e_per_sec=true_dc,
            gain=true_K,
            read_noise_e=1.0,
            black_level=black_lvl,
            exposure_times=[0.5, 2.0, 5.0, 10.0],
            frames_per_step=4,
        )
        dc_est, dc_err = noise.fit_dark_current(
            dark_series, gain_K=true_K, black_level=black_lvl
        )
        self.assertAlmostEqual(dc_est, true_dc, delta=0.15)
        self.assertGreaterEqual(dc_err, 0.0)


if __name__ == "__main__":
    unittest.main()