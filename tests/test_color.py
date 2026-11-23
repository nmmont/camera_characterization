"""
Unit tests for Stage 4: Color science, von Kries white balance, CCM fitting, and CIEDE2000.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import color, simulator


class TestColor(unittest.TestCase):

    def test_reference_colorchecker_data(self):
        """Checks 24-patch reference ColorChecker XYZ and patch names loading."""
        ref_xyz, white_xyz = color.get_reference_colorchecker_xyz()
        names = color.get_reference_patch_names()
        self.assertEqual(ref_xyz.shape, (24, 3))
        self.assertEqual(white_xyz.shape, (3,))
        self.assertAlmostEqual(float(white_xyz[1]), 1.0, places=6)
        self.assertEqual(len(names), 24)

    def test_xyz_to_lab_shape(self):
        """Checks XYZ to CIELAB conversion output shape."""
        xyz = np.array([[0.2, 0.2, 0.2]], dtype=np.float64)
        lab = color.xyz_to_lab(xyz)
        self.assertEqual(lab.shape, (1, 3))

    def test_ciede2000_zero_error_identical(self):
        """Checks CIEDE2000 returns zero error for identical colors."""
        lab = np.array([[50.0, 10.0, -5.0]], dtype=np.float64)
        de = color.ciede2000(lab, lab)
        self.assertAlmostEqual(float(de[0]), 0.0, places=4)

    def test_align_patch_means_and_lens_shading(self):
        """Checks 6x4 portrait-to-landscape patch rotation and flat-field division."""
        raw_grid = np.arange(24 * 3, dtype=np.float64).reshape(24, 3)
        aligned = color.align_patch_means(raw_grid)
        self.assertEqual(aligned.shape, (24, 3))

        sample = np.full((100, 100, 3), 0.25, dtype=np.float64)
        white = np.full((100, 100, 3), 0.50, dtype=np.float64)
        corrected = color.apply_lens_shading_correction(sample, white)
        np.testing.assert_allclose(corrected, 0.25, atol=1e-5)

    def test_compute_white_balance_gains(self):
        """Checks von Kries white balance gain computation on neutral Patch 21 (idx 20)."""
        mock_means = np.full((24, 3), 0.5, dtype=np.float64)
        mock_means[20] = np.array([0.4, 0.5, 0.6])
        gains, balanced = color.compute_white_balance_gains(mock_means, neutral_patch_idx=20)

        self.assertAlmostEqual(gains[0], 0.5 / 0.4)
        self.assertAlmostEqual(gains[1], 1.0)
        self.assertAlmostEqual(gains[2], 0.5 / 0.6)
        self.assertAlmostEqual(balanced[20, 0], balanced[20, 1])

    def test_fit_ccm_and_synthetic_recovery(self):
        """Checks CCM least-squares recovery produces ~0.00 CIEDE2000 on noiseless linear data."""
        ref = color.COLORCHECKER_REF_XYZ.copy()
        CCM_id = color.fit_color_correction_matrix(ref, ref)
        self.assertEqual(CCM_id.shape, (3, 3))
        np.testing.assert_allclose(CCM_id, np.eye(3), atol=1e-3)

        sim_rgb, _ = simulator.generate_synthetic_color_patches(
            ref_xyz=ref,
            illuminant_rgb_gains=np.array([0.94, 1.03, 1.12]),
            noise_std=0.0,
            seed=42,
        )
        _, bal_rgb = color.compute_white_balance_gains(sim_rgb, neutral_patch_idx=20)
        CCM = color.fit_color_correction_matrix(bal_rgb, ref)
        corr_xyz = bal_rgb @ CCM.T
        mean_de, _, worst_de, _ = color.evaluate_delta_e(corr_xyz, ref)
        self.assertAlmostEqual(mean_de, 0.0, places=4)
        self.assertAlmostEqual(worst_de, 0.0, places=4)

    def test_process_color_accuracy_pipeline(self):
        """Checks full ColorChecker accuracy pipeline payload structure."""
        sim_rgb, _ = simulator.generate_synthetic_color_patches(
            ref_xyz=color.COLORCHECKER_REF_XYZ,
            noise_std=0.001,
            seed=10,
        )
        res = color.process_color_accuracy_pipeline(sim_rgb)
        self.assertIn("de_mean_before", res)
        self.assertIn("de_mean_after", res)
        self.assertIn("de_worst_after", res)
        self.assertIn("ccm_matrix", res)
        self.assertLess(res["de_mean_after"][0], res["de_mean_before"][0])


if __name__ == "__main__":
    unittest.main()