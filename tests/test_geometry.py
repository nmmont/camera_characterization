"""
Unit tests for Stage 1: Geometric calibration, corner detection, line straightness, and pixel pitch.
"""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import geometry, simulator, utils


class TestGeometry(unittest.TestCase):

    def test_generate_object_points_shape(self):
        """Checks object points shape and initial zero Z coordinates."""
        obj_p = geometry.generate_object_points(rows=6, cols=9, square_size_mm=19.0)
        self.assertEqual(obj_p.shape, (54, 3))
        np.testing.assert_array_equal(obj_p[:, 2], 0.0)
        self.assertAlmostEqual(float(obj_p[1, 0]), 19.0)

    def test_detect_checkerboard_corners_synthetic(self):
        """Checks sub-pixel corner detection on a clean synthetic checkerboard image."""
        rows, cols, sq_px = 6, 9, 40
        board_h = (rows + 2) * sq_px
        board_w = (cols + 2) * sq_px
        img = np.full((board_h, board_w), 230, dtype=np.uint8)

        for r in range(rows + 1):
            for c in range(cols + 1):
                if (r + c) % 2 == 0:
                    y0 = (r + 1) * sq_px
                    x0 = (c + 1) * sq_px
                    img[y0 : y0 + sq_px, x0 : x0 + sq_px] = 25

        success, corners = geometry.detect_checkerboard_corners(img, rows=rows, cols=cols)
        self.assertTrue(success)
        self.assertEqual(corners.shape, (rows * cols, 2))

    def test_run_extended_calibration_synthetic_session(self):
        """Checks Zhang extended calibration parameter recovery on synthetic multi-view data."""
        syn_obj, syn_img, img_shape, K_true, _ = simulator.generate_synthetic_checkerboard_session(
            n_views=12, image_shape=(4080, 3072), noise_std_px=0.15, seed=42
        )
        calib = geometry.run_extended_calibration(syn_obj, syn_img, img_shape)

        self.assertIn("focal_length_x", calib)
        self.assertIn("radial_distortion_k1", calib)
        self.assertIn("per_view_rms_errors", calib)
        self.assertEqual(calib["n_views"], 12)
        self.assertAlmostEqual(calib["focal_length_x"][0], K_true[0, 0], delta=30.0)
        self.assertAlmostEqual(calib["focal_length_y"][0], K_true[1, 1], delta=30.0)
        self.assertLess(calib["rms_reprojection_error"][0], 1.0)

    def test_max_perpendicular_deviation_straight_line(self):
        """Checks that points strictly along a line yield 0.0 perpendicular deviation."""
        x = np.linspace(0, 100, 10)
        y = 2.0 * x + 5.0
        pts = np.column_stack([x, y])
        dev = geometry.max_perpendicular_deviation(pts)
        self.assertAlmostEqual(dev, 0.0, places=5)

    def test_evaluate_straightness_improvement(self):
        """Checks that undistortion reduces straightness deviation on radially distorted points."""
        syn_obj, syn_img, _, K_true, dist_true = simulator.generate_synthetic_checkerboard_session(
            n_views=1, image_shape=(4080, 3072), noise_std_px=0.0, seed=7
        )
        dev_before, dev_after = geometry.evaluate_straightness_improvement(
            syn_img, K_true, dist_true, target_row_idx=0, cols_per_row=9
        )
        self.assertLessEqual(dev_after, dev_before + 1e-4)

    def test_rectify_frame_preserves_shape(self):
        """Checks frame rectification output dimensions with alpha=1.0."""
        dummy = np.zeros((300, 400, 3), dtype=np.uint8)
        K = np.array([[300.0, 0.0, 200.0], [0.0, 300.0, 150.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([0.04, -0.1, 0.0, 0.0, 0.0], dtype=np.float64)
        rect = geometry.rectify_frame(dummy, K, dist, alpha=1.0)
        self.assertEqual(rect.shape, dummy.shape)

    def test_pixel_pitch_metrics_consistency(self):
        """Checks pixel pitch cross-validation metrics calculation."""
        mean_f_px = (3048.03 + 3043.67) / 2.0
        metrics = geometry.calculate_pixel_pitch_metrics(
            mean_focal_length_px=mean_f_px,
            f_mm=6.9,
            f_35mm=25.0,
            image_shape=(4080, 3072),
        )
        self.assertIn("pixel_pitch_method1_um", metrics)
        self.assertIn("pixel_pitch_method2_um", metrics)
        self.assertIn("pitch_percentage_difference", metrics)
        self.assertAlmostEqual(metrics["pixel_pitch_method1_um"], 2.2654, places=3)
        self.assertLess(metrics["pitch_percentage_difference"], 5.0)


if __name__ == "__main__":
    unittest.main()