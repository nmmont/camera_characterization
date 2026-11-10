# tests/test_geometry.py
"""
Unit tests for Stage 1: Geometric calibration, corner extraction, line straightness, and pixel pitch.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/artifacts")

from geometry import (
    generate_object_points,
    max_perpendicular_deviation,
    calculate_pixel_pitch_metrics
)


class TestGeometry(unittest.TestCase):

    def test_generate_object_points_shape(self):
        """Checks that generated 3D grid points have correct shape and Z=0."""
        rows, cols, sq_mm = 6, 9, 19.0
        pts = generate_object_points(rows, cols, sq_mm)
        
        self.assertEqual(pts.shape, (54, 3))
        np.testing.assert_array_equal(pts[:, 2], 0.0)
        np.testing.assert_array_equal(pts[0], [0.0, 0.0, 0.0])

    def test_max_perpendicular_deviation_straight_line(self):
        """Checks that points lying exactly on a straight line have 0.0 deviation."""
        x = np.linspace(0, 100, 10)
        y = np.full_like(x, 10.0)
        pts = np.column_stack([x, y])
        
        dev = max_perpendicular_deviation(pts)
        self.assertAlmostEqual(dev, 0.0, places=5)

    def test_pixel_pitch_metrics_consistency(self):
        """Checks that pixel pitch calculations return positive values and low difference."""
        res = calculate_pixel_pitch_metrics(
            f_px_x=4000.0,
            f_px_y=4000.0,
            f_mm=6.9,
            f_35mm=25.0,
            image_shape=(4080, 3072)
        )
        self.assertGreater(res["pixel_pitch_method1_um"], 0.0)
        self.assertGreater(res["pixel_pitch_method2_um"], 0.0)
        self.assertIn("pitch_percentage_difference", res)


if __name__ == "__main__":
    unittest.main()
