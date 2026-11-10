# tests/test_core_io.py
"""
Unit tests for core I/O, EXIF metadata parsing, lens fingerprinting, and pre-flight assertions.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

# Add toolkit paths
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/artifacts")

from utils import format_shutter_speed, verify_environment, reconstruct_k_matrix, reconstruct_dist_vector


class TestCoreIO(unittest.TestCase):

    def test_verify_environment(self):
        """Checks that core required dependencies pass verification."""
        try:
            verify_environment()
        except SystemExit:
            self.fail("verify_environment() raised SystemExit unexpectedly.")

    def test_format_shutter_speed_fractions(self):
        """Checks formatting of shutter speeds into clean sub-second strings (1/#s)."""
        self.assertEqual(format_shutter_speed((1, 100)), "1/100s")
        self.assertEqual(format_shutter_speed(0.005), "1/200s")
        self.assertEqual(format_shutter_speed(2.5), "2.5s")
        self.assertEqual(format_shutter_speed(None), "Unknown")

    def test_linear_light_normalization_bounds(self):
        """Checks that linear light normalization maps raw DN values strictly into [0.0, 1.0]."""
        raw_dn = np.array([64.0, 300.0, 500.0, 1023.0], dtype=np.float64)
        black_level = 64.0
        white_level = 1023.0
        
        linear_light = (raw_dn - black_level) / (white_level - black_level)
        linear_light = np.clip(linear_light, 0.0, 1.0)

        self.assertEqual(linear_light[0], 0.0)
        self.assertEqual(linear_light[-1], 1.0)
        self.assertTrue(np.all(linear_light >= 0.0))
        self.assertTrue(np.all(linear_light <= 1.0))

    def test_bayer_block_indexing(self):
        """Checks that 2x2 Bayer block slicing extracts correct single-plane channels."""
        mock_bayer = np.array([
            [10, 20, 10, 20],
            [30, 40, 30, 40],
            [10, 20, 10, 20],
            [30, 40, 30, 40]
        ], dtype=np.float64)

        r_plane = mock_bayer[0::2, 0::2]
        g1_plane = mock_bayer[0::2, 1::2]
        g2_plane = mock_bayer[1::2, 0::2]
        b_plane = mock_bayer[1::2, 1::2]

        self.assertEqual(r_plane.shape, (2, 2))
        np.testing.assert_array_equal(r_plane, 10)
        np.testing.assert_array_equal(g1_plane, 20)
        np.testing.assert_array_equal(g2_plane, 30)
        np.testing.assert_array_equal(b_plane, 40)

    def test_reconstruct_k_and_dist(self):
        """Verifies matrix and vector reconstruction from payload dictionaries."""
        payload = {
            "focal_length_x": [3000.0, 2.0],
            "focal_length_y": [3000.0, 2.0],
            "principal_point_x": [1500.0, 1.5],
            "principal_point_y": [2000.0, 1.5],
            "radial_distortion_k1": [0.02, 0.001],
            "radial_distortion_k2": [-0.05, 0.002],
            "tangential_distortion_p1": [0.001, 0.0001],
            "tangential_distortion_p2": [0.001, 0.0001],
            "radial_distortion_k3": [0.0, 0.0],
        }
        K = reconstruct_k_matrix(payload)
        dist = reconstruct_dist_vector(payload)
        
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(dist.shape, (5,))
        self.assertEqual(K[0, 0], 3000.0)
        self.assertEqual(dist[0], 0.02)


if __name__ == "__main__":
    unittest.main()
