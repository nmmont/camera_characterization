"""
Unit tests for core I/O, EXIF metadata parsing, lens fingerprinting, and JSON export.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from unittest import mock
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import utils


class TestCoreIO(unittest.TestCase):

    def test_verify_environment(self):
        """Checks that Python >= 3.12, OpenCV >= 5.0, and core dependencies pass verification."""
        self.assertTrue(utils.verify_environment())
        self.assertGreaterEqual(sys.version_info[:2], (3, 12))
        cv2_major = int(utils.cv2.__version__.split(".")[0])
        self.assertGreaterEqual(cv2_major, 5)

    def test_verify_environment_rejects_old_versions(self):
        """Checks that verify_environment raises RuntimeError for Python < 3.12 or OpenCV < 5."""
        with mock.patch.object(utils.cv2, "__version__", "4.10.0"):
            with self.assertRaises(RuntimeError):
                utils.verify_environment()

        fake_py = types.SimpleNamespace(major=3, minor=11, micro=9)
        with mock.patch.object(utils.sys, "version_info", fake_py):
            with self.assertRaises(RuntimeError):
                utils.verify_environment()
    
    def test_format_shutter_speed_fractions(self):
        """Checks formatting of shutter speeds into clean sub-second strings (1/#s)."""
        self.assertEqual(utils.format_shutter_speed((1, 100)), "1/100s")
        self.assertEqual(utils.format_shutter_speed(0.005), "1/200s")
        self.assertEqual(utils.format_shutter_speed(2.5), "2.5s")
        self.assertEqual(utils.format_shutter_speed(None), "Unknown")

    def test_reconstruct_k_and_dist(self):
        """Verifies matrix and vector reconstruction from payload dictionaries."""
        payload = {
            "focal_length_x": [3048.03, 6.54],
            "focal_length_y": [3043.67, 6.52],
            "principal_point_x": [1509.05, 2.81],
            "principal_point_y": [2065.80, 2.68],
            "radial_distortion_k1": [0.0418, 0.0017],
            "radial_distortion_k2": [-0.1081, 0.0056],
            "tangential_distortion_p1": [0.0012, 0.0002],
            "tangential_distortion_p2": [0.0012, 0.0002],
            "radial_distortion_k3": [0.1003, 0.0062],
        }
        K = utils.reconstruct_k_matrix(payload)
        dist = utils.reconstruct_dist_vector(payload)

        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(dist.shape, (5,))
        self.assertAlmostEqual(float(K[0, 0]), 3048.03, places=2)
        self.assertAlmostEqual(float(K[1, 1]), 3043.67, places=2)
        self.assertAlmostEqual(float(K[0, 2]), 1509.05, places=2)
        self.assertAlmostEqual(float(K[1, 2]), 2065.80, places=2)
        self.assertAlmostEqual(float(dist[0]), 0.0418, places=4)
        self.assertAlmostEqual(float(dist[4]), 0.1003, places=4)

    def test_extract_dng_metadata_nonexistent(self):
        """Checks graceful handling of missing file metadata without crashes."""
        meta = utils.extract_dng_metadata("nonexistent.dng")
        self.assertEqual(meta["black_level"], 0.0)
        self.assertEqual(meta["white_level"], 0.0)
        self.assertEqual(meta["focal_35mm"], 25.0)
        self.assertEqual(meta["cfa_pattern"], "Not detected.")

    def test_get_exposure_time_and_cfa_missing_file(self):
        """Checks safe fallbacks when reading nonexistent paths."""
        self.assertIsNone(utils.get_exposure_time("nonexistent.dng"))
        self.assertEqual(utils.extract_cfa_pattern("nonexistent.dng"), "Not detected.")
        self.assertEqual(utils.get_dng_files("nonexistent_dir_xyz"), [])

    def test_export_project1_results_json(self):
        """Checks schema formatting and file writing in export_project1_results_json."""
        exif_meta = {
            "lens_model": "Google Pixel 9",
            "capture_app": "Open Camera (Camera2 API)",
            "cfa_pattern": "GBRG",
            "width": 4080,
            "height": 3072,
            "binned": True,
            "black_level": 63.98,
            "black_level_std": 0.01,
        }
        geom_results = {
            "f_px": [3045.85, 4.62],
            "f_mm": [6.9, 0.1],
            "cx": [1509.05, 2.81],
            "cy": [2065.80, 2.68],
            "k1": [0.0418, 0.0017],
            "k2": [-0.1081, 0.0056],
            "p1": [0.0012, 0.0002],
            "p2": [0.0012, 0.0002],
            "rms": [0.9595, 0.0],
            "n_views": 68,
            "sq_mm": [19.0, 0.5],
        }
        res_results = {
            "mtf50_center": [0.3239, 0.0631],
            "mtf50_corner_mean": [0.2381, 0.0761],
            "mtf50_corner_worst": [0.1804, 0.0301],
        }
        noise_results = {
            "K": [0.0148, 3.69e-5],
            "read_noise": [57.76, 2.89],
            "dark_current": [16.46, 0.31],
            "iso": 22,
            "black_level": [63.9841, 0.0138],
            "white_level": 1023,
            "vendor_noiseprofile_present": True,
            "dr_stops": [10.14, 0.07],
            "dr_criterion": "SNR = 1 at low end",
        }
        color_results = {
            "de_mean_before": [12.34, 4.72],
            "de_mean_after": [3.53, 0.06],
            "de_worst_after": [11.19, 0.13],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = utils.export_project1_results_json(
                exif_meta=exif_meta,
                geom_results=geom_results,
                res_results=res_results,
                noise_results=noise_results,
                color_results=color_results,
                output_dir=tmpdir,
            )
            out_file = Path(tmpdir) / "results.json"
            self.assertTrue(out_file.exists())

            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["camera"], "Google Pixel 9")
            self.assertEqual(loaded["n_calibration_images_used"], 68)
            self.assertEqual(loaded["iso_of_noise_measurement"], 22)
            self.assertEqual(payload["delta_e_metric"], "CIEDE2000")


if __name__ == "__main__":
    unittest.main()