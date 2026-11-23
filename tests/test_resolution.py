"""
Unit tests for Stage 2: Slanted-edge MTF50, ESF/LSF, QC checks, and diffraction limits.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import resolution, simulator


class TestResolution(unittest.TestCase):

    def test_extract_green_roi_and_normalize_view(self):
        """Checks green-plane subsampling and linear light normalization."""
        raw_bayer = np.full((3072, 4080), 64.0, dtype=np.float64)
        # Set GBRG green positions (0,1) and (1,0) to 543.5 DN (midpoint of [64, 1023])
        raw_bayer[0::2, 1::2] = 543.5
        raw_bayer[1::2, 0::2] = 543.5

        roi_coords = {"x": 500, "y": 500, "w": 60, "h": 120}
        green_roi = resolution.extract_green_roi(
            raw_visible_array=raw_bayer,
            roi_coords=roi_coords,
            preview_scale=2.0,
            row_offset=0,
            col_offset=1,
            rotate_k=-1,
        )
        self.assertEqual(green_roi.ndim, 2)
        self.assertGreater(green_roi.size, 0)

        norm_roi = resolution.normalize_view(green_roi, black_level=64.0, white_level=1023.0)
        self.assertAlmostEqual(float(np.mean(norm_roi)), 0.5, places=4)

    def test_slanted_edge_mtf_and_qc_synthetic(self):
        """Checks QC helpers and slanted-edge MTF50 recovery against analytic Gaussian truth."""
        sigma = 1.0
        edge = simulator.capture_edge(sigma=sigma, angle_deg=5.0, h=120, w=120, seed=42)

        clip_res = resolution.check_green_roi_clipping(
            edge, black_level=0.0, white_level=1.0, sat_margin=0.01, label="Test"
        )
        self.assertTrue(clip_res["is_clean"])

        margin_res = resolution.check_green_roi_edge_margins(
            edge, min_margin_px=20.0, label="Test", show_plot=False
        )
        self.assertTrue(margin_res["edge_ok"])

        angle_res = resolution.measure_slant_edge_angle(edge, label="Test")
        self.assertAlmostEqual(angle_res["angle_from_vertical_deg"], 5.0, delta=0.5)

        freq, mtf, esf_x, esf = resolution.slanted_edge_mtf(edge)
        mtf50_meas = resolution.compute_mtf50(freq, mtf)
        mtf50_truth = resolution.analytic_mtf50(sigma)

        self.assertGreater(len(freq), 0)
        self.assertEqual(len(esf_x), len(esf))
        self.assertAlmostEqual(mtf50_meas, mtf50_truth, delta=0.015)

    def test_compute_mtf50_synthetic_crossing(self):
        """Checks linear interpolation solver for the 50% modulation threshold (MTF50)."""
        freq = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        mtf = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=np.float64)
        mtf50 = resolution.compute_mtf50(freq, mtf)
        self.assertAlmostEqual(mtf50, 0.25, places=4)

    def test_summarize_corner_mtf50(self):
        """Checks uniformity, repeatability, and pooled std calculations across corners."""
        mock_regions = {
            "top_left": {"frames": [{"mtf50_px": 0.18}, {"mtf50_px": 0.20}]},
            "top_right": {"frames": [{"mtf50_px": 0.32}, {"mtf50_px": 0.34}]},
            "bottom_left": {"frames": [{"mtf50_px": 0.20}, {"mtf50_px": 0.22}]},
            "bottom_right": {"frames": [{"mtf50_px": 0.22}, {"mtf50_px": 0.24}]},
        }
        summary = resolution.summarize_corner_mtf50(mock_regions, metric_key="mtf50_px")
        self.assertEqual(summary["corner_worst_region"], "top_left")
        self.assertAlmostEqual(summary["corner_worst_mean"], 0.19, places=4)
        self.assertEqual(summary["n_corners"], 4)
        self.assertEqual(summary["n_total_frames"], 8)
        self.assertGreater(summary["total_pooled_std"], 0.0)

    def test_calculate_diffraction_limits(self):
        """Checks optical diffraction cutoff frequency and crossover F-number calculations."""
        res = resolution.calculate_diffraction_limits(
            pixel_pitch_um=2.2654, f_number=1.68, wavelength_um=0.55
        )
        self.assertIn("v_c_um", res)
        self.assertIn("v_nyquist_um", res)
        self.assertIn("crossover_f_number", res)
        self.assertIn("modulation_at_nyquist", res)
        self.assertGreater(res["v_c_um"], 0.0)
        self.assertAlmostEqual(res["crossover_f_number"], (2.0 * 2.2654) / 0.55, places=3)


if __name__ == "__main__":
    unittest.main()