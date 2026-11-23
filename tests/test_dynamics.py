"""
Unit tests for Stage 3/4: Engineering vs. Photographic Dynamic Range and Full-Well Capacity.
"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_toolkit import dynamics


class TestDynamics(unittest.TestCase):

    def test_compute_full_well_capacity(self):
        """Checks photoelectron full well capacity conversion from saturation DN and gain K."""
        fw_e = dynamics.compute_full_well_capacity(
            saturation_dn=1023.0, black_level_dn=64.0, conversion_gain_k=0.02
        )
        self.assertAlmostEqual(fw_e, 47950.0, places=2)

        with self.assertRaises(ValueError):
            dynamics.compute_full_well_capacity(1023.0, 64.0, 0.0)

    def test_calculate_dynamic_range_stops_db_and_uncertainty(self):
        """Checks Engineering Dynamic Range in stops, dB, and propagated uncertainty."""
        full_well_e = 16384.0
        read_noise_e = 4.0
        dr_stops, dr_db, std_dr_stops = dynamics.calculate_dynamic_range(
            full_well_e,
            read_noise_e,
            snr_threshold=1.0,
            std_gain_k=0.0002,
            gain_k=0.02,
            std_read_noise_e=0.2,
        )
        self.assertAlmostEqual(dr_stops, 12.0, places=3)
        self.assertAlmostEqual(dr_db, 20.0 * np.log10(4096.0), places=3)
        self.assertGreater(std_dr_stops, 0.0)

    def test_usable_dynamic_range_narrower_than_engineering(self):
        """Checks that requiring a higher SNR threshold (SNR=4) yields a narrower DR than SNR=1."""
        full_well_e = 9000.0
        read_noise_e = 2.5

        dr_eng_stops, _, _ = dynamics.calculate_dynamic_range(
            full_well_e, read_noise_e, snr_threshold=1.0
        )
        dr_photo_stops, _, _ = dynamics.calculate_dynamic_range(
            full_well_e, read_noise_e, snr_threshold=4.0
        )

        self.assertGreater(dr_eng_stops, dr_photo_stops)


if __name__ == "__main__":
    unittest.main()