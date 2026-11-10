# tests/test_dynamics.py
"""
Unit tests for Stage 4: Engineering vs. Photographic Dynamic Range and Full-Well Capacity.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/artifacts")

from dynamics import (
    compute_full_well_capacity,
    calculate_dynamic_range
)


class TestDynamics(unittest.TestCase):

    def test_compute_full_well_capacity(self):
        """Checks photoelectron full well capacity conversion from saturation DN and gain K."""
        fw_e = compute_full_well_capacity(saturation_dn=1023.0, black_level_dn=64.0, conversion_gain_k=0.02)
        self.assertAlmostEqual(fw_e, 47950.0, places=2)

    def test_calculate_dynamic_range_stops_and_db(self):
        """Checks Engineering Dynamic Range in stops and dB at SNR=1 threshold."""
        full_well_e = 16384.0
        read_noise_e = 4.0
        dr_stops, dr_db = calculate_dynamic_range(full_well_e, read_noise_e, snr_threshold=1.0)
        self.assertAlmostEqual(dr_stops, 12.0, places=3)
        self.assertAlmostEqual(dr_db, 20.0 * np.log10(4096.0), places=3)

    def test_usable_dynamic_range_narrower_than_engineering(self):
        """Checks that requiring a higher SNR threshold (SNR=4) yields a narrower DR than SNR=1."""
        full_well_e = 9000.0
        read_noise_e = 2.5
        
        dr_eng_stops, _ = calculate_dynamic_range(full_well_e, read_noise_e, snr_threshold=1.0)
        dr_photo_stops, _ = calculate_dynamic_range(full_well_e, read_noise_e, snr_threshold=4.0)
        
        self.assertGreater(dr_eng_stops, dr_photo_stops)


if __name__ == "__main__":
    unittest.main()
