# src/camera_toolkit/dynamics.py
"""
Dynamic Range and Full Well Capacity calculations for camera characterization.
Computes engineering dynamic range (SNR = 1) and photographic/usable dynamic range.
"""
import numpy as np


def compute_full_well_capacity(saturation_dn: float, black_level_dn: float, conversion_gain_k: float) -> float:
    """
    Calculates sensor full-well capacity in photo-electrons.
    N_sat = (D_sat - B) / K
    """
    if conversion_gain_k <= 0:
        raise ValueError("Conversion gain K must be positive.")
    return float((saturation_dn - black_level_dn) / conversion_gain_k)


def calculate_dynamic_range(full_well_e: float, read_noise_e: float, snr_threshold: float = 1.0) -> tuple:
    """
    Computes dynamic range in stops and decibels (dB) for a given low-end SNR threshold.
    Engineering DR uses SNR = 1; Usable photographic DR uses SNR = 4 or 10.
    
    Returns:
        tuple: (dr_stops, dr_db)
    """
    if read_noise_e <= 0 or full_well_e <= 0:
        return 0.0, 0.0

    noise_floor_e = snr_threshold * read_noise_e
    dr_ratio = full_well_e / noise_floor_e
    dr_stops = float(np.log2(dr_ratio))
    dr_db = float(20.0 * np.log10(dr_ratio))
    return dr_stops, dr_db
