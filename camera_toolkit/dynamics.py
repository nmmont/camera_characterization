"""
Dynamic Range and Full Well Capacity calculations for camera characterization.
Computes engineering dynamic range (SNR = 1) and photographic/usable dynamic range.
"""
import numpy as np


def compute_full_well_capacity(
    saturation_dn: float,
    black_level_dn: float,
    conversion_gain_k: float,
) -> float:
    """
    Converts the linear digital saturation span `(D_sat - B)` into photo-electrons:
    `N_sat = (saturation_dn - black_level_dn) / conversion_gain_k`.
    Sensor full-well capacity
    """
    if not np.isfinite(conversion_gain_k) or conversion_gain_k <= 0.0:
        raise ValueError("Conversion gain K must be a positive finite number.")
    return float((float(saturation_dn) - float(black_level_dn)) / float(conversion_gain_k))


def calculate_dynamic_range(
    full_well_e: float,
    read_noise_e: float,
    snr_threshold: float = 1.0,
    std_gain_k: float = None,
    gain_k: float = None,
    std_read_noise_e: float = None,
) -> tuple[float, float, float]:
    """
    1. Computes engineering Dynamic Range at SNR = 1:
       - `DR_stops = log2(N_sat / (snr_threshold * read_noise_e))`
       - `DR_dB    = 20 * log10(N_sat / (snr_threshold * read_noise_e))`
    2. Propagates relative uncertainties from both `K` (which sets `N_sat`) and `read_noise_e`
       in quadrature: `delta_DR_stops = (1 / ln(2)) * sqrt((std_K / K)^2 + (std_rn / rn)^2)`.
    """
    if (
        not np.isfinite(full_well_e)
        or not np.isfinite(read_noise_e)
        or not np.isfinite(snr_threshold)
        or full_well_e <= 0.0
        or read_noise_e <= 0.0
        or snr_threshold <= 0.0
    ):
        nan = float("nan")
        return nan, nan, nan

    noise_floor_e = float(snr_threshold) * float(read_noise_e)
    dr_ratio = float(full_well_e) / noise_floor_e
    if dr_ratio <= 0.0:
        nan = float("nan")
        return nan, nan, nan

    dr_stops = float(np.log2(dr_ratio))
    dr_db = float(20.0 * np.log10(dr_ratio))

    rel_var = 0.0
    if (
        std_read_noise_e is not None
        and np.isfinite(std_read_noise_e)
        and std_read_noise_e >= 0.0
    ):
        rel_var += (float(std_read_noise_e) / float(read_noise_e)) ** 2
    if (
        std_gain_k is not None
        and gain_k is not None
        and np.isfinite(std_gain_k)
        and np.isfinite(gain_k)
        and gain_k > 0.0
        and std_gain_k >= 0.0
    ):
        rel_var += (float(std_gain_k) / float(gain_k)) ** 2

    std_dr_stops = float(np.sqrt(rel_var) / np.log(2.0)) if rel_var > 0.0 else 0.0
    return dr_stops, dr_db, std_dr_stops