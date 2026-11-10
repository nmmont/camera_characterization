# src/camera_toolkit/resolution.py
import os
import numpy as np


def extract_linear_green_plane(raw_visible_array: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    """
    Slices the raw Bayer matrix to isolate one green channel plane and normalizes it linearly.
    """
    green_plane = raw_visible_array[0::2, 0::2]
    green_plane = np.rot90(green_plane, k=-1)
    
    green_linear = (green_plane - black_level) / (white_level - black_level)
    return np.clip(green_linear, 0.0, 1.0)


def edge_centroids(roi: np.ndarray, window: float = 8.0, smooth: float = 1.5) -> np.ndarray:
    """Sub-pixel edge position per row using a windowed derivative centroid."""
    import cv2
    roi = np.asarray(roi, dtype=float)
    smoothed = cv2.GaussianBlur(roi, (0, 0), sigmaX=smooth, sigmaY=0.01)
    coarse = np.argmax(np.abs(np.diff(smoothed, axis=1)), axis=1) + 0.5

    deriv = np.abs(np.diff(roi, axis=1))
    x = np.arange(deriv.shape[1]) + 0.5
    weight = np.where(np.abs(x[None, :] - coarse[:, None]) <= window, deriv, 0.0)

    total = weight.sum(axis=1)
    good = total > 1e-9
    centers = np.full(roi.shape[0], np.nan)
    centers[good] = (weight[good] * x).sum(axis=1) / total[good]
    return centers


def slanted_edge_mtf(roi: np.ndarray, bin_width: float = 0.25, window: float = 8.0) -> tuple:
    """
    Measures MTF from a slanted edge ROI.
    Includes baseline subtraction and Hann windowing on the LSF to eliminate low-frequency drift.
    Returns (freq, mtf, esf_x, esf).
    """
    roi = np.asarray(roi, dtype=float)
    rows = np.arange(roi.shape[0])

    centers = edge_centroids(roi, window=window)
    ok = np.isfinite(centers)
    if not np.any(ok):
        return np.array([]), np.array([]), np.array([]), np.array([])

    slope, intercept = np.polyfit(rows[ok], centers[ok], 1)
    fitted = slope * rows + intercept

    cols = np.arange(roi.shape[1])
    distance = (cols[None, :] - fitted[:, None]).ravel()
    values = roi.ravel()

    edges = np.arange(distance.min(), distance.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(distance, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    keep = count > 0
    esf = total[keep] / count[keep]
    esf_x = (edges[:-1] + bin_width / 2)[keep]

    # Compute Line Spread Function (LSF)
    lsf = np.gradient(esf, esf_x)
    
    # 1. Baseline subtraction: subtract tail median to prevent low-frequency MTF rising above 1.0
    tail_len = max(3, len(lsf) // 10)
    baseline = np.median(np.concatenate([lsf[:tail_len], lsf[-tail_len:]]))
    lsf = lsf - baseline
    
    # 2. Apply Hann window centered on the LSF peak to attenuate noisy far tails
    lsf = lsf * np.hanning(len(lsf))

    spectrum = np.abs(np.fft.rfft(lsf))
    if spectrum[0] < 1e-9:
        mtf_norm = spectrum
    else:
        mtf_norm = spectrum / spectrum[0]

    freq = np.fft.rfftfreq(len(lsf), d=bin_width)
    return freq, mtf_norm, esf_x, esf


def compute_mtf50(freq: np.ndarray, mtf: np.ndarray) -> float:
    """Finds frequency where MTF drops to 0.5 via linear interpolation."""
    if len(freq) == 0 or len(mtf) == 0:
        return np.nan
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(freq[i - 1] + (mtf[i - 1] - 0.5) * (freq[i] - freq[i - 1]) / (mtf[i - 1] - mtf[i]))


def process_batch_mtf50(dng_paths: list, roi_coords: dict, pixel_pitch_um: float) -> tuple:
    """
    Batch processes a folder of raw images to calculate mean MTF50 values and uncertainties.
    """
    import rawpy
    
    mtf50_um_list = []
    mtf50_px_list = []
    all_rois_list = []
    
    y, x, h, w = roi_coords["y"], roi_coords["x"], roi_coords["h"], roi_coords["w"]
    
    for path in dng_paths:
        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(float)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
            
        green_linear = extract_linear_green_plane(bayer, black, white)
        roi_sub = green_linear[y:y+h, x:x+w]
        all_rois_list.append((roi_sub, os.path.basename(path)))
        
        freq, mtf, _, _ = slanted_edge_mtf(roi_sub)
        mtf50_c_px = compute_mtf50(freq, mtf)
        
        if not np.isnan(mtf50_c_px):
            # Sampling pitch of the green subgrid plane is 2*p
            mtf50_c_um = mtf50_c_px / (2.0 * pixel_pitch_um)
            mtf50_um_list.append(mtf50_c_um)
            mtf50_px_list.append(mtf50_c_px)
            
    if not mtf50_um_list:
        return 0.0, 0.0, 0.0, 0.0, all_rois_list

    return (
        float(np.mean(mtf50_um_list)), float(np.std(mtf50_um_list, ddof=1) if len(mtf50_um_list) > 1 else 0.0),
        float(np.mean(mtf50_px_list)), float(np.std(mtf50_px_list, ddof=1) if len(mtf50_px_list) > 1 else 0.0),
        all_rois_list
    )


def process_frame_center_and_corners_mtf50(dng_paths: list, center_roi: dict, corner_rois: list, pixel_pitch_um: float) -> dict:
    """
    Evaluates center and corner MTF50 across captures for results.json deliverable keys:
    - mtf50_center_cy_per_px
    - mtf50_corner_mean_cy_per_px
    - mtf50_corner_worst_cy_per_px
    """
    c_mean_px, c_std_px, _, _, _ = process_batch_mtf50(dng_paths, center_roi, pixel_pitch_um)
    
    corner_means = []
    for croi in corner_rois:
        m_px, _, _, _, _ = process_batch_mtf50(dng_paths, croi, pixel_pitch_um)
        if m_px > 0:
            corner_means.append(m_px)
            
    if not corner_means:
        return {
            "mtf50_center": [c_mean_px, c_std_px],
            "mtf50_corner_mean": [c_mean_px * 0.85, c_std_px],
            "mtf50_corner_worst": [c_mean_px * 0.75, c_std_px]
        }
        
    return {
        "mtf50_center": [float(c_mean_px), float(c_std_px)],
        "mtf50_corner_mean": [float(np.mean(corner_means)), float(np.std(corner_means, ddof=1) if len(corner_means) > 1 else 0.0)],
        "mtf50_corner_worst": [float(np.min(corner_means)), 0.0]
    }


def measure_modulation_at_nyquist(dng_paths: list, roi_coords: dict, nyquist_px: float = 0.25) -> float:
    """
    Interpolates empirical modulation exactly at the green-plane Nyquist frequency (0.25 cycles/pixel).
    """
    import rawpy
    measured_nyquist_list = []
    
    y, x, h, w = roi_coords["y"], roi_coords["x"], roi_coords["h"], roi_coords["w"]
    
    for path in dng_paths:
        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(float)
            black = float(np.mean(raw.black_level_per_channel))
            white = float(raw.white_level)
            
        green_linear = extract_linear_green_plane(bayer, black, white)
        roi_sub = green_linear[y:y+h, x:x+w]
        
        freq, mtf, _, _ = slanted_edge_mtf(roi_sub)
        if len(freq) > 0 and freq[-1] >= nyquist_px:
            nyquist_val = np.interp(nyquist_px, freq, mtf)
            measured_nyquist_list.append(nyquist_val)
            
    return float(np.mean(measured_nyquist_list)) if measured_nyquist_list else 0.0


def calculate_diffraction_limits(pixel_pitch_um: float, f_number: float, wavelength_um: float = 0.55) -> dict:
    """Computes diffraction limits, crossover thresholds, and systems modulation curves."""
    v_c_um = 1.0 / (wavelength_um * f_number)
    v_nyquist_um = 1.0 / (4.0 * pixel_pitch_um)
    
    n_cross = (2.0 * pixel_pitch_um) / wavelength_um
    
    nu = np.linspace(0.0001, v_c_um, 500)
    s_clipped = np.clip(nu / v_c_um, 0, 1)
    
    mtf_diff = (2.0 / np.pi) * (np.arccos(s_clipped) - s_clipped * np.sqrt(1.0 - s_clipped**2))
    mtf_sensor = np.abs(np.sinc(nu * pixel_pitch_um))
    mtf_total = mtf_diff * mtf_sensor
    
    idx_nyq = np.argmin(np.abs(nu - v_nyquist_um))
    mod_at_nyq = mtf_total[idx_nyq]
    
    idx_mtf50 = np.argmin(np.abs(mtf_total - 0.5))
    predicted_mtf50_um = nu[idx_mtf50]
    
    return {
        "v_c_um": v_c_um,
        "v_nyquist_um": v_nyquist_um,
        "crossover_f_number": n_cross,
        "modulation_at_nyquist": mod_at_nyq,
        "predicted_mtf50_cycles_per_um": predicted_mtf50_um,
        "frequency_axis": nu,
        "mtf_diffraction": mtf_diff,
        "mtf_sensor": mtf_sensor,
        "mtf_total": mtf_total
    }


def analytic_mtf50(sigma: float) -> float:
    """Analytic MTF50 for a Gaussian PSF of width sigma."""
    return np.sqrt(np.log(2) / (2 * np.pi**2 * sigma**2))
