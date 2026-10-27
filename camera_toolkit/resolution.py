# src/camera_toolkit/resolution.py
import os
import numpy as np
# ==========================================
# 1. CORE INGESTION & LOW-LEVEL MATH ENGINES
# ==========================================
def extract_linear_green_plane(raw_visible_array: np.ndarray, black_level: float, white_level: float) -> np.ndarray:
    """
    Slices the raw Bayer matrix to isolate one green channel plane and normalizes it.
    Applies the -90 degree rotation to map onto the physical horizontal edge space.
    """
    # Slice the green channel plane (Row 0, Col 1 or Row 0, Col 0 depending on the Bayer variant)
    green_plane = raw_visible_array[0::2, 0::2]
    green_plane = np.rot90(green_plane, k=-1)
    
    # Subtract black pedestal and scale linearly by active dynamic range [INDEX: 0.1.2]
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
    """Measures MTF from a slanted edge ROI. Returns (freq, mtf, esf_x, esf)."""
    roi = np.asarray(roi, dtype=float)
    rows = np.arange(roi.shape[0])

    centers = edge_centroids(roi, window=window)
    ok = np.isfinite(centers)
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

    lsf = np.gradient(esf, esf_x)
    lsf = lsf - np.median(np.concatenate([lsf[:5], lsf[-5:]]))
    lsf = lsf * np.hanning(len(lsf))

    spectrum = np.abs(np.fft.rfft(lsf))
    return np.fft.rfftfreq(len(lsf), d=bin_width), spectrum / spectrum[0], esf_x, esf

def compute_mtf50(freq: np.ndarray, mtf: np.ndarray) -> float:
    """Finds frequency where MTF drops to 0.5 via linear interpolation."""
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(freq[i - 1] + (mtf[i - 1] - 0.5) * (freq[i] - freq[i - 1]) / (mtf[i - 1] - mtf[i]))

# ==========================================
# 2. BATCH PROCESSING & WORKFLOW ORCHESTRATION
# ==========================================

def process_batch_mtf50(dng_paths: list, roi_coords: dict, pixel_pitch_um: float) -> tuple:
    """
    Batch processes a folder of raw images to calculate mean MTF50 values and uncertainties [INDEX: 0.1.3].
    
    Inputs:
        roi_coords: dict containing {"y": 700, "x": 390, "h": 400, "w": 200}
    Returns:
        tuple: (mean_c_um, std_c_um, mean_c_px, std_c_px, all_rois_list)
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
        
        # Call your slanted-edge binning routines
        freq, mtf, _, _ = slanted_edge_mtf(roi_sub)
        mtf50_c_px = compute_mtf50(freq, mtf)
        
        if not np.isnan(mtf50_c_px):
            # Sampling pitch of the green subgrid plane is 2*p [INDEX: 0.1.2]
            mtf50_c_um = mtf50_c_px / (2.0 * pixel_pitch_um)
            mtf50_um_list.append(mtf50_c_um)
            mtf50_px_list.append(mtf50_c_px)
            
    # Calculate statistics with standard deviations (ddof=1) for uncertainty pairs [INDEX: 0.1.3]
    return (
        float(np.mean(mtf50_um_list)), float(np.std(mtf50_um_list, ddof=1)),
        float(np.mean(mtf50_px_list)), float(np.std(mtf50_px_list, ddof=1)),
        all_rois_list
    )

def measure_modulation_at_nyquist(dng_paths: list, roi_coords: dict, nyquist_px: float = 0.25) -> float:
    """Interpolates empirical modulation exactly at the green-plane Nyquist frequency."""
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

# ==========================================
# 3. THEORETICAL & DIAGNOSTIC MODELS
# ==========================================

def calculate_diffraction_limits(pixel_pitch_um: float, f_number: float, wavelength_um: float = 0.55) -> dict:
    """Computes diffraction limits, crossover thresholds, and systems modulation curves."""
    v_c_um = 1.0 / (wavelength_um * f_number)
    v_nyquist_um = 1.0 / (4.0 * pixel_pitch_um)
    
    # Deliverable 6: Crossover F-number
    n_cross = (2.0 * pixel_pitch_um) / wavelength_um
    
    # Deliverable 7: Modulation curve profiles
    nu = np.linspace(0.0001, v_c_um, 500)
    s_clipped = np.clip(nu / v_c_um, 0, 1)
    
    mtf_diff = (2.0 / np.pi) * (np.arccos(s_clipped) - s_clipped * np.sqrt(1.0 - s_clipped**2))
    mtf_sensor = np.abs(np.sinc(nu * pixel_pitch_um))
    mtf_total = mtf_diff * mtf_sensor
    
    idx_nyq = np.argmin(np.abs(nu - v_nyquist_um))
    mod_at_nyq = mtf_total[idx_nyq]
    
    # Deliverable 8: Predicted MTF50 crossing
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
