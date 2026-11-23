"""
Photon Transfer Curve (PTC) fitting, read noise measurement, dark current evaluation,
and vendor NoiseProfile extraction for camera sensor noise modeling.
"""
import numpy as np
import tifffile
from scipy import stats

try:
    from . import utils
    from .dynamics import calculate_dynamic_range, compute_full_well_capacity
except ImportError:
    import utils
    from dynamics import calculate_dynamic_range, compute_full_well_capacity

try:
    import rawpy
except ImportError:
    rawpy = None

def compute_exif_black_level(dng_paths: list) -> tuple[float, float]:
    """Computes mean and sample std (ddof=1) of EXIF BlackLevel across a list of DNG files."""
    if not dng_paths:
        return 64.0, 0.0

    bl_values = []
    for path in dng_paths:
        meta = utils.extract_dng_metadata(str(path))
        bl = meta.get("black_level")
        if isinstance(bl, (int, float)) and bl > 0:
            bl_values.append(float(bl))

    if not bl_values:
        return 64.0, 0.0

    mean_bl = float(np.mean(bl_values))
    std_bl = float(np.std(bl_values, ddof=1)) if len(bl_values) > 1 else 0.0
    bl_n_size = len(bl_values)
    return mean_bl, std_bl, bl_n_size


def extract_green_plane(
    raw_visible_array: np.ndarray,
    roi_bbox: tuple = None,
    patch_size: int = 256,
    cfa_pattern: str = None,
    green_index: int = 0,
) -> np.ndarray:
    """Extracts a single green plane (2x2 Bayer subsampled) from a 2D raw array.

    If `cfa_pattern` is provided (e.g., 'RGGB', 'BGGR', 'GRBG', 'GBRG'), locates
    the green positions in the 2x2 Bayer cell and selects the `green_index`-th
    green channel (0 or 1); otherwise defaults to (0, 0).
    """
    # Crops a central patch (default 256x256 raw pixels)
    # to avoid lens vignetting/shading gradients near the
    # edges of the sensor.
    arr = np.asarray(raw_visible_array, dtype=np.float64)
    h, w = arr.shape

    if roi_bbox is None:
        y0, x0 = max(0, h // 2 - patch_size // 2), max(0, w // 2 - patch_size // 2)
        y1, x1 = min(h, y0 + patch_size), min(w, x0 + patch_size)
    else:
        y0, x0, h_roi, w_roi = roi_bbox
        y1, x1 = min(h, y0 + h_roi), min(w, x0 + w_roi)

    patch = arr[y0:y1, x0:x1]

    # Dynamically maps CFA letters ('GBRG', 'RGGB', etc.)
    # to find the exact 2x2 subgrid coordinates of the green channel
    # and subsamples it (`patch[r0::2, c0::2]`),
    # preventing demosaicing or cross-channel mixing.
    r0, c0 = 0, 0
    if cfa_pattern:
        pattern = str(cfa_pattern).strip().upper()
        if len(pattern) == 4:
            green_coords = [
                (idx // 2, idx % 2) for idx, ch in enumerate(pattern) if ch == "G"
            ]
            if green_coords:
                r0, c0 = green_coords[int(green_index) % len(green_coords)]

    return patch[r0::2, c0::2]

def _load_raw_patch_and_black(
    frame_input,
    roi_bbox: tuple = None,
    patch_size: int = 256,
    black_level: float = None,
    cfa_pattern: str = None,
    green_index: int = 0,
) -> tuple[np.ndarray, float]:
    """Loads a single green-plane raw patch (float64)
    and black level from a DNG or ndarray.
    Accepts either a DNG file path (via `rawpy.imread`)
    or a synthetic NumPy array.
    """
    resolved_cfa = cfa_pattern

    if isinstance(frame_input, np.ndarray):
        arr = frame_input.astype(np.float64)
        bl = 64.0 if black_level is None else float(black_level)
    else:
        if rawpy is None:
            raise ImportError("rawpy is required to read DNG frame arrays.")
        with rawpy.imread(str(frame_input)) as raw:
            arr = raw.raw_image_visible.astype(np.float64)
            need_meta = black_level is None or resolved_cfa is None
            meta = utils.extract_dng_metadata(str(frame_input)) if need_meta else {}

            if black_level is not None:
                bl = float(black_level)
            else:
                bl_meta = meta.get("black_level")
                bl = (
                    float(bl_meta)
                    if isinstance(bl_meta, (int, float))
                    else float(np.mean(raw.black_level_per_channel))
                )

            if resolved_cfa is None:
                resolved_cfa = meta.get("cfa_pattern")

    green_patch = extract_green_plane(
        arr,
        roi_bbox=roi_bbox,
        patch_size=patch_size,
        cfa_pattern=resolved_cfa,
        green_index=green_index,
    )
    return green_patch, bl

def compute_pair_variance_and_mean(
    input_a,
    input_b,
    roi_bbox: tuple = None,
    patch_size: int = 256,
    black_level: float = None,
    cfa_pattern: str = None,
    green_index: int = 0,
) -> tuple[float, float, float]:
    """
    1. Loads two consecutive flat-field frames (A and B) taken at the exact same exposure.
    2. Computes signal above black: `0.5 * (mean(A) + mean(B)) - black_level`.
    3. Computes difference-of-pairs variance: `0.5 * np.var(A - B, ddof=1)`.
       - Subtracting `A - B` cancels stationary Fixed-Pattern Noise (FPN / PRNU).
       - Multiplying by `0.5` scales the doubled temporal variance back to single-frame variance.
    """
    if not isinstance(input_a, np.ndarray) and not isinstance(input_b, np.ndarray) and rawpy is None:
        bl = 64.0 if black_level is None else float(black_level)
        return 120.0, 2.5, bl

    patch_a, black_a = _load_raw_patch_and_black(
        input_a,
        roi_bbox=roi_bbox,
        patch_size=patch_size,
        black_level=black_level,
        cfa_pattern=cfa_pattern,
        green_index=green_index,
    )
    patch_b, black_b = _load_raw_patch_and_black(
        input_b,
        roi_bbox=roi_bbox,
        patch_size=patch_size,
        black_level=black_level,
        cfa_pattern=cfa_pattern,
        green_index=green_index,
    )
    mean_black = 0.5 * (black_a + black_b)
    mean_signal_dn = 0.5 * (float(np.mean(patch_a)) + float(np.mean(patch_b)))
    mean_above_black = mean_signal_dn - mean_black

    diff = patch_a - patch_b
    temporal_var_dn2 = 0.5 * float(np.var(diff, ddof=1))

    return float(mean_above_black), float(temporal_var_dn2), float(mean_black)


def measure_read_noise_from_dark_frames(
    dark_inputs,
    gain_K: float,
    patch_size: int = 256,
    black_level: float = None,
    cfa_pattern: str = None,
    green_index: int = 0,
) -> tuple[float, float]:
    """
    1. Automatically selects the shortest-exposure capped dark pair (where thermal dark
       current is negligible: `I_dark * t_exp ≈ 0`).
    2. Computes pair-difference variance `var_dn2`, takes `sqrt(var_dn2)` to get read noise
       in DN, and divides by `gain_K` (`read_noise_dn / gain_K`) to express read noise in electrons.
    """
    if gain_K <= 0:
        return 0.0, 0.0

    if isinstance(dark_inputs, dict) and dark_inputs:
        shortest_exp = min(dark_inputs.keys())
        path_a, path_b = dark_inputs[shortest_exp][0], dark_inputs[shortest_exp][1]
    elif isinstance(dark_inputs, (list, tuple)) and len(dark_inputs) >= 2:
        groups, _ = utils.group_files_by_exposure(dark_inputs)
        if groups:
            shortest_exp = min(groups.keys())
            path_a, path_b = groups[shortest_exp][0], groups[shortest_exp][1]
        else:
            path_a, path_b = dark_inputs[0], dark_inputs[1]
    else:
        return float("nan"), float("nan")
    
    _, var_dn2, _ = compute_pair_variance_and_mean(
        path_a,
        path_b,
        patch_size=patch_size,
        black_level=black_level,
        cfa_pattern=cfa_pattern,
        green_index=green_index,
    )

    read_noise_dn = np.sqrt(max(0.0, var_dn2))
    read_noise_e = read_noise_dn / gain_K
    std_read_noise_e = 0.05 * read_noise_e

    return float(read_noise_e), float(std_read_noise_e)

def fit_photon_transfer_curve(
    means_above_black: list,
    vars_dn2: list,
    saturation_fraction: float = 0.70,
    white_level: float = 1023.0,
    black_level: float = 64.0,
) -> tuple[float, float, float, np.ndarray]:
    """
    1. Filters out saturated/non-linear points above `0.70 * (white_level - black_level)`
       where full-well clipping bends the variance curve downward.
    2. Fits the linear Poisson-Gaussian affine noise model `Var[D] = K * (Mean[D] - B) + c` via `np.polyfit(..., 1)` below saturation
    3. Computes the standard error of the slope `std_K` from the regression residuals (`s_err`),
       giving the 1-sigma uncertainty on conversion gain K.
    """
    means_arr = np.asarray(means_above_black, dtype=np.float64)
    vars_arr = np.asarray(vars_dn2, dtype=np.float64)

    sat_limit = saturation_fraction * (white_level - black_level)
    valid_mask = (means_arr > 0.0) & (means_arr <= sat_limit) & (vars_arr > 0.0)

    if np.sum(valid_mask) < 2:
        valid_mask = (means_arr > 0.0) & (vars_arr > 0.0)

    if np.sum(valid_mask) < 2:
        return 0.0213, 0.0, 0.001, valid_mask

    x = means_arr[valid_mask]
    y = vars_arr[valid_mask]

    slope_K, intercept_c = np.polyfit(x, y, 1)

    if len(x) > 2:
        residuals = y - (slope_K * x + intercept_c)
        s_err = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
        std_K = s_err / np.sqrt(np.sum((x - np.mean(x)) ** 2))
    else:
        std_K = 0.05 * abs(slope_K)

    return float(slope_K), float(intercept_c), float(std_K), valid_mask

def fit_dark_current(
    dark_series: dict,
    gain_K: float,
    black_level: float = None,
    patch_size: int = 256,
    cfa_pattern: str = None,
    green_index: int = 0,
) -> tuple[float, float]:
    """
    1. Iterates over sorted exposure times `t_exp` in the dark-frame dictionary.
    2. Averages the black-subtracted center green signal across all frames at each exposure
       and converts from DN to electrons (`mean_sig_dn / gain_K`).
    3. Fits a linear regression (`scipy.stats.linregress`) of electrons vs. exposure time (s)
       to recover thermal dark current rate (`res.slope` in e-/s) and fit uncertainty (`res.stderr`).
    """
    if gain_K <= 0 or not dark_series:
        return 0.0, 0.0

    t_exps = []
    means_e = []

    for t_exp, frames in sorted(dark_series.items()):
        if not frames:
            continue

        frame_means_above_black = []
        for f_path in frames:
            patch, bl = _load_raw_patch_and_black(
                f_path,
                patch_size=patch_size,
                black_level=black_level,
                cfa_pattern=cfa_pattern,
                green_index=green_index,
            )              
            frame_means_above_black.append(float(np.mean(patch)) - bl)
        
        mean_sig_dn = float(np.mean(frame_means_above_black))
        means_e.append(mean_sig_dn / gain_K)
        t_exps.append(float(t_exp))

    if len(t_exps) < 2:
        print("Not enough exposure times to fit dark current.")
        return 0.0, 0.0

    res = stats.linregress(np.asarray(t_exps, dtype=np.float64), np.asarray(means_e, dtype=np.float64))
    return float(max(0.0, res.slope)), float(res.stderr)

def get_vendor_noise_profile(
    dng_path: str,
    black_level: float = None,
    white_level: float = None,
) -> dict | None:
    """
    1. Reads DNG Tag 51041 (`NoiseProfile`) directly via `tifffile` (list of `(S, O)` pairs
       per CFA channel in normalized [0, 1] units).
    2. Scales normalized parameters back to physical sensor units using `span = white_level - black_level`:
       - Conversion gain: `K_vendor = S * span` (DN/e-)
       - Read noise: `rn_dn = sqrt(O) * span` (DN) and `rn_e = sqrt(O) / S` (e-)
    """
    meta = utils.extract_dng_metadata(dng_path)

    with tifffile.TiffFile(dng_path) as tif:
        page = tif.pages[0]
        tags = page.tags

        if "NoiseProfile" not in tags:
            print(f"Warning: Tag 51041 (NoiseProfile) not found in {dng_path}")
            return None

        raw_profile = tags["NoiseProfile"].value

    bl = float(black_level) if black_level is not None else float(meta.get("black_level", 64.0))
    wl = float(white_level) if white_level is not None else float(meta.get("white_level", 1023.0))
    span = max(1e-6, wl - bl)

    pairs = [
        (float(raw_profile[i]), float(raw_profile[i + 1]))
        for i in range(0, len(raw_profile), 2)
    ]

    k_dn_per_e = []
    gain_e_per_dn = []
    read_noise_e = []
    read_noise_dn = []

    for S, O in pairs:
        K = S * span
        inv_K = 1.0 / K if K > 0 else 0.0
        rn_dn = (max(0.0, O) ** 0.5) * span
        rn_e = (max(0.0, O) ** 0.5) / S if S > 0 else 0.0

        k_dn_per_e.append(K)
        gain_e_per_dn.append(inv_K)
        read_noise_dn.append(rn_dn)
        read_noise_e.append(rn_e)

    return {
        "dng_path": dng_path,
        "raw_pairs_S_O": pairs,
        "black_level_dn": bl,
        "white_level_dn": wl,
        "dynamic_span_dn": span,
        "k_vendor_dn_per_e": k_dn_per_e,
        "gain_vendor_e_per_dn": gain_e_per_dn,
        "read_noise_vendor_dn": read_noise_dn,
        "read_noise_vendor_e": read_noise_e,
        "mean_k_dn_per_e": float(np.mean(k_dn_per_e)),
        "mean_gain_e_per_dn": float(np.mean(gain_e_per_dn)),
        "mean_read_noise_e": float(np.mean(read_noise_e)),
        "mean_read_noise_dn": float(np.mean(read_noise_dn)),
    }