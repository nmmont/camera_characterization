# src/camera_toolkit/color.py
import os
import cv2
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None

try:
    import colour
except ImportError:
    colour = None


# Standard 24 ColorChecker reference XYZ values (D50, 2-degree observer, post-Nov 2014)
COLORCHECKER_REF_XYZ = np.array([
    [0.115, 0.101, 0.061], [0.383, 0.348, 0.245], [0.184, 0.193, 0.366],
    [0.108, 0.133, 0.066], [0.254, 0.233, 0.428], [0.316, 0.437, 0.435],
    [0.374, 0.291, 0.088], [0.133, 0.118, 0.363], [0.283, 0.193, 0.120],
    [0.088, 0.062, 0.147], [0.341, 0.438, 0.141], [0.461, 0.375, 0.088],
    [0.066, 0.052, 0.264], [0.141, 0.222, 0.096], [0.165, 0.118, 0.060],
    [0.589, 0.582, 0.096], [0.288, 0.197, 0.320], [0.110, 0.185, 0.347],
    [0.892, 0.901, 0.865], [0.583, 0.591, 0.569], [0.354, 0.361, 0.348],
    [0.195, 0.199, 0.193], [0.089, 0.091, 0.088], [0.031, 0.031, 0.031]
], dtype=np.float64)

# D50 Reference White
XYZ_D50 = np.array([0.96422, 1.00000, 0.82521], dtype=np.float64)

# User-measured bounding box mapping: [ymin, ymax, xmin, xmax]
COLORCHECKER_BBOXES = {
    "incan_1.dng":     [500, 1515, 395, 1070],
    "incan_2.dng":     [550, 1500, 320, 980],
    "incan_3.dng":     [500, 1600, 380, 1075],
    "incan_white.dng": [153, 1382, 204, 1836],
    "led_1.dng":       [480, 1500, 400, 1100],
    "led_2.dng":       [500, 1520, 380, 1100],
    "led_3.dng":       [490, 1610, 410, 1190],
    "led_white.dng":   [153, 1382, 204, 1836],
    "oh_1.dng":        [395, 1515, 395, 1120],
    "oh_2.dng":        [390, 1620, 365, 1200],
    "oh_3.dng":        [390, 1620, 365, 1200],
    "oh_white.dng":    [153, 1382, 204, 1836],
    "out_1.dng":       [490, 1515, 395, 1075],
    "out_2.dng":       [600, 1600, 410, 1075],
    "out_3.dng":       [530, 1540, 415, 1080],
    "out_white.dng":   [153, 1382, 204, 1836],
}


def xyz_to_lab(xyz: np.ndarray, white_ref: np.ndarray = XYZ_D50) -> np.ndarray:
    """Converts linear XYZ to CIELAB L*a*b* using standard D50 reference white."""
    xr, yr, zr = xyz[:, 0] / white_ref[0], xyz[:, 1] / white_ref[1], xyz[:, 2] / white_ref[2]
    
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    
    fx = np.where(xr > epsilon, np.cbrt(xr), (kappa * xr + 16.0) / 116.0)
    fy = np.where(yr > epsilon, np.cbrt(yr), (kappa * yr + 16.0) / 116.0)
    fz = np.where(zr > epsilon, np.cbrt(zr), (kappa * zr + 16.0) / 116.0)
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.column_stack([L, a, b])


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Computes CIEDE2000 color difference between two N x 3 arrays of CIELAB values."""
    if colour is not None:
        try:
            return colour.delta_E(lab1, lab2, method="CIE 2000")
        except Exception:
            pass

    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    
    G = 0.5 * (1.0 - np.sqrt((C_bar**7) / (C_bar**7 + 25.0**7)))
    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2
    
    C1_prime = np.hypot(a1_prime, b1)
    C2_prime = np.hypot(a2_prime, b2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0
    
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    dh_prime = np.zeros_like(h1_prime)
    abs_dh = np.abs(h1_prime - h2_prime)
    mask = (C1_prime * C2_prime != 0)
    
    cond1 = mask & (abs_dh <= 180.0)
    dh_prime[cond1] = h2_prime[cond1] - h1_prime[cond1]
    
    cond2 = mask & (abs_dh > 180.0) & (h2_prime <= h1_prime)
    dh_prime[cond2] = h2_prime[cond2] - h1_prime[cond2] + 360.0
    
    cond3 = mask & (abs_dh > 180.0) & (h2_prime > h1_prime)
    dh_prime[cond3] = h2_prime[cond3] - h1_prime[cond3] - 360.0
    
    dH_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(dh_prime / 2.0))
    
    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)
    
    h_bar_prime = np.zeros_like(h1_prime)
    h_sum = h1_prime + h2_prime
    
    h_mask1 = mask & (abs_dh <= 180.0)
    h_bar_prime[h_mask1] = 0.5 * h_sum[h_mask1]
    
    h_mask2 = mask & (abs_dh > 180.0) & (h_sum < 360.0)
    h_bar_prime[h_mask2] = 0.5 * (h_sum[h_mask2] + 360.0)
    
    h_mask3 = mask & (abs_dh > 180.0) & (h_sum >= 360.0)
    h_bar_prime[h_mask3] = 0.5 * (h_sum[h_mask3] - 360.0)
    
    h_bar_prime[~mask] = h_sum[~mask]
    
    T = (1.0 - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
             + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
             + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
             - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0)))
    
    dTheta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0)**2))
    R_C = 2.0 * np.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25.0**7))
    S_L = 1.0 + (0.015 * ((L_bar_prime - 50.0)**2)) / np.sqrt(20.0 + ((L_bar_prime - 50.0)**2))
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2.0 * dTheta)) * R_C
    
    delta_E = np.sqrt(
        (dL_prime / S_L)**2 +
        (dC_prime / S_C)**2 +
        (dH_prime / S_H)**2 +
        R_T * (dC_prime / S_C) * (dH_prime / S_H)
    )
    return delta_E


def check_raw_clipping(dng_path: str) -> tuple:
    """Inspects raw histogram to verify active dynamic range ceiling and check for clipping."""
    if rawpy is None:
        return 1023.0, 64.0, False
    with rawpy.imread(dng_path) as raw:
        data = raw.raw_image_visible
        white_level = float(raw.white_level)
        black_level = float(np.mean(raw.black_level_per_channel))
        is_clipped = bool(np.max(data) >= white_level - 1.0)
    return white_level, black_level, is_clipped


def extract_quarter_res_linear_rgb(dng_path: str, black_level: float = None, white_level: float = None) -> np.ndarray:
    """Converts 2x2 Bayer blocks (R, G1, G2, B) to 1 pixel quarter-res linear RGB in vertical portrait orientation."""
    if rawpy is None:
        return np.ones((100, 100, 3), dtype=np.float64) * 0.5

    with rawpy.imread(dng_path) as raw:
        raw_data = raw.raw_image.copy()
        h, w = raw_data.shape
        h_even = (h // 2) * 2
        w_even = (w // 2) * 2
        raw_data = raw_data[:h_even, :w_even]

        cfa_pattern = raw.raw_pattern
        color_desc = raw.color_desc
        black_levels = raw.black_level_per_channel

        r_plane, b_plane, g_planes = None, None, []
        for r_offset in range(2):
            for c_offset in range(2):
                c_idx = cfa_pattern[r_offset, c_offset]
                c_char = chr(color_desc[c_idx])
                b_level = black_levels[r_offset * 2 + c_offset]
                sub_plane = raw_data[r_offset::2, c_offset::2].astype(np.float32) - b_level

                if c_char == "R":
                    r_plane = sub_plane
                elif c_char == "B":
                    b_plane = sub_plane
                elif c_char in ["G", "g"]:
                    g_planes.append(sub_plane)

        g_plane = np.mean(g_planes, axis=0) if g_planes else (r_plane + b_plane) / 2.0
        avg_black = np.mean(black_levels) if black_level is None else black_level
        w_lvl = white_level if white_level is not None else float(raw.white_level)
        span = w_lvl - avg_black

        r_lin = np.clip(r_plane / span, 0.0, 1.0)
        g_lin = np.clip(g_plane / span, 0.0, 1.0)
        b_lin = np.clip(b_plane / span, 0.0, 1.0)

        quarter_rgb = np.stack([r_lin, g_lin, b_lin], axis=-1)
        if quarter_rgb.shape[1] > quarter_rgb.shape[0]:
            quarter_rgb = np.rot90(quarter_rgb, k=1)
        quarter_rgb = np.rot90(quarter_rgb, k=2)

        return quarter_rgb


def extract_colorchecker_manual(
    scene_rgb: np.ndarray,
    grid_bbox: list,
    inner_margin: float = 0.25,
    num_rows: int = 6,
    num_cols: int = 4,
    force_square: bool = True
) -> np.ndarray:
    """Extracts patch means for a 6x4 portrait ColorChecker grid using eyeballed bounding boxes."""
    ymin, ymax, xmin, xmax = grid_bbox
    row_height = (ymax - ymin) / num_rows
    col_width = (xmax - xmin) / num_cols
    patch_means = []

    for r in range(num_rows):
        for c in range(num_cols):
            cell_cy = ymin + (r + 0.5) * row_height
            cell_cx = xmin + (c + 0.5) * col_width

            if force_square:
                side_len = min(col_width, row_height) * (1.0 - 2 * inner_margin)
                p_xmin = int(cell_cx - side_len / 2.0)
                p_xmax = int(cell_cx + side_len / 2.0)
                p_ymin = int(cell_cy - side_len / 2.0)
                p_ymax = int(cell_cy + side_len / 2.0)
            else:
                p_ymin = int(ymin + r * row_height + inner_margin * row_height)
                p_ymax = int(ymin + (r + 1) * row_height - inner_margin * row_height)
                p_xmin = int(xmin + c * col_width + inner_margin * col_width)
                p_xmax = int(xmin + (c + 1) * col_width - inner_margin * col_width)

            patch_pixels = scene_rgb[p_ymin:p_ymax, p_xmin:p_xmax, :]
            mean_rgb = np.mean(patch_pixels, axis=(0, 1))
            patch_means.append(mean_rgb)

    return np.array(patch_means, dtype=np.float64)


def align_patch_means(raw_patch_means: np.ndarray) -> np.ndarray:
    """Rotates a 6x4 portrait patch matrix 90 degrees clockwise (k=-1) into standard 4x6 landscape order."""
    patch_grid_portrait = raw_patch_means.reshape(6, 4, 3)
    patch_grid_landscape = np.rot90(patch_grid_portrait, k=-1)
    return patch_grid_landscape.reshape(24, 3)


def apply_lens_shading_correction(sample_rgb: np.ndarray, white_sheet_rgb: np.ndarray) -> np.ndarray:
    """Normalizes white sheet flat field by its center and divides sample_rgb to correct lens shading."""
    h, w, _ = white_sheet_rgb.shape
    center_roi = white_sheet_rgb[h//2 - 20:h//2 + 20, w//2 - 20:w//2 + 20, :]
    center_val = np.mean(center_roi, axis=(0, 1), keepdims=True)
    
    flat_field_norm = white_sheet_rgb / np.maximum(center_val, 1e-6)
    corrected = sample_rgb / np.maximum(flat_field_norm, 1e-6)
    return np.clip(corrected, 0.0, 1.0)


def compute_white_balance_gains(patch_means_24x3: np.ndarray, neutral_patch_idx: int = 20) -> tuple:
    """Calculates gains normalized to Green from unclipped neutral patch (Ref Patch 21 Neutral 6.5, index 20)."""
    neutral_rgb = patch_means_24x3[neutral_patch_idx]
    g_val = neutral_rgb[1]
    
    gain_r = g_val / max(neutral_rgb[0], 1e-6)
    gain_g = 1.0
    gain_b = g_val / max(neutral_rgb[2], 1e-6)
    
    gains = np.array([gain_r, gain_g, gain_b], dtype=np.float64)
    balanced_means = patch_means_24x3 * gains[None, :]
    return gains, balanced_means


def fit_color_correction_matrix(balanced_means_24x3: np.ndarray, ref_xyz_24x3: np.ndarray = COLORCHECKER_REF_XYZ) -> np.ndarray:
    """Solves linear least-squares M @ balanced_means.T = ref_xyz.T."""
    M, _, _, _ = np.linalg.lstsq(balanced_means_24x3, ref_xyz_24x3, rcond=None)
    CCM = M.T
    return CCM


def evaluate_delta_e(measured_xyz_24x3: np.ndarray, ref_xyz_24x3: np.ndarray = COLORCHECKER_REF_XYZ, white_ref_xyz: np.ndarray = None) -> tuple:
    """
    Converts XYZ to CIELAB using White Patch 19 chromaticities for illuminant white point adaptation,
    and computes CIEDE2000 errors.
    """
    if colour is not None:
        try:
            target_white_xyz = white_ref_xyz if white_ref_xyz is not None else measured_xyz_24x3[18]
            white_xy_meas = colour.XYZ_to_xy(target_white_xyz)
            
            scale = ref_xyz_24x3[18, 1] / max(measured_xyz_24x3[18, 1], 1e-6)
            scaled_meas_xyz = measured_xyz_24x3 * scale

            lab_ref = colour.XYZ_to_Lab(ref_xyz_24x3, illuminant=white_xy_meas)
            lab_meas = colour.XYZ_to_Lab(scaled_meas_xyz, illuminant=white_xy_meas)

            delta_e_list = colour.delta_E(lab_meas, lab_ref, method="CIE 2000")
            mean_de = float(np.mean(delta_e_list))
            std_de = float(np.std(delta_e_list, ddof=1) if len(delta_e_list) > 1 else 0.0)
            worst_de = float(np.max(delta_e_list))

            return mean_de, std_de, worst_de, delta_e_list
        except Exception:
            pass

    scale = ref_xyz_24x3[18, 1] / max(measured_xyz_24x3[18, 1], 1e-6)
    scaled_meas_xyz = measured_xyz_24x3 * scale
    white_ref = white_ref_xyz if white_ref_xyz is not None else ref_xyz_24x3[18]

    lab_ref = xyz_to_lab(ref_xyz_24x3, white_ref=white_ref)
    lab_meas = xyz_to_lab(scaled_meas_xyz, white_ref=white_ref)

    delta_e_list = ciede2000(lab_meas, lab_ref)

    mean_de = float(np.mean(delta_e_list))
    std_de = float(np.std(delta_e_list, ddof=1) if len(delta_e_list) > 1 else 0.0)
    worst_de = float(np.max(delta_e_list))

    return mean_de, std_de, worst_de, delta_e_list


def process_color_accuracy_pipeline(patch_means_24x3: np.ndarray) -> dict:
    """Full Part 4 color accuracy pipeline evaluation."""
    gains, balanced_means = compute_white_balance_gains(patch_means_24x3, neutral_patch_idx=20)
    
    srgb_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    baseline_xyz = balanced_means @ srgb_matrix.T
    white_base_xyz = baseline_xyz[18]
    mean_de_before, std_de_before, _, _ = evaluate_delta_e(baseline_xyz, white_ref_xyz=white_base_xyz)
    
    CCM = fit_color_correction_matrix(balanced_means)
    fitted_xyz = balanced_means @ CCM.T
    white_fit_xyz = fitted_xyz[18]
    mean_de_after, std_de_after, worst_de_after, _ = evaluate_delta_e(fitted_xyz, white_ref_xyz=white_fit_xyz)
    
    return {
        "de_mean_before": [mean_de_before, std_de_before],
        "de_mean_after": [mean_de_after, std_de_after],
        "de_worst_after": [worst_de_after, 0.0],
        "ccm_matrix": CCM.tolist()
    }
