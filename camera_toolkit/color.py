"""
Color science, ColorChecker patch extraction, von Kries white balancing,
CCM fitting, and CIEDE2000 evaluation for camera characterization.
"""
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
    "oh_3.dng":        [420, 1660, 370, 1190],
    "oh_white.dng":    [153, 1382, 204, 1836],
    "out_1.dng":       [490, 1515, 395, 1075],
    "out_2.dng":       [600, 1600, 410, 1075],
    "out_3.dng":       [530, 1540, 415, 1080],
    "out_white.dng":   [153, 1382, 204, 1836],
}

SRGB_TO_XYZ = np.asarray(
    colour.models.RGB_COLOURSPACE_sRGB.matrix_RGB_to_XYZ, dtype=np.float64
)

def get_reference_colorchecker_xyz(
    chart_name: str = "ColorChecker24 - After November 2014",
) -> tuple[np.ndarray, np.ndarray]:
    """Loads published 24-patch XYZ tristimulus values and white point dynamically from colour-science."""
    if colour is None:
        raise ImportError("colour-science is required to load reference ColorChecker data.")

    cc = colour.CCS_COLOURCHECKERS[chart_name]
    ref_xyY = np.array(list(cc.data.values()), dtype=np.float64)
    ref_xyz = colour.xyY_to_XYZ(ref_xyY)

    # Dynamically convert the chart's published illuminant xy chromaticity (D50) to XYZ (Y = 1.0)
    white_xyz = colour.xy_to_XYZ(cc.illuminant)
    white_xyz = white_xyz / white_xyz[1]

    return ref_xyz, np.asarray(white_xyz, dtype=np.float64)


COLORCHECKER_REF_XYZ, XYZ_D50 = get_reference_colorchecker_xyz()

def get_reference_patch_names(
    chart_name: str = "ColorChecker24 - After November 2014",
) -> list[str]:
    """Returns numbered patch names directly from colour-science's ColorChecker dataset."""
    if colour is None:
        return [f"{i + 1}. Patch {i + 1}" for i in range(24)]
    cc = colour.CCS_COLOURCHECKERS[chart_name]
    return [f"{i + 1}. {name.title()}" for i, name in enumerate(cc.data.keys())]


def xyz_to_lab(xyz: np.ndarray, white_ref: np.ndarray = XYZ_D50) -> np.ndarray:
    """Converts linear XYZ to CIELAB L*a*b* using D50 reference white."""
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

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_prime - 63.0))
    )

    dTheta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt((C_bar_prime**7) / (C_bar_prime**7 + 25.0**7))
    S_L = 1.0 + (0.015 * ((L_bar_prime - 50.0) ** 2)) / np.sqrt(
        20.0 + ((L_bar_prime - 50.0) ** 2)
    )
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.radians(2.0 * dTheta)) * R_C

    delta_E = np.sqrt(
        (dL_prime / S_L) ** 2
        + (dC_prime / S_C) ** 2
        + (dH_prime / S_H) ** 2
        + R_T * (dC_prime / S_C) * (dH_prime / S_H)
    )
    return delta_E


def analyze_raw_saturation(dng_path: str) -> dict:
    """Computes raw histogram and empirical saturation ceiling (v_sat) for a DNG file."""
    if rawpy is None:
        return {
            "white_level": 1023.0,
            "black_level": 64.0,
            "v_sat": 1023,
            "clipped_fraction": 0.0,
            "is_clipped": False,
            "hist": np.array([]),
            "bin_centers": np.array([]),
        }

    with rawpy.imread(dng_path) as raw:
        raw_data = raw.raw_image.copy()
        if raw_data.shape[1] > raw_data.shape[0]:
            raw_data = np.rot90(raw_data)

        total_pixels = raw_data.size
        declared_white = float(raw.white_level)
        black_level = float(round(np.mean(raw.black_level_per_channel)))

        hist, bin_edges = np.histogram(
            raw_data, bins=200, range=(0, declared_white)
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        valid_bins = np.where(hist > 50)[0]
        if len(valid_bins) > 0:
            rightmost_idx = valid_bins[-1]
            edge_val = int(bin_centers[rightmost_idx])
            is_truly_clipped = (edge_val >= (declared_white - 15)) or (
                hist[rightmost_idx] > 5000 and edge_val > 800
            )
        else:
            edge_val = int(declared_white)
            is_truly_clipped = False

        if is_truly_clipped:
            v_sat = edge_val
            clipped_count = int(np.sum(raw_data >= (v_sat - 5)))
            clipped_fraction = float(clipped_count / total_pixels)
        else:
            v_sat = int(declared_white)
            clipped_fraction = 0.0

    return {
        "white_level": declared_white,
        "black_level": black_level,
        "v_sat": v_sat,
        "clipped_fraction": clipped_fraction,
        "is_clipped": bool(is_truly_clipped),
        "hist": hist,
        "bin_centers": bin_centers,
    }


def check_raw_clipping(dng_path: str) -> tuple:
    """Returns (v_sat, black_level, is_clipped) using empirical histogram ceiling."""
    info = analyze_raw_saturation(dng_path)
    return float(info["v_sat"]), float(info["black_level"]), bool(info["is_clipped"])


def extract_quarter_res_linear_rgb(
    dng_path: str, black_level: float = None, white_level: float = None
) -> np.ndarray:
    """Converts Bayer CFA blocks to 1 pixel quarter-res linear RGB [R, mean(G1,G2), B]."""
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
                sub_plane = (
                    raw_data[r_offset::2, c_offset::2].astype(np.float32)
                    - b_level
                )

                if c_char == "R":
                    r_plane = sub_plane
                elif c_char == "B":
                    b_plane = sub_plane
                elif c_char in ["G", "g"]:
                    g_planes.append(sub_plane)

        g_plane = (
            np.mean(g_planes, axis=0)
            if g_planes
            else (r_plane + b_plane) / 2.0
        )
        avg_black = (
            float(round(np.mean(black_levels)))
            if black_level is None
            else black_level
        )
        w_lvl = (
            white_level if white_level is not None else float(raw.white_level)
        )
        span = w_lvl - avg_black

        r_lin = np.clip(r_plane / span, 0.0, 1.0)
        g_lin = np.clip(g_plane / span, 0.0, 1.0)
        b_lin = np.clip(b_plane / span, 0.0, 1.0)

        quarter_rgb = np.stack([r_lin, g_lin, b_lin], axis=-1)

        if quarter_rgb.shape[1] > quarter_rgb.shape[0]:
            quarter_rgb = np.rot90(quarter_rgb, k=1)
        quarter_rgb = np.rot90(quarter_rgb, k=2)

        return quarter_rgb.astype(np.float64)


def extract_colorchecker_manual(
    quarter_rgb: np.ndarray,
    grid_bbox: list,
    inner_margin: float = 0.25,
    num_rows: int = 6,
    num_cols: int = 4,
    force_square: bool = True,
) -> np.ndarray:
    """Extracts ColorChecker patch means from specified bounding box."""
    ymin, ymax, xmin, xmax = grid_bbox
    box = quarter_rgb[ymin:ymax, xmin:xmax]
    h, w, _ = box.shape

    step_y = h / num_rows
    step_x = w / num_cols

    patch_means = []

    for r in range(num_rows):
        for c in range(num_cols):
            py0 = int(r * step_y)
            py1 = int((r + 1) * step_y)
            px0 = int(c * step_x)
            px1 = int((c + 1) * step_x)

            pw = px1 - px0
            ph = py1 - py0
            margin_y = int(ph * inner_margin)
            margin_x = int(pw * inner_margin)

            crop = box[
                py0 + margin_y : py1 - margin_y,
                px0 + margin_x : px1 - margin_x,
            ]
            if crop.size > 0:
                mean_val = np.mean(crop, axis=(0, 1))
            else:
                mean_val = np.array([0.5, 0.5, 0.5])
            patch_means.append(mean_val)

    return np.array(patch_means, dtype=np.float64)


def align_patch_means(raw_patch_means: np.ndarray) -> np.ndarray:
    """Rotates 6x4 portrait grid 90-deg clockwise to match 4x6 landscape reference table."""
    patch_grid_portrait = raw_patch_means.reshape(6, 4, 3)
    patch_grid_landscape = np.rot90(patch_grid_portrait, k=-1)
    return patch_grid_landscape.reshape(24, 3)


def apply_lens_shading_correction(
    sample_rgb: np.ndarray, white_sheet_rgb: np.ndarray
) -> np.ndarray:
    """Normalizes white sheet flat field by its center and divides sample_rgb to correct lens shading."""
    h, w, _ = white_sheet_rgb.shape
    center_roi = white_sheet_rgb[
        h // 2 - 20 : h // 2 + 20, w // 2 - 20 : w // 2 + 20, :
    ]
    center_val = np.mean(center_roi, axis=(0, 1), keepdims=True)
    flat_normalized = np.clip(white_sheet_rgb / max(np.max(center_val), 1e-6), 1e-4, 1.0)
    corrected_sample = sample_rgb / flat_normalized
    return np.maximum(corrected_sample, 0.0)
    #return np.clip(corrected_sample, 0.0, 1.0)


def compute_white_balance_gains(
    patch_means_24x3: np.ndarray, neutral_patch_idx: int = 20
) -> tuple:
    """Computes von Kries white balance gains using neutral Patch 21 (0-indexed 20)."""
    neutral_rgb = patch_means_24x3[neutral_patch_idx]
    g_val = max(neutral_rgb[1], 1e-6)
    r_gain = g_val / max(neutral_rgb[0], 1e-6)
    b_gain = g_val / max(neutral_rgb[2], 1e-6)
    gains = np.array([r_gain, 1.0, b_gain], dtype=np.float64)
    balanced_means = patch_means_24x3 * gains
    return gains, balanced_means


def fit_color_correction_matrix(
    balanced_means_24x3: np.ndarray,
    ref_xyz_24x3: np.ndarray = COLORCHECKER_REF_XYZ,
) -> np.ndarray:
    """Solves linear least-squares M @ balanced_means.T = ref_xyz.T."""
    M, _, _, _ = np.linalg.lstsq(balanced_means_24x3, ref_xyz_24x3, rcond=None)
    CCM = M.T
    return CCM


def evaluate_delta_e(
    measured_xyz_24x3: np.ndarray,
    ref_xyz_24x3: np.ndarray = COLORCHECKER_REF_XYZ,
    white_ref_xyz: np.ndarray = None,
) -> tuple:
    """Converts XYZ to CIELAB using White Patch 19 for illuminant white point adaptation.
    Computes CIEDE2000 color errors.
    """
    if colour is not None:
        try:
            target_white_xyz = (
                white_ref_xyz
                if white_ref_xyz is not None
                else measured_xyz_24x3[18]
            )
            white_xy_meas = colour.XYZ_to_xy(target_white_xyz)

            scale = ref_xyz_24x3[18, 1] / max(measured_xyz_24x3[18, 1], 1e-6)
            scaled_meas_xyz = measured_xyz_24x3 * scale

            lab_ref = colour.XYZ_to_Lab(ref_xyz_24x3, illuminant=white_xy_meas)
            lab_meas = colour.XYZ_to_Lab(scaled_meas_xyz, illuminant=white_xy_meas)

            delta_e_list = colour.delta_E(lab_meas, lab_ref, method="CIE 2000")
            mean_de = float(np.mean(delta_e_list))
            std_de = float(
                np.std(delta_e_list, ddof=1) if len(delta_e_list) > 1 else 0.0
            )
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
    gains, balanced_means = compute_white_balance_gains(
        patch_means_24x3, neutral_patch_idx=20
    )

    srgb_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    baseline_xyz = balanced_means @ srgb_matrix.T
    white_base_xyz = baseline_xyz[18]
    mean_de_before, std_de_before, _, _ = evaluate_delta_e(
        baseline_xyz, white_ref_xyz=white_base_xyz
    )

    CCM = fit_color_correction_matrix(balanced_means)
    fitted_xyz = balanced_means @ CCM.T
    white_fit_xyz = fitted_xyz[18]
    mean_de_after, std_de_after, worst_de_after, _ = evaluate_delta_e(
        fitted_xyz, white_ref_xyz=white_fit_xyz
    )

    return {
        "de_mean_before": [mean_de_before, std_de_before],
        "de_mean_after": [mean_de_after, std_de_after],
        "de_worst_after": [worst_de_after, 0.0],
        "ccm_matrix": CCM.tolist(),
    }

def evaluate_ccm_column_cosine_similarity(
    color_file_paths: list,
    bboxes_dict: dict = None,
    white_sheet_path: str = None,
    black_level: float = None,
    white_level: float = None,
    neutral_patch_idx: int = 20,
    verbose: bool = True,
) -> dict:
    """
    Fits an independent 3x3 Color Correction Matrix (CCM) for each DNG capture in
    `color_file_paths` and compares their columns as 3D direction vectors using
    cosine similarity (Part 3, Item 9).
    """
    if bboxes_dict is None:
        bboxes_dict = COLORCHECKER_BBOXES

    ws_rgb = None
    if white_sheet_path and os.path.exists(white_sheet_path):
        ws_w, ws_b, _ = check_raw_clipping(white_sheet_path)
        ws_rgb = extract_quarter_res_linear_rgb(
            white_sheet_path,
            black_level=black_level if black_level is not None else ws_b,
            white_level=white_level if white_level is not None else ws_w,
        )

    fitted_ccms = []
    chart_names = []

    for path in sorted(color_file_paths):
        if not isinstance(path, str) or not os.path.exists(path):
            continue

        fname = os.path.basename(path)
        bbox = bboxes_dict.get(fname, [480, 1500, 400, 1100])

        file_w, file_b, _ = check_raw_clipping(path)
        b_lvl = black_level if black_level is not None else file_b
        w_lvl = white_level if white_level is not None else file_w

        sc_rgb = extract_quarter_res_linear_rgb(
            path, black_level=b_lvl, white_level=w_lvl
        )
        if ws_rgb is not None:
            sc_rgb = apply_lens_shading_correction(sc_rgb, ws_rgb)

        r_means = extract_colorchecker_manual(sc_rgb, grid_bbox=bbox)
        alg_means = align_patch_means(r_means)
        _, bal_means = compute_white_balance_gains(
            alg_means, neutral_patch_idx=neutral_patch_idx
        )

        ccm_i = fit_color_correction_matrix(bal_means, COLORCHECKER_REF_XYZ)
        fitted_ccms.append(ccm_i)
        chart_names.append(fname)

        if verbose:
            print(f"Fitted 3x3 CCM for {fname}:")
            print(np.round(ccm_i, 4))
            print()

    channel_names = [
        "Col 1 (Red Map)",
        "Col 2 (Green Map)",
        "Col 3 (Blue Map)",
    ]
    summary_rows = []
    pairwise_details = {}

    for i in range(len(fitted_ccms)):
        for j in range(i + 1, len(fitted_ccms)):
            mat_a, mat_b = fitted_ccms[i], fitted_ccms[j]
            name_a, name_b = chart_names[i], chart_names[j]
            pair_key = f"{name_a} vs. {name_b}"

            if verbose:
                print("==================================================")
                print(f" MATH BREAKDOWN: {name_a} vs {name_b}")
                print("==================================================")

            sims = []
            angles_deg = []

            for col_idx in range(3):
                v1 = mat_a[:, col_idx]
                v2 = mat_b[:, col_idx]

                norm_v1 = float(np.linalg.norm(v1))
                norm_v2 = float(np.linalg.norm(v2))
                dot_prod = float(np.dot(v1, v2))

                denom = max(norm_v1 * norm_v2, 1e-12)
                cos_sim = float(np.clip(dot_prod / denom, -1.0, 1.0))
                angle_deg = float(np.degrees(np.arccos(cos_sim)))

                sims.append(cos_sim)
                angles_deg.append(angle_deg)

                if verbose:
                    print(f"--- {channel_names[col_idx]} ---")
                    print(
                        f"  {name_a} vector : [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}] "
                        f"(Norm: {norm_v1:.4f})"
                    )
                    print(
                        f"  {name_b} vector : [{v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f}] "
                        f"(Norm: {norm_v2:.4f})"
                    )
                    print(f"  Dot Product    : {dot_prod:.6f}")
                    print(
                        f"  Cosine Similarity (v1 · v2 / ||v1|| ||v2||) : "
                        f"{cos_sim:.4f} ({angle_deg:.2f}°)"
                    )
                    print()

            summary_rows.append((pair_key, sims[0], sims[1], sims[2]))
            pairwise_details[pair_key] = {
                "cosine_similarities": sims,
                "angles_deg": angles_deg,
            }

    if summary_rows:
        print("\n==========================================================")
        print("     SUMMARY TABLE: COLUMN-WISE COSINE SIMILARITY        ")
        print("==========================================================")
        print(
            f"{'Comparison Pair':<28} | {'Col 1 (Red)':<12} | "
            f"{'Col 2 (Green)':<12} | {'Col 3 (Blue)':<12}"
        )
        print("-" * 73)
        for pair_label, col1_sim, col2_sim, col3_sim in summary_rows:
            print(
                f"{pair_label:<28} | {col1_sim:<12.4f} | "
                f"{col2_sim:<12.4f} | {col3_sim:<12.4f}"
            )
        print("----------------------------------------------------------")

    return {
        "chart_names": chart_names,
        "ccms": fitted_ccms,
        "summary_rows": summary_rows,
        "pairwise_details": pairwise_details,
    }

def evaluate_multi_chart_de(
    color_file_paths: list,
    CCM: np.ndarray,
    illuminant_name: str = "LED",
    white_sheet_path: str = None,
    bboxes_dict: dict = None,
    black_level: float = None,
    white_level: float = None,
) -> dict:
    """Evaluates mean and worst-patch CIEDE2000 [mean, std] across repeated charts."""
    if bboxes_dict is None:
        bboxes_dict = COLORCHECKER_BBOXES

    patch_names = get_reference_patch_names()
    ws_rgb = (
        extract_quarter_res_linear_rgb(white_sheet_path, black_level=black_level, white_level=white_level)
        if white_sheet_path and os.path.exists(white_sheet_path)
        else None
    )

    capture_means, capture_worsts, all_de = [], [], []
    print(f"--- Per-Capture {illuminant_name} Results (After CCM) ---")

    for path in sorted(color_file_paths):
        if not os.path.exists(path):
            continue
        fname = os.path.basename(path)
        bbox = bboxes_dict.get(fname, [480, 1500, 400, 1100])

        sc_rgb = extract_quarter_res_linear_rgb(path, black_level=black_level, white_level=white_level)
        if ws_rgb is not None:
            sc_rgb = apply_lens_shading_correction(sc_rgb, ws_rgb)

        alg_means = align_patch_means(extract_colorchecker_manual(sc_rgb, grid_bbox=bbox))
        _, bal_means = compute_white_balance_gains(alg_means, neutral_patch_idx=20)

        corrected_xyz = bal_means @ CCM.T
        m_de, _, w_de, de_list = evaluate_delta_e(corrected_xyz, COLORCHECKER_REF_XYZ)

        capture_means.append(m_de)
        capture_worsts.append(w_de)
        all_de.append(de_list)
        print(f"{fname:<14} | Mean ΔE_00 = {m_de:.2f} | Worst = {w_de:.2f} ({patch_names[int(np.argmax(de_list))]})")

    n_charts = len(capture_means)
    mean_pair = [float(np.mean(capture_means)), float(np.std(capture_means, ddof=1)) if n_charts > 1 else 0.0]
    worst_pair = [float(np.mean(capture_worsts)), float(np.std(capture_worsts, ddof=1)) if n_charts > 1 else 0.0]

    mean_per_patch = np.mean(np.asarray(all_de), axis=0)
    top3 = [(patch_names[i], float(mean_per_patch[i])) for i in np.argsort(mean_per_patch)[::-1][:3]]

    print(f"\n--- Summary Across All {n_charts} {illuminant_name} Captures ---")
    print(f"Mean Patch ΔE_00 ({n_charts}-chart mean ± std) : {mean_pair[0]:.2f} ± {mean_pair[1]:.2f}")
    print(f"Worst Patch ΔE_00 ({n_charts}-chart mean ± std): {worst_pair[0]:.2f} ± {worst_pair[1]:.2f}")

    print(f"\n--- Top 3 Worst Patches (Averaged Across All {n_charts} {illuminant_name} Captures) ---")
    for rank, (name, val) in enumerate(top3, start=1):
        print(f"{rank}. {name:<22} | Mean ΔE_00 = {val:.2f}")

    return {"de_mean_after": mean_pair, "de_worst_after": worst_pair, "top3_worst": top3}

def inspect_capture_diagnostics(
    color_file_paths: list,
    CCM: np.ndarray = None,
    white_sheet_path: str = None,
    bboxes_dict: dict = None,
    inspect_indices: list = None,
    black_level: float = None,
    white_level: float = None,
    neutral_patch_idx: int = 20,
) -> list[dict]:
    """
    Audits raw clipping, matrix condition numbers (cond(bal), cond(CCM)),
    and per-patch RGB/ΔE_00 values across any list of ColorChecker DNGs.
    Uses a fixed CCM if provided; otherwise fits an individual CCM per capture.
    """
    if bboxes_dict is None:
        bboxes_dict = COLORCHECKER_BBOXES
    if inspect_indices is None:
        inspect_indices = [14, 17, 23]  # 0-indexed: 15. Red, 18. Cyan, 24. Black 2

    patch_names = get_reference_patch_names()
    ws_rgb = (
        extract_quarter_res_linear_rgb(white_sheet_path, black_level=black_level, white_level=white_level)
        if white_sheet_path and os.path.exists(white_sheet_path)
        else None
    )

    diagnostics = []
    for path in sorted(color_file_paths):
        if not os.path.exists(path):
            continue
        fname = os.path.basename(path)
        bbox = bboxes_dict.get(fname, [480, 1500, 400, 1100])

        sat_info = analyze_raw_saturation(path)
        sc_rgb = extract_quarter_res_linear_rgb(path, black_level=black_level, white_level=white_level)
        if ws_rgb is not None:
            sc_rgb = apply_lens_shading_correction(sc_rgb, ws_rgb)

        aligned = align_patch_means(extract_colorchecker_manual(sc_rgb, grid_bbox=bbox))
        _, bal = compute_white_balance_gains(aligned, neutral_patch_idx=neutral_patch_idx)

        ccm_i = CCM if CCM is not None else fit_color_correction_matrix(bal, COLORCHECKER_REF_XYZ)
        corr_xyz = bal @ ccm_i.T
        _, _, _, de_i = evaluate_delta_e(corr_xyz, COLORCHECKER_REF_XYZ)

        cond_bal = float(np.linalg.cond(bal))
        cond_ccm = float(np.linalg.cond(ccm_i))

        print(
            f"\n=== {fname} | clipped={sat_info['is_clipped']} (v_sat={sat_info['v_sat']}) "
            f"| cond(bal)={cond_bal:.2f} | cond(CCM)={cond_ccm:.2f} ==="
        )
        for idx in inspect_indices:
            r, g, b = aligned[idx]
            rb, gb, bb = bal[idx]
            print(
                f"  {patch_names[idx]:<22} | aligned RGB=[{r:.4f}, {g:.4f}, {b:.4f}] "
                f"| bal RGB=[{rb:.4f}, {gb:.4f}, {bb:.4f}] | ΔE_00={de_i[idx]:.3f}"
            )

        diagnostics.append({
            "filename": fname,
            "is_clipped": sat_info["is_clipped"],
            "v_sat": sat_info["v_sat"],
            "cond_bal": cond_bal,
            "cond_ccm": cond_ccm,
            "delta_e": de_i,
        })

    return diagnostics