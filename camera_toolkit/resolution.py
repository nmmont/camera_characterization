"""
Slanted-edge MTF50, spatial resolution analysis, ESF/LSF derivation,
and diffraction limits for camera characterization.
"""
import os
import cv2
import numpy as np
import contextlib
import io
import matplotlib.pyplot as plt

try:
    import rawpy
except ImportError:
    rawpy = None


def extract_green_roi(
    raw_visible_array: np.ndarray,
    roi_coords: dict,
    preview_scale: float = 2.0,
    row_offset: int = 0,
    col_offset: int = 0,
    rotate_k: int = -1,
) -> np.ndarray:
    """
    Translates portrait preview selection coordinates onto the native unrotated
    landscape sensor matrix (e.g., 3072 x 4080), crops a single-plane green
    channel without inter-plane averaging, and rotates it into MTF50 orientation.

    Returns a 2D float64 array in raw Digital Numbers (DN).
    """
    arr = np.asarray(raw_visible_array, dtype=np.float64)
    max_h, max_w = arr.shape

    scale = float(preview_scale)
    py = roi_coords.get("y", 0) * scale
    px = roi_coords.get("x", 0) * scale
    ph = roi_coords.get("h", 60) * scale
    pw = roi_coords.get("w", 30) * scale

    raw_x = int(np.clip(round(py), 0, max(0, max_w - 1)))
    raw_y = int(np.clip(round(max_h - px - pw), 0, max(0, max_h - 1)))
    raw_w = int(np.clip(round(ph), 8, max(8, max_w - raw_x)))
    raw_h = int(np.clip(round(pw), 8, max(8, max_h - raw_y)))

    raw_roi = arr[raw_y : raw_y + raw_h, raw_x : raw_x + raw_w]
    if raw_roi.size == 0 or raw_roi.shape[0] < 8 or raw_roi.shape[1] < 8:
        return np.zeros((64, 64), dtype=np.float64)

    # Extract a single green plane (2x2 Bayer subsampling)
    r0 = int(row_offset) % 2
    c0 = int(col_offset) % 2
    green_roi = raw_roi[r0::2, c0::2]

    if rotate_k != 0:
        green_roi = np.rot90(green_roi, k=rotate_k)

    return green_roi

def normalize_view(
    roi_dn: np.ndarray,
    black_level: float,
    white_level: float,
    clip: bool = True,
    preview_scale = 2.0
) -> np.ndarray:
    """
    Converts a raw DN ROI into linear light normalized to [0.0, 1.0] using
    black_level and white_level.
    """
    roi = np.asarray(roi_dn, dtype=np.float64)
    span = max(1e-6, float(white_level) - float(black_level))
    normalized = (roi - float(black_level)) / span
    if clip:
        return np.clip(normalized, 0.0, 1.0)
    return normalized

def measure_slant_edge_angle(green_roi: np.ndarray, label: str = "") -> dict:
    """
    Estimates the angle of a predominantly vertical slanted edge in a 2D
    green-channel ROI (already oriented/rotated as returned by
    `extract_linear_green_roi`).

    Returns a dict with slope (dx/dy), angle from vertical in degrees,
    angle from horizontal in degrees, and R^2 goodness-of-fit.
    """
    roi = np.asarray(green_roi, dtype=np.float64)
    h, w = roi.shape

    if h < 4 or w < 4:
        print(f"[{label}] ROI too small ({h}x{w}) to measure edge angle.")
        return {}

    # 1. Compute row-wise horizontal derivative (signed so dark-to-bright and
    #    bright-to-dark edges both work) and locate peak transition per row
    dx_profile = np.diff(roi, axis=1)  # shape (h, w - 1)
    peak_cols = np.argmax(np.abs(dx_profile), axis=1)

    ys = []
    xs_subpix = []

    # 2. Sub-pixel refinement around each row's peak gradient using a small window
    half_win = 2
    for r in range(h):
        c_peak = int(peak_cols[r])
        c_start = max(0, c_peak - half_win)
        c_end = min(dx_profile.shape[1], c_peak + half_win + 1)

        # Use magnitude weights shifted above local baseline so both positive and
        # negative derivatives behave identically
        weights = np.abs(dx_profile[r, c_start:c_end])
        weight_sum = np.sum(weights)
        if weight_sum <= 1e-12:
            continue

        # +0.5 aligns np.diff index c with pixel boundary between col c and c+1
        col_positions = np.arange(c_start, c_end, dtype=np.float64) + 0.5
        x_sub = float(np.sum(col_positions * weights) / weight_sum)

        ys.append(float(r))
        xs_subpix.append(x_sub)

    ys = np.asarray(ys, dtype=np.float64)
    xs_subpix = np.asarray(xs_subpix, dtype=np.float64)

    if len(ys) < 2:
        print(f"[{label}] Could not find enough valid edge points across rows.")
        return {}

    # 3. Fit x(y) = slope * y + intercept, where slope = dx / dy
    slope_dx_dy, intercept = np.polyfit(ys, xs_subpix, 1)

    # Angle relative to the vertical (y) axis:
    # slope = dx / dy = tan(theta_from_vertical)
    angle_from_vertical_deg = float(np.degrees(np.arctan(slope_dx_dy)))
    angle_from_horizontal_deg = float(90.0 - abs(angle_from_vertical_deg))

    # Goodness of fit (R^2) to flag noisy or non-linear edge fits
    x_pred = slope_dx_dy * ys + intercept
    ss_res = float(np.sum((xs_subpix - x_pred) ** 2))
    ss_tot = float(np.sum((xs_subpix - np.mean(xs_subpix)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

    prefix = f"[{label}] " if label else ""
    print(
        f"{prefix}Slanted Edge Angle: {angle_from_vertical_deg:+.2f}° from vertical "
        f"({angle_from_horizontal_deg:.2f}° from horizontal) | "
        f"dx/dy = {slope_dx_dy:+.4f} | R² = {r_squared:.4f}"
    )

    return {
        "slope_dx_dy": float(slope_dx_dy),
        "intercept": float(intercept),
        "angle_from_vertical_deg": angle_from_vertical_deg,
        "angle_from_horizontal_deg": angle_from_horizontal_deg,
        "r_squared": r_squared,
    }

def check_green_roi_clipping(
    green_roi: np.ndarray,
    black_level: float = 63.95,
    white_level: float = 1023.0,
    sat_margin: float = 5.0,
    label: str = "Green ROI",
) -> dict:
    """
    Checks a 2D green-channel ROI array (in DN) for highlight saturation
    and shadow clipping.
    """
    roi = np.asarray(green_roi, dtype=np.float64)
    total_px = roi.size
    if total_px == 0:
        print(f"[{label}] Empty green ROI.")
        return {"label": label, "is_clipped": True, "is_clean": False}

    min_dn = float(np.min(roi))
    max_dn = float(np.max(roi))

    sat_threshold = float(white_level) - float(sat_margin)
    high_clip_px = int(np.sum(roi >= sat_threshold))
    low_clip_px = int(np.sum(roi <= float(black_level)))

    high_pct = (high_clip_px / total_px) * 100.0
    low_pct = (low_clip_px / total_px) * 100.0
    is_clipped = (high_clip_px > 0) or (low_clip_px > 0)

    status = "!!CLIPPED!!" if is_clipped else "NO CLIPPING"
    print(f"[{label}] {status}")
    print(
        f"  Range: [{min_dn:.2f}, {max_dn:.2f}] DN ({roi.shape[1]}x{roi.shape[0]} px) | "
        f"Black: {black_level:.2f} DN | White: {white_level:.1f} DN"
    )
    if is_clipped:
        print(f"  High Clip (>={sat_threshold:.1f} DN): {high_clip_px} px ({high_pct:.2f}%)")
        print(f"  Low Clip  (<={black_level:.2f} DN): {low_clip_px} px ({low_pct:.2f}%)")

    return {
        "label": label,
        "is_clipped": is_clipped,
        "is_clean": not is_clipped,
        "shape": roi.shape,
        "min_dn": min_dn,
        "max_dn": max_dn,
        "high_clip_px": high_clip_px,
        "high_clip_pct": high_pct,
        "low_clip_px": low_clip_px,
        "low_clip_pct": low_pct,
    }

def check_green_roi_edge_margins(
    green_roi: np.ndarray,
    min_margin_px: float = 12.0,
    label: str = "Green ROI",
    show_plot: bool = True,
) -> dict:
    """
    Estimates row-wise slanted-edge positions across a 2D green-channel ROI,
    checks horizontal margin from the left and right ROI borders, and optionally
    plots the ROI with detected edge points.
    """
    roi = np.asarray(green_roi, dtype=np.float64)
    h, w = roi.shape
    if h < 4 or w < 4:
        print(f"!!! [{label}] Green ROI too small ({h}x{w}) for edge margin check.")
        return {
            "label": label,
            "edge_ok": False,
            "min_edge_x": np.nan,
            "max_edge_x": np.nan,
            "left_margin_px": 0.0,
            "right_margin_px": 0.0,
            "edge_reason": "Green ROI too small for edge margin check",
        }

    # Edge-pad horizontally before smoothing so bright/dark plateaus don't drop to 0 at borders
    kx = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    ky = np.array([0.2, 0.6, 0.2], dtype=np.float64)

    pad_x = len(kx) // 2
    padded_x = np.pad(roi, ((0, 0), (pad_x, pad_x)), mode="edge")
    smoothed = np.apply_along_axis(lambda r: np.convolve(r, kx, mode="valid"), 1, padded_x)

    pad_y = len(ky) // 2
    padded_y = np.pad(smoothed, ((pad_y, pad_y), (0, 0)), mode="edge")
    smoothed = np.apply_along_axis(lambda c: np.convolve(c, ky, mode="valid"), 0, padded_y)

    grad_x = np.abs(np.diff(smoothed, axis=1))  # shape (h, w - 1)

    edge_x_rows = []
    edge_y_rows = []
    half_win = 2
    search_start = 1
    search_end = max(search_start + 1, grad_x.shape[1] - 1)

    for r in range(h):
        row_grad = grad_x[r]
        c_peak = search_start + int(np.argmax(row_grad[search_start:search_end]))
        c0 = max(0, c_peak - half_win)
        c1 = min(len(row_grad), c_peak + half_win + 1)
        weights = row_grad[c0:c1]
        w_sum = float(np.sum(weights))
        if w_sum > 1e-12:
            cols = np.arange(c0, c1, dtype=np.float64) + 0.5
            edge_x_rows.append(float(np.sum(cols * weights) / w_sum))
            edge_y_rows.append(float(r))

    edge_x_arr = np.asarray(edge_x_rows, dtype=np.float64)
    edge_y_arr = np.asarray(edge_y_rows, dtype=np.float64)

    if len(edge_x_arr) < 2:
        return {
            "label": label,
            "edge_ok": False,
            "min_edge_x": np.nan,
            "max_edge_x": np.nan,
            "left_margin_px": 0.0,
            "right_margin_px": 0.0,
            "edge_reason": "Could not reliably locate edge across rows",
        }

    med_x = float(np.median(edge_x_arr))
    inlier_mask = np.abs(edge_x_arr - med_x) <= 10.0
    inliers_x = edge_x_arr[inlier_mask] if np.sum(inlier_mask) >= 2 else edge_x_arr
    inliers_y = edge_y_arr[inlier_mask] if np.sum(inlier_mask) >= 2 else edge_y_arr

    min_edge_x = float(np.min(inliers_x))
    max_edge_x = float(np.max(inliers_x))

    left_margin_px = min_edge_x
    right_margin_px = float((w - 1) - max_edge_x)

    edge_ok = (left_margin_px >= min_margin_px) and (right_margin_px >= min_margin_px)
    reason = (
        "OK"
        if edge_ok
        else (
            f"Tight/clipping edge margin "
            f"(left={left_margin_px:.1f} px, right={right_margin_px:.1f} px, min_req={min_margin_px:.1f} px)"
        )
    )

    status = "MARGIN OK" if edge_ok else "!TIGHT MARGIN!"
    print(f"[{label}] {status}")
    print(
        f"  Edge X Range: [{min_edge_x:.1f}, {max_edge_x:.1f}] px ({w}x{h} px) | "
        f"Margins (L/R): {left_margin_px:.1f} / {right_margin_px:.1f} px"
    )
    if not edge_ok:
        print(f"  Warning: {reason}")

    if show_plot:
        fig, (ax_img, ax_prof) = plt.subplots(1, 2, figsize=(10, 4.2))

        ax_img.imshow(roi, cmap="gray", origin="upper")
        ax_img.plot(inliers_x, inliers_y, "c.", markersize=3, label="Inlier edge pts")
        outlier_x = edge_x_arr[~inlier_mask]
        outlier_y = edge_y_arr[~inlier_mask]
        if len(outlier_x) > 0:
            ax_img.plot(outlier_x, outlier_y, "rx", markersize=4, label="Outlier pts")
        ax_img.axvline(min_margin_px, color="y", linestyle="--", linewidth=1, label=f"Min margin ({min_margin_px}px)")
        ax_img.axvline((w - 1) - min_margin_px, color="y", linestyle="--", linewidth=1)
        ax_img.set_title(f"{label} — Received ROI ({w}x{h})")
        ax_img.set_xlabel("X [px]")
        ax_img.set_ylabel("Y [px]")
        ax_img.legend(loc="best", fontsize=7)

        mean_grad_x = np.mean(grad_x, axis=0)
        x_grad_axis = np.arange(len(mean_grad_x)) + 0.5
        ax_prof.plot(x_grad_axis, mean_grad_x, color="tab:blue", linewidth=1.5)
        ax_prof.axvspan(med_x - 10.0, med_x + 10.0, color="tab:green", alpha=0.15, label="Median ±10px window")
        ax_prof.set_title("Mean |d(ROI)/dX| Across Rows")
        ax_prof.set_xlabel("X [px]")
        ax_prof.set_ylabel("Mean |ΔDN|")
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(loc="best", fontsize=7)

        plt.tight_layout()
        plt.show()

    return {
        "label": label,
        "edge_ok": edge_ok,
        "shape": roi.shape,
        "min_edge_x": min_edge_x,
        "max_edge_x": max_edge_x,
        "left_margin_px": left_margin_px,
        "right_margin_px": right_margin_px,
        "edge_reason": reason,
    }

# Context Manager to Suppress Noisy Helper Prints. the print 
@contextlib.contextmanager
def _silence_prints(enabled: bool = True):
    """
    Temporarily redirects stdout/stderr into memory buffers so individual
    helper functions don't clutter the notebook output, while still allowing
    Matplotlib figures (plt.show()) to render normally.
    """
    if not enabled:
        yield
        return
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield

def run_qc_audit(
    dng_paths: list,
    roi_coords: dict,
    black_level: float = None,
    white_level: float = None,
    sat_margin: float = 5.0,
    min_margin_px: float = 20.0,
    min_r_squared: float = 0.90,
    preview_scale: float = 2.0,
    rotate_k: int = -1,
    show_edge_plots: bool = False,
    silence_helpers: bool = True,
) -> dict:
    """
    Audits a list of raw DNG slanted-edge captures across three ISO 12233
    quality-control criteria before running MTF50 calculations:
      1. Edge Angle & Line Linearity (R² >= min_r_squared)
      2. Dynamic Range Clipping (no highlight saturation or floor truncation)
      3. Left/Right ESF Tail Margins (>= min_margin_px on both sides of the edge)
    """
    if not dng_paths:
        raise ValueError("dng_paths list is empty.")

    passed_paths = [] # Stores file paths that pass all 3 QC gates
    rows = [] # Stores detailed per-file diagnostic dictionaries

    for path in dng_paths:
        fname = os.path.basename(path)

        # --- Step A: Load Raw Bayer Sensor Array & Pedestal/Ceiling Levels ---
        with rawpy.imread(path) as raw:
            bayer = raw.raw_image_visible.astype(np.float64)
            # Fall back to DNG metadata black/white levels if not explicitly passed
            b_lvl = (
                float(np.mean(raw.black_level_per_channel))
                if black_level is None
                else float(black_level)
            )
            w_lvl = (
                float(raw.white_level)
                if white_level is None
                else float(white_level)
            )
        
        # --- Step B: Extract Un-Demosaiced Single Green-Plane ROI ---
        # Crops the target ROI from a single green subgrid (no demosaic interpolation)
        # and rotates it so the slanted edge is oriented predominantly vertically
        green_roi = extract_green_roi(
            raw_visible_array=bayer,
            roi_coords=roi_coords,
            preview_scale=preview_scale,
            rotate_k=rotate_k,
        )

        # --- Step C: Run the 3 ISO 12233 Preflight Checks ---
        # Run helper checks with stdout/stderr silenced (plt.show still works)
        with _silence_prints(enabled=silence_helpers):
            # 1. Measure edge slant angle (dx/dy) and straight-line R² fit quality
            angle_info = measure_slant_edge_angle(green_roi, label=fname)

            # 2. Check for pixel clipping near white_level or below the black floor
            clip_res = check_green_roi_clipping(
                green_roi=green_roi,
                black_level=b_lvl,
                white_level=w_lvl,
                sat_margin=sat_margin,
                label=fname,
            )

            # 3. Verify sufficient flat dark/bright tail room on left & right of the edge
            edge_res = check_green_roi_edge_margins(
                green_roi=green_roi,
                min_margin_px=min_margin_px,
                label=fname,
                show_plot=show_edge_plots,
            )

        # --- Step D: Evaluate Boolean Pass/Fail Flags ---
        angle_deg = angle_info.get("angle_from_vertical_deg", np.nan)
        r_sq = angle_info.get("r_squared", np.nan)
        angle_ok = np.isfinite(angle_deg) and (np.isnan(r_sq) or r_sq >= min_r_squared)

        clip_ok = bool(clip_res.get("is_clean", not clip_res.get("is_clipped", True)))
        edge_ok = bool(edge_res.get("edge_ok", False))

        # Only accept captures that pass all three checks simultaneously
        all_ok = clip_ok and edge_ok and angle_ok
        if all_ok:
            passed_paths.append(path)

        # --- Step E: Format & Print Per-Capture Diagnostic Report ---
        if np.isfinite(angle_deg):
            r_part = f", R²={r_sq:.4f}" if np.isfinite(r_sq) else ""
            angle_str = f"{'PASS' if angle_ok else 'FAIL'} ({angle_deg:+.2f}°{r_part})"
        else:
            angle_str = "FAIL (N/A)"

        min_dn = clip_res.get("min_dn", np.nan)
        max_dn = clip_res.get("max_dn", np.nan)
        hi_px = clip_res.get("high_clip_px", 0)
        lo_px = clip_res.get("low_clip_px", 0)
        clip_str = (
            f"{'PASS' if clip_ok else 'FAIL'} | Range: [{min_dn:.1f}, {max_dn:.1f}] DN | H:{hi_px}, L:{lo_px}"
        )

        left_m = edge_res.get("left_margin_px", np.nan)
        right_m = edge_res.get("right_margin_px", np.nan)
        edge_str = (
            f"{'PASS' if edge_ok else 'FAIL'} | L/R Margins: {left_m:.1f} / {right_m:.1f} px (min {min_margin_px:.1f} px)"
        )

        print(f"\n{fname}")
        print(f"  Angle     : {angle_str}")
        print(f"  Clip Info : {clip_str}")
        print(f"  Edge Info : {edge_str}")

        rows.append({
            "filename": fname,
            "path": path,
            "passed": all_ok,
            "clip": clip_res,
            "edge": edge_res,
            "angle": angle_info,
        })

    # Print final summary count of valid captures ready for MTF50 processing
    print(f"\nPassed quality control: {len(passed_paths)}/{len(dng_paths)}")

    return {
        "passed_paths": passed_paths,
        "rows": rows,
    }

def edge_centroids(roi: np.ndarray, window: float = 8.0, smooth: float = 1.5) -> np.ndarray:
    """
    Sub-pixel edge position using a windowed derivative centroid.
    Dynamically identifies edge orientation to support both vertical and horizontal slanted edges.
    """
    roi = np.asarray(roi, dtype=float)
    h, w = roi.shape

    if h < 4 or w < 4:
        return np.full(h, np.nan)

    #Check if edge is vertical or horizontal
    deriv_col = np.abs(np.diff(roi, axis=1))
    deriv_row = np.abs(np.diff(roi, axis=0))
    is_vertical_edge = deriv_col.sum() > deriv_row.sum()

    #Apply 1D smoother across edge direction to locate the coarse peak
    #Computes sub-pixel centroid using raw unsmoothed gradients within 8px window
    if is_vertical_edge:
        smoothed = cv2.GaussianBlur(roi, (0, 0), sigmaX=smooth, sigmaY=0.01)
        coarse = np.argmax(np.abs(np.diff(smoothed, axis=1)), axis=1) + 0.5
        deriv = np.abs(np.diff(roi, axis=1))
        x_space = np.arange(deriv.shape[1]) + 0.5
        weight = np.where(np.abs(x_space[None, :] - coarse[:, None]) <= window, deriv, 0.0)

        total = weight.sum(axis=1)
        good = total > 1e-9
        centers = np.full(h, np.nan)
        centers[good] = (weight[good] * x_space).sum(axis=1) / total[good]
    else:
        smoothed = cv2.GaussianBlur(roi, (0, 0), sigmaX=0.01, sigmaY=smooth)
        coarse = np.argmax(np.abs(np.diff(smoothed, axis=0)), axis=0) + 0.5
        deriv = np.abs(np.diff(roi, axis=0))
        y_space = np.arange(deriv.shape[0]) + 0.5
        weight = np.where(np.abs(y_space[:, None] - coarse[None, :]) <= window, deriv, 0.0)

        total = weight.sum(axis=0)
        good = total > 1e-9
        centers = np.full(w, np.nan)
        centers[good] = (weight[:, good] * y_space[:, None]).sum(axis=0) / total[good]

    return centers

def slanted_edge_mtf(roi: np.ndarray, bin_width: float = 0.25, window: float = 8.0) -> tuple:
    """
    Measures MTF from a slanted edge ROI with true DC area normalization and grid resampling.
    """
    roi = np.asarray(roi, dtype=float)
    h, w = roi.shape

    if h < 4 or w < 4:
        return np.array([]), np.array([]), np.array([]), np.array([])

    deriv_col = np.abs(np.diff(roi, axis=1))
    deriv_row = np.abs(np.diff(roi, axis=0))
    is_vertical_edge = deriv_col.sum() > deriv_row.sum()

    # Applies 1D smoothing across the edge direction to locate a coarse peak (coarse), then computes a sub-pixel centroid using raw unsmoothed gradients
    centers = edge_centroids(roi, window=window)
    ok = np.isfinite(centers)
    if not np.any(ok) or np.sum(ok) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # linear edge fit
    if is_vertical_edge:
        independent_axis = np.arange(h)
        dependent_axis = np.arange(w)
        slope, intercept = np.polyfit(independent_axis[ok], centers[ok], 1) 
        fitted = slope * independent_axis + intercept
        distance = (dependent_axis[None, :] - fitted[:, None]).ravel()
    else:
        independent_axis = np.arange(w)
        dependent_axis = np.arange(h)
        slope, intercept = np.polyfit(independent_axis[ok], centers[ok], 1)
        fitted = slope * independent_axis + intercept
        distance = (dependent_axis[:, None] - fitted[None, :]).ravel()

    # Bins all 2D pixels into 0.25 px (4× oversampled) bins along the edge normal
    # Averages each bin (total / count), and interpolates onto a uniform 0.25 px grid (esf)
    values = roi.ravel()
    edges = np.arange(distance.min(), distance.max() + bin_width, bin_width)
    idx = np.clip(np.digitize(distance, edges) - 1, 0, len(edges) - 2)
    total = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    count = np.bincount(idx, minlength=len(edges) - 1)
    keep = count > 0

    if not np.any(keep):
        return np.array([]), np.array([]), np.array([]), np.array([])

    esf_raw = total[keep] / count[keep]
    esf_x_raw = (edges[:-1] + bin_width / 2)[keep]

    esf_x = np.arange(esf_x_raw.min(), esf_x_raw.max(), bin_width)
    esf = np.interp(esf_x, esf_x_raw, esf_raw)

    # Differentiates the Edge Spread Function into the Line Spread Function (lsf)
    # flipping sign if the edge goes bright-to-dark (np.abs(np.min(lsf)) > np.max(lsf)).
    lsf = np.gradient(esf, bin_width)

    if np.abs(np.min(lsf)) > np.max(lsf):
        lsf = -lsf

    # Subtracts residual background slope from the outer 10% tails (baseline)
    # Shifts the LSF peak to the center (np.roll)
    # And multiplies by a np.hanning window
    tail_len = max(3, len(lsf) // 10)
    baseline = np.median(np.concatenate([lsf[:tail_len], lsf[-tail_len:]]))
    lsf = lsf - baseline

    peak_idx = np.argmax(lsf)
    shift = (len(lsf) // 2) - peak_idx
    lsf_centered = np.roll(lsf, shift)

    lsf_windowed = lsf_centered * np.hanning(len(lsf_centered))

    # Zero-pads to 512 samples for smooth frequency resolution
    # Computes the real Fourier magnitude (np.abs(np.fft.rfft(...)))
    # And normalizes by window_area (DC = 1.0)
    pad_len = 512
    if len(lsf_windowed) < pad_len:
        pad_width = pad_len - len(lsf_windowed)
        lsf_padded = np.pad(lsf_windowed, (pad_width // 2, pad_width - (pad_width // 2)), mode="constant")
    else:
        lsf_padded = lsf_windowed

    spectrum = np.abs(np.fft.rfft(lsf_padded))
    window_area = np.sum(lsf_windowed)
    mtf_norm = spectrum / window_area if window_area > 1e-9 else spectrum

    freq = np.fft.rfftfreq(len(lsf_padded), d=bin_width)
    return freq, mtf_norm, esf_x, esf

def compute_mtf50(freq: np.ndarray, mtf: np.ndarray) -> float:
    """Finds frequency where MTF drops to 0.5 via linear interpolation."""
    # Locates where MTF first drops below 0.50 (np.where(mtf < 0.5))
    # linearly interpolates the exact MTF50 frequency (cy/green-px)
    # then scales by 1 / (2 * pixel_pitch_um) to convert to cy/µm and cy/mm. 
    if len(freq) == 0 or len(mtf) == 0:
        return np.nan
    below = np.where(mtf < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    return float(freq[i - 1] + (mtf[i - 1] - 0.5) * (freq[i] - freq[i - 1]) / (mtf[i - 1] - mtf[i]))

def measure_single_frame_mtf50(
    dng_path: str,
    roi_coords: dict,
    black_level: float = None,
    white_level: float = None,
    pixel_pitch_um: float = 2.2654,
    preview_scale: float = 2.0,
    rotate_k: int = -1,
    row_offset: int = 0,
    col_offset: int = 0,
) -> dict:
    """
    Single-frame MTF50 measurement using modular extract_green_roi and normalize_view.
    """
    if rawpy is None:
        raise ImportError("rawpy is required to read DNG files.")
    if not os.path.exists(dng_path):
        raise FileNotFoundError(f"DNG file not found: {dng_path}")

    #set black and white level
    with rawpy.imread(dng_path) as raw:
        bayer = raw.raw_image_visible.astype(np.float64)
        b_lvl = (
            float(np.mean(raw.black_level_per_channel))
            if black_level is None
            else float(black_level)
        )
        w_lvl = (
            float(raw.white_level)
            if white_level is None
            else float(white_level)
        )

    #slice green plane for one green channel
    green_roi = extract_green_roi(
        raw_visible_array=bayer,
        roi_coords=roi_coords,
        preview_scale=preview_scale,
        row_offset=row_offset,
        col_offset=col_offset,
        rotate_k=rotate_k,
    )

    #normalize the view
    roi_sub = normalize_view(green_roi, black_level=b_lvl, white_level=w_lvl, clip=False)

    #run mtf50 function
    freq, mtf, esf_x, esf = slanted_edge_mtf(roi_sub)
    mtf50_px = compute_mtf50(freq, mtf)

    if np.isnan(mtf50_px):
        raise ValueError(f"Could not compute valid MTF50 for '{os.path.basename(dng_path)}'")

    mtf50_um = mtf50_px / (2.0 * pixel_pitch_um)
    mtf50_mm = mtf50_um * 1000.0

    return {
        "filename": os.path.basename(dng_path),
        "mtf50_px": float(mtf50_px),
        "mtf50_cy_per_um": float(mtf50_um),
        "mtf50_cy_per_mm": float(mtf50_mm),
        "black_level": b_lvl,
        "white_level": w_lvl,
        "green_roi": green_roi,
        "roi_sub": roi_sub,
        "freq": freq,
        "mtf": mtf,
        "esf_x": esf_x,
        "esf": esf,
    }

def _resolve_roi_for_path(dng_path: str, roi_coords_spec) -> dict | None:
    """Resolves a per-file or shared ROI coordinate dict for `dng_path`."""
    if not isinstance(roi_coords_spec, dict):
        return None
    fname = os.path.basename(dng_path)
    if fname in roi_coords_spec and isinstance(roi_coords_spec[fname], dict):
        return roi_coords_spec[fname]
    if "y" in roi_coords_spec:
        return roi_coords_spec
    return None

def measure_modulation_at_nyquist(
    dng_paths: list,
    roi_coords_spec,
    nyquist_px: float = 0.5,
    black_level: float = None,
    white_level: float = None,
    pixel_pitch_um: float = 2.2654,
    preview_scale: float = 2.0,
    rotate_k: int = -1,
    row_offset: int = 0,
    col_offset: int = 0,
) -> float:
    """Measures mean MTF modulation at green-plane Nyquist (0.5 cy/green-px) across captures."""
    mod_vals = []

    for path in dng_paths:
        roi_coords = _resolve_roi_for_path(path, roi_coords_spec)
        if roi_coords is None:
            continue

        res = measure_single_frame_mtf50(
            dng_path=path,
            roi_coords=roi_coords,
            black_level=black_level,
            white_level=white_level,
            pixel_pitch_um=pixel_pitch_um,
            preview_scale=preview_scale,
            rotate_k=rotate_k,
            row_offset=row_offset,
            col_offset=col_offset,
        )
        freq = np.asarray(res["freq"], dtype=float)
        mtf = np.asarray(res["mtf"], dtype=float)

        if len(freq) > 0 and len(mtf) > 0:
            order = np.argsort(freq)
            mod_vals.append(float(np.interp(nyquist_px, freq[order], mtf[order])))

    return float(np.mean(mod_vals)) if mod_vals else float("nan")

def process_dng_list_mtf50(
    dng_paths: list,
    roi_coords: dict,
    pixel_pitch_um: float,
    black_level: float = None,
    white_level: float = None,
    preview_scale: float = 2.0,
    rotate_k: int = -1,
) -> dict:
    """
    Computes MTF50 across a list of DNG paths, prints per-frame and summary
    metrics, and returns mean/std in both cy/green-px and cy/um plus frame records.
    """
    if not dng_paths:
        raise ValueError("Provided dng_paths list is empty.")

    print(f"=== Processing {len(dng_paths)} DNG images ===")
    frame_results = []

    for path in dng_paths:
        res = measure_single_frame_mtf50(
            dng_path=path,
            roi_coords=roi_coords,
            black_level=black_level,
            white_level=white_level,
            pixel_pitch_um=pixel_pitch_um,
            preview_scale=preview_scale,
            rotate_k=rotate_k,
        )
        frame_results.append(res)
        print(
            f"[{res['filename']}] MTF50: {res['mtf50_px']:.4f} cy/green-px "
            f"({res['mtf50_cy_per_um']:.4f} cy/µm | {res['mtf50_cy_per_mm']:.1f} cy/mm)"
        )

    px_vals = [r["mtf50_px"] for r in frame_results]
    um_vals = [r["mtf50_cy_per_um"] for r in frame_results]

    mean_px = float(np.mean(px_vals))
    std_px = float(np.std(px_vals, ddof=1) if len(px_vals) > 1 else 0.0)
    mean_um = float(np.mean(um_vals))
    std_um = float(np.std(um_vals, ddof=1) if len(um_vals) > 1 else 0.0)

    print("-" * 65)
    print(f"List Mean MTF50: {mean_px:.4f} ± {std_px:.4f} cy/green-px ({mean_um:.4f} ± {std_um:.4f} cy/µm)")
    print("-" * 65)

    return {
        "mean_px": mean_px,
        "std_px": std_px,
        "mean_um": mean_um,
        "std_um": std_um,
        "frames": frame_results,
    }

def summarize_corner_mtf50(
    mtf_results_by_region: dict,
    corner_regions: tuple = ("top_left", "top_right", "bottom_left", "bottom_right"),
    metric_key: str = "mtf50_px",
) -> dict:
    """
    Computes corner MTF50 mean, worst-corner mean & std, and three standard deviations:
    - uniformity_std    : std across per-corner means (corner-to-corner spread)
    - repeatability_std : pooled within-corner std across repeated frames
    - total_pooled_std  : std across all individual frame measurements pooled
    """
    corner_means = []
    corner_stds = []
    corner_counts = []
    valid_regions = []
    all_values = []

    num_rep = 0.0
    den_rep = 0

    for region in corner_regions:
        res = mtf_results_by_region.get(region)
        if not res:
            continue

        frames = res.get("frames", [])
        vals = np.asarray(
            [f[metric_key] for f in frames if metric_key in f and np.isfinite(f[metric_key])],
            dtype=np.float64,
        )
        n = len(vals)
        if n == 0:
            continue

        c_mean = float(np.mean(vals))
        c_std = float(np.std(vals, ddof=1)) if n > 1 else 0.0

        valid_regions.append(region)
        corner_means.append(c_mean)
        corner_stds.append(c_std)
        corner_counts.append(n)
        all_values.append(vals)

        if n > 1:
            num_rep += (n - 1) * (c_std ** 2)
            den_rep += (n - 1)

    if not corner_means:
        return {
            "corner_mean": np.nan,
            "corner_worst_mean": np.nan,
            "corner_worst_std": np.nan,
            "corner_worst_region": None,
            "uniformity_std": np.nan,
            "repeatability_std": np.nan,
            "total_pooled_std": np.nan,
            "corner_means": [],
            "n_corners": 0,
            "n_total_frames": 0,
        }

    corner_means_arr = np.asarray(corner_means, dtype=np.float64)
    pooled_arr = np.concatenate(all_values)

    worst_idx = int(np.argmin(corner_means_arr))
    corner_mean = float(np.mean(corner_means_arr))
    corner_worst_mean = float(corner_means_arr[worst_idx])
    corner_worst_std = float(corner_stds[worst_idx])
    corner_worst_region = valid_regions[worst_idx]

    uniformity_std = (
        float(np.std(corner_means_arr, ddof=1)) if len(corner_means_arr) > 1 else 0.0
    )
    repeatability_std = float(np.sqrt(num_rep / den_rep)) if den_rep > 0 else 0.0
    total_pooled_std = float(np.std(pooled_arr, ddof=1)) if len(pooled_arr) > 1 else 0.0

    return {
        "corner_mean": corner_mean,
        "corner_worst_mean": corner_worst_mean,
        "corner_worst_std": corner_worst_std,
        "corner_worst_region": corner_worst_region,
        "uniformity_std": uniformity_std,
        "repeatability_std": repeatability_std,
        "total_pooled_std": total_pooled_std,
        "corner_means": corner_means,
        "corner_stds": corner_stds,
        "corner_counts": corner_counts,
        "n_corners": len(corner_means),
        "n_total_frames": int(len(pooled_arr)),
    }

def calculate_diffraction_limits(
    pixel_pitch_um: float,
    f_number: float,
    wavelength_um: float = 0.55,
    nyquist_green_px: float = 0.5,
) -> dict:
    """Computes diffraction/detector MTF curves in both cycles/um and cycles/green-px."""
    green_pitch_um = 2.0 * pixel_pitch_um

    # Spatial frequencies in cycles/um
    v_c_um = 1.0 / (wavelength_um * f_number)
    v_nyquist_um = nyquist_green_px / green_pitch_um
    n_cross = green_pitch_um / wavelength_um

    # Frequency axis up to diffraction cutoff (in cycles/um and cycles/green-px)
    nu_um = np.linspace(1e-6, v_c_um, 1000)
    freq_green_px = nu_um * green_pitch_um

    s_clipped = np.clip(nu_um / v_c_um, 0.0, 1.0)
    mtf_diff = (2.0 / np.pi) * (
        np.arccos(s_clipped) - s_clipped * np.sqrt(1.0 - s_clipped**2)
    )
    # Detector aperture along green-plane sampling pitch
    mtf_sensor = np.abs(np.sinc(nu_um * green_pitch_um))
    mtf_total = mtf_diff * mtf_sensor

    # Modulation at green-plane Nyquist (0.5 cycles/green-px)
    mod_at_nyq = float(np.interp(nyquist_green_px, freq_green_px, mtf_total))

    # Predicted MTF50 (where mtf_total crosses 0.5, scanning from low to high frequency)
    below_half = np.where(mtf_total <= 0.5)[0]
    if len(below_half) > 0 and below_half[0] > 0:
        idx = below_half[0]
        predicted_mtf50_green_px = float(
            np.interp(
                0.5,
                [mtf_total[idx], mtf_total[idx - 1]],
                [freq_green_px[idx], freq_green_px[idx - 1]],
            )
        )
    else:
        predicted_mtf50_green_px = float("nan")

    predicted_mtf50_um = (
        predicted_mtf50_green_px / green_pitch_um
        if np.isfinite(predicted_mtf50_green_px)
        else float("nan")
    )

    return {
        "green_pitch_um": float(green_pitch_um),
        "v_c_um": float(v_c_um),
        "v_c_green_px": float(v_c_um * green_pitch_um),
        "v_nyquist_um": float(v_nyquist_um),
        "nyquist_green_px": float(nyquist_green_px),
        "crossover_f_number": float(n_cross),
        "modulation_at_nyquist": mod_at_nyq,
        "predicted_mtf50_cycles_per_um": predicted_mtf50_um,
        "predicted_mtf50_green_px": predicted_mtf50_green_px,
        "frequency_axis": nu_um,
        "frequency_axis_green_px": freq_green_px,
        "mtf_diffraction": mtf_diff,
        "mtf_sensor": mtf_sensor,
        "mtf_total": mtf_total,
    }

def analytic_mtf50(sigma: float) -> float:
    """Analytic MTF50 for a Gaussian PSF of width sigma."""
    return float(np.sqrt(np.log(2) / (2 * np.pi**2 * sigma**2)))
