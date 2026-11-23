"""
Plotting functions for camera characterization pipeline visualization.
Generates publication-quality figures for calibration residuals, MTF curves,
Photon Transfer Curves (PTC), SNR curves, and color accuracy across illuminants.
"""
import math
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    import rawpy
except ImportError:
    rawpy = None


def _save_figure(
    fig,
    output_dir: str,
    filename: str,
    dpi: int = 300,
    pad_inches: float = 0.1,
) -> str | None:
    """Saves a Matplotlib figure into <output_dir>/figures/<filename> using an absolute path."""
    if not output_dir or not filename:
        return None
    fig_dir = os.path.abspath(os.path.join(output_dir, "figures"))
    os.makedirs(fig_dir, exist_ok=True)
    save_path = os.path.join(fig_dir, filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    print(f"[PLOT] Saved at: {save_path}")
    return save_path


def plot_directory_contact_sheet(
    input_dir: str,
    cols: int = 5,
    max_images: int = None,
    thumb_max_dim: int = 400,
    gap: float = 0.03,
    output_dir: str = "images/outputs",
    filename: str = "contact_sheet.png",
):
    """
    Creates a portrait-oriented contact sheet strictly from .dng files in `input_dir`
    with small gaps, no filenames, and no title.
    """
    if rawpy is None:
        raise ImportError("rawpy is required to decode DNG files.")

    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return None

    paths = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(".dng")
    ])

    if max_images is not None:
        paths = paths[:max_images]

    if not paths:
        print(f"No .dng files found in {input_dir}")
        return None

    thumbs = []
    for p in paths:
        with rawpy.imread(p) as raw:
            img_rgb = raw.postprocess(
                half_size=True,
                use_camera_wb=True,
                no_auto_bright=False,
            )

        # Enforce portrait orientation (height > width)
        if img_rgb.shape[1] > img_rgb.shape[0]:
            img_rgb = np.rot90(img_rgb, k=1)

        # Downscale to fast thumbnail size
        h, w = img_rgb.shape[:2]
        scale = thumb_max_dim / max(h, w)
        if scale < 1.0:
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        thumbs.append(img_rgb)

    n = len(thumbs)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    sample_h, sample_w = thumbs[0].shape[:2]
    aspect = sample_h / max(sample_w, 1)
    fig_w = min(cols * 2.0, 16.0)
    tile_w = fig_w / cols
    fig_h = min(rows * tile_w * aspect, 40.0)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, ax in enumerate(axes_flat):
        if idx < n:
            ax.imshow(thumbs[idx])
        ax.axis("off")

    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.99,
        bottom=0.01,
        wspace=gap,
        hspace=gap,
    )

    save_path = _save_figure(fig, output_dir, filename, dpi=200, pad_inches=0.02)
    plt.show()
    plt.close(fig)
    return save_path


def save_distortion_comparison(
    original_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    output_dir: str = "images/outputs",
    filename: str = "distortion_check.png",
):
    """Generates and saves a clean, side-by-side diagnostic chart comparing the distorted image directly against the unwarped rectilinear image."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB) if len(original_bgr.shape) == 3 else original_bgr
    rect_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB) if len(rectified_bgr.shape) == 3 else rectified_bgr

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original Capture (Distorted)", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(rect_rgb)
    axes[1].set_title("Undistorted Frame", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout()
    target_path = _save_figure(fig, output_dir, filename)
    plt.close(fig)
    return target_path


def plot_radial_distortion_curve(
    k1: float,
    k2: float,
    k3: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_width: int = 3072,
    img_height: int = 4080,
    max_calibrated_r: float = 0.55,
    output_dir: str = "images/outputs",
    filename: str = "radial_distortion_curve.png",
):
    """
    Plots radial distortion (%) as a function of normalized radial distance r
    from the principal point to the farthest image corner, marking the
    calibrated checkerboard region vs. peripheral extrapolation.
    """
    corners = np.array([
        [0.0, 0.0],
        [img_width - 1.0, 0.0],
        [0.0, img_height - 1.0],
        [img_width - 1.0, img_height - 1.0],
    ], dtype=np.float64)

    x_norm = (corners[:, 0] - cx) / fx
    y_norm = (corners[:, 1] - cy) / fy
    r_corners = np.hypot(x_norm, y_norm)
    r_max = float(np.max(r_corners))

    r = np.linspace(0.0, r_max, 300)
    r2 = r ** 2
    pct_distortion = 100.0 * (k1 * r2 + k2 * (r2 ** 2) + k3 * (r2 ** 3))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(0.0, max_calibrated_r, color="forestgreen", alpha=0.08, label="Calibrated Board Zone")
    if r_max > max_calibrated_r:
        ax.axvspan(max_calibrated_r, r_max, color="darkorange", alpha=0.10, label="Corner Extrapolation Zone")
        ax.axvline(max_calibrated_r, color="darkorange", linestyle=":", lw=1.3)

    ax.axhline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    ax.plot(r, pct_distortion, color="royalblue", lw=2.2, label=r"Radial Model ($k_1 r^2 + k_2 r^4 + k_3 r^6$)")

    corner_dist = float(pct_distortion[-1])
    ax.scatter([r_max], [corner_dist], color="crimson", zorder=5, s=55)
    ax.annotate(
        f"Corner (r={r_max:.2f}): {corner_dist:+.2f}%",
        xy=(r_max, corner_dist),
        xytext=(r_max * 0.62, corner_dist * 0.85),
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
        fontsize=9,
        fontweight="bold",
        color="crimson",
    )

    ax.set_xlabel("Normalized Radial Distance from Principal Point, r (tan θ)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Radial Distortion (%)", fontsize=10, fontweight="bold")
    ax.set_title(
        f"Radial Lens Distortion Profile\n(k₁={k1:+.5f}, k₂={k2:+.5f}, k₃={k3:+.5f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()

    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_rectified_comparison(
    sample_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    title_distorted: str = "Original Capture (Distorted)",
    title_rectified: str = "Undistorted Frame",
    output_dir: str = "images/outputs",
    filename: str = "rectified_comparison.png",
):
    """Displays and saves side-by-side rectified image comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    orig_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB) if len(sample_bgr.shape) == 3 else sample_bgr
    rect_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB) if len(rectified_bgr.shape) == 3 else rectified_bgr

    axes[0].imshow(orig_rgb)
    axes[0].set_title(title_distorted, fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(rect_rgb)
    axes[1].set_title(title_rectified, fontsize=11, fontweight="bold")
    axes[1].axis("off")

    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_calibration_residuals_2panel(
    all_object_points: list,
    all_image_points: list,
    rvecs: list,
    tvecs: list,
    K: np.ndarray,
    dist: np.ndarray,
    img_shape: tuple,
    per_view: list = None,
    rms: float = None,
    output_dir: str = "images/outputs",
    filename: str = "calibration_residuals_2panel.png",
):
    """Generates a 2-panel calibration residual diagnostic plot:
    1. Spatial vector quiver plot showing 2D reprojection residuals across (x, y) scaled 500x.
    2. Per-view RMS reprojection error line plot identifying outlier captures dragging the fit.
    """
    res, pos = [], []
    for i in range(len(all_object_points)):
        proj, _ = cv2.projectPoints(all_object_points[i], rvecs[i], tvecs[i], K, dist)
        res.append(proj.reshape(-1, 2) - all_image_points[i].reshape(-1, 2))
        pos.append(all_image_points[i].reshape(-1, 2))
    res, pos = np.vstack(res), np.vstack(pos)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    quiv = a1.quiver(
        pos[:, 0],
        pos[:, 1],
        res[:, 0],
        res[:, 1],
        np.linalg.norm(res, axis=1),
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=0.002,
        width=0.003,
    )
    a1.add_patch(
        plt.Rectangle((0, 0), img_shape[0], img_shape[1], fill=False, ec="#213921", lw=1.2)
    )
    a1.set_xlim(-200, img_shape[0] + 200)
    a1.set_ylim(img_shape[1] + 200, -200)
    a1.set_aspect("equal")
    a1.grid(False)
    a1.set_xlabel("X [px]", fontsize=9)
    a1.set_ylabel("Y [px]", fontsize=9)
    a1.set_title("Residuals by Position (Drawn 500x)", fontsize=11, fontweight="bold")
    fig.colorbar(quiv, ax=a1, label="Residual Mag [px]", fraction=0.046, pad=0.04)

    if per_view is None or rms is None:
        errors = []
        for i in range(len(all_object_points)):
            p_proj, _ = cv2.projectPoints(all_object_points[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(all_image_points[i], p_proj, cv2.NORM_L2) / np.sqrt(len(p_proj))
            errors.append(err)
        per_view = errors
        rms = np.sqrt(np.mean(np.square(errors)))

    per_view_arr = np.array(per_view).ravel()

    a2.plot(
        np.arange(1, len(per_view_arr) + 1),
        per_view_arr,
        "o-",
        color="#6d8d24",
        lw=1.5,
        ms=5,
    )
    a2.axhline(rms, color="#213921", ls="--", lw=1.3, label=f"Overall {rms:.4f} px")
    a2.set_xlabel("View Index", fontsize=9)
    a2.set_ylabel("Per-view RMS [px]", fontsize=9)
    a2.set_title("Which Capture is Dragging the Fit", fontsize=11, fontweight="bold")
    a2.legend(fontsize=9)
    a2.grid(True, linestyle="--", alpha=0.5)

    print(
        f"residual mean [{res[:, 0].mean():.4f} {res[:, 1].mean():.4f}] px, sd"
        f" [{res[:, 0].std():.4f} {res[:, 1].std():.4f}] px"
    )

    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_calibration_summary_table(
    rms: float,
    fx: float,
    sfx: float,
    fy: float,
    sfy: float,
    cx: float,
    scx: float,
    cy: float,
    scy: float,
    k1: float,
    sk1: float,
    k2: float,
    sk2: float,
    p1: float,
    sp1: float,
    p2: float,
    sp2: float,
    k3: float,
    sk3: float,
    focal_length_mm: float,
    f_stop: float,
    focal_35mm: float,
    img_shape: tuple,
    output_dir: str = "images/outputs",
    filename: str = "calibration_summary_table.png",
):
    """Renders and saves formatted camera calibration summary table."""
    table_data = [
        ["RMS Reprojection Error", f"{rms:.4f}", "-", "px"],
        ["Focal Length (fx)", f"{fx:.2f}", f"+/- {sfx:.2f}", "px"],
        ["Focal Length (fy)", f"{fy:.2f}", f"+/- {sfy:.2f}", "px"],
        ["Principal Point (cx)", f"{cx:.2f}", f"+/- {scx:.2f}", f"px (Center: {img_shape[0]/2:.1f})"],
        ["Principal Point (cy)", f"{cy:.2f}", f"+/- {scy:.2f}", f"px (Center: {img_shape[1]/2:.1f})"],
        ["Radial Dist (k1)", f"{k1:.4f}", f"+/- {sk1:.4f}", ""],
        ["Radial Dist (k2)", f"{k2:.4f}", f"+/- {sk2:.4f}", ""],
        ["Radial Dist (k3)", f"{k3:.4f}", f"+/- {sk3:.4f}", ""],
        ["Tangential Dist (p1)", f"{p1:.4f}", f"+/- {sp1:.4f}", ""],
        ["Tangential Dist (p2)", f"{p2:.4f}", f"+/- {sp2:.4f}", ""],
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Parameter", "Value", "Uncertainty", "Units"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    ax.set_title("Google Pixel 9 Camera Calibration Summary", fontsize=13, weight="bold", pad=25)
    ax.text(
        0.5,
        1.02,
        f"Physical Focal Length: {focal_length_mm} mm  |  Aperture: {f_stop}  |  Size: {img_shape} px  |  35mm Equiv: {focal_35mm} mm",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_predicted_system_mtf(
    nu: np.ndarray,
    mtf_diff: np.ndarray,
    mtf_sensor: np.ndarray,
    mtf_total: np.ndarray,
    v_nyquist_um: float,
    p_um: float,
    f_aperture: float,
    wavelength_um: float = 0.55,
    output_dir: str = "images/outputs",
    filename: str = "predicted_system_mtf.png",
):
    """Plots and saves predicted system MTF curves (diffraction, sinc sensor MTF, and combined total)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(nu, mtf_diff, "--", color="darkblue", label="MTF_diffraction")
    ax.plot(nu, mtf_sensor, ":", color="goldenrod", label="MTF_sensor (sinc)")
    ax.plot(nu, mtf_total, "-", color="black", lw=2, label="Combined System MTF Product")
    ax.axvline(
        v_nyquist_um,
        color="green",
        linestyle="-.",
        lw=1.5,
        label=f"Green Nyquist ({v_nyquist_um:.4f} c/um)",
    )

    ax.set_xlabel("Spatial Frequency (cycles / um)", fontsize=11)
    ax.set_ylabel("Modulation Transfer Function (MTF)", fontsize=11)
    ax.set_title(
        f"Predicted System MTF Curves (N = {f_aperture}, p = {p_um:.2f} um, lambda = {wavelength_um:.2f} um)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_slanted_edge_results(
    esf_x: np.ndarray,
    esf: np.ndarray,
    lsf: np.ndarray,
    freq: np.ndarray,
    mtf: np.ndarray,
    sigma: float = None,
    edge_window: tuple = (-20.0, 20.0),
    max_freq: float = 1.5,
    output_dir: str = "images/outputs",
    filename: str = "slanted_edge_esf_lsf_mtf.png",
):
    """Renders and saves 3-panel ESF, LSF, and MTF slanted-edge diagnostic plot."""
    esf_x = np.asarray(esf_x)
    esf = np.asarray(esf)
    lsf = np.asarray(lsf)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(esf_x, esf, "-", ms=2.5, color="blue")
    if edge_window is not None:
        axes[0].set_xlim(edge_window)
    axes[0].set_title("Edge Spread Function (ESF)")
    axes[0].set_xlabel("Distance from edge (green pixels)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(esf_x, lsf, color="green", lw=1.5)
    if edge_window is not None:
        axes[1].set_xlim(edge_window)
    axes[1].set_title("Line Spread Function (LSF)")
    axes[1].set_xlabel("Distance from edge (green pixels)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(freq, mtf, lw=2, color="darkgreen", label="Measured MTF")
    if sigma is not None:
        analytic_truth = np.exp(-2 * np.pi**2 * sigma**2 * freq**2)
        axes[2].plot(freq, analytic_truth, "--", color="orange", label="Analytic Truth")
    axes[2].axhline(0.5, color="gray", ls=":", label="MTF = 0.5")
    if max_freq is not None:
        axes[2].set_xlim(0, max_freq)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("Modulation Transfer Function (MTF)")
    axes[2].set_xlabel("Cycles / green-plane pixel")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_regional_mtf50(
    mtf_results_by_region: dict,
    res_payload: dict,
    pixel_pitch_um: float,
    output_dir: str = "images/outputs",
    filename: str = "regional_mtf50_comparison.png",
):
    """
    Plots per-frame MTF50 scatter points, region mean ± 1 SD error bars,
    and overall corner mean reference line across center and corner ROIs.
    """
    preferred_order = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
    regions = [r for r in preferred_order if r in mtf_results_by_region and mtf_results_by_region[r]]
    regions += [r for r in mtf_results_by_region if r not in regions and mtf_results_by_region[r]]

    if not regions:
        print("No regional MTF50 results available to plot.")
        return None

    data = {
        r: [float(frame["mtf50_px"]) for frame in mtf_results_by_region[r].get("frames", [])]
        for r in regions
    }

    pitch = float(pixel_pitch_um)
    x_pos = np.arange(len(regions))
    means = np.array([float(mtf_results_by_region[r]["mean_px"]) for r in regions])
    stds = np.array([float(mtf_results_by_region[r]["std_px"]) for r in regions])

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    rng = np.random.default_rng(42)
    for idx, r in enumerate(regions):
        vals = np.asarray(data[r], dtype=float)
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.10, 0.10, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            color="tab:blue",
            alpha=0.75,
            s=36,
            zorder=3,
            label="Individual frames" if idx == 0 else None,
        )

    ax.errorbar(
        x_pos,
        means,
        yerr=stds,
        fmt="o",
        color="black",
        ecolor="crimson",
        elinewidth=1.8,
        capsize=5,
        markersize=7,
        zorder=4,
        label="Region Mean ± 1 SD",
    )

    for idx, (m, s) in enumerate(zip(means, stds)):
        ax.text(
            idx,
            m + s + 0.012,
            f"{m:.3f}±{s:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    corner_mean_entry = res_payload.get("mtf50_corner_mean", [np.nan, 0.0])
    corner_mean_val = float(
        corner_mean_entry[0] if isinstance(corner_mean_entry, (list, tuple, np.ndarray)) else corner_mean_entry
    )
    if np.isfinite(corner_mean_val):
        ax.axhline(
            corner_mean_val,
            color="gray",
            linestyle="--",
            linewidth=1.1,
            label=f"Corner Mean ({corner_mean_val:.3f} cy/green-px)",
        )

    all_vals = np.concatenate([np.asarray(data[r], dtype=float) for r in regions if len(data[r]) > 0])
    y_min = max(0.0, float(np.min(all_vals)) - 0.05) if len(all_vals) > 0 else 0.0
    y_max = float(np.max(means + stds)) + 0.07

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.replace("_", " ").title() for r in regions])
    ax.set_ylabel("MTF50 [cycles / green-px]")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_title("Regional MTF50 Across Center and Corners (Passing QC Frames)")

    to_cy_mm = lambda v: (v / (2.0 * pitch)) * 1000.0
    from_cy_mm = lambda v: (v / 1000.0) * (2.0 * pitch)
    sec_ax = ax.secondary_yaxis("right", functions=(to_cy_mm, from_cy_mm))
    sec_ax.set_ylabel("MTF50 [cycles / mm]")

    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()

    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_ptc_curve(
    means_above_black: list,
    vars_dn2: list,
    valid_mask: np.ndarray,
    gain_K: float,
    intercept_c: float,
    output_dir: str = "images/outputs",
    filename: str = "ptc_curve.png",
):
    """Plots and saves Photon Transfer Curve scatter and linear fit."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means_arr = np.array(means_above_black, dtype=float)
    vars_arr = np.array(vars_dn2, dtype=float)

    ax.scatter(
        means_arr[valid_mask],
        vars_arr[valid_mask],
        color="blue",
        label="Linear Region (Included)",
    )
    if np.any(~valid_mask):
        ax.scatter(
            means_arr[~valid_mask],
            vars_arr[~valid_mask],
            color="red",
            marker="x",
            label="Saturated (>70% Excluded)",
        )

    if len(means_arr) > 0:
        x_fit = np.linspace(0, max(means_arr), 100)
        ax.plot(
            x_fit,
            gain_K * x_fit + intercept_c,
            "r--",
            label=f"PTC Fit (K = {gain_K:.4f} DN/e-)",
        )

    ax.set_xlabel("Mean Signal Above Black (DN)")
    ax.set_ylabel("Pair Variance (DN^2)")
    ax.set_title("Photon Transfer Curve (PTC)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_snr_curve(
    means: np.ndarray,
    var_pair: np.ndarray,
    K: float,
    read_noise_e: float,
    output_dir: str = "images/outputs",
    filename: str = "snr_chart.png",
    valid_mask: np.ndarray = None,
):
    """Generates and saves the Signal-to-Noise Ratio (SNR) curve on a log-log scale."""
    means = np.asarray(means, dtype=float)
    var_pair = np.asarray(var_pair, dtype=float)

    positive = (means > 0) & (var_pair > 0)
    if not np.any(positive):
        means = np.array([10.0, 50.0, 100.0, 500.0])
        var_pair = np.array([1.0, 3.0, 6.0, 30.0])
        positive = np.ones_like(means, dtype=bool)

    if valid_mask is None:
        mask = positive
    else:
        mask = positive & np.asarray(valid_mask, dtype=bool)
        if not np.any(mask):
            mask = positive

    excl_mask = positive & (~mask)

    m_clean = means[mask]
    v_clean = var_pair[mask]

    signal_e = m_clean / K if K > 0 else m_clean
    snr_measured = m_clean / np.sqrt(v_clean)

    sig_model_e = np.logspace(
        np.log10(max(1.0, signal_e.min() / 2)),
        np.log10(signal_e.max() * 1.5),
        300,
    )
    snr_model = sig_model_e / np.sqrt(sig_model_e + read_noise_e**2)
    snr_shot_limit = np.sqrt(sig_model_e)
    knee_e = read_noise_e**2

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        signal_e,
        snr_measured,
        "o",
        color="#0072b2",
        ms=6,
        label="Measured SNR (Included)",
    )

    if np.any(excl_mask):
        m_excl = means[excl_mask]
        v_excl = var_pair[excl_mask]
        sig_excl_e = m_excl / K if K > 0 else m_excl
        snr_excl = m_excl / np.sqrt(v_excl)
        ax.scatter(
            sig_excl_e,
            snr_excl,
            color="red",
            marker="x",
            s=45,
            linewidths=1.8,
            zorder=4,
            label="Saturated (>70% Excluded)",
        )

    ax.loglog(
        sig_model_e,
        snr_model,
        color="#009e73",
        lw=2,
        label=rf"Model ($\sigma_r = {read_noise_e:.1f}\ e^-$)",
    )
    ax.loglog(
        sig_model_e,
        snr_shot_limit,
        "--",
        color="#e69f00",
        lw=1.5,
        label=r"Photon Shot Noise Limit ($\sqrt{\mu_e}$)",
    )
    ax.axvline(
        knee_e,
        color="#cc79a7",
        linestyle=":",
        lw=1.5,
        label=f"Read Noise Knee ({knee_e:.0f} $e^-$)",
    )

    ax.set_xlabel(r"Signal Intensity $\mu_e$ (photoelectrons $e^-$)", fontsize=11)
    ax.set_ylabel("Signal-to-Noise Ratio (SNR)", fontsize=11)
    ax.set_title("Signal-to-Noise Ratio (SNR) & Read Noise Knee", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()

    target_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return target_path


def plot_dark_current(
    dark_exposure_groups: dict,
    gain_K: float,
    dark_current_e_sec: float,
    dark_current_err: float,
    black_level: float = 64.0,
    patch_size: int = 256,
    cfa_pattern: str = None,
    green_index: int = 0,
    output_dir: str = "images/outputs",
    filename: str = "dark_current_fit.png",
):
    """Plots per-frame and group-mean dark signals using the exact same center green patch as fit_dark_current."""
    from . import noise

    t_frames, sig_frames = [], []
    t_groups, sig_groups = [], []

    for exp_time in sorted(dark_exposure_groups.keys()):
        frames = dark_exposure_groups[exp_time]
        if not frames:
            continue

        group_vals = []
        for file_path in frames:
            patch, bl = noise._load_raw_patch_and_black(
                file_path,
                patch_size=patch_size,
                black_level=black_level,
                cfa_pattern=cfa_pattern,
                green_index=green_index,
            )
            val_dn = float(np.mean(patch))
            t_frames.append(float(exp_time))
            sig_frames.append(val_dn)
            group_vals.append(val_dn)

        t_groups.append(float(exp_time))
        sig_groups.append(float(np.mean(group_vals)))

    t_arr = np.asarray(t_groups, dtype=float)
    sig_arr = np.asarray(sig_groups, dtype=float)

    slope_dn, intercept_dn = np.polyfit(t_arr, sig_arr, 1)
    slope_err = dark_current_err * gain_K

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        t_frames,
        sig_frames,
        color="black",
        alpha=0.65,
        s=40,
        label="Individual Dark Frames",
        zorder=3,
    )

    fit_line = slope_dn * t_arr + intercept_dn
    ax.plot(
        t_arr,
        fit_line,
        color="red",
        linestyle="--",
        linewidth=1.8,
        label=f"Fit Slope: {slope_dn:.4f} ± {slope_err:.4f} DN/s",
        zorder=4,
    )

    ax.set_xlabel("Exposure Time (seconds)", fontsize=10)
    ax.set_ylabel("Mean Dark Signal (DN)", fontsize=10)
    ax.set_title(
        f"Dark Current Evaluation\nDark Current: {dark_current_e_sec:.4f} ± {dark_current_err:.4f} e⁻/s/pixel",
        fontsize=11,
        fontweight="bold",
    )

    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()

    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_color_delta_e_across_illuminants(
    per_patch_errors: dict,
    ccm_source: str = "LED",
    output_dir: str = "images/outputs",
    filename: str = "color_delta_e_across_illuminants.png",
):
    """Plots per-patch CIEDE2000 color error curves across illuminants."""
    fig, ax = plt.subplots(figsize=(10, 5))

    markers = {"LED": "^", "Outdoors": "o", "Overhead": "s", "Incandescent": "d"}
    colors = {
        "LED": "forestgreen",
        "Outdoors": "darkorange",
        "Overhead": "steelblue",
        "Incandescent": "crimson",
    }

    patches_x = np.arange(1, 25)

    for illum_name, errors in per_patch_errors.items():
        m = markers.get(illum_name, "o")
        c = colors.get(illum_name, "blue")
        mean_val = float(np.mean(errors))
        ax.plot(
            patches_x,
            errors,
            marker=m,
            color=c,
            label=f"{illum_name} Illuminant (Mean: {mean_val:.2f})",
            lw=1.5,
        )

    ax.set_xticks(patches_x)
    ax.set_xlabel("ColorChecker Patch Index (1 to 24)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"CIEDE2000 ($\Delta E_{00}$)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Per-Patch Color Error Across Illuminants\n(Using Fixed {ccm_source} Color Correction Matrix)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()

    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_raw_saturation_histograms(
    dng_paths: list,
    row_labels: list = None,
    output_dir: str = "images/outputs",
    filename: str = "color_raw_histograms.png",
):
    """Plots a 4x4 grid of log-scale raw histograms with empirical v_sat markers."""
    from . import color

    if not dng_paths:
        print("No DNG files provided for histogram plot.")
        return None

    if row_labels is None:
        row_labels = ["Incandescent", "LED", "Overhead", "Outdoors"]

    cols = 4
    rows = len(row_labels)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows))

    for idx, path in enumerate(dng_paths):
        r = idx // cols
        c = idx % cols
        if r >= rows:
            break

        ax = axes[r, c] if rows > 1 else axes[c]
        ax.set_box_aspect(1.33)

        info = color.analyze_raw_saturation(path)
        v_sat = info["v_sat"]
        clipped_fraction = info["clipped_fraction"]

        ax.plot(info["bin_centers"], info["hist"], color="crimson", lw=1.2)
        ax.axvline(
            v_sat,
            color="blue",
            linestyle="--",
            alpha=0.8,
            label=f"v_sat ({v_sat})",
        )

        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=7)

        title_text = (
            f"{os.path.basename(path)}\n"
            f"V_sat: {v_sat} | Clipped: {clipped_fraction * 100:.3f}%"
        )
        ax.set_title(title_text, fontsize=8)
        ax.tick_params(axis="both", labelsize=8)

        if c == 0:
            ax.text(
                -0.28,
                0.5,
                row_labels[r],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
                ha="right",
                rotation=90,
            )

    for idx in range(len(dng_paths), rows * cols):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c] if rows > 1 else axes[c]
        ax.axis("off")

    fig.tight_layout()
    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_color_delta_e_before_after(
    delta_e_before: np.ndarray,
    delta_e_after: np.ndarray,
    illuminant_name: str = "LED",
    output_dir: str = "images/outputs",
    filename: str = "color_delta_e_before_after.png",
):
    """Plots a grouped side-by-side bar chart of per-patch CIEDE2000 error before and after CCM."""
    de_bef = np.asarray(delta_e_before, dtype=float)
    de_aft = np.asarray(delta_e_after, dtype=float)

    mean_bef = float(np.mean(de_bef))
    mean_aft = float(np.mean(de_aft))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, len(de_aft) + 1)

    ax.bar(
        x - 0.2,
        de_bef,
        width=0.4,
        label=f"Before CCM (Mean: {mean_bef:.2f})",
        color="indianred",
    )
    ax.bar(
        x + 0.2,
        de_aft,
        width=0.4,
        label=f"After CCM (Mean: {mean_aft:.2f})",
        color="steelblue",
    )

    ax.set_xticks(x)
    ax.set_xlabel("ColorChecker Patch Index (1–24)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"CIEDE2000 ($\Delta E_{00}$)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Per-Patch Color Error Before and After CCM Application ({illuminant_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    fig.tight_layout()

    save_path = _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)
    return save_path


def plot_white_sheet_horizontal_profiles(
    white_sheet_paths: dict,
    black_level: float = None,
    white_level: float = None,
    patch_size: int = 50,
    print_ratios: bool = True,
    output_dir: str = "images/outputs",
    filename: str = "white_sheet_horizontal_profiles.png",
) -> dict:
    """
    Plots horizontal center profiles (Green channel) across multiple white-sheet
    flat fields and computes before/after corner-to-center vignetting ratios.
    """
    from . import color

    fig, ax = plt.subplots(figsize=(10, 5.5))
    all_round_ratios = {}
    channels_names = ["Red", "Green", "Blue"]

    for round_name, path in white_sheet_paths.items():
        if not isinstance(path, str) or not os.path.exists(path):
            print(f"Skipping missing white sheet for {round_name}: {path}")
            continue

        v_sat_det, b_lvl_det, _ = color.check_raw_clipping(path)
        b_lvl = float(black_level) if black_level is not None else b_lvl_det
        w_lvl = float(white_level) if white_level is not None else v_sat_det

        ws_rgb = color.extract_quarter_res_linear_rgb(
            path, black_level=b_lvl, white_level=w_lvl
        )
        hh, ww, _ = ws_rgb.shape
        r_c, c_c = hh // 2, ww // 2
        half_p = patch_size // 2

        g_ch = ws_rgb[:, :, 1]
        profile = np.mean(g_ch[r_c - 5 : r_c + 5, :], axis=0)
        ax.plot(profile, lw=1.8, label=f"{round_name} ({os.path.basename(path)})")

        round_ratios = {"Before": [], "After": []}
        for ch_idx in range(3):
            ch = ws_rgb[:, :, ch_idx]
            center_val = max(
                float(np.mean(ch[r_c - half_p : r_c + half_p, c_c - half_p : c_c + half_p])),
                1e-6,
            )
            corner_val = float(np.mean(ch[0:patch_size, 0:patch_size]))
            round_ratios["Before"].append(corner_val / center_val)

            ws_ff_norm = ch / center_val
            ch_corrected = np.clip(ch / np.maximum(ws_ff_norm, 1e-4), 0.0, 1.0)
            after_center = max(
                float(np.mean(ch_corrected[r_c - half_p : r_c + half_p, c_c - half_p : c_c + half_p])),
                1e-6,
            )
            after_corner = float(np.mean(ch_corrected[0:patch_size, 0:patch_size]))
            round_ratios["After"].append(after_corner / after_center)

        all_round_ratios[round_name] = round_ratios

    ax.set_title("Horizontal Center Profiles of White Sheets Across All Rounds", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horizontal Pixel Index (Quarter-Res)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Intensity (Green Channel)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    _save_figure(fig, output_dir, filename)
    plt.show()
    plt.close(fig)

    if print_ratios and all_round_ratios:
        print("\n=== Corner-to-Center Ratios Across All White Sheets ===")
        for round_name, ratios in all_round_ratios.items():
            print(f"\n--- {round_name} White Sheet ---")
            for i, ch_name in enumerate(channels_names):
                print(
                    f"  {ch_name:<5} Channel -> Before: {ratios['Before'][i]:.4f} | "
                    f"After: {ratios['After'][i]:.4f}"
                )

    return all_round_ratios