# src/camera_toolkit/plots.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def save_distortion_comparison(original_bgr: np.ndarray, rectified_bgr: np.ndarray, output_dir: str, filename: str = "distortion_check.png"):
    """
    Generates and saves a clean, side-by-side diagnostic chart comparing the 
    distorted image directly against the unwarped rectilinear image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    
    pairs = [
        (orig_rgb, "Original (Distorted)"),
        (rect_rgb, "Undistorted (CALIB_FIX_K3)")
    ]
    
    for ax, (im, ttl) in zip(axes, pairs):
        ax.imshow(im)
        ax.set_title(ttl, fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        
    fig.tight_layout()
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, filename)
    
    plt.savefig(target_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[PLOT] Geometric rectification chart saved at: {target_path}")


def plot_predicted_system_mtf(metrics: dict, f_number: float, pixel_pitch: float, wavelength: float, output_dir: str):
    """Plots and saves the predicted analytical system MTF curves."""
    nu = metrics["frequency_axis"]
    
    plt.figure(figsize=(8, 5))
    plt.plot(nu, metrics["mtf_diffraction"], label="MTF_diffraction", linestyle="--", color="blue")
    plt.plot(nu, metrics["mtf_sensor"], label="MTF_sensor (sinc)", linestyle=":", color="orange")
    plt.plot(nu, metrics["mtf_total"], label="Combined MTF Product", color="black", linewidth=2)
    plt.axvline(metrics["v_nyquist_um"], color="green", linestyle="-.", 
                label=f"Green Nyquist ({metrics['v_nyquist_um']:.4f} c/um)")

    plt.xlabel("Spatial Frequency (cycles / um)")
    plt.ylabel("Modulation Transfer Function (MTF)")
    plt.title(f"Predicted System MTF Curves (N = {f_number}, p = {pixel_pitch:.2f} um, lambda = {wavelength} um)")
    plt.grid(True)
    plt.legend()
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, "predicted_mtf_curves.png")
    
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Theoretical limits curves saved at: {target_path}")


def plot_slanted_edge_results(freq: np.ndarray, mtf: np.ndarray, esf_x: np.ndarray, esf: np.ndarray, sigma: float, output_dir: str):
    """Generates and saves the empirical ESF, LSF, and MTF evaluation profiles."""
    if len(freq) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(esf_x, esf, ".", ms=2.5, color="blue")
    axes[0].set_title("Edge Spread Function (ESF)")
    axes[0].set_xlabel("Distance from edge (px)")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True)

    lsf = np.gradient(esf, esf_x)
    axes[1].plot(esf_x, lsf, color="green", lw=1.5)
    axes[1].set_title("Line Spread Function (LSF)")
    axes[1].set_xlabel("Distance from edge (px)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(freq, mtf, lw=2, color="darkgreen", label="Measured MTF")
    if sigma > 0:
        axes[2].plot(freq, np.exp(-2 * np.pi**2 * sigma**2 * freq**2), "--", lw=1.5, color="orange", label="Analytic Truth")
    axes[2].axhline(0.5, color="gray", ls=":", label="50% Modulation")
    axes[2].set_title("Modulation Transfer Function (MTF)")
    axes[2].set_xlabel("Cycles per pixel")
    axes[2].set_ylabel("Modulation")
    axes[2].set_xlim(0, 0.5)
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, "slanted_edge_measurement_results.png")
    
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Empirical MTF curves saved at: {target_path}")


def save_linear_green_plane_plot(green_linear_array: np.ndarray, filename_str: str, output_dir: str):
    """Saves the extracted linear radiance green plane map."""
    plt.figure(figsize=(6, 6))
    plt.imshow(green_linear_array, cmap="gray", origin="upper")
    plt.title(f"Extracted Linear Green Plane\n({filename_str})", fontsize=11, weight="bold")
    plt.xlabel("X [green pixels]")
    plt.ylabel("Y [green pixels]")
    plt.colorbar(label="Normalized Radiance")
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, "extracted_green_plane.png")
    
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Radiance channel image map saved at: {target_path}")


def save_roi_contact_sheet(all_rois_list: list, output_dir: str):
    """Assembles and saves a multi-column diagnostic thumbnail sheet of edge cuts."""
    if not all_rois_list:
        return
    num_files = len(all_rois_list)
    ncols = 2
    nrows = (num_files + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, (roi_sub, filename) in enumerate(all_rois_list):
        ax = axes[i]
        ax.imshow(roi_sub, cmap="gray")
        ax.set_title(filename, fontsize=10, fontweight="bold")
        ax.set_xlabel("X [px]", fontsize=8)
        ax.set_ylabel("Y [px]", fontsize=8)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Slant-Edge ROI Contact Sheet", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, "slant_edge_roi_contact_sheet.png")
    
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Multi-view contact thumbnail sheet saved at: {target_path}")


def plot_ptc_curve(means_above_black: list, variances_dn2: list, gain_K: float, intercept_c: float, output_dir: str):
    """Plots and saves Photon Transfer Curve (PTC) line fitting."""
    means = np.array(means_above_black)
    vars_dn2 = np.array(variances_dn2)
    
    plt.figure(figsize=(7, 5))
    plt.plot(means, vars_dn2, "o", color="blue", label="Measured Pair Variances")
    
    x_line = np.linspace(0, max(means) * 1.1, 100)
    y_line = gain_K * x_line + intercept_c
    plt.plot(x_line, y_line, "--", color="red", label=f"Fit Line (K = {gain_K:.4f} DN/e-)")
    
    plt.xlabel("Mean Signal Above Black (DN)")
    plt.ylabel("Temporal Variance (DN^2)")
    plt.title("Photon Transfer Curve (PTC)")
    plt.grid(True)
    plt.legend()
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    target_path = os.path.join(fig_dir, "ptc_curve.png")
    
    plt.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] PTC curve saved at: {target_path}")


def plot_calibration_residuals_2panel(
    all_object_points: list, 
    all_image_points: list, 
    rvecs: list, 
    tvecs: list, 
    K: np.ndarray, 
    dist: np.ndarray, 
    img_shape: tuple, 
    per_view: list = None, 
    rms: float = None
) -> plt.Figure:
    """
    Generates a 2-panel calibration residual diagnostic plot:
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
    a1.quiver(pos[:, 0], pos[:, 1], res[:, 0], res[:, 1],
              np.linalg.norm(res, axis=1), cmap="viridis",
              angles="xy", scale_units="xy", scale=0.002, width=0.003)
    a1.add_patch(plt.Rectangle((0, 0), img_shape[0], img_shape[1], fill=False, ec="#213921", lw=1.2))
    a1.set_xlim(-200, img_shape[0] + 200)
    a1.set_ylim(img_shape[1] + 200, -200)
    a1.set_aspect("equal")
    a1.grid(alpha=0)
    a1.set_title("residuals by position, drawn 500x", fontsize=11)

    if per_view is None or rms is None:
        errors = []
        for i in range(len(all_object_points)):
            p_proj, _ = cv2.projectPoints(all_object_points[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(all_image_points[i], p_proj, cv2.NORM_L2) / np.sqrt(len(p_proj))
            errors.append(err)
        per_view = errors
        rms = np.sqrt(np.mean(np.square(errors)))

    per_view_arr = np.array(per_view).ravel()

    a2.plot(np.arange(1, len(per_view_arr) + 1), per_view_arr, "o-",
            color="#6d8d24", lw=1.5, ms=5)
    a2.axhline(rms, color="#213921", ls="--", lw=1.3, label=f"overall {rms:.3f} px")
    a2.set_xlabel("view")
    a2.set_ylabel("per-view RMS (px)")
    a2.set_title("which capture is dragging the fit", fontsize=11)
    a2.legend(fontsize=9)

    fig.tight_layout()
    print(f"residual mean {res.mean(axis=0)} px, sd {res.std(axis=0)} px")
    return fig


def plot_color_delta_e_across_illuminants(per_patch_errors: dict, output_dir: str = None) -> plt.Figure:
    """
    Plots per-patch CIEDE2000 (Delta E_00) color error curves across all illuminants
    (Outdoors, Overhead, LED, Incandescent).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 25)

    colors = {
        "Outdoors": "tab:orange",
        "Overhead": "tab:blue",
        "LED": "tab:green",
        "Incandescent": "tab:red"
    }
    markers = {
        "Outdoors": "o",
        "Overhead": "s",
        "LED": "^",
        "Incandescent": "d"
    }

    for name, errors in per_patch_errors.items():
        c = colors.get(name, "black")
        m = markers.get(name, "o")
        ax.plot(x, errors, marker=m, linestyle="-", color=c, linewidth=2, label=f"{name} Illuminant")

    ax.set_xlabel("ColorChecker Patch Index (1 to 24)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"CIEDE2000 ($\Delta E_{00}$)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Patch Color Error Across Illuminants\n(Using Fixed Outdoor Color Correction Matrix)", fontsize=12, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if output_dir:
        fig_dir = os.path.join(output_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        target_path = os.path.join(fig_dir, "color_delta_e_across_illuminants.png")
        plt.savefig(target_path, dpi=300, bbox_inches="tight")
        print(f"[PLOT] Color error plot saved at: {target_path}")

    return fig
