# src/camera_toolkit/plots.py
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_distortion_comparison(original_bgr: np.ndarray, rectified_bgr: np.ndarray, output_dir: str, filename: str = "distortion_check.png"):
    """
    Generates and saves a clean, side-by-side diagnostic chart comparing the 
    distorted image directly against the unwarped rectilinear image [INDEX: 0.1.3].
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rect_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    
    pairs = [
        (orig_rgb, "Original (Distorted)"),
        (rect_rgb, "Undistorted (Optimized)")
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
    """Plots and saves the predicted analytical system MTF curves [INDEX: 0.1.3]."""
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
    """Generates and saves the empirical analytical ESF, LSF, and MTF evaluation profiles [INDEX: 0.1.3]."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(esf_x, esf, ".", ms=2.5, color="blue")
    axes[0].set_title("Edge Spread Function (ESF)")
    axes[0].set_xlabel("Distance from edge (px)")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True)

    axes[1].plot(esf_x, np.gradient(esf, esf_x), color="green", lw=1.5)
    axes[1].set_title("Line Spread Function (LSF)")
    axes[1].set_xlabel("Distance from edge (px)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(freq, mtf, lw=2, color="darkgreen", label="Measured MTF")
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
    """Saves the extracted linear radiance green plane map (Deliverable 14) [INDEX: 0.1.3]."""
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
    """Assembles and saves a multi-column diagnostic thumbnail sheet of your edge cuts [INDEX: 0.1.3]."""
    num_files = len(all_rois_list)
    ncols = 2
    nrows = (num_files + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4 * nrows))
    axes = axes.flatten()
    
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
