# src/camera_toolkit/report_generator.py
"""
Generates a markdown summary report from results.json for portfolio presentation.
"""
import os
import json


def generate_markdown_report(results_json_path: str, output_md_path: str = None) -> str:
    """Reads results.json and formats a structured Markdown camera characterization report."""
    if not os.path.exists(results_json_path):
        raise FileNotFoundError(f"results.json not found at {results_json_path}")
        
    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    camera = data.get("camera", "Camera Characterization")
    app = data.get("capture_app", "Default")
    cfa = data.get("cfa_pattern", "RGGB")
    w = data.get("raw_width_px", 0)
    h = data.get("raw_height_px", 0)
    
    f_px = data.get("focal_length_px", [0.0, 0.0])
    f_mm = data.get("focal_length_mm", [0.0, 0.0])
    cx = data.get("principal_point_x_px", [0.0, 0.0])
    cy = data.get("principal_point_y_px", [0.0, 0.0])
    k1 = data.get("distortion_k1", [0.0, 0.0])
    k2 = data.get("distortion_k2", [0.0, 0.0])
    rms = data.get("reprojection_rms_px", [0.0, 0.0])
    n_views = data.get("n_calibration_images_used", 0)
    
    mtf_ctr = data.get("mtf50_center_cy_per_px", [0.0, 0.0])
    mtf_crn = data.get("mtf50_corner_mean_cy_per_px", [0.0, 0.0])
    
    K = data.get("conversion_gain_dn_per_e", [0.0, 0.0])
    rn = data.get("read_noise_e", [0.0, 0.0])
    dark = data.get("dark_current_e_per_s", [0.0, 0.0])
    dr = data.get("dynamic_range_stops", [0.0, 0.0])
    
    de_before = data.get("delta_e_mean_before_ccm", [0.0, 0.0])
    de_after = data.get("delta_e_mean_after_ccm", [0.0, 0.0])
    de_worst = data.get("delta_e_worst_after_ccm", [0.0, 0.0])

    report = f"""# Camera Characterization Report: {camera}

## 1. System Metadata & Sensor Specifications
- **Device / Lens Model**: {camera}
- **Capture Application**: {app}
- **Color Filter Array (CFA)**: {cfa}
- **Sensor Resolution**: {w} x {h} pixels

## 2. Geometric Calibration & Lens Distortion
- **Focal Length ($f_{{px}}$)**: {f_px[0]:.2f} ± {f_px[1]:.2f} px
- **Focal Length ($f_{{mm}}$)**: {f_mm[0]:.2f} mm
- **Principal Point ($c_x, c_y$)**: ({cx[0]:.2f} ± {cx[1]:.2f}, {cy[0]:.2f} ± {cy[1]:.2f}) px
- **Radial Distortion ($k_1, k_2$)**: {k1[0]:.4f} ± {k1[1]:.4f}, {k2[0]:.4f} ± {k2[1]:.4f} ($k_3 = 0$ fixed)
- **Reprojection RMS Error**: {rms[0]:.4f} px ({n_views} views used)

## 3. Spatial Resolution & Optical Transfer Function
- **Center MTF50**: {mtf_ctr[0]:.4f} ± {mtf_ctr[1]:.4f} cycles/pixel
- **Corner Mean MTF50**: {mtf_crn[0]:.4f} ± {mtf_crn[1]:.4f} cycles/pixel

## 4. Sensor Noise & Dynamic Range
- **Conversion Gain ($K$)**: {K[0]:.4f} ± {K[1]:.4f} DN/$e^-$
- **Read Noise ($\sigma_r$)**: {rn[0]:.2f} ± {rn[1]:.2f} $e^-$
- **Dark Current**: {dark[0]:.4f} ± {dark[1]:.4f} $e^-$/s
- **Dynamic Range**: {dr[0]:.2f} ± {dr[1]:.2f} stops ({data.get("dynamic_range_criterion", "SNR=1")})

## 5. Color Science & Accuracy ($\Delta E_{{2000}}$)
- **Mean $\Delta E$ Before CCM**: {de_before[0]:.2f} ± {de_before[1]:.2f}
- **Mean $\Delta E$ After CCM**: {de_after[0]:.2f} ± {de_after[1]:.2f}
- **Worst Patch $\Delta E$ After CCM**: {de_worst[0]:.2f}
"""
    if output_md_path:
        os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(report)
    return report
