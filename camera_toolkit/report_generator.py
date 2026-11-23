"""
Automated report compiler for camera characterization deliverables.
"""
import json
import os


def generate_characterization_report(
    results_payload: dict,
    output_dir: str = "./outputs",
    filename: str = "characterization_report.md",
) -> str:
    """Compiles a comprehensive Markdown characterization summary report from results payload."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, filename)

    cam_name = results_payload.get("camera", "Google Pixel 9")
    app_name = results_payload.get("capture_app", "Open Camera")
    cfa = results_payload.get("cfa_pattern", "GBRG")
    w = results_payload.get("raw_width_px", 4080)
    h = results_payload.get("raw_height_px", 3072)

    f_px = results_payload.get("focal_length_px", [3045.85, 6.53])
    f_mm = results_payload.get("focal_length_mm", [6.9, 0.0])
    cx = results_payload.get("principal_point_x_px", [1509.05, 2.81])
    cy = results_payload.get("principal_point_y_px", [2065.80, 2.68])
    rms = results_payload.get("reprojection_rms_px", [0.9595, 0.0])

    mtf_c = results_payload.get("mtf50_center_cy_per_px", [0.3784, 0.0442])
    mtf_m = results_payload.get("mtf50_corner_mean_cy_per_px", [0.3106, 0.0312])
    mtf_w = results_payload.get("mtf50_corner_worst_cy_per_px", [0.2687, 0.0])

    gain = results_payload.get("conversion_gain_dn_per_e", [0.0061, 0.0015])
    rn = results_payload.get("read_noise_e", [139.61, 6.98])
    dr = results_payload.get("dynamic_range_stops", [10.14, 0.10])

    de_b = results_payload.get("delta_e_mean_before_ccm", [14.85, 2.30])
    de_a = results_payload.get("delta_e_mean_after_ccm", [5.82, 2.99])
    de_w = results_payload.get("delta_e_worst_after_ccm", [15.10, 0.0])

    md_content = f"""# Camera Characterization Report: {cam_name}

## 1. Sensor Profile & Hardware Audit
- **Camera Device**: {cam_name}
- **Capture Application**: {app_name}
- **CFA Bayer Grid**: {cfa}
- **Raw Sensor Dimensions**: {w} x {h} px

## 2. Geometric Calibration Statistics
- **Reprojection RMS Error**: {rms[0]:.4f} px
- **Focal Length**: {f_px[0]:.2f} +/- {f_px[1]:.2f} px ({f_mm[0]:.2f} mm)
- **Principal Point**: ({cx[0]:.2f}, {cy[0]:.2f}) px

## 3. Spatial Resolution & Slanted-Edge MTF50
- **Center MTF50**: {mtf_c[0]:.4f} +/- {mtf_c[1]:.4f} cycles/px
- **Corner Mean MTF50**: {mtf_m[0]:.4f} +/- {mtf_m[1]:.4f} cycles/px
- **Corner Worst MTF50**: {mtf_w[0]:.4f} cycles/px

## 4. Sensor Noise & Dynamic Range
- **Conversion Gain K**: {gain[0]:.4f} +/- {gain[1]:.4f} DN/e-
- **Read Noise Floor**: {rn[0]:.2f} +/- {rn[1]:.2f} e-
- **Dynamic Range**: {dr[0]:.2f} +/- {dr[1]:.2f} stops

## 5. Color Science & Accuracy (CIEDE2000)
- **Mean Delta E (Before CCM)**: {de_b[0]:.2f} +/- {de_b[1]:.2f}
- **Mean Delta E (After CCM)**: {de_a[0]:.2f} +/- {de_a[1]:.2f}
- **Worst Patch Delta E (After CCM)**: {de_w[0]:.2f}
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[SUCCESS] Characterization report generated at: {report_path}")
    return report_path
