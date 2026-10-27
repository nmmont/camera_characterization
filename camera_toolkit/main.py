# src/camera_toolkit/main.py
import os
import sys
import yaml
import rawpy
import numpy as np
from pathlib import Path

# Embedded path helper to prevent package namespace failures on Windows
sys.path.append(str(Path(__file__).resolve().parents))

from camera_toolkit import utils
from camera_toolkit import geometry
from camera_toolkit import plots
from camera_toolkit import resolution

def load_config(config_path="config.yaml") -> dict:
    """Loads operational configuration parameters from the root tree directory."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    utils.verify_environment()
    
    config = load_config("config.yaml")
    geom_cfg = config["geometry"]
    proj_cfg = config["project"]
    
    print(f"\n--- Initiating Calibration Pipeline for: {proj_cfg['camera_name']} ---")
    
    # 1. Fetch RAW DNG file paths using your 'file_extension' key [INDEX: 0.1.2]
    dng_paths = utils.find_target_images(geom_cfg["input_dir"], extension=geom_cfg["file_extension"])
    if not dng_paths:
        raise FileNotFoundError(f"No source verification DNG files found inside: {geom_cfg['input_dir']}")
        
    # Extract structural sensor information from your DNG file EXIF header tags [INDEX: 0.1.2]
    lens_model, f_mm, f_aperture, f_35mm, img_shape = utils.lens_id(dng_paths[0])
    
    if f_35mm is None:
        f_35mm = geom_cfg.get("focal_35mm_fallback", 25.0)
        print(f"⚠️ EXIF 35mm tag missing. Applying config fallback: {f_35mm} mm")
              
    print(f"✅ True RAW Verification Audited: {lens_model}")
    print(f"   EXIF Lens Specs: {f_mm} mm | 35mm Equivalent: {f_35mm} mm | Shape: {img_shape} px\n")
    
    # 2. Fetch matching JPEG paths using your 'jpg_extension' key
    jpg_paths = utils.find_target_images(geom_cfg["input_dir"], extension=geom_cfg["jpg_extension"])
    assert len(dng_paths) == len(jpg_paths), "Data Error: Mismatched file counts between your DNG and JPEG datasets."

    board_cfg = geom_cfg["checkerboard"]
    rows, cols, sq_size = board_cfg["rows"], board_cfg["cols"], board_cfg["square_size_mm"]
    ideal_obj_points = geometry.generate_object_points(rows, cols, sq_size)
    
    all_object_points = []
    all_image_points = []
    
    # 3. Fast processing loop utilizing your 8-bit JPEG frame assets
    print(f"Processing {len(jpg_paths)} JPEG target frames for sub-pixel feature tracking...")
    for idx, path in enumerate(jpg_paths):
        print(f" -> [Frame {idx+1}/{len(jpg_paths)}] Processing: {os.path.basename(path)}...", end="", flush=True)
        
        # Load standard 8-bit image matrix instantly
        bgr_array = utils.load_processed_bgr_image(path)
        success, corners = geometry.detect_checkerboard_corners(bgr_array, rows, cols)
        
        if success:
            all_object_points.append(ideal_obj_points)
            all_image_points.append(corners)
            print(" Success! Grid localized.")
        else:
            print(" ⚠️ Failed to find checkerboard corners.")
            
    print(f"\nSuccessfully resolved full grids on {len(all_image_points)} of {len(jpg_paths)} captures.")
    
    # 4. Compute camera calibration matrices and uncertainties
    print("Computing matrix parameter optimizations...")
    calibration_payload = geometry.run_extended_calibration(all_object_points, all_image_points, img_shape)
    
    # 5. Quantify straightness improvements and pixel pitch validations
    dev_before, dev_after = geometry.evaluate_straightness_improvement(
        all_image_points=all_image_points,  # Analyzes the first valid target frame
        camera_matrix=utils.reconstruct_k_matrix(calibration_payload),
        dist_coeffs=utils.reconstruct_dist_vector(calibration_payload),
        target_row_idx=2,
        cols_per_row=cols
    )
    
    pitch_metrics = geometry.calculate_pixel_pitch_metrics(
        f_px_x=calibration_payload["focal_length_x"][0],
        f_px_y=calibration_payload["focal_length_y"][0],
        f_mm=f_mm,
        f_35mm=f_35mm,
        image_shape=img_shape  # Width dimension
    )
    
    # 6. Format final results storage dictionary
    results = {
        "camera_name": lens_model if proj_cfg["camera_name"] == "AUTO" else proj_cfg["camera_name"],
        "geometry": calibration_payload,
        "straightness_verification": {
            "max_perpendicular_deviation_before_px": [dev_before, 0.0],
            "max_perpendicular_deviation_after_px": [dev_after, 0.0]
        },
        "pixel_pitch_cross_validation": {
            "pixel_pitch_method1_um": [pitch_metrics["pixel_pitch_method1_um"], 0.0],
            "pixel_pitch_method2_um": [pitch_metrics["pixel_pitch_method2_um"], 0.0],
            "percentage_difference": [pitch_metrics["pitch_percentage_difference"], 0.0]
        },
        "noise": {},
        "resolution": {}
    }
    
    # 7. Unwarp image and save charts to outputs/ [INDEX: 0.1.3]
    print("\nGenerating visual verification charts and saving parameters...")
    sample_warped_img = utils.load_processed_bgr_image(jpg_paths[0])
    clean_rectilinear_img = geometry.rectify_frame(
        image_array=sample_warped_img,
        camera_matrix=utils.reconstruct_k_matrix(calibration_payload),
        dist_coeffs=utils.reconstruct_dist_vector(calibration_payload),
        alpha=1.0
    )
    
    plots.save_distortion_comparison(sample_warped_img, clean_rectilinear_img, proj_cfg["output_dir"])
    
    print("\n--- Geometry Calibration Pipeline Complete ---")

    # =========================================================================
    # PART 2: SPATIAL RESOLUTION PIPELINE (Integrated) [INDEX: 0.1.3]
    # =========================================================================
    print("\n--- Initiating Spatial Resolution Pipeline ---")
    res_cfg = config.get("resolution", {"input_dir": "./images/checkerboard", "file_extension": ".dng"})
    
    p = pitch_metrics["pixel_pitch_method1_um"]  # Calculated pitch microns
    
    # 1. Compute Analytical System MTF & Diffraction Limitations
    diffraction_payload = resolution.calculate_diffraction_limits(
        pixel_pitch_um=p,
        f_number=f_aperture
    )
    
    # Save the predicted crossover curves figure directly into outputs/figures/ [INDEX: 0.1.3]
    plots.plot_predicted_system_mtf(
        metrics=diffraction_payload,
        f_number=f_aperture,
        pixel_pitch=p,
        wavelength=0.55,
        output_dir=proj_cfg["output_dir"]
    )
    
    # 2. Grab Slanted-Edge Target Path List
    slant_paths = utils.find_target_images(res_cfg["input_dir"], extension=res_cfg["file_extension"])
    if not slant_paths:
        raise FileNotFoundError(f"No spatial resolution targets found inside: {res_cfg['input_dir']}")
        
    # 3. Extract Single-Channel Linear Green Plane Diagnostic (Deliverable 14)
    with rawpy.imread(slant_paths[0]) as raw:
        sample_bayer = raw.raw_image_visible.astype(float)
        black_level = float(np.mean(raw.black_level_per_channel))
        white_level = float(raw.white_level)
        
    green_linear = resolution.extract_linear_green_plane(sample_bayer, black_level, white_level)
    plots.save_linear_green_plane_plot(green_linear, os.path.basename(slant_paths[0]), proj_cfg["output_dir"])
    
    # 4. Batch Process All Frame Crops to Compute Mean MTF50 Statistics (Deliverable 15) [INDEX: 0.1.3]
    roi_coords = {"y": 700, "x": 390, "h": 400, "w": 200}
    
    mean_um, std_um, mean_px, std_px, rois_list = resolution.process_batch_mtf50(
        dng_paths=slant_paths,
        roi_coords=roi_coords,
        pixel_pitch_um=p
    )
    
    # Save out the multi-pane thumbnails contact sheet figure [INDEX: 0.1.3]
    plots.save_roi_contact_sheet(rois_list, proj_cfg["output_dir"])
    
        # Locate line 132 in main.py and change it from this:
    # freq, mtf_curve, esf_x, esf = resolution.slanted_edge_mtf(rois_list)
    
    # To this:
    # Extracts just the first raw 2D pixel array from the (patch_array, filename) tuple list
    sample_roi_patch = rois_list[0][0] 
    freq, mtf_curve, esf_x, esf = resolution.slanted_edge_mtf(sample_roi_patch)
    
    plots.plot_slanted_edge_results(freq, mtf_curve, esf_x, esf, sigma=1.0, output_dir=proj_cfg["output_dir"])

    # 5. Intersect Empirical Modulation Values Exactly at Nyquist Bounds (0.25 c/px)
    measured_mod_nyq = resolution.measure_modulation_at_nyquist(slant_paths, roi_coords, nyquist_px=0.25)
    
    # Pack metrics into your final parameters deliverable dictionary [INDEX: 0.1.3]
    results["resolution"] = {
        "crossover_f_number": [float(diffraction_payload["crossover_f_number"]), 0.0],
        "predicted_modulation_at_nyquist": [float(diffraction_payload["modulation_at_nyquist"]), 0.0],
        "measured_modulation_at_nyquist": [float(measured_mod_nyq), 0.0],
        "mean_mtf50_cycles_per_um": [mean_um, std_um],
        "mean_mtf50_cycles_per_pixel": [mean_px, std_px]
    }
    
    # 6. Export Unified Value-Uncertainty Records and Finalize Session [INDEX: 0.1.3]
    utils.save_json_results(results, proj_cfg["output_dir"])
    print("\n--- Spatial Resolution Pipeline Complete ---")


if __name__ == "__main__":
    main()
