# src/camera_toolkit/utils.py
import os
import json
import sys
import cv2
import numpy as np
import tifffile

try:
    import rawpy
except ImportError:
    rawpy = None

MODEL_ID = 272              # Root TIFF tag for Device Model
FOCAL_LENGTH_ID = 37386     # Focal length tag
FOCAL_35MM_ID = 41989       # 35mm equivalent tag
F_NUMBER_ID = 33437         # F-Number tag
ISO_ID = 34855              # ISO speed ratings tag
EXPOSURE_TIME_ID = 33434    # Exposure time (shutter speed) tag
SUBJECT_DISTANCE_ID = 37382  # Subject distance (focus) tag


def verify_environment():
    """Validates Python version and core environment library installations and prints their versions."""
    required_packages = ["rawpy", "colour", "numpy", "cv2", "yaml", "matplotlib", "tifffile"]
    missing_packages = []
    
    print("--- Environment Validation ---")
    
    # 1. Verify and print the Python Version
    # sys.version_info yields a clean format (e.g., "3.10.12")
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 {"Python":<12} : {py_version}")
    
    # 2. Verify and print package versions
    for pkg in required_packages:
        try:
            # Dynamically import the package if it isn't already loaded
            module = __import__(pkg)
            
            # Retrieve the version (handle packages that might use different attributes)
            version = getattr(module, "__version__", "Version unknown")
            
            print(f"✅ {pkg:<12} : {version}")
        except ImportError:
            print(f"❌ {pkg:<12} : NOT INSTALLED")
            missing_packages.append(pkg)
            
    if missing_packages:
        print(f"\nCRITICAL: Missing dependencies: {missing_packages}")
        sys.exit(1)
        
    print("\nEnvironment Validation Passed Successfully.\n")


def verify_true_bayer_raw(file_path: str):
    """
    Inspects a DNG file container to confirm it contains a genuine un-demosaiced 
    Bayer mosaic grid.
    """
    if rawpy is None:
        return
    try:
        with rawpy.imread(file_path) as raw:
            raw_shape = raw.raw_image.shape
            if len(raw_shape) != 2:
                raise ValueError(
                    f"Invalid Array Shape {raw_shape}: Image is already multi-channel."
                )
            color_description = raw.color_desc
            if len(color_description) != 4 or b'R' not in color_description:
                raise ValueError(
                    f"Invalid CFA Tag Layout '{color_description}'."
                )
    except Exception as error:
        raise AssertionError(
            f"Data Validation Failure for file '{os.path.basename(file_path)}': {str(error)}"
        )


def find_target_images(directory_path: str, extension: str) -> list:
    """Scans folders for files matching a specific extension choice."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    matched = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith(extension.lower())]
    return sorted(matched)


def format_shutter_speed(exp_time) -> str:
    """Converts exposure time into a clean 1/# format for sub-second values."""
    if not exp_time:
        return "Unknown"
    if isinstance(exp_time, (tuple, list)) and len(exp_time) == 2:
        num, den = exp_time[0], exp_time[1]
        if num == 0 or den == 0:
            return "0s"
        val = num / den
    elif isinstance(exp_time, (int, float)):
        val = float(exp_time)
        num, den = val, 1
    else:
        return str(exp_time)

    if val >= 1.0:
        return f"{round(val, 2)}s"
    else:
        reciprocal = round(den / num) if (isinstance(exp_time, (tuple, list)) and num != 0) else round(1.0 / val)
        return f"1/{reciprocal}s"


def lens_id(path: str) -> tuple:
    """Extracts raw metadata from DNG containers."""
    verify_true_bayer_raw(path)
    
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]

        device_model = p.tags.get(MODEL_ID)
        if device_model:
            device_model = device_model.value
            if isinstance(device_model, bytes):
                device_model = device_model.decode("utf-8", errors="ignore").strip()
        else:
            device_model = "Google Pixel 9"

        if "ExifTag" in p.tags:
            exif = p.tags["ExifTag"].value
        else:
            exif = {tag.code: tag.value for tag in p.tags.values()}

        fl = exif.get(FOCAL_LENGTH_ID) or exif.get("FocalLength")
        if fl and isinstance(fl, (tuple, list)) and len(fl) == 2:
            calculated_fl = round(fl[0] / fl[1], 3)
        else:
            calculated_fl = float(fl) if fl else 6.9

        f_num = exif.get(F_NUMBER_ID) or exif.get("FNumber")
        if f_num and isinstance(f_num, (tuple, list)) and len(f_num) == 2:
            calculated_f = round(f_num[0] / f_num[1], 2)
        else:
            calculated_f = float(f_num) if f_num else 1.68

        focal_35mm = exif.get(FOCAL_35MM_ID) or exif.get("FocalLengthIn35mmFilm", 25.0)
        width = p.tags["ImageWidth"].value
        length = p.tags["ImageLength"].value

        return device_model, calculated_fl, calculated_f, focal_35mm, (width, length)


def load_processed_bgr_image(file_path: str) -> np.ndarray:
    """Standard OpenCV color loader for drawing and fast grid processing."""
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Failed to decode image: {file_path}")
    return img


def save_json_results(results_dict: dict, output_dir: str, filename: str = "results.json"):
    """Writes pipeline metrics cleanly out to your required results JSON deliverable."""
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, filename)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[SUCCESS] Metrics exported cleanly to: {target}")


def reconstruct_k_matrix(calibration_payload: dict) -> np.ndarray:
    """Assembles a standard 3x3 OpenCV intrinsic matrix from dict values."""
    return np.array([
        [calibration_payload["focal_length_x"][0], 0.0, calibration_payload["principal_point_x"][0]],
        [0.0, calibration_payload["focal_length_y"][0], calibration_payload["principal_point_y"][0]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


def reconstruct_dist_vector(calibration_payload: dict) -> np.ndarray:
    """Assembles lens distortion coefficient vectors from dict values."""
    return np.array([
        calibration_payload["radial_distortion_k1"][0],
        calibration_payload["radial_distortion_k2"][0],
        calibration_payload["tangential_distortion_p1"][0],
        calibration_payload["tangential_distortion_p2"][0],
        calibration_payload["radial_distortion_k3"][0]
    ], dtype=np.float32)


def export_project1_results_json(
    exif_meta: dict,
    geom_results: dict,
    res_results: dict,
    noise_results: dict,
    color_results: dict,
    output_dir: str
):
    """
    Formats and exports the full results dictionary strictly matching results.template.txt schema.
    """
    results_payload = {
        "camera": str(exif_meta.get("lens_model", "Google Pixel 9")),
        "capture_app": str(exif_meta.get("capture_app", "Open Camera")),
        "cfa_pattern": str(exif_meta.get("cfa_pattern", "GRBG")),
        "raw_width_px": int(exif_meta.get("width", 4080)),
        "raw_height_px": int(exif_meta.get("height", 3072)),
        "binned_from_higher_resolution_sensor": bool(exif_meta.get("binned", False)),

        "focal_length_px": [float(geom_results.get("f_px", [0.0, 0.0])[0]), float(geom_results.get("f_px", [0.0, 0.0])[1])],
        "focal_length_mm": [float(geom_results.get("f_mm", [0.0, 0.0])[0]), 0.0],
        "principal_point_x_px": [float(geom_results.get("cx", [0.0, 0.0])[0]), float(geom_results.get("cx", [0.0, 0.0])[1])],
        "principal_point_y_px": [float(geom_results.get("cy", [0.0, 0.0])[0]), float(geom_results.get("cy", [0.0, 0.0])[1])],
        "distortion_k1": [float(geom_results.get("k1", [0.0, 0.0])[0]), float(geom_results.get("k1", [0.0, 0.0])[1])],
        "distortion_k2": [float(geom_results.get("k2", [0.0, 0.0])[0]), float(geom_results.get("k2", [0.0, 0.0])[1])],
        "distortion_p1": [float(geom_results.get("p1", [0.0, 0.0])[0]), float(geom_results.get("p1", [0.0, 0.0])[1])],
        "distortion_p2": [float(geom_results.get("p2", [0.0, 0.0])[0]), float(geom_results.get("p2", [0.0, 0.0])[1])],
        "reprojection_rms_px": [float(geom_results.get("rms", [0.0, 0.0])[0]), 0.0],
        "n_calibration_images_used": int(geom_results.get("n_views", 0)),
        "checkerboard_square_mm_measured": [float(geom_results.get("sq_mm", [19.0, 0.0])[0]), float(geom_results.get("sq_mm", [19.0, 0.0])[1])],

        "mtf50_center_cy_per_px": [float(res_results.get("mtf50_center", [0.0, 0.0])[0]), float(res_results.get("mtf50_center", [0.0, 0.0])[1])],
        "mtf50_corner_mean_cy_per_px": [float(res_results.get("mtf50_corner_mean", [0.0, 0.0])[0]), float(res_results.get("mtf50_corner_mean", [0.0, 0.0])[1])],
        "mtf50_corner_worst_cy_per_px": [float(res_results.get("mtf50_corner_worst", [0.0, 0.0])[0]), float(res_results.get("mtf50_corner_worst", [0.0, 0.0])[1])],

        "conversion_gain_dn_per_e": [float(noise_results.get("K", [0.0, 0.0])[0]), float(noise_results.get("K", [0.0, 0.0])[1])],
        "read_noise_e": [float(noise_results.get("read_noise", [0.0, 0.0])[0]), float(noise_results.get("read_noise", [0.0, 0.0])[1])],
        "dark_current_e_per_s": [float(noise_results.get("dark_current", [0.0, 0.0])[0]), float(noise_results.get("dark_current", [0.0, 0.0])[1])],
        "iso_of_noise_measurement": int(noise_results.get("iso", 100)),
        "black_level_dn": [float(noise_results.get("black_level", [64.0, 0.0])[0]), 0.0],
        "white_level_dn": int(noise_results.get("white_level", 1023)),
        "vendor_noiseprofile_present": bool(noise_results.get("vendor_noiseprofile_present", False)),

        "dynamic_range_stops": [float(noise_results.get("dr_stops", [0.0, 0.0])[0]), float(noise_results.get("dr_stops", [0.0, 0.0])[1])],
        "dynamic_range_criterion": str(noise_results.get("dr_criterion", "SNR = 1 at low end")),

        "delta_e_metric": "CIEDE2000",
        "colorchecker_reference_set": "ColorChecker24 - After November 2014",
        "delta_e_mean_before_ccm": [float(color_results.get("de_mean_before", [0.0, 0.0])[0]), float(color_results.get("de_mean_before", [0.0, 0.0])[1])],
        "delta_e_mean_after_ccm": [float(color_results.get("de_mean_after", [0.0, 0.0])[0]), float(color_results.get("de_mean_after", [0.0, 0.0])[1])],
        "delta_e_worst_after_ccm": [float(color_results.get("de_worst_after", [0.0, 0.0])[0]), float(color_results.get("de_worst_after", [0.0, 0.0])[1])]
    }
    
    save_json_results(results_payload, output_dir, filename="results.json")
