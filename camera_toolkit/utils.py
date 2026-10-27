# src/camera_toolkit/utils.py
import os
import json
import sys
import cv2
import numpy as np
import rawpy
import tifffile

# Standard numeric EXIF and TIFF tag IDs
MODEL_ID = 272              # Root TIFF tag for Device Model
FOCAL_LENGTH_ID = 37386     # Focal length tag
FOCAL_35MM_ID = 41989       # 35mm equivalent tag
F_NUMBER_ID = 33437         # F-Number tag
ISO_ID = 34855              # ISO speed ratings tag
EXPOSURE_TIME_ID = 33434    # Exposure time (shutter speed) tag
SUBJECT_DISTANCE_ID = 37382  # Subject distance (focus) tag


def verify_environment():
    """Validates core environment library installations."""
    required_packages = ["numpy", "cv2", "docx", "yaml", "matplotlib", "tifffile", "rawpy"]
    missing_packages = [pkg for pkg in required_packages if pkg not in sys.modules and __import__(pkg) is None]
    if missing_packages:
        print(f" Missing dependencies: {missing_packages}")
        print("Please run: uv add pyyaml python-docx rawpy tifffile")
        sys.exit(1)
    print("Environment Validation Passed.")


def verify_true_bayer_raw(file_path: str):
    """
    Inspects a DNG file container to confirm it contains a genuine un-demosaiced 
    Bayer mosaic grid. Prevents pre-processed files from throwing off models [INDEX: 0.1.5].
    """
    try:
        with rawpy.imread(file_path) as raw:
            # 1. True raw files must be an un-demosaiced 2D single-channel mosaic grid
            raw_shape = raw.raw_image.shape
            if len(raw_shape) != 2:
                raise ValueError(
                    f"Invalid Array Shape {raw_shape}: Image is already multi-channel. "
                    f"It has been demosaiced by an external ISP framework."
                )
                
            # 2. Check Color Filter Array layout tags: Must expose a standard 4-channel pattern layout
            color_description = raw.color_desc
            if len(color_description) != 4 or b'R' not in color_description:
                raise ValueError(
                    f"Invalid CFA Tag Layout '{color_description}': "
                    f"Does not match standard uncompressed color filter geometries."
                )
                
    except Exception as error:
        raise AssertionError(
            f"Data Validation Failure for file '{os.path.basename(file_path)}':\n"
            f"-> Reason: {str(error)}\n"
            f"-> Note: Apple ProRAW or standard JPEGs cannot be used as source sensor raw files "
            f"as their raw Bayer values have already been altered [INDEX: 0.1.5]."
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
    """Extracts raw metadata from DNG containers after executing a mandatory Bayer audit check [INDEX: 0.1.2, 0.1.5]."""
    # Defensive safeguard check runs right here before any headers are trusted! [INDEX: 0.1.5]
    verify_true_bayer_raw(path)
    
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]

        device_model = p.tags.get(MODEL_ID)
        if device_model:
            device_model = device_model.value
            if isinstance(device_model, bytes):
                device_model = device_model.decode("utf-8", errors="ignore").strip()
        else:
            device_model = "Google Pixel"

        if "ExifTag" in p.tags:
            exif = p.tags["ExifTag"].value
        else:
            exif = {tag.code: tag.value for tag in p.tags.values()}

        fl = exif.get(FOCAL_LENGTH_ID) or exif.get("FocalLength")
        if fl and isinstance(fl, (tuple, list)) and len(fl) == 2:
            calculated_fl = round(fl[0] / fl[1], 3)
        else:
            calculated_fl = fl if fl else 0.0

        f_num = exif.get(F_NUMBER_ID) or exif.get("FNumber")
        if f_num and isinstance(f_num, (tuple, list)) and len(f_num) == 2:
            calculated_f = round(f_num[0] / f_num[1], 2)
        else:
            calculated_f = f_num if f_num else None

        focal_35mm = exif.get(FOCAL_35MM_ID) or exif.get("FocalLengthIn35mmFilm", None)
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
    """Writes pipeline metrics cleanly out to your required results JSON deliverable [INDEX: 0.1.3]."""
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, filename)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
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
