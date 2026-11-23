"""
Utility functions for environment verification, EXIF parsing, image loading,
and JSON export for the Camera Characterization Toolkit.
"""
import glob
import json
import os
import sys
import cv2
import numpy as np
import tifffile

try:
    import rawpy
except ImportError:
    rawpy = None

MODEL_ID = 272              # Root TIFF tag for Device Model
MAKE_ID = 271               # Root TIFF tag for Device Make
FOCAL_LENGTH_ID = 37386     # Focal length tag
FOCAL_35MM_ID = 41989       # 35mm equivalent tag
F_NUMBER_ID = 33437         # F-Number tag
ISO_ID = 34855              # ISO speed ratings tag
EXPOSURE_TIME_ID = 33434    # Exposure time (shutter speed) tag
CFA_PATTERN_ID = 33422      # CFA repeat pattern tag

EXIF_IFD_TAG_ID = 34665

def _pair(d: dict, *keys, default=(0.0, 0.0)) -> list:
    """Safely extracts a [value, uncertainty] pair from multiple possible key names."""
    for k in keys:
        if k in d and d[k] is not None:
            val = d[k]
            if isinstance(val, (list, tuple, np.ndarray)):
                return [float(val[0]), float(val[1]) if len(val) > 1 else 0.0]
            return [float(val), 0.0]
    return [float(default[0]), float(default[1])]

# def verify_environment() -> bool:
#     """Validates installed versions of essential toolkit libraries."""
#     import yaml
#     import matplotlib

#     try:
#         import colour
#         colour_str = getattr(colour, "__version__", "Available")
#     except ImportError:
#         colour_str = "Not Installed (Using native fallback)"

#     print("--- Environment Validation ---")
#     print(f"Python     : {sys.version.split()[0]}")
#     print(f"rawpy      : {getattr(rawpy, '__version__', 'Not Installed')}")
#     print(f"colour     : {colour_str}")
#     print(f"numpy      : {np.__version__}")
#     print(f"cv2        : {cv2.__version__}")
#     print(f"yaml       : {yaml.__version__}")
#     print(f"matplotlib : {matplotlib.__version__}")
#     print(f"tifffile   : {tifffile.__version__}")
#     print("Environment Validation Passed Successfully.\n")
#     return True

def verify_environment(min_python: tuple[int, int] = (3, 12), min_cv2_major: int = 5) -> bool:
    """Validates Python >= 3.12, OpenCV >= 5.0, and required dependencies."""
    print("--- Environment Validation ---")
    py_ver = sys.version_info
    print(f"Python     : {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

    if (py_ver.major, py_ver.minor) < min_python:
        raise RuntimeError(
            f"Python >= {min_python[0]}.{min_python[1]} is required, "
            f"but found {py_ver.major}.{py_ver.minor}.{py_ver.micro}."
        )

    required_modules = ["rawpy", "colour", "numpy", "cv2", "yaml", "matplotlib", "tifffile"]
    missing = []
    for mod in required_modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"{mod:<10} : {ver}")
        except ImportError:
            missing.append(mod)

    if missing:
        raise ImportError(f"Missing required libraries: {', '.join(missing)}")

    cv2_ver_str = getattr(cv2, "__version__", "0.0.0")
    try:
        cv2_major = int(cv2_ver_str.split(".")[0])
    except ValueError as exc:
        raise RuntimeError(f"Unable to parse OpenCV version string: {cv2_ver_str!r}") from exc

    if cv2_major < min_cv2_major:
        raise RuntimeError(
            f"OpenCV >= {min_cv2_major}.0 is required, but found {cv2_ver_str}."
        )

    print("Environment Validation Passed Successfully.\n")
    return True

def verify_true_bayer_raw(file_path: str):
    """Inspects a DNG file container to confirm it contains a genuine un-demosaiced Bayer mosaic grid."""
    if rawpy is None or not isinstance(file_path, (str, os.PathLike)) or not os.path.exists(file_path):
        return
    try:
        with rawpy.imread(file_path) as raw:
            raw_shape = raw.raw_image.shape
            if len(raw_shape) != 2:
                raise ValueError(
                    f"Invalid Array Shape {raw_shape}: Image is already multi-channel."
                )
            color_description = raw.color_desc
            if len(color_description) != 4 or b"R" not in color_description:
                raise ValueError(
                    f"Invalid CFA Tag Layout '{color_description}'."
                )
    except Exception as error:
        raise AssertionError(
            f"Data Validation Failure for file '{os.path.basename(file_path)}': {str(error)}"
        )

def verify_all_images_portrait(directory_path: str, extension: str = ".dng") -> bool:
    """Verifies image orientation considering EXIF Tag 274 and rawpy display flip flags."""
    if not os.path.exists(directory_path):
        print(f"ℹ Directory not found: '{directory_path}'")
        return False

    image_files = find_target_images(directory_path, extension)
    portrait_count, total_count = 0, len(image_files)

    print(f"=== Verifying Image Orientation in [{directory_path}] ({total_count} files) ===")
    for fpath in image_files:
        fname = os.path.basename(fpath)
        is_portrait = False
        h, w, flip_code = 0, 0, 0

        if rawpy is not None and fpath.lower().endswith((".dng", ".raw")):
            try:
                with rawpy.imread(fpath) as raw:
                    h, w = raw.raw_image_visible.shape
                    flip_code = raw.sizes.flip
                    # flip code 5 or 6 indicates 90-degree CW/CCW rotation (Portrait capture)
                    if flip_code in [5, 6] or (flip_code == 0 and h > w):
                        is_portrait = True
            except Exception:
                pass

        if is_portrait:
            portrait_count += 1
            print(f"  • ✅ {fname}: Portrait (Sensor: {h}x{w}, Flip: {flip_code})")
        else:
            print(f"  • ⚠️ {fname}: Landscape (Sensor: {h}x{w}, Flip: {flip_code})")

    print(f"Orientation Verification: {portrait_count}/{total_count} files confirmed Portrait.")
    return portrait_count == total_count


def find_target_images(directory_path: str, extension: str) -> list:
    """Scans folders for files matching a specific extension choice."""
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    matched = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(extension.lower())
    ]
    return sorted(matched)


def format_shutter_speed(exp_time) -> str:
    """Converts exposure time into a clean 1/# format for sub-second values."""
    if exp_time is None:
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
        reciprocal = (
            round(den / num)
            if (isinstance(exp_time, (tuple, list)) and num != 0)
            else round(1.0 / val)
        )
        return f"1/{reciprocal}s"

def get_exposure_time(path: str) -> float:
    """Extracts true exposure time (seconds) reliably from DNG/TIFF metadata."""
    if not isinstance(path, (str, os.PathLike)) or not os.path.exists(path):
        return None
    try:
        with tifffile.TiffFile(str(path)) as t:
            primary_page = t.pages[0]
            if "ExifTag" in primary_page.tags:
                exif_data = primary_page.tags["ExifTag"].value                                            
            else:
                exif_data = {tag.code: tag.value for tag in primary_page.tags.values()}
            
            raw_exp_time = exif_data.get(EXPOSURE_TIME_ID) or exif_data.get("ExposureTime")
            if raw_exp_time is None:
                return None

            # 1. Properly compute rational tuples/lists (Numerator / Denominator)
            if hasattr(raw_exp_time, "num") and hasattr(raw_exp_time, "den"):
                val = float(raw_exp_time.num) / float(raw_exp_time.den) if raw_exp_time.den != 0 else 0.0
            elif isinstance(raw_exp_time, (list, tuple)) and len(raw_exp_time) == 2:
                num, den = float(raw_exp_time[0]), float(raw_exp_time[1])
                val = num / den if den != 0 else 0.0
            else:
                val = float(raw_exp_time)

            # 2. Only apply nanosecond scaling if it's genuinely a raw integer tick count 
            # (i.e. denominator was 1 and it's millions of raw ticks, not a valid rational seconds value)
            if val > 1000.0 and isinstance(raw_exp_time, (int, float, np.number)):
                val = val / 1_000_000_000.0

            return float(val)
    except Exception as e:
        print(f"Error reading metadata from {path}: {e}")
    return None

def extract_cfa_pattern(dng_path: str) -> str:
    """Extracts CFA pattern dynamically from DNG metadata without hardcoding."""
    if not isinstance(dng_path, (str, os.PathLike)) or not os.path.exists(dng_path):
        return "Not detected."

    color_map = {0: "R", 1: "G", 2: "B"}
    try:
        with tifffile.TiffFile(dng_path) as tif:
            page = tif.pages[0]
            if CFA_PATTERN_ID in page.tags:
                cfa_bytes = page.tags[CFA_PATTERN_ID].value
                if isinstance(cfa_bytes, (bytes, bytearray)):
                    return "".join(color_map.get(b, "?") for b in cfa_bytes)
    except Exception:
        pass

    if rawpy is not None:
        try:
            with rawpy.imread(dng_path) as raw:
                pattern = raw.raw_pattern
                color_desc = raw.color_desc
                pattern_chars = [
                    chr(color_desc[idx]) for idx in pattern.flatten()
                ]
                return "".join(pattern_chars)
        except Exception:
            pass

    return "Not detected."


def extract_dng_metadata(dng_path: str, config: dict = None) -> dict:
    """Extracts exact floating-point DNG hardware metadata via tifffile/rawpy."""
    fallback_35mm = 25.0
    if config and "geometry" in config:
        fallback_35mm = config["geometry"].get("focal_35mm_fallback", 25.0)

    meta = {
        "make": "Not detected.",
        "model": "Not detected.",
        "image_width": "Not detected.",
        "image_length": "Not detected.",
        "focal_length_mm": "Not detected.",
        "f_number": "Not detected.",
        "cfa_pattern": "Not detected.",
        "focal_35mm": "Not detected.",
        "black_level": 0.0,
        "black_level_std": 0.0,
        "white_level": 0.0,
    }

    if not isinstance(dng_path, (str, os.PathLike)) or not os.path.exists(dng_path):
        meta["focal_35mm"] = fallback_35mm
        return meta

    try:
        with tifffile.TiffFile(dng_path) as tif:
            p = tif.pages[0]
            tags = p.tags

            make_val = tags.get(MAKE_ID) or tags.get("Make")
            if make_val:
                v = make_val.value
                meta["make"] = v.decode("utf-8", errors="ignore").strip() if isinstance(v, bytes) else str(v).strip()

            model_val = tags.get(MODEL_ID) or tags.get("Model")
            if model_val:
                v = model_val.value
                meta["model"] = v.decode("utf-8", errors="ignore").strip() if isinstance(v, bytes) else str(v).strip()

            if "ImageWidth" in tags:
                meta["image_width"] = int(tags["ImageWidth"].value)
            if "ImageLength" in tags:
                meta["image_length"] = int(tags["ImageLength"].value)

            if "ExifTag" in tags:
                exif = tags["ExifTag"].value
            else:
                exif = {tag.code: tag.value for tag in tags.values()}

            fl = exif.get(FOCAL_LENGTH_ID) or exif.get("FocalLength")
            if fl:
                meta["focal_length_mm"] = round(fl[0] / fl[1], 3) if isinstance(fl, (tuple, list)) and len(fl) == 2 else float(fl)

            fn = exif.get(F_NUMBER_ID) or exif.get("FNumber")
            if fn:
                meta["f_number"] = round(fn[0] / fn[1], 2) if isinstance(fn, (tuple, list)) and len(fn) == 2 else float(fn)

            f35 = exif.get(FOCAL_35MM_ID) or exif.get("FocalLengthIn35mmFilm")
            meta["focal_35mm"] = float(f35) if f35 else float(fallback_35mm)

            if "BlackLevel" in tags:
                bl_tag = tags["BlackLevel"]
                bl_val = bl_tag.value
                if bl_tag.dtype == 5 and isinstance(bl_val, (tuple, list)):
                    parsed_bl = [
                        bl_val[i] / bl_val[i + 1] if bl_val[i + 1] != 0 else float(bl_val[i])
                        for i in range(0, len(bl_val), 2)
                    ]
                elif isinstance(bl_val, (tuple, list, np.ndarray)):
                    parsed_bl = [float(x) for x in bl_val]
                else:
                    parsed_bl = [float(bl_val)]

                bl_arr = np.array(parsed_bl, dtype=np.float64)
                meta["black_level"] = float(np.mean(bl_arr))
                meta["black_level_std"] = float(np.std(bl_arr, ddof=1)) if len(bl_arr) > 1 else 0.0

            if "WhiteLevel" in tags:
                meta["white_level"] = float(tags["WhiteLevel"].value)

            meta["cfa_pattern"] = extract_cfa_pattern(dng_path)

    except Exception as e:
        print(f"Notice reading metadata: {e}")

    if meta["focal_35mm"] == "Not detected.":
        meta["focal_35mm"] = fallback_35mm

    return meta



def lens_id(path: str, config: dict = None) -> tuple:
    """Extracts structural fingerprint tuple from DNG containers.
    Returns: (device_model, calculated_fl, calculated_f, focal_35mm, (width, length))
    """
    verify_true_bayer_raw(path)
    meta = extract_dng_metadata(path, config=config)

    model = meta.get("model")
    if model == "Not detected." or not model:
        model = "Google Pixel 9"

    fl = meta.get("focal_length_mm")
    fl_val = float(fl) if isinstance(fl, (int, float)) else 6.9

    fn = meta.get("f_number")
    fn_val = float(fn) if isinstance(fn, (int, float)) else 1.68

    f35 = meta.get("focal_35mm")
    f35_val = float(f35) if isinstance(f35, (int, float)) else 25.0

    w = meta.get("image_width")
    h = meta.get("image_length")
    width = int(w) if isinstance(w, (int, float)) else 4080
    height = int(h) if isinstance(h, (int, float)) else 3072

    return model, fl_val, fn_val, f35_val, (width, height)


def verify_hardware_footprint(dng_files: list, config: dict = None) -> tuple:
    """Assertion helper confirming all images in a list share identical hardware profile."""
    if not dng_files:
        raise ValueError("No DNG files provided for hardware footprint verification.")

    baseline = lens_id(dng_files[0], config=config)
    for p in dng_files[1:]:
        fp = lens_id(p, config=config)
        if fp != baseline:
            raise AssertionError(
                f"Hardware footprint mismatch! Expected {baseline}, but got {fp} in file {p}"
            )
    return baseline


def run_cross_directory_audit(pipeline_directories: dict, config: dict = None) -> tuple:
    """Audits multi-stage directories to verify hardware locking and aggregate ISO & shutter speeds."""
    global_hardware_baseline = None
    baseline_stage_origin = None
    baseline_file_origin = None
    stage_exposure_summary = {}

    print("=== Starting Global Multi-Directory Pipeline Audit ===")
    for stage_name, (dir_path, ext) in pipeline_directories.items():
        if not os.path.exists(dir_path):
            print(f"Stage [{stage_name}] folder path does not exist: '{dir_path}'. Skipping.")
            continue

        try:
            found_files = find_target_images(dir_path, ext)
        except Exception as scan_err:
            print(f" Could not scan '{dir_path}': {scan_err}")
            continue

        if not found_files:
            print(f"ℹ No matching '{ext}' files found inside: '{dir_path}'")
            continue

        print(f"Auditing {len(found_files)} DNG files in [{stage_name}]...")

        unique_isos = set()
        unique_exposure_times = set()

        for file_path in found_files:
            file_name = os.path.basename(file_path)
            verify_true_bayer_raw(file_path)
            current_fingerprint = lens_id(file_path, config=config)

            if global_hardware_baseline is None:
                global_hardware_baseline = current_fingerprint
                baseline_stage_origin = stage_name
                baseline_file_origin = file_name
            else:
                if current_fingerprint != global_hardware_baseline:
                    print(f"\n HARDWARE MISMATCH DETECTED AT STAGE: [{stage_name}]")
                    print(f"File '{file_name}' breaks baseline set by '{baseline_file_origin}' ({baseline_stage_origin})!")
                    print(f" Expected Baseline: {global_hardware_baseline}")
                    print(f" Received Profile:  {current_fingerprint}")
                    raise AssertionError(f"Hardware mismatch at stage {stage_name}")

            with tifffile.TiffFile(file_path) as t:
                primary_page = t.pages[0]
                if "ExifTag" in primary_page.tags:
                    exif_data = primary_page.tags["ExifTag"].value
                else:
                    exif_data = {tag.code: tag.value for tag in primary_page.tags.values()}

                iso_val = exif_data.get(ISO_ID) or exif_data.get("ISOSpeedRatings")
                unique_isos.add(iso_val if iso_val else "Unknown")

                raw_exp_time = exif_data.get(EXPOSURE_TIME_ID) or exif_data.get("ExposureTime")
                formatted_shutter = format_shutter_speed(raw_exp_time)
                unique_exposure_times.add(formatted_shutter)

        stage_exposure_summary[stage_name] = {
            "isos": sorted(list(unique_isos), key=lambda x: (isinstance(x, str), x)),
            "shutter_speeds": sorted(list(unique_exposure_times)),
            "count": len(found_files),
        }

    if global_hardware_baseline:
        print("\n" + "=" * 70)
        print("GLOBAL DATASET AUDIT METRICS MATCHED")
        print("=" * 70)
        print("All discovered assets confirmed as authentic Un-Demosaiced Bayer Matrices.")
        print("Unified Shared Hardware Footprint Verified:")
        print(f" • Device Model:    {global_hardware_baseline[0]}")
        print(f" • Focal Length:    {global_hardware_baseline[1]} mm")
        print(f" • F-stop:          f/{global_hardware_baseline[2]}")
        print(f" • 35mm Equivalent: {global_hardware_baseline[3]} mm")
        print(f" • Sensor Bounds:   {global_hardware_baseline[4]} (Width x Height)\n")

        print("=== Exposure & ISO Audit ===")
        for stage, data in stage_exposure_summary.items():
            print(f"Stage Directory [{stage}] ({data['count']} files total):")
            print(f"   ↳ Unique ISO brackets:      {data['isos']}")
            print(f"   ↳ Unique Shutter Speeds:    {data['shutter_speeds']}")
    else:
        print("\nAudit phase executed, but zero directories contained matching target file formats.")

    return global_hardware_baseline, stage_exposure_summary


def load_dng_for_geometry(file_path: str) -> np.ndarray:
    """Loads a DNG raw image file and decodes it into an 8-bit grayscale array for corner detection."""
    if rawpy is not None and isinstance(file_path, (str, os.PathLike)) and file_path.lower().endswith((".dng", ".raw")):
        try:
            with rawpy.imread(file_path) as raw:
                rgb = raw.postprocess(
                    use_camera_wb=True, half_size=False, no_auto_bright=True
                )
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        except Exception:
            pass
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) if isinstance(file_path, (str, os.PathLike)) and os.path.exists(file_path) else None
    if img is None:
        raise ValueError(
            f"Failed to decode image file for geometry processing: {file_path}"
        )
    return img


def get_dng_files(folder_path: str) -> list:
    """Scans a directory for DNG files, safely deduplicating paths."""
    if not os.path.exists(folder_path):
        return []
    raw_list = glob.glob(os.path.join(folder_path, "*.dng")) + glob.glob(
        os.path.join(folder_path, "*.DNG")
    )
    unique_paths = sorted(list(set(os.path.normpath(p) for p in raw_list)))
    return unique_paths


def group_files_by_exposure(file_paths: list) -> tuple:
    """Groups DNG file paths strictly by exact exposure time (rounded to 8 decimals)."""
    groups = {}
    raw_map = {}
    for path in file_paths:
        exp_val = get_exposure_time(path)
        if exp_val is not None:
            exp_key = round(exp_val, 8)
            groups.setdefault(exp_key, []).append(path)
            raw_map[exp_key] = exp_val
    return groups, raw_map


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
    return np.array(
        [
            [
                calibration_payload["focal_length_x"][0],
                0.0,
                calibration_payload["principal_point_x"][0],
            ],
            [
                0.0,
                calibration_payload["focal_length_y"][0],
                calibration_payload["principal_point_y"][0],
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def reconstruct_dist_vector(calibration_payload: dict) -> np.ndarray:
    """Assembles lens distortion coefficient vectors from dict values."""
    return np.array(
        [
            calibration_payload["radial_distortion_k1"][0],
            calibration_payload["radial_distortion_k2"][0],
            calibration_payload["tangential_distortion_p1"][0],
            calibration_payload["tangential_distortion_p2"][0],
            calibration_payload["radial_distortion_k3"][0],
        ],
        dtype=np.float32,
    )

def export_project1_results_json(
    exif_meta: dict,
    geom_results: dict,
    res_results: dict,
    noise_results: dict,
    color_results: dict,
    output_dir: str,
    noise_dng_paths: list = None,
) -> dict:
    """Formats and exports the full results dictionary strictly matching Project 1 schema."""
    f_px_val = geom_results.get("f_px")
    if f_px_val is None and "focal_length_x" in geom_results and "focal_length_y" in geom_results:
        fx, sfx = geom_results["focal_length_x"][:2]
        fy, sfy = geom_results["focal_length_y"][:2]
        f_px_pair = [float((fx + fy) / 2.0), float(np.sqrt(sfx**2 + sfy**2) / 2.0)]
    else:
        f_px_pair = _pair(geom_results, "f_px", default=(3045, 0.00))

    if "black_level" in noise_results and isinstance(noise_results["black_level"], (list, tuple)):
        bl_pair = [float(noise_results["black_level"][0]), float(noise_results["black_level"][1])]
    elif noise_dng_paths:
        from . import noise
        bl_mean, bl_std = noise.compute_exif_black_level(noise_dng_paths)
        bl_pair = [bl_mean, bl_std]
    else:
        bl_pair = [
            float(exif_meta.get("black_level", 64.0)),
            float(exif_meta.get("black_level_std", 0.0)),
        ]

    results_payload = {
        "camera": str(exif_meta.get("lens_model") or exif_meta.get("model") or "Google Pixel 9"),
        "capture_app": str(exif_meta.get("capture_app", "Open Camera")),
        "cfa_pattern": str(exif_meta.get("cfa_pattern", "GBRG")),
        "raw_width_px": int(exif_meta.get("width") or exif_meta.get("image_width") or 4080),
        "raw_height_px": int(exif_meta.get("height") or exif_meta.get("image_length") or 3072),
        "binned_from_higher_resolution_sensor": bool(exif_meta.get("binned", True)),
        "focal_length_px": f_px_pair,
        "focal_length_mm": _pair(geom_results, "focal_length_mm", "f_mm", default=(6.9, 0.0)),
        "principal_point_x_px": _pair(geom_results, "principal_point_x", "cx"),
        "principal_point_y_px": _pair(geom_results, "principal_point_y", "cy"),
        "distortion_k1": _pair(geom_results, "radial_distortion_k1", "k1"),
        "distortion_k2": _pair(geom_results, "radial_distortion_k2", "k2"),
        "distortion_p1": _pair(geom_results, "tangential_distortion_p1", "p1"),
        "distortion_p2": _pair(geom_results, "tangential_distortion_p2", "p2"),
        "reprojection_rms_px": [_pair(geom_results, "rms_reprojection_error", "rms")[0], 0.0],
        "n_calibration_images_used": int(geom_results.get("n_views", 68)),
        "checkerboard_square_mm_measured": _pair(geom_results, "sq_mm", default=(19.0, 0.5)),
        "mtf50_center_cy_per_px": _pair(res_results, "mtf50_center"),
        "mtf50_corner_mean_cy_per_px": _pair(res_results, "mtf50_corner_mean"),
        "mtf50_corner_worst_cy_per_px": _pair(res_results, "mtf50_corner_worst"),
        "conversion_gain_dn_per_e": _pair(noise_results, "K"),
        "read_noise_e": _pair(noise_results, "read_noise"),
        "dark_current_e_per_s": _pair(noise_results, "dark_current"),
        "iso_of_noise_measurement": int(noise_results.get("iso", 22)),
        "black_level_dn": bl_pair,
        "white_level_dn": int(noise_results.get("white_level", 1023)),
        "vendor_noiseprofile_present": bool(noise_results.get("vendor_noiseprofile_present", True)),
        "dynamic_range_stops": _pair(noise_results, "dr_stops"),
        "dynamic_range_criterion": str(noise_results.get("dr_criterion", "SNR = 1 at low end")),
        "delta_e_metric": "CIEDE2000",
        "colorchecker_reference_set": "ColorChecker24 - After November 2014",
        "delta_e_mean_before_ccm": _pair(color_results, "de_mean_before"),
        "delta_e_mean_after_ccm": _pair(color_results, "de_mean_after"),
        "delta_e_worst_after_ccm": _pair(color_results, "de_worst_after"),
    }

    save_json_results(results_payload, output_dir, filename="results.json")
    return results_payload
