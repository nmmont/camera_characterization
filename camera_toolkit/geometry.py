# src/camera_toolkit/geometry.py
import cv2
import numpy as np


def generate_object_points(rows: int, cols: int, square_size_mm: float) -> np.ndarray:
    """
    Generates the ideal 3D grid spatial coordinates (X, Y, Z) for the checkerboard target.
    The Z-coordinate for all points is initialized to 0.0 because the grid pattern is flat.
    """
    obj_p = np.zeros((rows * cols, 3), dtype=np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size_mm
    return obj_p


def detect_checkerboard_corners(image_array: np.ndarray, rows: int, cols: int) -> tuple:
    """
    Locates and refines checkerboard feature intersections at full resolution.
    """
    if len(image_array.shape) == 3:
        gray_full = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = image_array.copy()

    if gray_full.dtype == np.uint16:
        gray_full = cv2.normalize(gray_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    success, corners = cv2.findChessboardCorners(gray_full, (cols, rows), flags)
    
    if not success:
        return False, None
        
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), crit)
    
    return True, refined_corners


def run_extended_calibration(all_object_points: list, all_image_points: list, image_shape: tuple) -> dict:
    """
    Runs Zhang's calibration method with CALIB_FIX_K3 to prevent corner extrapolation errors.
    Calculates analytical parameter uncertainties.
    """
    # Fix k3 to 0.0 to prevent overfitting and extreme corner distortion extrapolation
    flags = cv2.CALIB_FIX_K3

    rms, K, dist, rvecs, tvecs, sd_int, sd_ext, per_view = cv2.calibrateCameraExtended(
        all_object_points, 
        all_image_points, 
        image_shape, 
        None, 
        None,
        flags=flags
    )
    
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    
    sfx, sfy, scx, scy, sk1, sk2, sp1, sp2, sk3 = sd_int.ravel()[:9]
    
    per_view_errors = compute_per_view_reprojection_errors(
        all_object_points, all_image_points, K, dist, rvecs, tvecs
    )
    
    return {
        "rms_reprojection_error": [float(rms), 0.0],
        "focal_length_x": [float(fx), float(sfx)],
        "focal_length_y": [float(fy), float(sfy)],
        "principal_point_x": [float(cx), float(scx)],
        "principal_point_y": [float(cy), float(scy)],
        "radial_distortion_k1": [float(k1), float(sk1)],
        "radial_distortion_k2": [float(k2), float(sk2)],
        "radial_distortion_k3": [float(k3), float(sk3)],
        "tangential_distortion_p1": [float(p1), float(sp1)],
        "tangential_distortion_p2": [float(p2), float(sp2)],
        "per_view_rms_errors": per_view_errors.tolist(),
        "n_views": len(all_image_points)
    }


def compute_per_view_reprojection_errors(all_object_points: list, all_image_points: list, K: np.ndarray, dist: np.ndarray, rvecs: list, tvecs: list) -> np.ndarray:
    """
    Computes per-view RMS reprojection errors to identify problematic face-on or warped frames.
    """
    errors = []
    for i in range(len(all_object_points)):
        img_points_projected, _ = cv2.projectPoints(
            all_object_points[i], rvecs[i], tvecs[i], K, dist
        )
        err = cv2.norm(all_image_points[i], img_points_projected, cv2.NORM_L2) / np.sqrt(len(img_points_projected))
        errors.append(err)
    return np.array(errors, dtype=np.float32)


def rectify_frame(image_array: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Remaps pixel coordinates using optimized camera profiles to reverse lens warping.
    """
    h, w = image_array.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, 
        dist_coeffs, 
        (w, h), 
        alpha, 
        (w, h)
    )
    undistorted_img = cv2.undistort(image_array, camera_matrix, dist_coeffs, None, new_K)
    x, y, crop_w, crop_h = roi
    if alpha < 1.0 and crop_w > 0 and crop_h > 0:
        return undistorted_img[y:y+crop_h, x:x+crop_w]
        
    return undistorted_img


def max_perpendicular_deviation(points: np.ndarray) -> float:
    """
    Calculates the maximum perpendicular distance from a set of 2D points 
    to their first-order linear polynomial best-fit line.
    """
    x, y = points[:, 0], points[:, 1]
    m, b = np.polyfit(x, y, 1)
    distances = np.abs(m * x - y + b) / np.sqrt(m**2 + 1)
    return float(np.max(distances))


def evaluate_straightness_improvement(all_image_points: list, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, target_row_idx: int = 2, cols_per_row: int = 9) -> tuple:
    """
    Quantifies the mathematical straightness correction for a specific checkerboard row.
    """
    start_idx = target_row_idx * cols_per_row
    end_idx = (target_row_idx + 1) * cols_per_row
    
    row_corners = all_image_points[0][start_idx:end_idx].reshape(-1, 2)
    undistorted_corners_nested = cv2.undistortPoints(
        row_corners.reshape(-1, 1, 2), 
        camera_matrix, 
        dist_coeffs, 
        P=camera_matrix
    )
    undistorted_corners = undistorted_corners_nested.reshape(-1, 2)
    
    deviation_before = max_perpendicular_deviation(row_corners)
    deviation_after = max_perpendicular_deviation(undistorted_corners)
    
    return deviation_before, deviation_after


def calculate_pixel_pitch_metrics(f_px_x: float, f_px_y: float, f_mm: float, f_35mm: float, image_shape: tuple) -> dict:
    """
    Cross-validates sensor pixel pitch using two independent models:
    Method 1: Empirical calibration focal length vs. EXIF physical focal length.
    Method 2: Crop factor & 35mm film equivalent projection (43.3 mm diagonal).
    """
    if f_px_x <= 0 or f_px_y <= 0 or f_mm <= 0 or f_35mm <= 0:
        raise ValueError("Focal length and pixel counts must be non-zero positive numbers.")
        
    f_mean_px = (f_px_x + f_px_y) / 2.0
    pitch_method1_um = (f_mm * 1000.0) / f_mean_px
    
    crop_factor = f_35mm / f_mm
    sensor_diagonal_mm = 43.3 / crop_factor
    
    w, h = image_shape[0], image_shape[1]
    image_diagonal_px = np.sqrt(w ** 2 + h ** 2)
    pitch_method2_um = (sensor_diagonal_mm * 1000.0) / image_diagonal_px
    
    pct_diff = (
        abs(pitch_method1_um - pitch_method2_um)
        / ((pitch_method1_um + pitch_method2_um) / 2.0)
    ) * 100.0
    
    return {
        "mean_focal_length_px": f_mean_px,
        "pixel_pitch_method1_um": pitch_method1_um,
        "pixel_pitch_method2_um": pitch_method2_um,
        "pitch_percentage_difference": pct_diff
    }
