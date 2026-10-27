# src/camera_toolkit/geometry.py
import cv2
import numpy as np

# ==========================================
# 1. CORE SOLVERS & LOW-LEVEL ENGINES
# ==========================================

def generate_object_points(rows: int, cols: int, square_size_mm: float) -> np.ndarray:
    """
    Generates the ideal 3D grid spatial coordinates (X, Y, Z) for the checkerboard target.
    The Z-coordinate for all points is initialized to 0.0 because the grid pattern is flat.
    """
    obj_p = np.zeros((rows * cols, 3), dtype=np.float32)
    # Match your grid transpose logic to coordinate shapes
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj_p *= square_size_mm
    return obj_p

def detect_checkerboard_corners(image_array: np.ndarray, rows: int, cols: int) -> tuple:
    """
    Locates and refines checkerboard feature intersections at full resolution.
    Uses hardware-accelerated normalization to drop runtime down to milliseconds.
    """
    # 1. Isolate single channel
    if len(image_array.shape) == 3:
        gray_full = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray_full = image_array.copy()

    # 2. FAST HARDWARE SCALING: Convert 16-bit RAW to 8-bit image instantly
    if gray_full.dtype == np.uint16:
        # cv2.normalize is written in native C++ and optimized for large arrays
        gray_full = cv2.normalize(gray_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. Locate corners on full pixel grid
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    success, corners = cv2.findChessboardCorners(gray_full, (cols, rows), flags)
    
    if not success:
        return False, None
        
    # 4. High-precision sub-pixel refinement window
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined_corners = cv2.cornerSubPix(gray_full, corners, (11, 11), (-1, -1), crit)
    
    return True, refined_corners

def run_extended_calibration(all_object_points: list, all_image_points: list, image_shape: tuple) -> dict:
    """
    Runs Zhang's calibration method to extract intrinsics, distortions, and analytical uncertainties.
    
    Inputs:
        image_shape: (width, height) pixel dimensions
    Returns:
        dict: Populated values and standard deviations paired for results.json saving [INDEX: 0.1.3]
    """
    # Extended solver calculates standard deviations for uncertainty parameters [INDEX: 0.1.3]
    rms, K, dist, rvecs, tvecs, sd_int, sd_ext, per_view = cv2.calibrateCameraExtended(
        all_object_points, 
        all_image_points, 
        image_shape, 
        None, 
        None
    )
    
    # Unpack intrinsic matrices parameters
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist.ravel()[:5]
    
    # Unpack matching calculated parameter standard deviations
    sfx, sfy, scx, scy, sk1, sk2, sp1, sp2, sk3 = sd_int.ravel()[:9]
    
    # Format into structured list pairs: [value, uncertainty] for results.json compliance [INDEX: 0.1.3]
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
        "tangential_distortion_p2": [float(p2), float(sp2)]
    }

# ==========================================
# 2. POST-PROCESSING & RECTIFICATION
# ==========================================

def rectify_frame(image_array: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Remaps pixel coordinates using optimized camera profiles to reverse lens warping.
    
    Inputs:
        alpha = 0.0: Crops out all outer black pixel borders (zooms in slightly)
        alpha = 1.0: Retains all original pixel data, preserving black unwarped borders
    """
    h, w = image_array.shape[:2]
    
    # 1. Compute the optimized new intrinsic camera matrix
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, 
        dist_coeffs, 
        (w, h), 
        alpha, 
        (w, h)
    )
    
    # 2. Undistort using the optimized mapping matrix
    undistorted_img = cv2.undistort(image_array, camera_matrix, dist_coeffs, None, new_K)
    
    # 3. If an inner crop is requested (alpha=0.0), crop to the valid ROI boundary box
    x, y, crop_w, crop_h = roi
    if alpha < 1.0 and crop_w > 0 and crop_h > 0:
        return undistorted_img[y:y+crop_h, x:x+crop_w]
        
    return undistorted_img

# ==========================================
# 3. VALIDATION, DIAGNOSTICS, & METRICS
# ==========================================

def max_perpendicular_deviation(points: np.ndarray) -> float:
    """
    Calculates the maximum perpendicular distance from a set of 2D points 
    to their first-order linear polynomial best-fit line.
    
    Inputs:
        points: NumPy array of shape (N, 2) representing coordinate positions.
    Returns:
        float: Maximum spatial error deviation in pixels.
    """
    x, y = points[:, 0], points[:, 1]
    
    # Fit a straight line (y = mx + b) through the spatial points coordinates
    m, b = np.polyfit(x, y, 1)
    
    # Analytical equation for perpendicular distance from a point to a line:
    # d = |m*x - y + b| / sqrt(m^2 + 1)
    distances = np.abs(m * x - y + b) / np.sqrt(m**2 + 1)
    return float(np.max(distances))

def evaluate_straightness_improvement(all_image_points: list, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, target_row_idx: int = 2, cols_per_row: int = 9) -> tuple:
    """
    Quantifies the mathematical straightness correction for a specific checkerboard row.
    Passes corners through cv2.undistortPoints to compute sub-pixel error reduction.
    
    Returns:
        tuple: (max_deviation_before_px, max_deviation_after_px)
    """
    # 1. Isolate the target row coordinate chunk from the very first calibration frame capture
    start_idx = target_row_idx * cols_per_row
    end_idx = (target_row_idx + 1) * cols_per_row
    
    # Extract raw 2D pixel coordinates for that horizontal target row
    row_corners = all_image_points[0][start_idx:end_idx].reshape(-1, 2)
    
    # 2. Map coordinates through the inverse lens distortion polynomial functions
    # Using P=camera_matrix retains the coordinates in the baseline pixel frame space
    undistorted_corners_nested = cv2.undistortPoints(
        row_corners.reshape(-1, 1, 2), 
        camera_matrix, 
        dist_coeffs, 
        P=camera_matrix
    )
    undistorted_corners = undistorted_corners_nested.reshape(-1, 2)
    
    # 3. Compute perpendicular metric errors for both states
    deviation_before = max_perpendicular_deviation(row_corners)
    deviation_after = max_perpendicular_deviation(undistorted_corners)
    
    return deviation_before, deviation_after
# Add this to src/camera_toolkit/geometry.py

def calculate_pixel_pitch_metrics(f_px_x: float, f_px_y: float, f_mm: float, f_35mm: float, image_shape: tuple) -> dict:
    """
    Cross-validates sensor pixel pitch using two independent models:
    Method 1: Uses your empirical calibration matrix vs. EXIF physical focal length.
    Method 2: Uses the 35mm film equivalent crop factor projection.
    
    Returns:
        dict: Compiled micron values and percentage error differences for your report.
    """
    # Defensive parameter checks utilizing the resolved configuration value passed from main
    if f_px_x <= 0 or f_px_y <= 0 or f_mm <= 0 or f_35mm <= 0:
        raise ValueError("Focal length and pixel counts must be non-zero positive numbers.")
        
    # 1. Method 1: Calibration focal length & EXIF physical focal length
    f_mean_px = (f_px_x + f_px_y) / 2.0
    pitch_method1_um = (f_mm * 1000.0) / f_mean_px
    
    # 2. Method 2: Crop factor & 35mm film standard diagonal (43.3 mm)
    crop_factor = f_35mm / f_mm
    sensor_diagonal_mm = 43.3 / crop_factor
    
    # # SAFE UNPACKING LOGIC: Handles tuple variables or nested list dimensions cleanly
    # if isinstance(image_shape, (tuple, list)) and len(image_shape) >= 2:
    #     w, h = image_shape[0], image_shape[1]
    # elif hasattr(image_shape, '__getitem__'):
    #     w, h = image_shape[0], image_shape[1]
    # else:
    #     raise TypeError(f"Expected image_shape to be a subscriptable tuple/list, got {type(image_shape)}")
    try:
        w, h = image_shape[0], image_shape[1]
    except (TypeError, IndexError):
        raise TypeError(f"Expected image_shape to be a subscriptable tuple/list with at least 2 elements, got {type(image_shape)}")
   
    image_diagonal_px = np.sqrt(w ** 2 + h ** 2)
    pitch_method2_um = (sensor_diagonal_mm * 1000.0) / image_diagonal_px
    
    # 3. Calculate cross-validation error metrics
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

