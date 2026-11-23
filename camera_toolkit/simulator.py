"""
Synthetic data generator for camera characterization pipeline testing and benchmarking.
"""
import cv2
import numpy as np
from scipy.special import erf

def generate_synthetic_checkerboard_data(
    rows: int = 6,
    cols: int = 9,
    square_size_mm: float = 19.0,
    fx: float = 3055.61,
    fy: float = 3055.61,
    cx: float = 2070.0,
    cy: float = 1500.0,
    k1: float = 0.04,
    k2: float = -0.10,
    p1: float = 0.001,
    p2: float = 0.001,
    k3: float = 0.10,
    noise_std_px: float = 0.25,
    seed: int = 42,
) -> tuple:
    """Generates synthetic 3D object points and projected 2D image points for testing Zhang's solver."""
    rng = np.random.default_rng(seed)
    obj_p = np.zeros((rows * cols, 3), dtype=np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    rvec = np.array([0.1, -0.15, 0.05], dtype=np.float64)
    tvec = np.array([0.0, 0.0, 500.0], dtype=np.float64)

    img_p, _ = cv2.projectPoints(obj_p, rvec, tvec, K, dist)
    noise = rng.normal(0, noise_std_px, size=img_p.shape)
    noisy_img_p = (img_p + noise).astype(np.float32)

    return obj_p, noisy_img_p, K, dist

def generate_synthetic_checkerboard_session(
    n_views: int = 15,
    image_shape: tuple = (4080, 3072),
    rows: int = 6,
    cols: int = 9,
    square_size_mm: float = 19.0,
    noise_std_px: float = 0.25,
    seed: int = 42,
) -> tuple:
    """Generates a complete multi-view synthetic checkerboard calibration dataset."""
    rng = np.random.default_rng(seed)
    obj_p_single = np.zeros((rows * cols, 3), dtype=np.float32)
    obj_p_single[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    fx_true, fy_true = 3055.61, 3055.61
    cx_true, cy_true = image_shape[0] / 2.0 - 20.0, image_shape[1] / 2.0 + 30.0
    K_true = np.array([[fx_true, 0, cx_true], [0, fy_true, cy_true], [0, 0, 1]], dtype=np.float64)
    dist_true = np.array([0.0418, -0.1081, 0.0012, 0.0012, 0.1003], dtype=np.float64)

    all_obj = []
    all_img = []

    for i in range(n_views):
        rvec = rng.uniform(-0.25, 0.25, size=3)
        tvec = np.array([
            rng.uniform(-100, 100),
            rng.uniform(-100, 100),
            rng.uniform(400, 700),
        ], dtype=np.float64)

        img_p, _ = cv2.projectPoints(obj_p_single, rvec, tvec, K_true, dist_true)
        noise = rng.normal(0, noise_std_px, size=img_p.shape)
        noisy_p = (img_p + noise).astype(np.float32)

        all_obj.append(obj_p_single)
        all_img.append(noisy_p)

    return all_obj, all_img, image_shape, K_true, dist_true

def capture_edge(
    sigma: float = 1.0,
    angle_deg: float = 5.0,
    h: int = 1024,
    w: int = 256,
    exposure: float = 0.55,
    seed: int = 0,
    noisy: bool = True,
) -> np.ndarray:
    """Simulates a continuous optical slanted-edge blurred by a Gaussian PSF prior to pixel sampling."""
    theta = np.deg2rad(angle_deg)
    slope = np.tan(theta)
    yy, xx = np.mgrid[0:h, 0:w]

    # Signed perpendicular distance from each pixel center to the continuous edge
    perp_dist = (xx - ((w / 2.0) + slope * (yy - h / 2.0))) * np.cos(theta)

    if sigma > 0:
        # Exact continuous convolution of a step edge with a 2D Gaussian PSF of width sigma
        edge = 0.5 * (1.0 + erf(perp_dist / (np.sqrt(2.0) * sigma)))
    else:
        edge = (perp_dist > 0).astype(float)

    dark, light = 0.05, 0.90
    scene = (dark + (light - dark) * edge) * exposure

    if noisy:
        rng = np.random.default_rng(seed)
        shot = rng.poisson(scene * 10000.0) / 10000.0
        read = rng.normal(0, 0.005, size=(h, w))
        scene = np.clip(shot + read, 0, 1)

    return scene.astype(np.float64)

def generate_synthetic_flat_field(
    mean_signal_electrons: float,
    gain_dn_per_e: float,
    read_noise_e: float,
    black_level_dn: float = 64.0,
    patch_size: int = 128,
    seed: int = None,
) -> np.ndarray:
    """Generates synthetic flat-field patch with sub-DN precision for noise modeling."""
    rng = np.random.default_rng(seed)
    lam = max(0.0, mean_signal_electrons)
    electrons = rng.poisson(lam, size=(patch_size, patch_size)).astype(np.float64)
    signal_dn = electrons * gain_dn_per_e

    read_noise_std_dn = read_noise_e * gain_dn_per_e
    gauss_noise = rng.normal(0.0, read_noise_std_dn, size=(patch_size, patch_size))

    return signal_dn + gauss_noise + black_level_dn

def generate_flat_field_dataset(
    true_gain: float = 0.0213,
    true_read_noise: float = 2.5,
    black_level: float = 64.0,
    num_pairs: int = 12,
) -> list:
    """Generates a full series of 12 flat-field image pairs spanning black to saturation limits."""
    dataset = []
    radiance_levels = np.logspace(0.5, 2.9, num_pairs)
    for idx, radiance in enumerate(radiance_levels):
        frame_a = generate_synthetic_flat_field(
            radiance, true_gain, true_read_noise, black_level, seed=idx * 2
        )
        frame_b = generate_synthetic_flat_field(
            radiance, true_gain, true_read_noise, black_level, seed=idx * 2 + 1
        )
        dataset.append((frame_a, frame_b))
    return dataset

def generate_synthetic_dark_series(
    dark_current_e_per_sec: float = 0.5,
    gain: float = 0.0213,
    read_noise_e: float = 2.5,
    black_level: float = 64.0,
    exposure_times: list = [0.1, 0.5, 1.0, 2.0],
    frames_per_step: int = 2,
) -> dict:
    """Generates simulated dark frames accumulating thermal dark current over time."""
    dark_series = {}
    for t_exp in exposure_times:
        frames = []
        thermal_dark_electrons = dark_current_e_per_sec * t_exp
        for i in range(frames_per_step):
            frame = generate_synthetic_flat_field(
                mean_signal_electrons=thermal_dark_electrons,
                gain_dn_per_e=gain,
                read_noise_e=read_noise_e,
                black_level_dn=black_level,
                seed=int(t_exp * 100) + i,
            )
            frames.append(frame)
        dark_series[t_exp] = frames
    return dark_series

def generate_synthetic_color_patches(
    ref_xyz: np.ndarray,
    sensor_matrix: np.ndarray = None,
    illuminant_rgb_gains: np.ndarray = None,
    black_level_norm: float = 0.0,
    white_level_norm: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic linear camera RGB patch means from reference XYZ.

    Returns:
        sim_rgb: (N, 3) simulated camera linear RGB values
        true_ccm: (3, 3) exact matrix mapping noiseless sim_rgb -> ref_xyz
    """
    rng = np.random.default_rng(seed)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)

    if sensor_matrix is None:
        # Plausible non-orthogonal sensor response matrix
        sensor_matrix = np.array([
            [2.85, -0.45, 0.10],
            [-0.35, 1.80, -0.20],
            [0.05, -0.30, 1.15],
        ], dtype=np.float64)

    if illuminant_rgb_gains is None:
        illuminant_rgb_gains = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    else:
        illuminant_rgb_gains = np.asarray(illuminant_rgb_gains, dtype=np.float64)

    # Forward model: XYZ -> camera linear RGB under a given illuminant
    sim_rgb = (ref_xyz @ sensor_matrix.T) * illuminant_rgb_gains

    # Optional normalized pedestal / scaling / noise to test preprocessing & robustness
    sim_rgb = black_level_norm + sim_rgb * (white_level_norm - black_level_norm)
    if noise_std > 0.0:
        sim_rgb = sim_rgb + rng.normal(0.0, noise_std, size=sim_rgb.shape)

    # Exact inverse relationship for noiseless reference check
    effective_fwd = sensor_matrix * illuminant_rgb_gains[:, None]
    true_ccm = np.linalg.inv(effective_fwd).T
    return sim_rgb, true_ccm