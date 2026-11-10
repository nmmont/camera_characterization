# src/camera_toolkit/simulator.py
import cv2
import numpy as np


def capture_edge(sigma: float = 1.0, angle_deg: float = 5.0, h: int = 256, w: int = 128, 
                 exposure: float = 0.55, seed: int = 0, noisy: bool = True) -> np.ndarray:
    """Simulates a monochrome slanted-edge capture in linear light for MTF validation."""
    slope = np.tan(np.deg2rad(angle_deg))
    yy, xx = np.mgrid[0:h, 0:w]
    edge = (xx > (w / 2) + slope * (yy - h / 2)).astype(float)
    dark, light = 0.05, 0.90
    scene = dark + (light - dark) * edge

    blurred = cv2.GaussianBlur(scene, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if not noisy:
        return blurred

    BLACK_LEVEL = 256
    WHITE_LEVEL = 4095
    FULL_WELL = 9000.0
    READ_NOISE = 2.8

    rng = np.random.default_rng(seed)
    electrons = np.clip(blurred, 0, None) * FULL_WELL * exposure
    signal = rng.poisson(np.clip(electrons, 0, None)).astype(float)
    signal = signal + rng.normal(0.0, READ_NOISE, electrons.shape)

    gain_dn_per_e = (WHITE_LEVEL - BLACK_LEVEL) / FULL_WELL
    dn = BLACK_LEVEL + signal * gain_dn_per_e
    raw = np.clip(np.round(dn), 0, WHITE_LEVEL).astype(np.uint16)

    return (raw.astype(float) - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)


def generate_synthetic_flat_field(mean_signal_electrons: float, gain_dn_per_e: float, 
                                  read_noise_e: float, black_level_dn: float = 64.0, 
                                  patch_size: int = 128, seed: int = None) -> np.ndarray:
    """
    Generates a uniform raw flat-field patch applying physical affine noise equations.
    """
    rng = np.random.default_rng(seed)
    lam = max(0.0, mean_signal_electrons)
    electrons = rng.poisson(lam, size=(patch_size, patch_size)).astype(np.float32)
    signal_dn = electrons * gain_dn_per_e
    
    read_noise_std_dn = read_noise_e * gain_dn_per_e
    gauss_noise = rng.normal(0.0, read_noise_std_dn, size=(patch_size, patch_size))
    
    raw_frame = signal_dn + gauss_noise + black_level_dn
    return np.clip(np.round(raw_frame), 0, 1023).astype(np.uint16)


def generate_flat_field_dataset(true_gain: float = 0.0213, true_read_noise: float = 2.5, black_level: float = 64.0, num_pairs: int = 12) -> list:
    """
    Generates a full series of 12 flat-field image pairs spanning black to saturation limits.
    """
    dataset = []
    radiance_levels = np.logspace(0.5, 2.9, num_pairs)
    for idx, radiance in enumerate(radiance_levels):
        frame_a = generate_synthetic_flat_field(radiance, true_gain, true_read_noise, black_level, seed=idx*2)
        frame_b = generate_synthetic_flat_field(radiance, true_gain, true_read_noise, black_level, seed=idx*2 + 1)
        dataset.append((frame_a, frame_b))
    return dataset


def generate_synthetic_dark_series(dark_current_e_per_sec: float = 0.5, gain: float = 0.0213, read_noise_e: float = 2.5, 
                                   black_level: float = 64.0, exposure_times: list = [0.1, 0.5, 1.0, 2.0], 
                                   frames_per_step: int = 2) -> dict:
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
                seed=int(t_exp * 100) + i
            )
            frames.append(frame)
        dark_series[t_exp] = frames
    return dark_series


def generate_synthetic_checkerboard_data(rows: int = 6, cols: int = 9, square_size_mm: float = 19.0) -> tuple:
    """Generates idealized geometric coordinate vectors to mock target checkerboards."""
    obj_points = np.zeros((rows * cols, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    
    f_px = 3000.0
    c_x, c_y = 2040.0, 1536.0
    K_true = np.array([[f_px,  0.0, c_x],
                       [ 0.0, f_px, c_y],
                       [ 0.0,  0.0, 1.0]], dtype=np.float32)
    
    dist_true = np.array([0.15, -0.04, 0.0, 0.0, 0.0], dtype=np.float32)
    rvec = np.array([0.05, -0.1, 0.02], dtype=np.float32)
    tvec = np.array([10.0, -20.0, 450.0], dtype=np.float32)
    
    img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K_true, dist_true)
    return obj_points, img_points.reshape(-1, 2), K_true, dist_true


def generate_synthetic_checkerboard_session(n: int = 15, rows: int = 6, cols: int = 9, 
                                          square_size_mm: float = 19.0, tilt_deg: float = 30.0, 
                                          noise_px: float = 0.15, seed: int = 513, 
                                          img_shape: tuple = (4080, 3072)) -> tuple:
    """Generates synthetic multi-view checkerboard session (n views) with ground truth intrinsics."""
    obj_p = np.zeros((rows * cols, 3), dtype=np.float32)
    obj_p[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm

    w, h = img_shape
    f_px = 3055.61
    cx, cy = 1500.96, 2072.98
    K_true = np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_true = np.array([0.0247, -0.0659, 0.001, 0.0012, 0.0], dtype=np.float32)

    rng = np.random.default_rng(seed)
    syn_obj_list, syn_img_list = [], []

    for i in range(n):
        rvec = rng.uniform(-np.radians(tilt_deg), np.radians(tilt_deg), size=(3, 1)).astype(np.float32)
        tvec = np.array([rng.uniform(-100, 100), rng.uniform(-80, 80), rng.uniform(400, 700)], dtype=np.float32).reshape(3, 1)

        proj, _ = cv2.projectPoints(obj_p, rvec, tvec, K_true, dist_true)
        proj = proj.reshape(-1, 2)
        if noise_px > 0:
            proj += rng.normal(0, noise_px, size=proj.shape).astype(np.float32)

        syn_obj_list.append(obj_p)
        syn_img_list.append(proj.reshape(-1, 1, 2))

    return syn_obj_list, syn_img_list, (w, h), K_true, dist_true
