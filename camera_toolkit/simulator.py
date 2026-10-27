# src/camera_toolkit/simulator.py
import cv2
import numpy as np

# =========================================================================
# 1. SPATIAL RESOLUTION SIMULATOR (Preserved from your code)
# =========================================================================

def capture_edge(sigma: float = 1.0, angle_deg: float = 5.0, h: int = 256, w: int = 128, 
                 exposure: float = 0.55, seed: int = 0, noisy: bool = True) -> np.ndarray:
    """Simulates a monochrome slanted-edge capture in linear light for MTF validation ``."""
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


# =========================================================================
# 2. SENSOR NOISE & PHOTON TRANSFER CURVE SIMULATOR (New)
# =========================================================================

def generate_synthetic_flat_field(mean_signal_electrons: float, gain_dn_per_e: float, 
                                  read_noise_e: float, black_level_dn: float = 64.0, 
                                  patch_size: int = 128, seed: int = None) -> np.ndarray:
    """
    Generates a uniform raw flat-field patch applying physical affine noise equations ``.
    Enforces Var[D] = K(E[D] - B) + K^2 * sigma_r^2 ``.
    """
    rng = np.random.default_rng(seed)
    
    # Photon Shot Noise (Poisson distribution on incoming light channel) ``
    lam = max(0.0, mean_signal_electrons)
    electrons = rng.poisson(lam, size=(patch_size, patch_size)).astype(np.float32)
    
    # Scale from electrons to Digital Numbers (DN) via conversion gain ``
    signal_dn = electrons * gain_dn_per_e
    
    # Electronic Read Noise (Gaussian distribution in DN space) ``
    read_noise_std_dn = read_noise_e * gain_dn_per_e
    gauss_noise = rng.normal(0.0, read_noise_std_dn, size=(patch_size, patch_size))
    
    # Combine signals and inject baseline black level sensor pedestal ``
    raw_frame = signal_dn + gauss_noise + black_level_dn
    return np.clip(np.round(raw_frame), 0, 1023).astype(np.uint16)


def generate_flat_field_dataset(true_gain: float, true_read_noise: float, black_level: float = 64.0, 
                                num_pairs: int = 12) -> list:
    """
    Generates a full batch series of 12 flat-field image pairs spanning black to saturation.
    Provides the perfect analytical test-twin for your PTC line-fitting loops ``.
    """
    dataset = []
    # Generates uniform radiance steps from nearly black up to near-saturation limits
    radiance_levels = np.logspace(0.5, 2.9, num_pairs)
    
    for idx, radiance in enumerate(radiance_levels):
        # Generate pairs with identical illumination parameters but discrete noise seeds ``
        frame_a = generate_synthetic_flat_field(radiance, true_gain, true_read_noise, black_level, seed=idx*2)
        frame_b = generate_synthetic_flat_field(radiance, true_gain, true_read_noise, black_level, seed=idx*2 + 1)
        dataset.append((frame_a, frame_b))
        
    return dataset


# =========================================================================
# 3. DARK CURRENT TIME-SERIES SIMULATOR (New)
# =========================================================================

def generate_synthetic_dark_series(dark_current_e_per_sec: float, gain: float, read_noise_e: float, 
                                   black_level: float = 64.0, exposure_times: list = [0.1, 0.5, 1.0, 2.0], 
                                   frames_per_step: int = 5) -> dict:
    """
    Generates simulated lens-capped dark frames accumulating thermal dark current over time ``.
    Used to verify that your noise.py slope-fitter can accurately extract true electron rates ``.
    """
    dark_series = {}
    for t_exp in exposure_times:
        frames = []
        # Calculate mean thermal noise accumulated over exposure duration
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


# =========================================================================
# 4. GEOMETRIC PROJECTION GRID SIMULATOR (New)
# =========================================================================

def generate_synthetic_checkerboard_data(rows: int = 6, cols: int = 9, square_size_mm: float = 19.0) -> tuple:
    """
    Generates idealized geometric coordinate vectors to mock target checkerboards ``.
    Projects perfect 3D coordinates into 2D camera pixels with intentional lens distortion.
    """
    # 1. Define ideal flat 3D object plane coordinates (Z = 0)
    obj_points = np.zeros((rows * cols, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    
    # 2. Hardcode a ground-truth intrinsic camera matrix (K)
    f_px = 3000.0
    c_x, c_y = 2040.0, 1536.0 # Central principal points for a standard sensor grid
    K_true = np.array([[f_px,  0.0, c_x],
                       [ 0.0, f_px, c_y],
                       [ 0.0,  0.0, 1.0]], dtype=np.float32)
    
    # 3. Inject an intentional baseline lens warping profile (Barrel distortion k1)
    dist_true = np.array([0.15, -0.04, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # 4. Setup a sample mock camera position orientation pose
    rvec = np.array([0.05, -0.1, 0.02], dtype=np.float32)
    tvec = np.array([10.0, -20.0, 450.0], dtype=np.float32) # Target placed 450mm away
    
    # 5. Execute forward mathematical projection
    img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K_true, dist_true)
    
    return obj_points, img_points.reshape(-1, 2), K_true, dist_true
