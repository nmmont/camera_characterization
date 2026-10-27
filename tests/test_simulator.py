# tests/test_simulator.py
import pytest
import numpy as np
from camera_toolkit import simulator

def test_flat_field_poisson_gaussian_math():
    """
    Validates that the flat-field noise simulator matches the physical 
    affine variance equation: Var[D] = K*(E[D] - B) + K^2 * sigma_r^2.
    """
    true_gain = 0.25         # K (DN/e-)
    true_read_noise = 3.0    # sigma_r (electrons)
    black_level = 64.0       # B (DN)
    mean_light_e = 400.0     # Photon mean (electrons)
    
    # Generate a large 512x512 patch to reduce statistical sampling error
    patch = simulator.generate_synthetic_flat_field(
        mean_signal_electrons=mean_light_e,
        gain_dn_per_e=true_gain,
        read_noise_e=true_read_noise,
        black_level_dn=black_level,
        patch_size=512,
        seed=42
    )
    
    # 1. Compute empirical statistics from the generated patch
    measured_mean = np.mean(patch)
    measured_var = np.var(patch)
    
    # 2. Compute theoretical expectations
    expected_mean = (mean_light_e * true_gain) + black_level
       
    # FIXED: True physics variance scales photon noise by true_gain^2
    expected_var = (true_gain ** 2 * mean_light_e) + (true_read_noise * true_gain) ** 2

    # 3. Assert within a tight statistical tolerance window (10% due to sample variance)
    assert measured_mean == pytest.approx(expected_mean, rel=1e-2)
    assert measured_var == pytest.approx(expected_var, rel=1e-1)


def test_dark_series_exposure_linearity():
    """Verifies that thermal dark current noise scales linearly with exposure duration."""
    dark_rate = 10.0  # 10 electrons/second
    gain = 0.5
    read_noise = 2.0
    exposure_times = [0.5, 2.0]
    
    dark_dataset = simulator.generate_synthetic_dark_series(
        dark_current_e_per_sec=dark_rate,
        gain=gain,
        read_noise_e=read_noise,
        exposure_times=exposure_times,
        frames_per_step=3
    )
    
    # Short exposure mean signal calculation (minus baseline pedestal)
    mean_short = np.mean(dark_dataset[0.5][0]) - 64.0
    # Long exposure mean signal calculation (minus baseline pedestal)
    mean_long = np.mean(dark_dataset[2.0][0]) - 64.0
    
    # Expect the long exposure to accumulate roughly 4x more thermal electrons
    assert mean_long > mean_short
    assert (mean_long / mean_short) == pytest.approx(4.0, rel=3e-1)


def test_geometric_projection_dimensions():
    """Confirms that the geometric grid projector maps arrays onto valid coordinates."""
    rows, cols = 6, 9
    obj_p, img_p, K_true, dist_true = simulator.generate_synthetic_checkerboard_data(rows, cols)
    
    # Grid shape must contain exactly rows * cols points
    assert obj_p.shape == (54, 3)
    assert img_p.shape == (54, 2)
    # Ensure the lens distortion k1 factor is non-zero
    assert dist_true[0] == 0.15
