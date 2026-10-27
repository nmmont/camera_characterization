# tests/test_geometry.py
import pytest
import numpy as np
from camera_toolkit import geometry

def test_object_point_generation_dimensions():
    """Confirms that the 3D target array layout coordinates match perfectly."""
    rows, cols, sq_size = 6, 9, 25.0
    obj_p = geometry.generate_object_points(rows, cols, sq_size)
    
    # Grid shape must contain exactly rows * cols points across 3 coordinate spatial axes
    assert obj_p.shape == (54, 3)
    # The Z axis for flat checkerboards must be completely initialized to zero
    assert np.all(obj_p[:, 2] == 0.0)

def test_max_perpendicular_deviation_on_perfect_line():
    """Asserts that a mathematically straight horizontal line yields exactly 0 pixels of error."""
    # Create 5 collinear coordinates lying perfectly on the line y = 2x + 1
    perfect_pts = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0], [4.0, 9.0]])
    deviation = geometry.max_perpendicular_deviation(perfect_pts)
    
    assert deviation == pytest.approx(0.0, abs=1e-5)
