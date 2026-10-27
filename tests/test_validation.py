import numpy as np
import pytest
from pqblocks_digital_anneal.validation import (
    validate_input,
    validate_parameters,
    compute_energy_scale_parameters,
    compute_schedule_parameters
)


def test_validate_input_valid():
    """Test validation with valid inputs."""
    J = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    # Should not raise any exception
    validated_J, validated_h = validate_input(J, h)
    assert validated_J.shape == J.shape
    assert validated_h.shape == h.shape


def test_validate_input_empty_model():
    """Test validation with empty model."""
    J = np.array([]).reshape(0, 0)
    h = np.array([])
    
    with pytest.raises(ValueError, match="Empty model"):
        validate_input(J, h)


def test_validate_input_dimension_mismatch():
    """Test validation with dimension mismatch."""
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="J shape mismatch"):
        validate_input(J, h)


def test_validate_input_asymmetric_J():
    """Test validation with asymmetric J."""
    J = np.array([[0, 1, 0], [2, 0, -1], [0, -1, 0]], dtype=np.float64)  # Not symmetric
    h = np.array([1, -1, 0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="J must be symmetric"):
        validate_input(J, h)


def test_validate_input_nonzero_diagonal():
    """Test validation with non-zero diagonal in J."""
    J = np.array([[1, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64)  # Diagonal not zero
    h = np.array([1, -1, 0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="J diagonal must be zero"):
        validate_input(J, h)


def test_validate_input_nan_inputs():
    """Test validation with NaN inputs."""
    J = np.array([[0, np.nan, 0], [np.nan, 0, -1], [0, -1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="Inputs must be finite"):
        validate_input(J, h)


def test_validate_input_inf_inputs():
    """Test validation with Inf inputs."""
    J = np.array([[0, np.inf, 0], [np.inf, 0, -1], [0, -1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    with pytest.raises(ValueError, match="Inputs must be finite"):
        validate_input(J, h)


def test_validate_parameters_valid():
    """Test parameter validation with valid inputs."""
    # Should not raise any exception
    validate_parameters(1000)


def test_validate_parameters_non_positive():
    """Test parameter validation with non-positive sweeps."""
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        validate_parameters(0)
    
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        validate_parameters(-1)


def test_compute_energy_scale_parameters_normal():
    """Test energy scale parameter computation with normal inputs."""
    J = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    delta_E_max, delta_E_min = compute_energy_scale_parameters(J, h)
    
    # Expected values:
    # For i=0: 2|h[0]| + sum_j 2|J[0,j]| = 2*1 + 2*(0+1+0) = 2 + 2 = 4
    # For i=1: 2|h[1]| + sum_j 2|J[1,j]| = 2*1 + 2*(1+0+1) = 2 + 4 = 6
    # For i=2: 2|h[2]| + sum_j 2|J[2,j]| = 2*0 + 2*(0+1+0) = 0 + 2 = 2
    # So delta_E_max = max(4, 6, 2) = 6
    
    # For delta_E_min:
    # Non-zero J entries: |1|, |1|, |-1|, |-1| = 1
    # So delta_E_min = 1
    
    assert np.isclose(delta_E_max, 6.0)
    assert np.isclose(delta_E_min, 1.0)


def test_compute_energy_scale_parameters_all_zero():
    """Test energy scale parameter computation with all-zero model."""
    J = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    h = np.array([0, 0, 0], dtype=np.float64)
    
    delta_E_max, delta_E_min = compute_energy_scale_parameters(J, h)
    
    # Special case for all-zero model
    assert np.isclose(delta_E_max, 0.0)
    assert np.isclose(delta_E_min, 1.0)


def test_compute_energy_scale_parameters_no_J_only_h():
    """Test energy scale parameter computation with only h terms."""
    J = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    h = np.array([1, -2, 0], dtype=np.float64)
    
    delta_E_max, delta_E_min = compute_energy_scale_parameters(J, h)
    
    # For delta_E_max:
    # For i=0: 2|h[0]| + sum_j 2|J[0,j]| = 2*1 + 0 = 2
    # For i=1: 2|h[1]| + sum_j 2|J[1,j]| = 2*2 + 0 = 4
    # For i=2: 2|h[2]| + sum_j 2|J[2,j]| = 2*0 + 0 = 0
    # So delta_E_max = max(2, 4, 0) = 4
    
    # For delta_E_min:
    # No non-zero J entries, but h has non-zero entries
    # Non-zero h entries: |1|, |-2| = 1, 2
    # So delta_E_min = min(1, 2) = 1
    
    assert np.isclose(delta_E_max, 4.0)
    assert np.isclose(delta_E_min, 1.0)


def test_compute_schedule_parameters_normal():
    """Test schedule parameter computation with normal inputs."""
    J = np.array([[0, 1, 0], [1, 0, -1], [0, -1, 0]], dtype=np.float64)
    h = np.array([1, -1, 0], dtype=np.float64)
    
    beta_0, beta_f, delta = compute_schedule_parameters(J, h)
    
    # From previous test, we know:
    # delta_E_max = 6.0, delta_E_min = 1.0
    
    # So:
    # beta_0 = 0.1 / 6.0 = 0.016666...
    # beta_f = 5.0 / 1.0 = 5.0
    # delta = 0.5 * 1.0 = 0.5
    
    assert np.isclose(beta_0, 0.1 / 6.0)
    assert np.isclose(beta_f, 5.0)
    assert np.isclose(delta, 0.5)


def test_compute_schedule_parameters_all_zero():
    """Test schedule parameter computation with all-zero model."""
    J = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
    h = np.array([0, 0, 0], dtype=np.float64)
    
    beta_0, beta_f, delta = compute_schedule_parameters(J, h)
    
    # Special case values from spec for all-zero model
    assert np.isclose(beta_0, 0.1)
    assert np.isclose(beta_f, 5.0)
    assert np.isclose(delta, 0.5)