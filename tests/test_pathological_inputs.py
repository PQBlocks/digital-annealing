import numpy as np
import pytest
from pqblocks_digital_anneal.validation import validate_input, validate_parameters
from pqblocks_digital_anneal.solver import DigitalAnnealerSampler

def test_N_equals_1_model():
    """Test that N=1 models are handled correctly."""
    J = np.array([[0.0]])
    h = np.array([1.0])
    
    # Should not raise any exception
    validated_J, validated_h = validate_input(J, h)
    assert validated_J.shape == (1, 1)
    assert validated_h.shape == (1,)

def test_N_equals_1_model_with_solver():
    """Test that N=1 models work with the solver."""
    J = np.array([[0.0]])
    h = np.array([1.0])
    
    sampler = DigitalAnnealerSampler(seed=42)
    results = sampler.sample(J, h, num_sweeps=100, num_reads=1)
    
    assert len(results) == 1
    assert 'sample' in results[0]
    assert 'energy' in results[0]
    assert len(results[0]['sample']) == 1
    assert results[0]['sample'][0] in [-1, 1]

def test_non_positive_num_reads():
    """Test that non-positive num_reads raises ValueError."""
    J = np.array([[0, 1], [1, 0]], dtype=np.float64)
    h = np.array([1, -1], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler(seed=42)
    
    with pytest.raises(ValueError, match="num_reads must be > 0"):
        sampler.sample(J, h, num_sweeps=100, num_reads=0)
    
    with pytest.raises(ValueError, match="num_reads must be > 0"):
        sampler.sample(J, h, num_sweeps=100, num_reads=-1)

# More direct tests of validate_parameters function
def test_validate_parameters_inf_nan_direct():
    """Direct test of validate_parameters with Inf/NaN values."""
    # Test infinite sweeps
    with pytest.raises(ValueError, match="sweeps must be finite"):
        validate_parameters(float('inf'))
    
    # Test NaN sweeps
    with pytest.raises(ValueError, match="sweeps must be finite"):
        validate_parameters(float('nan'))
    
    # Test infinite reads
    with pytest.raises(ValueError, match="num_reads must be finite"):
        validate_parameters(100, float('inf'))
    
    # Test NaN reads
    with pytest.raises(ValueError, match="num_reads must be finite"):
        validate_parameters(100, float('nan'))
        
def test_validate_parameters_non_positive_direct():
    """Direct test of validate_parameters with non-positive values."""
    # Test zero sweeps
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        validate_parameters(0)
    
    # Test negative sweeps
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        validate_parameters(-1)
    
    # Test zero reads
    with pytest.raises(ValueError, match="num_reads must be > 0"):
        validate_parameters(100, 0)
    
    # Test negative reads
    with pytest.raises(ValueError, match="num_reads must be > 0"):
        validate_parameters(100, -1)