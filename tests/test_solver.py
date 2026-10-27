import numpy as np
import pytest
from pqblocks_digital_anneal.solver import DigitalAnnealerSampler


def test_digital_annealer_sampler_initialization():
    """Test initialization of the DigitalAnnealerSampler."""
    # Test with default seed
    sampler1 = DigitalAnnealerSampler()
    assert hasattr(sampler1, 'seed')
    
    # Test with specific seed
    sampler2 = DigitalAnnealerSampler(seed=42)
    assert sampler2.seed == 42


def test_digital_annealer_sampler_sample_valid():
    """Test sampling with valid inputs."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler(seed=42)
    results = sampler.sample(J, h, num_sweeps=100, num_reads=1)
    
    # Check that we got the expected number of results
    assert len(results) == 1
    
    # Check that each result has the expected structure
    result = results[0]
    assert 'sample' in result
    assert 'energy' in result
    assert len(result['sample']) == 4
    assert result['sample'].dtype == np.float64
    # All spins should be either +1 or -1
    assert np.all(np.isin(result['sample'], [-1, 1]))


def test_digital_annealer_sampler_sample_multiple_reads():
    """Test sampling with multiple reads."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler(seed=42)
    results = sampler.sample(J, h, num_sweeps=100, num_reads=3)
    
    # Check that we got the expected number of results
    assert len(results) == 3
    
    # Check that each result has the expected structure
    for result in results:
        assert 'sample' in result
        assert 'energy' in result
        assert len(result['sample']) == 4
        assert result['sample'].dtype == np.float64
        # All spins should be either +1 or -1
        assert np.all(np.isin(result['sample'], [-1, 1]))


def test_digital_annealer_sampler_sample_empty_model():
    """Test sampling with empty model."""
    J = np.array([]).reshape(0, 0)
    h = np.array([])
    
    sampler = DigitalAnnealerSampler()
    
    with pytest.raises(ValueError, match="Empty model"):
        sampler.sample(J, h)


def test_digital_annealer_sampler_sample_invalid_parameters():
    """Test sampling with invalid parameters."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler()
    
    # Test with zero sweeps
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        sampler.sample(J, h, num_sweeps=0)
    
    # Test with negative sweeps
    with pytest.raises(ValueError, match="sweeps must be > 0"):
        sampler.sample(J, h, num_sweeps=-1)


def test_digital_annealer_sampler_sample_asymmetric_J():
    """Test sampling with asymmetric J matrix."""
    J = np.array([
        [0, 1, 0, 0],
        [2, 0, -1, 0],  # Not symmetric
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler()
    
    with pytest.raises(ValueError, match="J must be symmetric"):
        sampler.sample(J, h)


def test_digital_annealer_sampler_sample_nonzero_diagonal():
    """Test sampling with non-zero diagonal in J."""
    J = np.array([
        [1, 1, 0, 0],  # Diagonal not zero
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler()
    
    with pytest.raises(ValueError, match="J diagonal must be zero"):
        sampler.sample(J, h)


def test_digital_annealer_sampler_sample_nan_inputs():
    """Test sampling with NaN inputs."""
    J = np.array([
        [0, np.nan, 0, 0],  # Contains NaN
        [np.nan, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler()
    
    with pytest.raises(ValueError, match="Inputs must be finite"):
        sampler.sample(J, h)


def test_digital_annealer_sampler_sample_inf_inputs():
    """Test sampling with Inf inputs."""
    J = np.array([
        [0, np.inf, 0, 0],  # Contains Inf
        [np.inf, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    sampler = DigitalAnnealerSampler()
    
    with pytest.raises(ValueError, match="Inputs must be finite"):
        sampler.sample(J, h)


def test_digital_annealer_sampler_reproducibility():
    """Test that the sampler produces reproducible results with the same seed."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    h = np.array([1, -1, 0, 0], dtype=np.float64)
    
    # Run sampler twice with the same seed
    sampler1 = DigitalAnnealerSampler(seed=42)
    results1 = sampler1.sample(J, h, num_sweeps=100, num_reads=2)
    
    sampler2 = DigitalAnnealerSampler(seed=42)
    results2 = sampler2.sample(J, h, num_sweeps=100, num_reads=2)
    
    # Results should be identical
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert np.array_equal(r1['sample'], r2['sample'])
        assert np.isclose(r1['energy'], r2['energy'])