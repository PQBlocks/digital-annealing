import numpy as np
import pytest
from pqblocks_digital_anneal.core import (
    ising_energy,
    compute_local_fields,
    compute_delta_energies,
    update_delta_energies_after_flip
)

def test_ising_energy():
    """Test the Ising energy computation."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    h = np.array([1, -1, 0, 0], dtype=float)
    s = np.array([1, 1, 1, 1], dtype=float)
    
    # Expected energy calculation:
    # H = sum_{i<j} J_ij * s_i * s_j + sum_i h_i * s_i
    #   = J_12*s1*s2 + J_23*s2*s3 + J_34*s3*s4 + h1*s1 + h2*s2 + h3*s3 + h4*s4
    #   = 1*1*1 + (-1)*1*1 + 1*1*1 + 1*1 + (-1)*1 + 0*1 + 0*1
    #   = 1 - 1 + 1 + 1 - 1 + 0 + 0
    #   = 1
    expected_energy = 1.0
    computed_energy = ising_energy(s, J, h)
    
    assert np.isclose(computed_energy, expected_energy), f"Expected {expected_energy}, got {computed_energy}"

def test_compute_local_fields():
    """Test the computation of effective local fields."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    h = np.array([1, -1, 0, 0], dtype=float)
    s = np.array([1, 1, 1, 1], dtype=float)
    
    # Expected local fields:
    # h1^eff = J12*s2 + h1 = 1*1 + 1 = 2
    # h2^eff = J21*s1 + J23*s3 + h2 = 1*1 + (-1)*1 + (-1) = -1
    # h3^eff = J32*s2 + J34*s4 + h3 = (-1)*1 + 1*1 + 0 = 0
    # h4^eff = J43*s3 + h4 = 1*1 + 0 = 1
    expected_h_eff = np.array([2, -1, 0, 1], dtype=float)
    computed_h_eff = compute_local_fields(s, J, h)
    
    assert np.allclose(computed_h_eff, expected_h_eff), f"Expected {expected_h_eff}, got {computed_h_eff}"

def test_compute_delta_energies():
    """Test the computation of delta energies."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    h = np.array([1, -1, 0, 0], dtype=float)
    s = np.array([1, 1, 1, 1], dtype=float)
    
    # Expected delta energies:
    # ΔE_i = -2 * s_i * h_i^eff
    # ΔE1 = -2 * 1 * 2 = -4
    # ΔE2 = -2 * 1 * (-1) = 2
    # ΔE3 = -2 * 1 * 0 = 0
    # ΔE4 = -2 * 1 * 1 = -2
    expected_delta_E = np.array([-4, 2, 0, -2], dtype=float)
    computed_delta_E = compute_delta_energies(s, J, h)
    
    assert np.allclose(computed_delta_E, expected_delta_E), f"Expected {expected_delta_E}, got {computed_delta_E}"

def test_update_delta_energies_after_flip():
    """Test the incremental update of delta energies after a spin flip."""
    # 4-spin example from the spec
    J = np.array([
        [0, 1, 0, 0],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    
    h = np.array([1, -1, 0, 0], dtype=float)
    s = np.array([1, 1, 1, 1], dtype=float)
    
    # Initial delta energies (from previous test)
    delta_E_old = np.array([-4, 2, 0, -2], dtype=float)
    
    # Flip spin 1 (index 0)
    flipped_spin_index = 0
    s_k_old = s[flipped_spin_index]  # 1
    
    # Expected updated delta energies:
    # ΔE1_new = -ΔE1_old = -(-4) = 4
    # ΔE2_new = ΔE2_old + 4 * J21 * s2 * s1_old = 2 + 4 * 1 * 1 * 1 = 6
    # ΔE3_new = ΔE3_old + 4 * J31 * s3 * s1_old = 0 + 4 * 0 * 1 * 1 = 0
    # ΔE4_new = ΔE4_old + 4 * J41 * s4 * s1_old = -2 + 4 * 0 * 1 * 1 = -2
    expected_delta_E_new = np.array([4, 6, 0, -2], dtype=float)
    computed_delta_E_new = update_delta_energies_after_flip(delta_E_old, s, J, flipped_spin_index)
    
    assert np.allclose(computed_delta_E_new, expected_delta_E_new), f"Expected {expected_delta_E_new}, got {computed_delta_E_new}"

if __name__ == "__main__":
    pytest.main([__file__])