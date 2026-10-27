import numpy as np

def ising_energy(spin_configuration, J, h):
    """
    Compute the energy of an Ising model configuration.
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - energy: scalar energy value
    """
    # Ensure inputs are numpy arrays
    s = np.asarray(spin_configuration)
    J = np.asarray(J)
    h = np.asarray(h)
    
    # Compute energy: H = -sum_{i<j} J_ij * s_i * s_j - sum_i h_i * s_i
    # Using the canonical form H = sum_{i<j} J_ij * s_i * s_j + sum_i h_i * s_i
    # So we have H = sum_{i<j} J_ij * s_i * s_j + sum_i h_i * s_i
    energy = 0.0
    
    # Pairwise interactions
    for i in range(len(s)):
        for j in range(i+1, len(s)):
            energy += J[i, j] * s[i] * s[j]
    
    # Local field contributions
    energy += np.dot(h, s)
    
    return energy

def compute_local_fields(spin_configuration, J, h):
    """
    Compute the effective local fields for each spin.
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - h_eff: effective local fields (N)
    """
    s = np.asarray(spin_configuration)
    J = np.asarray(J)
    h = np.asarray(h)
    
    # Compute effective local fields: h_i^eff = sum_{j != i} J_ij * s_j + h_i
    h_eff = h.copy().astype(np.float64)
    for i in range(len(s)):
        for j in range(len(s)):
            if i != j:
                h_eff[i] += J[i, j] * s[j]
    
    return h_eff

def compute_delta_energies(spin_configuration, J, h):
    """
    Compute the energy changes for flipping each spin.
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - delta_E: energy changes for flipping each spin (N)
    """
    s = np.asarray(spin_configuration)
    J = np.asarray(J)
    h = np.asarray(h)
    
    # Compute effective local fields
    h_eff = compute_local_fields(s, J, h)
    
    # Compute delta energies: ΔE_i = -2 * s_i * h_i^eff
    delta_E = -2.0 * s * h_eff
    
    return delta_E

def update_delta_energies_after_flip(delta_E_old, spin_configuration, J, flipped_spin_index):
    """
    Update delta energies after a spin flip using incremental updates.
    
    Parameters:
    - delta_E_old: previous delta energies (N)
    - spin_configuration: current spin configuration before flip
    - J: coupling matrix (NxN)
    - flipped_spin_index: index of the flipped spin
    
    Returns:
    - delta_E_new: updated delta energies (N)
    """
    s = np.asarray(spin_configuration)
    J = np.asarray(J)
    k = flipped_spin_index
    
    # Copy old delta energies
    delta_E_new = delta_E_old.copy()
    
    # Update for the flipped spin: ΔE_k_new = -ΔE_k_old
    delta_E_new[k] = -delta_E_old[k]
    
    # Update for all other spins j != k:
    # ΔE_j_new = ΔE_j_old + 4 * J_jk * s_j * s_k_old
    s_k_old = s[k]
    for j in range(len(s)):
        if j != k:
            delta_E_new[j] = delta_E_old[j] + 4.0 * J[j, k] * s[j] * s_k_old
    
    return delta_E_new