import numpy as np
from numba import njit, prange


@njit
def xorshift32(state):
    """
    Simple XorShift32 PRNG.
    Returns the next state and a random float in [0, 1).
    """
    state ^= (state << 13) & 0xFFFFFFFF
    state ^= (state >> 17) & 0xFFFFFFFF
    state ^= (state << 5) & 0xFFFFFFFF
    return state, state / 4294967296.0  # 2^32


@njit
def ising_energy_numba(spin_configuration, J, h):
    """
    Compute the energy of an Ising model configuration (Numba version).
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - energy: scalar energy value
    """
    # Compute energy: H = sum_{i<j} J_ij * s_i * s_j + sum_i h_i * s_i
    energy = 0.0
    
    # Pairwise interactions
    for i in range(len(spin_configuration)):
        for j in range(i+1, len(spin_configuration)):
            energy += J[i, j] * spin_configuration[i] * spin_configuration[j]
    
    # Local field contributions
    for i in range(len(spin_configuration)):
        energy += h[i] * spin_configuration[i]
    
    return energy


@njit
def compute_local_fields_numba(spin_configuration, J, h):
    """
    Compute the effective local fields for each spin (Numba version).
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - h_eff: effective local fields (N)
    """
    # Compute effective local fields: h_i^eff = sum_{j != i} J_ij * s_j + h_i
    h_eff = h.copy().astype(np.float64)
    for i in range(len(spin_configuration)):
        for j in range(len(spin_configuration)):
            if i != j:
                h_eff[i] += J[i, j] * spin_configuration[j]
    
    return h_eff


@njit
def compute_delta_energies_numba(spin_configuration, J, h):
    """
    Compute the energy changes for flipping each spin (Numba version).
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - delta_E: energy changes for flipping each spin (N)
    """
    # Compute effective local fields
    h_eff = compute_local_fields_numba(spin_configuration, J, h)
    
    # Compute delta energies: ΔE_i = -2 * s_i * h_i^eff
    delta_E = np.empty(len(spin_configuration), dtype=np.float64)
    for i in range(len(spin_configuration)):
        delta_E[i] = -2.0 * spin_configuration[i] * h_eff[i]
    
    return delta_E


@njit
def update_delta_energies_after_flip_numba(delta_E_old, spin_configuration, J, flipped_spin_index):
    """
    Update delta energies after a spin flip using incremental updates (Numba version).
    
    Parameters:
    - delta_E_old: previous delta energies (N)
    - spin_configuration: current spin configuration before flip
    - J: coupling matrix (NxN)
    - flipped_spin_index: index of the flipped spin
    
    Returns:
    - delta_E_new: updated delta energies (N)
    """
    # Copy old delta energies
    delta_E_new = delta_E_old.copy()
    
    # Update for the flipped spin: ΔE_k_new = -ΔE_k_old
    delta_E_new[flipped_spin_index] = -delta_E_old[flipped_spin_index]
    
    # Update for all other spins j != k:
    # ΔE_j_new = ΔE_j_old + 4 * J_jk * s_j * s_k_old
    s_k_old = spin_configuration[flipped_spin_index]
    for j in range(len(spin_configuration)):
        if j != flipped_spin_index:
            delta_E_new[j] = delta_E_old[j] + 4.0 * J[j, flipped_spin_index] * spin_configuration[j] * s_k_old
    
    return delta_E_new


@njit
def compute_acceptance_probabilities_numba(delta_E, beta, E_off):
    """
    Compute acceptance probabilities for all spins (Numba version).
    
    Parameters:
    - delta_E: energy changes for flipping each spin (N)
    - beta: inverse temperature
    - E_off: energy offset
    
    Returns:
    - P: acceptance probabilities (N)
    """
    N = len(delta_E)
    P = np.empty(N, dtype=np.float64)
    
    for i in range(N):
        delta = delta_E[i] - E_off
        if delta <= 0:
            P[i] = 1.0
        else:
            x = -beta * delta
            if x < -709:  # Underflow safe
                P[i] = 0.0
            else:
                P[i] = np.exp(x)
    
    return P


@njit
def parallel_trial_mc_step_numba(spin_configuration, J, h, beta, E_off, delta, rng_seed):
    """
    Perform one parallel-trial Monte Carlo step (Numba version).
    
    Parameters:
    - spin_configuration: array of spins (+1 or -1)
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    - beta: inverse temperature
    - E_off: energy offset
    - delta: dynamic offset rate parameter
    - rng_seed: seed for the RNG
    
    Returns:
    - new_spin_configuration: updated spin configuration
    - new_E_off: updated energy offset
    - accepted: boolean indicating if any spin was flipped
    - delta_E: updated delta energies for next step (to enable O(1) updates)
    """
    # Compute delta energies
    delta_E = compute_delta_energies_numba(spin_configuration, J, h)
    
    # Compute acceptance probabilities
    P = compute_acceptance_probabilities_numba(delta_E, beta, E_off)
    
    # Initialize RNG state
    rng_state = rng_seed
    
    # Determine which flips are accepted
    accepted_indices = []
    for i in range(len(spin_configuration)):
        rng_state, rand_val = xorshift32(rng_state)
        if rand_val < P[i]:
            accepted_indices.append(i)
    
    # If no flips accepted, increment E_off by delta
    if len(accepted_indices) == 0:
        new_E_off = E_off + delta
        return spin_configuration.copy(), new_E_off, False, delta_E
    
    # If flips accepted, select one uniformly at random
    rng_state, rand_val = xorshift32(rng_state)
    selected_index = accepted_indices[int(rand_val * len(accepted_indices)) % len(accepted_indices)]
    
    # Update spin configuration
    new_spin_configuration = spin_configuration.copy()
    new_spin_configuration[selected_index] *= -1
    
    # Reset E_off
    new_E_off = 0.0
    
    # Update delta energies incrementally in O(1) time
    new_delta_E = update_delta_energies_after_flip_numba(delta_E, spin_configuration, J, selected_index)
    
    return new_spin_configuration, new_E_off, True, new_delta_E