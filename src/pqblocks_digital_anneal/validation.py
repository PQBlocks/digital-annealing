import numpy as np


def validate_input(J, h):
    """
    Validate the input parameters for the Digital Annealer solver.
    
    Parameters:
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Raises:
    - ValueError: if any validation condition fails
    """
    # Convert to numpy arrays if not already
    J = np.asarray(J, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    
    # Check for NaN/Inf inputs (do this first before other checks)
    if not np.isfinite(J).all() or not np.isfinite(h).all():
        raise ValueError("Inputs must be finite")
    
    # Check for empty model (N=0)
    if len(h) == 0:
        raise ValueError("Empty model")
    
    # Check dimension mismatch
    if J.shape != (len(h), len(h)):
        raise ValueError("J shape mismatch")
    
    # Check for asymmetric J
    if not np.allclose(J, J.T):
        raise ValueError("J must be symmetric")
    
    # Check for non-zero diagonal in J
    if np.any(np.diag(J) != 0):
        raise ValueError("J diagonal must be zero")
    
    return J, h


def validate_parameters(num_sweeps, num_reads=None):
    """
    Validate solver parameters.
    
    Parameters:
    - num_sweeps: number of sweeps
    - num_reads: number of reads (optional)
    
    Raises:
    - ValueError: if any validation condition fails
    """
    # Check that num_sweeps is finite
    if not np.isfinite(num_sweeps):
        raise ValueError("sweeps must be finite")
    
    # Check for non-positive sweeps
    if num_sweeps <= 0:
        raise ValueError("sweeps must be > 0")
    
    # If num_reads is provided, check it
    if num_reads is not None:
        # Check that num_reads is finite
        if not np.isfinite(num_reads):
            raise ValueError("num_reads must be finite")
        
        # Check for non-positive reads
        if num_reads <= 0:
            raise ValueError("num_reads must be > 0")


def compute_energy_scale_parameters(J, h):
    """
    Compute energy scale parameters for the Digital Annealer.
    
    Parameters:
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - delta_E_max: maximum energy scale
    - delta_E_min: minimum energy scale
    """
    # Convert to numpy arrays if not already
    J = np.asarray(J, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    
    N = len(h)
    
    # Compute ΔE_max = max_i (2|h_i| + sum_j 2|J_ij|)
    delta_E_max = 0.0
    for i in range(N):
        local_sum = 2 * abs(h[i])
        for j in range(N):  # Sum over all j, including j=i (but J_ii=0)
            local_sum += 2 * abs(J[i, j])
        delta_E_max = max(delta_E_max, local_sum)
    
    # Handle all-zero model case
    if delta_E_max == 0:
        return 0.0, 1.0  # Special case values
    
    # Compute ΔE_min = min_{i,j; J_ij ≠ 0} |J_ij|. If all J=0, use min_{i; h_i ≠ 0} |h_i|. If all h=0, set to 1.0.
    delta_E_min = float('inf')
    
    # Check for non-zero J entries
    for i in range(N):
        for j in range(N):
            if J[i, j] != 0:
                delta_E_min = min(delta_E_min, abs(J[i, j]))
    
    # If no non-zero J entries, check h
    if delta_E_min == float('inf'):
        for i in range(N):
            if h[i] != 0:
                delta_E_min = min(delta_E_min, abs(h[i]))
    
    # If still infinite, set to 1.0
    if delta_E_min == float('inf'):
        delta_E_min = 1.0
    
    return delta_E_max, delta_E_min


def compute_schedule_parameters(J, h):
    """
    Compute the schedule parameters for the Digital Annealer.
    
    Parameters:
    - J: coupling matrix (NxN)
    - h: local field vector (N)
    
    Returns:
    - beta_0: initial inverse temperature
    - beta_f: final inverse temperature
    - delta: dynamic offset rate
    """
    # Compute energy scale parameters
    delta_E_max, delta_E_min = compute_energy_scale_parameters(J, h)
    
    # Handle all-zero model case
    if delta_E_max == 0:
        return 0.1, 5.0, 0.5  # Special case values from spec
    
    # Compute beta schedule
    beta_0 = 0.1 / delta_E_max
    beta_f = 5.0 / delta_E_min
    
    # Compute dynamic offset rate
    delta = 0.5 * delta_E_min
    
    return beta_0, beta_f, delta