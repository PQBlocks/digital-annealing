import numpy as np
from .core import compute_delta_energies, update_delta_energies_after_flip
from .numba_kernel import (
    compute_delta_energies_numba, 
    update_delta_energies_after_flip_numba,
    compute_acceptance_probabilities_numba,
    parallel_trial_mc_step_numba,
    ising_energy_numba,
    xorshift32
)
from .validation import validate_input, validate_parameters, compute_schedule_parameters


class DigitalAnnealerSampler:
    """
    A quantum-inspired Digital Annealer sampler for solving Ising models.
    
    This solver is not a cryptographic primitive and provides no cryptographic security.
    """
    
    def __init__(self, seed=None):
        """
        Initialize the Digital Annealer sampler.
        
        Parameters:
        - seed: Random seed for reproducibility
        """
        self.seed = seed if seed is not None else np.random.randint(0, 2**32)
    
    def sample(self, J, h, num_sweeps=1000, num_reads=1):
        """
        Sample from the Ising model using the Digital Annealer algorithm.
        
        Parameters:
        - J: Coupling matrix (NxN)
        - h: Local field vector (N)
        - num_sweeps: Number of Monte Carlo sweeps (must be a positive integer)
        - num_reads: Number of independent reads (must be a positive integer)
        
        Returns:
        - results: List of dictionaries containing samples and energies
        """
        # Validate inputs
        J, h = validate_input(J, h)
        
        # Validate parameters before converting to int
        # We need to check finiteness before converting to int
        if not np.isfinite(num_sweeps):
            raise ValueError("sweeps must be finite")
        if not np.isfinite(num_reads):
            raise ValueError("num_reads must be finite")
            
        # Convert to int and validate
        num_sweeps = int(num_sweeps)
        num_reads = int(num_reads)
        validate_parameters(num_sweeps, num_reads)
        
        # Compute schedule parameters
        beta_0, beta_f, delta = compute_schedule_parameters(J, h)
        
        # Initialize results list
        results = []
        
        # Perform independent reads
        for read_index in range(num_reads):
            # Create a new random seed for this read
            read_seed = (self.seed + read_index) & 0xFFFFFFFF
            
            # Sample using the Digital Annealer algorithm
            sample_result = self._sample_single_read(J, h, num_sweeps, beta_0, beta_f, delta, read_seed)
            results.append(sample_result)
        
        return results
    
    def _sample_single_read(self, J, h, num_sweeps, beta_0, beta_f, delta, seed):
        """
        Perform a single read/sample from the Ising model.
        
        Parameters:
        - J: Coupling matrix (NxN)
        - h: Local field vector (N)
        - num_sweeps: Number of Monte Carlo sweeps
        - beta_0: Initial inverse temperature
        - beta_f: Final inverse temperature
        - delta: Dynamic offset rate
        - seed: Random seed for this read
        
        Returns:
        - result: Dictionary containing the sample and energy
        """
        N = len(h)
        
        # Initialize spin configuration randomly
        rng = np.random.default_rng(seed)
        spin_configuration = np.where(rng.random(N) < 0.5, -1, 1).astype(np.float64)
        
        # Initialize energy offset
        E_off = 0.0
        
        # Geometric cooling schedule
        beta_schedule = np.logspace(np.log10(beta_0), np.log10(beta_f), num_sweeps)
        
        # Initialize delta energies for incremental updates
        # Note: In first step, we compute them in the kernel. For subsequent steps, they're passed between steps.
        delta_E = None
        
        # Perform Monte Carlo sweeps
        for sweep in range(num_sweeps):
            beta = beta_schedule[sweep]
            
            # Perform one parallel-trial Monte Carlo step
            if delta_E is None:
                # First step - compute delta_E in the kernel
                spin_configuration, E_off, accepted, delta_E = parallel_trial_mc_step_numba(
                    spin_configuration, J, h, beta, E_off, delta, (seed + sweep) & 0xFFFFFFFF
                )
            else:
                # Subsequent steps - use incremental updates
                # We need to recompute acceptance probabilities with the current delta_E
                P = compute_acceptance_probabilities_numba(delta_E, beta, E_off)
                
                # Determine which flips are accepted
                accepted_indices = []
                rng_state = (seed + sweep) & 0xFFFFFFFF
                for i in range(len(spin_configuration)):
                    rng_state, rand_val = xorshift32(rng_state)
                    if rand_val < P[i]:
                        accepted_indices.append(i)
                
                # If no flips accepted, increment E_off by delta
                if len(accepted_indices) == 0:
                    E_off += delta
                    # delta_E remains unchanged for next step
                else:
                    # If flips accepted, select one uniformly at random
                    rng_state, rand_val = xorshift32(rng_state)
                    selected_index = accepted_indices[int(rand_val * len(accepted_indices)) % len(accepted_indices)]
                    
                    # Update delta energies incrementally in O(1) time (uses spin configuration before the flip)
                    delta_E = update_delta_energies_after_flip_numba(delta_E, spin_configuration, J, selected_index)
                    
                    # Update spin configuration
                    spin_configuration[selected_index] *= -1
                    
                    # Reset E_off
                    E_off = 0.0
        
        # Compute final energy
        final_energy = ising_energy_numba(spin_configuration, J, h)
        
        # Return result
        return {
            'sample': spin_configuration.copy(),
            'energy': final_energy
        }