# Autonomous Agent Integration Guide

> Operational instructions and best practices for AI and autonomous software agents integrating `pqblocks-digital-anneal`.

---

## 1. Package Purpose

`pqblocks-digital-anneal` provides a quantum-inspired Digital Annealer heuristic solver implementing parallel-trial Monte Carlo moves and dynamic energy offsets for dense Ising models.

---

## 2. Key APIs

```python
from pqblocks_digital_anneal import DigitalAnnealerSampler

# Initialize sampler
sampler = DigitalAnnealerSampler(seed=42)

# Execute optimization
results = sampler.sample(
    J=coupling_matrix,   # np.ndarray: (N, N) symmetric coupling matrix with 0 diagonal
    h=local_field_vector,# np.ndarray: (N,) linear bias vector
    num_sweeps=1000,     # int: number of Monte Carlo sweeps (default: 1000)
    num_reads=1          # int: number of independent runs (default: 1)
)

# Extract optimal configuration
best = min(results, key=lambda r: r['energy'])
print("Best sample:", best['sample'])    # np.ndarray of {-1, 1}
print("Best energy:", best['energy'])    # float
```

---

## 3. Best Practices for Agents

1. **Matrix Validation**: Ensure the coupling matrix $J$ is strictly symmetric ($J = J^T$) and has zeroes along the main diagonal ($J_{ii} = 0$). Use `(J + J.T) / 2.0` and `np.fill_diagonal(J, 0.0)` if converting arbitrary quadratic matrices.
2. **Multi-Read Sampling**: On complex optimization problems ($N > 20$), run `num_reads=10` or higher to sample diverse basins of attraction.
3. **Reproducibility**: Specify `seed` during `DigitalAnnealerSampler` construction for deterministic, verifiable behavior.
