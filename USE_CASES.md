# Use Cases & Integration Patterns

> Practical applications and integration scenarios for `pqblocks-digital-anneal`.

---

## 1. Escaping Degenerate Energy Plateaus

Standard Metropolis Simulated Annealing selects candidate spin flips sequentially at random. On combinatorial optimization problems featuring degenerate energy plateaus (where many transitions yield $\Delta E = 0$), classical walkers undergo slow diffusive random walks.

The Digital Annealer algorithm implements parallel-trial spin updates paired with a dynamic offset energy $\delta$, ensuring that plateau states are systematically traversed and exited:

```python
import numpy as np
from pqblocks_digital_anneal import DigitalAnnealerSampler

# Define dense coupling matrix with plateau states
N = 8
J = np.random.randn(N, N)
J = (J + J.T) / 2.0
np.fill_diagonal(J, 0.0)
h = np.zeros(N)

sampler = DigitalAnnealerSampler(seed=42)
results = sampler.sample(J, h, num_sweeps=1000, num_reads=5)

best_result = min(results, key=lambda x: x['energy'])
print(f"Optimal energy: {best_result['energy']:.4f}")
print(f"Best configuration: {best_result['sample']}")
```

---

## 2. Multi-Read Stochastic Exploration

Because combinatorial landscapes often contain multiple disconnected low-energy basins, running multiple independent reads allows agents to sample diverse low-lying configurations:

```python
# Perform 10 independent reads with distinct random seeds
results = sampler.sample(J, h, num_sweeps=500, num_reads=10)

energies = [r['energy'] for r in results]
print(f"Energy range across 10 reads: [{min(energies):.4f}, {max(energies):.4f}]")
```

---

## 3. Lossless Integration with QUBO Formulation Pipelines

When solving binary optimization problems ($x_i \in \{0, 1\}$), problems can be converted losslessly to Ising spin models ($s_i \in \{-1, +1\}$) via $x_i = (s_i + 1)/2$ using `pqblocks-qubo`:

```python
from pqblocks_qubo import qubo_to_ising
from pqblocks_digital_anneal import DigitalAnnealerSampler

# Convert QUBO matrix Q to Ising J, h, offset
J, h, offset = qubo_to_ising(Q)

# Solve Ising formulation
sampler = DigitalAnnealerSampler()
results = sampler.sample(J, h, num_sweeps=1000)

ising_best = min(results, key=lambda x: x['energy'])
final_qubo_energy = ising_best['energy'] + offset
```
