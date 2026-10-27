# pqblocks-digital-anneal

> Digital Annealer-style parallel-trial heuristic solver for Ising and QUBO optimization problems with dynamic plateau-escape mechanisms.

[![CI](https://github.com/PQBlocks/digital-annealing/actions/workflows/ci.yml/badge.svg)](https://github.com/PQBlocks/digital-annealing/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## Overview

`pqblocks-digital-anneal` is a classical quantum-inspired optimization algorithm modeled on the Fujitsu Digital Annealer architecture.

It implements a parallel-trial Monte Carlo method with dynamic energy offset plateau-escape mechanisms to solve fully-connected Quadratic Unconstrained Binary Optimization (QUBO) and Ising problems. By evaluating candidate spin-flip transitions in parallel and dynamically applying energy offsets ($E_{\text{off}}$) when rejected transitions accumulate, the algorithm avoids getting trapped in flat energy plateaus and shallow local minima.

### Key Highlights

- **Parallel-Trial Monte Carlo**: Evaluates multiple spin-flip candidates simultaneously across fully connected graphs.
- **Dynamic Plateau Escape**: Progressively boosts acceptance probabilities via an offset parameter $E_{\text{off}}$ when candidate transitions stagnate on energy plateaus.
- **Incremental Energy Updates**: Tracks local fields $h_i^{\text{eff}} = h_i + \sum_j J_{ij} s_j$ to update $\Delta E$ in $O(N)$ operations per accepted flip instead of $O(N^2)$ recomputation.
- **Numba JIT Acceleration**: Critical inner loops compiled to native machine instructions for high throughput.

---

## Installation

```bash
pip install pqblocks-digital-anneal
```

### Dependencies

Requires Python 3.11+, `numpy`, and `numba`.

---

## Quickstart

```python
import numpy as np
from pqblocks_digital_anneal import DigitalAnnealerSampler

# 1. Define 4-spin Ising problem
N = 4
J = np.zeros((N, N))
J[0, 1] = J[1, 0] = 1.0
J[1, 2] = J[2, 1] = -1.0
J[2, 3] = J[3, 2] = 1.0
h = np.array([1.0, -1.0, 0.0, 0.0])

# 2. Sample with Digital Annealer
sampler = DigitalAnnealerSampler(seed=42)
results = sampler.sample(J, h, num_sweeps=1000, num_reads=1)

# 3. Retrieve optimal solution
best = min(results, key=lambda x: x["energy"])
print(f"Best energy: {best['energy']:.6f}")
print("Spin State:", best["sample"].tolist())
```

---

## Architecture & Algorithm

1. **Local Field Maintenance**: Maintains effective local fields $h_i^{\text{eff}}$ such that the candidate spin-flip energy jump $\Delta E_i = 2 s_i h_i^{\text{eff}}$.
2. **Acceptance Probability with Offset**: Transitions are accepted with probability $P(\Delta E_i) = \min(1, \exp(-\beta (\Delta E_i - E_{\text{off}})))$.
3. **Offset Dynamics**: If no candidate flips are accepted in a trial step, $E_{\text{off}}$ increases incrementally by $\Delta E_{\text{off}}$, enabling escape from flat plateaus; upon accepting a transition, $E_{\text{off}}$ resets to zero.

---

## Fair Benchmarking Disclaimer

When benchmarking the Digital Annealer against classical simulated annealing (D-Wave Neal) and exact solvers (Google OR-Tools CP-SAT):
- **Landscape Dependence**: Digital Annealer parallel trials excel on fully-connected spin glasses where random parallel trials navigate unstructured landscapes efficiently.
- **Losing Cases Disclosed**: On highly degenerate landscapes with deep narrow minima and small gradient differences (such as `golden_04`), exact branch-and-cut solvers (CP-SAT) find verified global optima faster than stochastic parallel trials.

---

## Verification & Testing

Tested across Python 3.11, 3.12, 3.13, and 3.14.

```bash
pytest
```

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
