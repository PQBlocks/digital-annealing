# Testing & Test Suite Documentation

> Test suite coverage, execution procedures, and verification breakdown for `pqblocks-digital-anneal`.

---

## 1. Test Suite Summary

The test suite contains **41 comprehensive tests** covering the Digital Annealer parallel-trial Monte Carlo kernel, dynamic energy offset updates, local field calculations, statistical Kolmogorov-Smirnov equivalence, and strict boundary validation.

- **Status**: 41 passed, 0 failed
- **Runtime**: ~2.0 seconds
- **Statement Coverage**: 73% across 225 statements
- **Python Matrix**: Fully verified on Python 3.11, 3.12, 3.13, and 3.14

---

## 2. Test File Breakdown

| Test File | Tests | Focus Area |
|---|---|---|
| `test_core.py` | 4 | Pure Python local fields, delta energy calculations, and single-spin flip updates |
| `test_da.py` | 4 | Worked 4-spin validation, Python vs Numba numerical equality, bit-identical seed reproducibility, KS statistical equivalence |
| `test_numba_kernel.py` | 4 | JIT-compiled Ising energy, local field vectors, and flip update acceleration |
| `test_pathological_inputs.py` | 5 | Boundary and edge cases: $N=1$ models, non-positive reads, NaN/Inf floating-point bounds |
| `test_solver.py` | 10 | `DigitalAnnealerSampler` API, multiple reads, empty inputs, asymmetric matrices, non-zero diagonal rejection |
| `test_validation.py` | 14 | Input validation, matrix dimension checks, temperature schedule auto-parameterization ($\beta_0, \beta_f, \delta$) |

---

## 3. Running the Tests

To run the complete test suite:

```bash
PYTHONPATH=src pytest -v
```

To run with coverage reporting:

```bash
PYTHONPATH=src pytest --cov=pqblocks_digital_anneal --cov-report=term
```
