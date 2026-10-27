# Verification Report: pqblocks-digital-anneal

> Formal release verification report for `pqblocks-digital-anneal`.

---

## 1. Executive Summary

| Attribute | Value |
|---|---|
| **Product** | `pqblocks-digital-anneal` |
| **Package Name** | `pqblocks_digital_anneal` |
| **Version** | `0.1.0` |
| **Classification** | Quantum-inspired classical optimisation: not cryptography, no security claim |
| **Verification Status** | **PASSED** |
| **Test Matrix** | Python 3.11, 3.12, 3.13, 3.14 |
| **Test Results** | 41 passed, 0 failed across all supported versions |
| **Leak Gate** | 0 hits (22 files scanned) |

---

## 2. Verification Gate Results

| Check ID | Verification Item | Target | Result | Status |
|---|---|---|---|---|
| **CHK-01** | Test Suite Completeness | 41 passing tests | 41 passed, 0 failed | **PASS** |
| **CHK-02** | Python Compatibility | Python 3.11–3.14 | Clean execution on all 4 versions | **PASS** |
| **CHK-03** | Source & Wheel Build | PEP 517 / PEP 621 | Valid sdist and wheel generated | **PASS** |
| **CHK-04** | Package Integrity | `twine check --strict` | Passed without warnings | **PASS** |
| **CHK-05** | Public Leak Gate | 0 internal leaks | 0 hits across all files | **PASS** |
| **CHK-06** | Example Verification | $\ge 2$ functioning examples | `quickstart.py` & `02_parallel_trial_plateau_escape.py` passed | **PASS** |
| **CHK-07** | Statistical Equivalence | Two-tier KS test | Passed with $p > 0.05$ | **PASS** |

---

## 3. Mathematical & Algorithmic Verification

- **Parallel-Trial Monte Carlo Updates**: Evaluates all candidate single-spin flip candidates in parallel per sweep, accelerating exploration across uniform-energy states.
- **Dynamic Energy Offset Rate ($\delta$)**: Offsets the acceptance threshold dynamically during trapped states, enabling efficient escape from plateau configurations without requiring artificially high thermal excitation.
- **Statistical Equivalence**: Verified via two-sample Kolmogorov-Smirnov test comparing pure Python reference and Numba JIT-accelerated kernel distributions.

---

## 4. Conclusion

The package `pqblocks-digital-anneal` has satisfied all release verification requirements and is verified for release.
