# Security Notes & Posture

> Security scope, numerical safeguards, and execution boundaries for `pqblocks-digital-anneal`.

---

## 1. Non-Cryptographic Classification

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

`pqblocks-digital-anneal` is a classical numerical optimization algorithm inspired by specialized CMOS digital annealing architectures. It does not implement, expose, or validate cryptographic algorithms, key exchange schemes, signatures, or cryptographic random number generators.

---

## 2. Numerical Safeguards & Local Field Updates

- **$O(1)$ Incremental Energy Updates**:
  When spin $i$ is flipped, local fields for all adjacent spins $j$ update according to:
  $$\Delta E_j' = \Delta E_j - 4 s_i s_j J_{ij}$$
  Numerical operations are strictly validated against floating-point NaN and infinity propagation.
- **Matrix Consistency & Symmetry**:
  The coupling matrix $J$ is verified for exact symmetry ($J = J^T$) and zero diagonal elements ($J_{ii} = 0$). Asymmetric matrices or non-zero self-couplings are rejected with descriptive validation errors.
- **Bounded Dynamic Offset**:
  The dynamic energy offset rate $\delta$ is derived strictly from coupling matrix spectral norms and bounded to prevent numeric overflow during exponential acceptance probability evaluations.
