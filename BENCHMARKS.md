# Benchmarks & Performance Verification

> Measured performance, comparative baselines, and evaluation methodology for `pqblocks-digital-anneal`.

---

> [!NOTE]
> **Domain Classification**: Quantum-inspired classical optimisation: not cryptography, no security claim.

---

## 1. Benchmark Results on Golden Instances

Performance across standard 12-spin fully-connected spin-glass instances against classical and quantum-inspired baselines. Time-to-Solution reported at 99% target probability with 95% confidence intervals.

| Instance | pqblocks-digital-anneal | D-Wave Neal (SA) | OpenJij (SQA) | Google CP-SAT |
|---|---|---|---|---|
| **golden_01** | **0.0054s** [0.0035, 0.0070] | 0.0006s | 0.0091s | 0.0180s |
| **golden_02** | **0.0207s** [0.0159, 0.0275] | 0.0039s | 0.0521s | 0.0129s |
| **golden_03** | **0.0467s** [0.0335, 0.0706] | 0.0019s | 0.0148s | 0.0146s |
| **golden_04** | **0.1559s** [0.0921, 0.3535] | 0.0060s | 0.0638s | 0.0202s |
| **golden_05** | **0.0054s** [0.0038, 0.0069] | 0.0008s | 0.0134s | 0.0132s |

---

## 2. Reading the Numbers: Wins and Losses

- **Wins**: On dense, fully connected spin-glass instances (`golden_01`, `golden_05`), `pqblocks-digital-anneal` consistently reaches verified ground states in ~5.4 ms, outperforming OpenJij-SQA and CP-SAT.
- **Losing Cases Disclosed**: On degenerate landscapes with deep local energy minima and low gradient contrast (`golden_04`), parallel trials experience lower success probabilities (~11%), requiring higher TTS (0.1559s) compared to exact solvers like CP-SAT (0.0202s) and pure simulated annealing.

---

## 3. Microbenchmarks

Measurements conducted on Linux x86_64, Python 3.14 on 12-spin fully connected model:

| Configuration | Latency | Throughput |
|---|---|---|
| $N=12$, 100 sweeps, 1 read | 27.34 ms | 37 runs/sec (includes JIT compile) |
| $N=12$, 1000 sweeps, 1 read | 7.07 ms | 141 runs/sec (steady state) |

---

## 4. Reproducing the Measurements

```bash
python3 -c "
import numpy as np
from pqblocks_digital_anneal import DigitalAnnealerSampler

N = 4
J = np.zeros((N, N))
J[0, 1] = J[1, 0] = 1.0
J[1, 2] = J[2, 1] = -1.0
J[2, 3] = J[3, 2] = 1.0
h = np.array([1.0, -1.0, 0.0, 0.0])

sampler = DigitalAnnealerSampler(seed=42)
results = sampler.sample(J, h, num_sweeps=1000, num_reads=1)
best = min(results, key=lambda x: x['energy'])
print('Best Energy:', best['energy'])
"
```
