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
print(f'Best energy: {best["energy"]:.6f}')
