"""Example 02: Solving fully-connected spin-glass problems via Digital Annealer parallel trials."""

import numpy as np
from pqblocks_digital_anneal import DigitalAnnealerSampler


def main():
    # 8-spin fully-connected spin glass problem
    N = 8
    np.random.seed(42)
    J = np.random.randn(N, N)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = np.random.randn(N) * 0.5

    print("--- Digital Annealer Parallel-Trial Optimization ---")
    sampler = DigitalAnnealerSampler(seed=42)

    # Execute with 5 independent parallel-trial reads
    results = sampler.sample(
        J=J,
        h=h,
        num_sweeps=500,
        num_reads=5,
    )

    best = min(results, key=lambda x: x["energy"])
    print(f"Optimal Energy Found: {best['energy']:.6f}")
    print(f"Best Spin State:      {best['sample'].tolist()}")
    print(f"Total Reads Evaluated: {len(results)}")

    print("\nParallel-trial optimization completed successfully.")


if __name__ == "__main__":
    main()
