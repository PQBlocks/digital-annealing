# Changelog

All notable changes to `pqblocks-digital-anneal` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-26

### Added
- Quantum-inspired Digital Annealer heuristic solver for dense Ising optimization problems.
- Parallel-trial Monte Carlo update mechanism evaluating candidate spin flips concurrently.
- Dynamic energy offset rate ($\delta$) escaping degenerate energy plateaus.
- High-performance Numba JIT-compiled kernels for local field updates and energy evaluation.
- Strict input validation ensuring symmetric zero-diagonal coupling matrices and finite parameters.
- Comprehensive test suite with 41 tests passing across Python 3.11, 3.12, 3.13, and 3.14.
- Quickstart and parallel-trial plateau escape examples.
