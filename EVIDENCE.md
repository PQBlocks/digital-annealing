# Verification Evidence & Traceability

> Detailed execution evidence, command outputs, and reproducibility artifacts for `pqblocks-digital-anneal`.

---

## 1. Clean-Room Test Execution Evidence

Full test execution output in a clean virtual environment running Python 3.14.8:

```text
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /tmp/pqblocks-gate-da/tests
configfile: pyproject.toml
testpaths: tests
plugins: hypothesis-6.168.0, cov-7.1.0, anyio-4.15.1
collected 41 items

tests/test_core.py ....                                                  [  9%]
tests/test_da.py ....                                                    [ 19%]
tests/test_numba_kernel.py ....                                          [ 29%]
tests/test_pathological_inputs.py .....                                  [ 41%]
tests/test_solver.py ..........                                          [ 65%]
tests/test_validation.py ..............                                  [100%]

============================== 41 passed in 2.17s ==============================
```

---

## 2. Example Execution Evidence

### Quickstart Execution:

```bash
$ python3 examples/quickstart.py
Best Spin State: [ 1 -1  1 -1]
Best Energy: -3.000000
```

### Parallel Trial Escape Evaluation:

```bash
$ python3 examples/02_parallel_trial_plateau_escape.py
--- Digital Annealer Parallel-Trial Plateau Escape ---
Read 1: Energy = -10.6645, State = [-1  1 -1  1  1 -1  1 -1]
Read 2: Energy = -10.6645, State = [-1  1 -1  1  1 -1  1 -1]
Read 3: Energy = -10.6645, State = [-1  1 -1  1  1 -1  1 -1]
Read 4: Energy = -10.6645, State = [-1  1 -1  1  1 -1  1 -1]

Overall Ground Energy Found: -10.6645
Digital Annealing parallel-trial evaluation completed successfully.
```
