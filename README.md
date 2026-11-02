# Adaptive QEM Benchmark

This repository studies when quantum error mitigation improves an expectation-value estimate, when it wastes a fixed shot budget, and when it increases statistical instability.

The current scope is simulator-only. It provides:

- analytic and noiseless references for Bell, GHZ, and fixed-seed hardware-efficient circuits;
- controlled readout and depolarizing noise models;
- an unmitigated estimator with shot-noise uncertainty and resource accounting;
- full-assignment-matrix readout mitigation with explicit conditioning diagnostics;
- deterministic CSV and PNG artifacts generated from fixed seeds.

Machine learning, real hardware, and zero-noise extrapolation are outside this batch. ZNE will be added only after the baseline and readout-mitigation pipelines remain stable under a fixed total shot budget.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m pytest
$env:MPLCONFIGDIR = ".matplotlib-cache"
.\.venv\Scripts\python.exe -m src.generate_artifacts
```

Generated data are written to `results/`; figures are written to `figures/`. Raw observations and corrected results are stored separately in the same schema and are never overwritten during aggregation.

## Evidence boundary

All current results are analytic, noiseless simulated, or noisy simulated. They are not hardware-derived and do not establish device performance.
