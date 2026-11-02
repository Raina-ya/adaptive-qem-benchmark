# Repository Contract

## Research question

Under a fixed total shot budget, when do unmitigated estimation and readout mitigation improve or degrade expectation-value accuracy and stability across controlled circuit and noise conditions?

## Required boundaries

- Use deterministic classical simulation only until the simulator benchmark is stable.
- Do not add machine learning or real-hardware execution in the current scope.
- Preserve analytic, noiseless simulated, noisy simulated, and hardware-derived labels.
- Count calibration circuits and all other extra executions in resource overhead.
- Preserve repeat-level raw data; never replace it with corrected or aggregated values.
- Keep modules small, explicit, and covered by regression tests.

## Reproducibility

- Derive all stochastic seeds from `src.config.ROOT_SEED`.
- Save tabular artifacts under `results/` and plots under `figures/`.
- Include grid parameters in generated filenames or rows.
- Treat committed result schemas and analytic relations as regression contracts.
