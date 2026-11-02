# Adaptive QEM Benchmark

A reproducible simulator study of when quantum error mitigation helps, wastes a fixed
shot budget, or fails explicitly.

The benchmark compares raw estimation, full-assignment-matrix readout mitigation, and
linear zero-noise extrapolation (ZNE) for the GHZ observable `X^w`. It crosses widths
2, 4, and 6; four mixed-noise levels; total budgets of 1,024, 4,096, and 16,384 shots;
and 50 deterministic repeats.

## Verified result

Among 34 operating cells where all three methods completed 50 repeats, readout
mitigation had the lowest RMSE in 21 cells, raw estimation in 9, and linear ZNE in 4.
Readout mitigation also produced 28 explicit `ill_conditioned` failures at width 6
and a 1,024-shot budget. No method is uniformly reliable across the declared grid.

![Operating-regime map showing raw, readout, ZNE, and failed cells](paper/figures/operating_regime_map.png)

Read the full paper: **[Adaptive Quantum Error Mitigation Under Fixed Shot Budgets](paper/main.pdf)**

## Evidence boundary

- **Analytic:** Bell/GHZ stabilizer expectations and shot-noise relations.
- **Noiseless simulated:** statevector references and folding equivalence checks.
- **Noisy simulated:** all finite-shot benchmark results, using idealized Qiskit Aer
  readout and depolarizing channels.
- **Hardware-derived:** none.

The noise levels are controlled study parameters, not calibrations from an IBM or
other physical device.

## Methods and resource accounting

- Raw estimation spends the full budget on the target circuit.
- Readout mitigation spends 20% on all `2^w` calibration states and 80% on the target.
- Linear ZNE splits the same total budget across global folds 1, 3, and 5 and retains
  unclipped extrapolates.
- Every stochastic seed derives from root seed `20261102`.
- Failed, raw, corrected, and factor-level observations remain explicit.

## Reproduce

Python 3.11 or 3.12 is required. The full Day 8-12 benchmark takes several minutes on
a typical desktop.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = ".matplotlib-cache"

.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_artifacts
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_days08_12
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_paper_assets
tectonic paper/main.tex --outdir paper
```

Verify committed deterministic outputs:

```powershell
git diff --exit-code -- results experiments/manifests paper/generated paper/figures
```

## Generation run logs

Each wrapped generation command writes a compact JSON record under `results/runs/`.
The record contains timing, environment and Git metadata, status, and SHA-256 hashes
for files written under `results/` and `figures/`; it does not embed result data.

```powershell
ls results/runs/
```

## Limitations

This is a small, simulator-only GHZ benchmark. It does not model device drift,
crosstalk, leakage, relaxation, correlated readout, or calibration uncertainty. The
declared GHZ depth values 3 and 6 are retained as `not_applicable`; only depth 1 is
executed. Full-matrix calibration scales exponentially with width. Aggregate metrics
are conditional on completed rows, while operating-cell winners require all methods
to complete all 50 repeats.

## Repository structure

```text
src/                         Benchmark and mitigation implementations
tests/                       Analytic, statistical, and reproducibility invariants
experiments/                 Frozen configuration and manifest
results/raw/                 Immutable repeat-level benchmark rows
results/processed/           Coverage and runtime summaries
results/runs/                Per-run generation metadata and output hashes
figures/                     Generated project figures
paper/main.tex               Technical paper source
paper/main.pdf               Compiled paper
paper/generated/             Tables and macros derived from authoritative CSV data
paper/figures/               Paper figures generated from repository outputs
```

The project contains no license file because repository licensing intent has not been
established.
