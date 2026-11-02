# Phase 0 Through Readout Mitigation Completion Checklist

| Requirement | Evidence | Status |
|---|---|---|
| Repository terminology, seed policy, labels, dependencies, and non-goals | `AGENTS.md`, `docs/definitions.md`, `docs/limitations.md` | Complete |
| Reproducible Python environment | `pyproject.toml`, `requirements-ci.txt`, `.github/workflows/ci.yml` | Complete |
| Root-seed implementation | `src/config.py`, `tests/test_config.py` | Complete |
| Bell XX analytic, statevector, and 1,024-shot example | `src/minimal_example.py`, `results/day02_bell_xx_shots-1024_seed-20261102.csv` | Complete |
| Bell estimate figure | `figures/day02_bell_xx_estimate_distribution.png` | Complete |
| Ordering, seed, and analytic regression tests | `tests/test_minimal_example.py` | Complete |
| GHZ widths 2, 4, and 6 with global X and neighboring ZZ | `src/circuits.py`, `src/observables.py` | Complete |
| GHZ analytic/statevector references and metadata | `src/references.py`, `tests/test_ghz_references.py`, `results/day03_ghz_references_widths-2-4-6.csv` | Complete |
| Fixed-seed hardware-efficient width-depth grid | `src/hardware_efficient_circuits.py`, `tests/test_hardware_efficient_references.py` | Complete |
| Nontrivial hardware-efficient statevector references | `results/day04_he_references_w-2-4-6_d-1-3-6_seed-20261102.csv` | Complete |
| Readout, one-qubit, two-qubit, and mixed noise | `src/noise_models.py`, `tests/test_noise_models.py` | Complete |
| Zero/low/medium/high noise validation and plot | `results/day05_noise_response_bell_xx_shots-4096.csv`, `figures/day05_error_vs_noise_by_family.png` | Complete |
| Logical and transpiled execution metadata | `src/circuits.py`, `src/estimators/unmitigated.py` and generated CSV columns | Complete |
| Unmitigated estimate and analytic uncertainty | `src/estimators/unmitigated.py`, `tests/test_unmitigated.py` | Complete |
| Fixed-budget resource ledger and raw-count preservation | `src/resource_accounting.py`, `tests/test_resource_accounting.py` | Complete |
| Readout calibration with 20/80 budget split | `src/mitigation/readout.py`, `tests/test_readout_mitigation.py` | Complete |
| Conditioning diagnostics and constrained correction | `constrained_readout_correction()` and ill-conditioned regression test | Complete |
| Raw, calibration, and corrected evidence retained | `results/day07_readout_calibration_budget-4096_repeats-30.csv`, `results/day07_readout_mitigation_budget-4096_repeats-30.csv` | Complete |
| Raw-versus-corrected figure | `figures/day07_readout_raw_vs_corrected.png` | Complete |

No task in the authoritative batch requires additional information. Subsequent work added
fixed-budget gate folding and linear ZNE, the frozen volumetric manifest, the repeat-level
pilot and GHZ benchmark, and the reproducible paper assets documented in the repository root.
