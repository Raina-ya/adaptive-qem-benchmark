# QEM Project — Daily Task Checklist

## Standing Requirements (copy into every day)

> [!important] Daily execution contract
> - Generate today's requested outputs directly without asking for confirmation first.
> - Before starting, strictly review the previous day's pushed or committed content. Flag inconsistencies, failed checks, and unfinished work.
> - Keep every result reproducible: use fixed random seeds, save data under `results/`, save plots under `figures/`, and encode the parameter grid in generated filenames.
> - Do not introduce machine learning or real-hardware execution until the simulator benchmark is stable and its acceptance checks pass.
> - Compare QEM methods under the same total shot budget. Count calibration circuits, folded circuits, noise-factor runs, and other extra executions as overhead.
> - Preserve repeat-level raw data. Never overwrite raw observations with mitigated or aggregated values.
> - Label results as analytic, noiseless simulated, noisy simulated, or hardware-derived. Do not overstate evidence.
> - End the day by answering 2–3 self-check questions and writing a short note about the impact on later phases.

## Day 1

**Phase:** Phase 0 — Environment and definitions
**Task:** Establish the repository contract and reproducible environment.

### Input → Steps → Output

- **Input:** Project scope, Python 3.11/3.12, Qiskit/Aer requirements.
- **Steps:** Define terminology, seed policy, result labels, dependency strategy, and explicit non-goals.
- **Output:** Environment manifests and a written experiment contract.

### Deliverables

- `pyproject.toml`
- `requirements-ci.txt`
- `docs/definitions.md`
- `docs/limitations.md`
- `src/config.py`

### Definition of Done

- [ ] A clean environment can import all required packages.
- [ ] `shot`, `repeat`, `benchmark cell`, and `total shot budget` are defined.
- [ ] Simulator-only and no-ML boundaries are documented.
- [ ] A root-seed policy is written down.

### Self-check questions

1. Could another person reproduce the environment without guessing versions?
2. Are simulation claims clearly separated from hardware claims?
3. Is every non-goal explicit?

### Impact on later phases

This contract prevents later comparisons from silently changing definitions, dependencies, or evidence labels.

## Day 2

**Phase:** Phase 0 — Minimal runnable example
**Task:** Validate one Bell-state expectation from exact theory to finite-shot output.

### Input → Steps → Output

- **Input:** Two-qubit Bell circuit and the observable `XX`.
- **Steps:** Derive the exact value, run noiseless statevector evaluation, then run a fixed-seed 1,024-shot sample.
- **Output:** One tested CSV and one estimate-distribution figure.

### Deliverables

- `src/minimal_example.py`
- `tests/test_minimal_example.py`
- `results/day02_bell_xx_shots-1024_seed-2026.csv`
- `figures/day02_bell_xx_estimate_distribution.png`

### Definition of Done

- [ ] The noiseless result matches the analytic value within `1e-12`.
- [ ] Re-running with the same seed reproduces the CSV.
- [ ] The figure shows the analytic reference and sampled estimate.
- [ ] Bit and qubit ordering are covered by tests.

### Self-check questions

1. Can I derive the expected `XX` value without running code?
2. Does the seed control all stochastic steps?
3. Would a reversed bit order be caught by a test?

### Impact on later phases

This creates the smallest trusted vertical slice that every later circuit and mitigation method must preserve.

## Day 3

**Phase:** Phase 1 — Circuit family and references
**Task:** Build GHZ benchmark circuits with analytic observables.

### Input → Steps → Output

- **Input:** GHZ widths `2, 4, 6`; observables `X^⊗w` and neighboring `ZZ`.
- **Steps:** Generate circuits, derive expected values, and compare them with statevector simulation.
- **Output:** Circuit factory, reference table, and analytic regression tests.

### Deliverables

- `src/circuits.py`
- `src/observables.py`
- `src/references.py`
- `tests/test_ghz_references.py`
- `results/day03_ghz_references_widths-2-4-6.csv`

### Definition of Done

- [ ] Widths 2, 4, and 6 produce valid GHZ circuits.
- [ ] Every reference is labeled analytic or noiseless simulated.
- [ ] Statevector and analytic values agree within `1e-12`.
- [ ] Circuit metadata records width, logical depth, and gate counts.

### Self-check questions

1. Which GHZ observables must equal `+1`, and why?
2. Are reference values independent of finite-shot sampling?

### Impact on later phases

The GHZ family supplies hand-checkable controls for validating noise and mitigation behavior.

## Day 4

**Phase:** Phase 1 — Circuit family and references
**Task:** Add fixed-seed hardware-efficient circuits with nontrivial references.

### Input → Steps → Output

- **Input:** Widths `2, 4, 6`, depths `1, 3, 6`, fixed parameter seed.
- **Steps:** Generate parameterized circuits, select fixed Pauli observables, and calculate statevector references.
- **Output:** A deterministic second circuit family and reference dataset.

### Deliverables

- `src/hardware_efficient_circuits.py`
- `tests/test_hardware_efficient_references.py`
- `results/day04_he_references_w-2-4-6_d-1-3-6_seed-2026.csv`

### Definition of Done

- [ ] Circuit parameters regenerate identically from the root seed.
- [ ] References cover all declared width-depth combinations.
- [ ] At least some exact expectations are not near `±1`.
- [ ] Repeated generation is byte-stable or numerically identical by schema.

### Self-check questions

1. Why is a nontrivial expectation family needed in addition to GHZ?
2. Can the reference be regenerated without sampled data?

### Impact on later phases

This family tests whether QEM conclusions generalize beyond stabilizer-like endpoint values.

## Day 5

**Phase:** Phase 1 — Noise models
**Task:** Validate readout and depolarizing noise models.

### Input → Steps → Output

- **Input:** Both circuit families; zero, low, medium, and high noise settings.
- **Steps:** Add readout, one-qubit depolarizing, two-qubit depolarizing, and mixed-noise configurations; run endpoint checks.
- **Output:** Tested noise-model module and noise-response plots.

### Deliverables

- `src/noise_models.py`
- `tests/test_noise_models.py`
- `results/day05_noise_response_grid.csv`
- `figures/day05_error_vs_noise_by_family.png`

### Definition of Done

- [ ] Zero noise returns the noiseless reference.
- [ ] Noise is attached to the intended basis gates and qubits.
- [ ] Logical and transpiled depth are both recorded.
- [ ] Each curve identifies the simulated noise model.

### Self-check questions

1. Which error source is each noise configuration intended to isolate?
2. Could transpilation invalidate the intended noise placement?
3. Do any results suggest an implementation error rather than physics?

### Impact on later phases

These controlled models provide the independent variables used to identify each QEM method's operating regime.

## Day 6

**Phase:** Phase 2 — Unmitigated baseline
**Task:** Establish the raw estimator and resource ledger.

### Input → Steps → Output

- **Input:** Trusted circuits, observables, noise models, and shot budgets.
- **Steps:** Compute raw expectation estimates and analytic shot-noise uncertainty; record every circuit and shot.
- **Output:** Baseline estimator, resource schema, and validation data.

### Deliverables

- `src/estimators/unmitigated.py`
- `src/resource_accounting.py`
- `tests/test_unmitigated.py`
- `results/day06_unmitigated_shots-1024-4096-16384.csv`

### Definition of Done

- [ ] Pauli estimates remain in `[-1, 1]`.
- [ ] Total shots equal the declared budget.
- [ ] Raw counts remain available after aggregation.
- [ ] Analytic uncertainty agrees with repeated sampling within tolerance.

### Self-check questions

1. What uncertainty exists even when the simulator noise is zero?
2. Can every reported estimate be traced to raw counts and a shot total?

### Impact on later phases

All mitigation gains and overheads will be measured relative to this baseline.

## Day 7

**Phase:** Phase 2 — Readout mitigation
**Task:** Calibrate and correct measurement error under a fixed budget.

### Input → Steps → Output

- **Input:** Readout-noise model and a budget split of 20% calibration, 80% target shots.
- **Steps:** Build calibration circuits, estimate the confusion matrix, diagnose conditioning, and apply constrained correction.
- **Output:** Readout-mitigation module, calibration data, and correction figure.

### Deliverables

- `src/mitigation/readout.py`
- `tests/test_readout_mitigation.py`
- `results/day07_readout_calibration_budget-4096.csv`
- `figures/day07_readout_raw_vs_corrected.png`

### Definition of Done

- [ ] Calibration shots are included in total overhead.
- [ ] Zero readout error yields an identity confusion matrix within sampling error.
- [ ] Ill-conditioned matrices return an explicit status.
- [ ] Raw and corrected results are both preserved.

### Self-check questions

1. What errors can readout mitigation correct, and what errors can it not correct?
2. Does matrix inversion amplify sampling noise in any condition?
3. Is calibration reuse reported separately from standalone cost?

### Impact on later phases

This establishes the first method whose accuracy gain must be weighed against calibration cost and variance inflation.

## Day 8

**Phase:** Phase 2 — Zero-noise extrapolation
**Task:** Implement and validate gate folding at noise factors 1, 3, and 5.

### Input → Steps → Output

- **Input:** Target circuits and folding factors `1, 3, 5`.
- **Steps:** Fold supported gates, confirm ideal-unitary equivalence, and record depth and gate-count growth.
- **Output:** Folding module and structural validation table.

### Deliverables

- `src/mitigation/folding.py`
- `tests/test_gate_folding.py`
- `results/day08_fold_validation_lambda-1-3-5.csv`
- `figures/day08_depth_vs_noise_factor.png`

### Definition of Done

- [ ] Folded circuits preserve the ideal expectation within `1e-12`.
- [ ] Actual transpiled depth and two-qubit gate counts are recorded.
- [ ] Unsupported folding cases fail explicitly.
- [ ] Noise factor is encoded in every result filename or row.

### Self-check questions

1. Why should folding preserve the ideal unitary?
2. Does physical noise scale as cleanly as circuit depth?

### Impact on later phases

Reliable folding is the prerequisite for interpreting ZNE fits as controlled extrapolation rather than arbitrary curve fitting.

## Day 9

**Phase:** Phase 2 — ZNE comparison
**Task:** Add preregistered linear ZNE and compare all three methods fairly.

### Input → Steps → Output

- **Input:** Folded circuits, three noise factors, and equal total shot budgets.
- **Steps:** Split shots across factors, fit a linear model, save residuals, and compare raw, readout, and ZNE on selected cells.
- **Output:** ZNE estimator, validation CSV, and extrapolation plot.

### Deliverables

- `src/mitigation/zne.py`
- `tests/test_zne.py`
- `results/day09_method_validation_budget-4096_repeats-20.csv`
- `figures/day09_zne_extrapolation_lambda-1-3-5.png`

### Definition of Done

- [ ] ZNE shots sum to the same total budget as other methods.
- [ ] Raw factor-level estimates and fit residuals are saved.
- [ ] Linear extrapolation is fixed before viewing full benchmark results.
- [ ] Out-of-range estimates are retained and flagged, not silently clipped.

### Self-check questions

1. Did ZNE reduce bias, variance, or only one of them?
2. Are the scaled expectation values sufficiently smooth for extrapolation?
3. Would a different fit be a justified sensitivity analysis or result shopping?

### Impact on later phases

This completes the three comparable strategies required for the volumetric benchmark.

## Day 10

**Phase:** Phase 3 — Benchmark design
**Task:** Freeze the volumetric grid and result schema before the full run.

### Input → Steps → Output

- **Input:** Two circuit families, validated methods, and runtime measurements.
- **Steps:** Define width, depth, noise, shots, method, repeat, status, and seed columns; estimate workload.
- **Output:** Immutable v1 config and manifest.

### Deliverables

- `experiments/configs/volumetric_v1.yaml`
- `experiments/manifests/volumetric_v1_manifest.csv`
- `docs/benchmark_schema.md`
- `tests/test_manifest.py`

### Definition of Done

- [ ] The grid includes widths `2, 4, 6`, depths `1, 3, 6`, and three shot budgets.
- [ ] Zero plus low, medium, and high noise are represented.
- [ ] Invalid combinations are marked `not_applicable`.
- [ ] Expected cells, circuits, and shots are calculated before execution.

### Self-check questions

1. Can any grid dimension be changed after results without creating a new version?
2. Are all methods compared on identical experimental conditions?

### Impact on later phases

Freezing the manifest prevents cherry-picking and gives every later result a traceable experimental identity.

## Day 11

**Phase:** Phase 3 — Pilot benchmark
**Task:** Run a 20-repeat endpoint pilot and audit failures.

### Input → Steps → Output

- **Input:** Manifest endpoint subset and validated pipelines.
- **Steps:** Run endpoint cells, profile runtime, inspect missing rows, and classify failures.
- **Output:** Pilot raw data, audit report, and preliminary heatmap.

### Deliverables

- `src/benchmark_runner.py`
- `results/raw/day11_pilot_repeats-20.csv`
- `results/processed/day11_pilot_audit.csv`
- `figures/day11_pilot_rmse_heatmap.png`

### Definition of Done

- [ ] Every pilot cell has `complete`, `failed`, or `not_applicable` status.
- [ ] No failed cell is silently dropped.
- [ ] Runtime and storage estimates are updated.
- [ ] Only implementation defects, not unfavorable outcomes, trigger changes.

### Self-check questions

1. Which failures indicate numerical instability versus a code defect?
2. Is the full benchmark feasible at the planned repeat count?
3. Did any result tempt me to alter the preregistered grid?

### Impact on later phases

The pilot validates execution scale and exposes pipeline defects before expensive full-grid runs.

## Day 12

**Phase:** Phase 3 — Full GHZ benchmark
**Task:** Execute and validate the complete GHZ grid.

### Input → Steps → Output

- **Input:** Frozen manifest, GHZ circuits, and 50 repeat seeds.
- **Steps:** Run all applicable GHZ cells, checkpoint results, and audit coverage.
- **Output:** Repeat-level GHZ dataset and coverage report.

### Deliverables

- `results/raw/day12_ghz_volumetric_v1_repeats-50.csv`
- `results/processed/day12_ghz_coverage.csv`
- `figures/day12_ghz_error_vs_noise.png`

### Definition of Done

- [ ] All applicable GHZ cells have 50 complete repeats or an explicit failure status.
- [ ] Seed IDs are unique and reproducible.
- [ ] Raw repeat-level rows remain immutable.
- [ ] Coverage agrees with the manifest.

### Self-check questions

1. Are method differences larger than sampling uncertainty?
2. Which regimes show mitigation-induced variance?

### Impact on later phases

The analytically controlled GHZ results provide the safest foundation for interpreting the regime map.

## Day 13

**Phase:** Phase 3 — Full hardware-efficient benchmark
**Task:** Execute and validate the nontrivial circuit grid.

### Input → Steps → Output

- **Input:** Frozen manifest, fixed-seed hardware-efficient circuits, and 50 repeat seeds.
- **Steps:** Run all applicable cells, checkpoint data, and compare coverage with GHZ results.
- **Output:** Repeat-level second-family dataset and audit.

### Deliverables

- `results/raw/day13_he_volumetric_v1_repeats-50.csv`
- `results/processed/day13_he_coverage.csv`
- `figures/day13_he_error_vs_depth.png`

### Definition of Done

- [ ] All applicable cells are accounted for.
- [ ] Exact references remain linked by circuit ID.
- [ ] Resource accounting includes transpiled and folded circuits.
- [ ] No aggregation overwrites repeat-level data.

### Self-check questions

1. Do GHZ conclusions persist for nontrivial expectation values?
2. Which outcomes depend strongly on circuit family?
3. Are depth effects confounded with changed observables?

### Impact on later phases

This tests whether selector rules can generalize beyond one specially structured circuit family.

## Day 14

**Phase:** Phase 3 — Metrics and uncertainty
**Task:** Aggregate repeats into bias, variance, RMSE, overhead, and 95% intervals.

### Input → Steps → Output

- **Input:** Day 12–13 repeat-level data and exact references.
- **Steps:** Compute metrics, paired BCa bootstrap intervals, Wilson intervals where applicable, and primary figures.
- **Output:** Benchmark summary and reproducible plots.

### Deliverables

- `src/metrics.py`
- `tests/test_metrics.py`
- `results/processed/day14_benchmark_summary_v1.csv`
- `figures/day14_rmse_heatmap.png`
- `figures/day14_bias_variance_tradeoff.png`
- `figures/day14_overhead_vs_rmse_gain.png`

### Definition of Done

- [ ] Summary values can be recomputed from raw rows.
- [ ] Paired comparisons resample matching repeat IDs.
- [ ] Confidence method and resample count are recorded.
- [ ] Negative mitigation gains remain visible.

### Self-check questions

1. Does lower bias ever coincide with worse RMSE?
2. Are intervals wide enough to make some winners inconclusive?
3. Is overhead shown alongside accuracy?

### Impact on later phases

These metrics become the evidence used to label operating regimes and evaluate the selector.

## Day 15

**Phase:** Phase 4 — Operating-regime labels
**Task:** Define winners, practical ties, and inconclusive cells.

### Input → Steps → Output

- **Input:** Benchmark summary with uncertainty intervals.
- **Steps:** Define the RMSE oracle, practical-equivalence threshold, and inconclusive rule; reserve holdout conditions.
- **Output:** Training labels, holdout manifest, and first regime map.

### Deliverables

- `src/regime_map.py`
- `experiments/configs/holdout_v1.yaml`
- `results/processed/day15_regime_labels.csv`
- `figures/day15_operating_regime_map.png`

### Definition of Done

- [ ] At least 20% of conditions are hidden from rule design.
- [ ] Ties and inconclusive cells are allowed.
- [ ] Oracle labels depend on benchmark outcomes, not runtime-accessible inputs.
- [ ] The practical threshold is documented before holdout evaluation.

### Self-check questions

1. Is a forced winner scientifically justified in every cell?
2. Could the label rule unfairly favor a high-overhead method?

### Impact on later phases

Clear labels prevent the selector from learning or encoding arbitrary distinctions unsupported by uncertainty.

## Day 16

**Phase:** Phase 4 — Rule-based selector
**Task:** Write a one-page, interpretable QEM selection policy.

### Input → Steps → Output

- **Input:** Training labels and runtime-observable diagnostics only.
- **Steps:** Define rules using noise type, shot budget, gate count, fit residual, and calibration conditioning.
- **Output:** Frozen selector v1 and human-readable rule document.

### Deliverables

- `src/selector.py`
- `docs/selector_rules.md`
- `tests/test_selector.py`
- `results/processed/day16_selector_training_predictions.csv`

### Definition of Done

- [ ] The selector uses no ML.
- [ ] The selector never reads exact-reference error at inference time.
- [ ] Every decision can be explained by a short rule trace.
- [ ] The rules are frozen before viewing holdout outcomes.

### Self-check questions

1. Can I explain every rule without reading the source code?
2. Are any rules proxies for hidden exact-reference information?
3. Is `inconclusive` available when evidence is weak?

### Impact on later phases

The frozen rules turn the benchmark into an actionable but auditable recommendation system.

## Day 17

**Phase:** Phase 4 — Holdout evaluation
**Task:** Evaluate selector accuracy and regret on unseen conditions.

### Input → Steps → Output

- **Input:** Frozen selector and holdout seeds/noise levels.
- **Steps:** Run holdout experiments, compare selections with oracle labels, and calculate RMSE regret.
- **Output:** Unmodified holdout evaluation and confusion matrix.

### Deliverables

- `results/raw/day17_holdout_repeats-50.csv`
- `results/processed/day17_selector_evaluation.csv`
- `figures/day17_selector_confusion_matrix.png`
- `figures/day17_selector_rmse_regret.png`

### Definition of Done

- [ ] Selector accuracy, regret, and inconclusive rate are reported.
- [ ] Original rules are not changed after seeing holdout outcomes.
- [ ] New seeds and intermediate noise levels are used.
- [ ] Failures and null improvements remain in the report.

### Self-check questions

1. Where does the selector fail, and is the failure explainable?
2. Is low accuracy caused by poor rules or uncertain oracle labels?
3. Does the selector save resources relative to always using ZNE?

### Impact on later phases

Holdout performance determines whether the regime rules are credible enough to extend to decision tasks or hardware.

## Day 18

**Phase:** Phase 4 — Core benchmark release
**Task:** Freeze the simulator-first v1 result and reproducibility package.

### Input → Steps → Output

- **Input:** All Phase 0–4 code, data, figures, tests, and limitations.
- **Steps:** Regenerate artifacts from a clean environment, run checks, document claims, and tag incomplete work.
- **Output:** Reproducible v1 release candidate.

### Deliverables

- `README.md`
- `docs/methodology.md`
- `docs/releases/v0.1.0.md`
- `results/processed/core_v1_summary.csv`
- `figures/core_v1_regime_map.png`

### Definition of Done

- [ ] Tests and lint checks pass in a clean environment.
- [ ] Committed tables and figures regenerate from declared inputs.
- [ ] Claims distinguish supported findings from hypotheses.
- [ ] Limitations cover model, sampling, measurement, and generalization boundaries.

### Self-check questions

1. Can every headline claim be traced to a figure or table?
2. Are negative and inconclusive findings reported honestly?
3. Could another researcher reproduce the core result?

### Impact on later phases

Phase 5 may begin only after this simulator benchmark is stable; otherwise extensions would amplify unresolved defects.

## Day 19

**Phase:** Phase 5 — Decision-error probability
**Task:** Build a controlled two-choice decision benchmark.

### Input → Steps → Output

- **Input:** Two observables with exact decision margins `0.02, 0.05, 0.10, 0.20`.
- **Steps:** Generate repeated raw, readout, and ZNE estimates; calculate sign or argmax errors with Wilson intervals.
- **Output:** Decision-error dataset and margin curve.

### Deliverables

- `src/decision_metrics.py`
- `tests/test_decision_metrics.py`
- `results/day19_decision_error_margins-002-005-010-020.csv`
- `figures/day19_decision_margin_vs_error.png`

### Definition of Done

- [ ] Exact decisions are known independently of sampling.
- [ ] Every method uses the same total shot budget.
- [ ] Decision-error probability includes a Wilson interval.
- [ ] Expectation RMSE and decision-error results are shown separately.

### Self-check questions

1. Can lower expectation RMSE coexist with higher decision-error probability?
2. Which decision margins are unresolved at the current shot budget?
3. Does mitigation variance flip near-ties?

### Impact on later phases

This connects estimator quality to the downstream reliability question that motivated the project.

## Day 20

**Phase:** Phase 5 — Combined mitigation
**Task:** Test readout mitigation plus ZNE without hiding compounded overhead.

### Input → Steps → Output

- **Input:** Stable readout and ZNE pipelines; selected benchmark and decision cells.
- **Steps:** Allocate calibration and noise-factor shots within one fixed budget, run paired repeats, and compare combined versus single methods.
- **Output:** Combined-method results and overhead tradeoff figure.

### Deliverables

- `src/mitigation/combined.py`
- `tests/test_combined_mitigation.py`
- `results/day20_combined_qem_budget-4096-16384.csv`
- `figures/day20_combined_overhead_vs_gain.png`

### Definition of Done

- [ ] Calibration and ZNE executions are both counted.
- [ ] The total budget matches all comparison methods.
- [ ] Combined mitigation is evaluated on preregistered selected cells.
- [ ] Worse or inconclusive combined results remain visible.

### Self-check questions

1. Does combining methods correct distinct error sources?
2. Is any gain large enough to justify the compounded overhead?
3. Are interactions between correction and extrapolation documented?

### Impact on later phases

This shows whether modular QEM components cooperate under realistic resource accounting before hardware is considered.

## Day 21

**Phase:** Phase 5 — Hardware-readiness gate
**Task:** Decide whether the project is ready for optional IBM hardware execution.

### Input → Steps → Output

- **Input:** Stable simulator release, decision benchmark, and combined-QEM evidence.
- **Steps:** Freeze simulator predictions, define backend metadata requirements, specify a minimal hardware subset, and perform a go/no-go review without submitting jobs.
- **Output:** Hardware protocol or a documented decision to remain simulator-only.

### Deliverables

- `experiments/hardware/hardware_manifest_v1.csv`
- `docs/hardware_protocol.md`
- `docs/hardware_readiness_review.md`
- `results/processed/day21_final_project_summary.csv`

### Definition of Done

- [ ] Simulator predictions are frozen before any hardware result is seen.
- [ ] The protocol records backend, job ID, calibration time, shots, and transpilation metadata.
- [ ] The selected hardware subset fits a declared execution budget.
- [ ] A no-go decision is allowed and justified.
- [ ] No hardware-derived claim exists without actual hardware data.

### Self-check questions

1. Which simulator findings are genuinely testable on hardware?
2. What device drift or calibration effects would limit comparison?
3. Would hardware execution answer a research question or merely decorate the project?

### Impact on later phases

This gate protects the project's evidence quality and turns hardware access into a controlled extension rather than a requirement for completion.

## Progress Tracker

| Day | Phase | Status | Notes |
|---:|:---:|:---|:---|
| 1 | Phase 0 | Complete | Contract, environment, and seed policy verified |
| 2 | Phase 0 | Complete | Bell XX exact and finite-shot artifacts generated |
| 3 | Phase 1 | Complete | GHZ analytic references verified |
| 4 | Phase 1 | Complete | Fixed-seed hardware-efficient grid verified |
| 5 | Phase 1 | Complete | Controlled noise models and response plot generated |
| 6 | Phase 2 | Complete | Unmitigated estimator and resource ledger verified |
| 7 | Phase 2 | Complete | Readout mitigation and calibration evidence generated |
| 8 | Phase 2 | Not started | |
| 9 | Phase 2 | Not started | |
| 10 | Phase 3 | Not started | |
| 11 | Phase 3 | Not started | |
| 12 | Phase 3 | Not started | |
| 13 | Phase 3 | Not started | |
| 14 | Phase 3 | Not started | |
| 15 | Phase 4 | Not started | |
| 16 | Phase 4 | Not started | |
| 17 | Phase 4 | Not started | |
| 18 | Phase 4 | Not started | |
| 19 | Phase 5 | Not started | |
| 20 | Phase 5 | Not started | |
| 21 | Phase 5 | Not started | |
