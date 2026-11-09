# Adaptive QEM Benchmark

A reproducible simulator study of when quantum error mitigation helps, wastes a fixed
shot budget, or fails explicitly.

The benchmark compares raw estimation, full-assignment-matrix readout mitigation, and
linear zero-noise extrapolation (ZNE) for GHZ and fixed-seed hardware-efficient
circuits. It crosses widths 2, 4, and 6; depths 1, 3, and 6 where applicable; multiple
noise regimes; total budgets of 1,024, 4,096, and 16,384 shots; and 50 deterministic
repeats.

## Verified result

Across 144 GHZ and hardware-efficient operating cells, the cost-aware oracle found 77 raw
winners, 26 practical ties, 18 readout winners, 14 ZNE winners, and 9 inconclusive cells. An
earlier version labeled all 77 raw wins as ties because of an implementation error; those
labels are kept in `regime_labels_v1.csv`. Of the raw wins, 63 have references near zero,
where noise moves estimates toward the reference.

The frozen rule selector reached 41.4% decision compatibility on a deterministic 29-cell
holdout and 50.0% under four synthetic noise shifts, with mean regret 0.0210 and 0.0081. These
are diagnostic results, and always-readout had lower holdout regret (0.0077). A repository-precommitted
runtime selector with charged pilot shots then failed all three frozen criteria on 120 cells of
unseen circuits, a no-go result. Details are in
[the selector audit](docs/selector_audit_protocol.md).

Under synthetic readout drift over 100 decisions, triggered recalibration used about 40% fewer
shots than recalibrating at every decision, with similar error (0.067 versus 0.062). Never
recalibrating had an error of 0.234, and this depended strongly on drift direction.

The hardware-efficient run retained 99 explicit readout failures, concentrated in
width-6, low-shot, medium/high-noise cells. Negative mitigation gains are also kept:
179 of 432 method summaries had worse RMSE than the raw estimator.

A repository-precommitted combined-QEM study compares raw, readout, ZNE, and readout-plus-ZNE
under the same 4,096- and 16,384-shot budgets. Combined QEM had the lowest expectation
RMSE in 12 of 16 circuit/budget/allocation cells, but all 16 combined-ZNE validity
assessments were rejected by at least one frozen negative-control criterion. On the
0.02-margin decision task, combined mitigation produced error rates from 0.26 to 0.48
and did not consistently improve on readout mitigation. These simulated results show
estimation gains, not credible universal ZNE behavior or improved decisions.

The application case study asks whether those estimator changes improve a complete
fixed-budget decision: selecting one of five repository-precommitted $p=1$ QAOA candidates for
each of three exact MaxCut instances. Exhaustive classical search fixes each graph's
MaxCut optimum, while noiseless statevector simulation fixes candidate energies and
ranks before noisy sampling. Clear rankings were always selected correctly, but
near-tie error remained method-dependent. At 4,096 shots it was 0.28 for raw, 0.32
for readout among completed selections, and 0.36 for ZNE; at 16,384 shots it was
0.17, 0.179, and 0.42. On the completed 4,096-shot ring and sparse strata, readout
reduced energy RMSE but worsened top-1 selection relative to raw; its medium/dense
stratum failed entirely. Only 2 of 828 completed selections met the
repository-precommitted simultaneous-interval certification rule. Thus lower energy RMSE did
not reliably buy a better application decision.

A sequential certification study then asked when a two-candidate decision can stop
before the maximum budget. With Wilson stopping at 16,384 shots on the symmetric task,
raw estimation used 9,583 mean shots instead of 16,384, but left 40.7% of decisions
unresolved instead of 27.1%. Correcting for repeated looks kept false certification at
margin 0 below 3%, compared with 22-25% for uncorrected peeking. Near ties at margin
0.02 used 98-100% of the budget. Certification does not detect bias: when gate noise
differed between candidates (high noise, margin 0.05), raw certified the wrong winner
in every repeat after about 2,000 shots. Details are in
[the protocol](docs/sequential_decision_protocol.md).

A hardware readiness review of six candidate cells ended in a documented **no-go**, and no job was
submitted. Only one of seven go requirements is fully met and all six no-go triggers apply: the plan
would need about 72.1 million shots with no declared budget, no code captures device metadata, and a
near-tie cell is already inconclusive in simulation. See
[the review](docs/hardware_readiness_review.md),
[the protocol](docs/hardware_protocol.md), and the
[evidence matrix](results/processed/final_evidence_matrix_v1.csv).

![Operating-regime map with the corrected raw, tie, readout, ZNE, and inconclusive labels](paper/figures/operating_regime_map_v2.png)

Read the current paper: **[When Does Quantum Error Mitigation Pay Off Under a Fixed Shot Budget?](paper/main(2).pdf)**
(source `paper/main_v2.tex`). It covers every study, including the signal, confirmatory,
selector, certification, and readout-boundary results. The earlier paper,
[Adaptive Quantum Error Mitigation Under Fixed Shot Budgets](paper/main.pdf), is kept unchanged.

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
- Combined mitigation reserves 20% for assignment-matrix calibration, then allocates
  the remainder across folds using frozen balanced or base-heavy policies.
- Every stochastic seed derives from root seed `20261102`.
- Failed, raw, corrected, and factor-level observations remain explicit.

## Reproduce

Python 3.11 or 3.12 is required. The expanded nontrivial-circuit generation took about
43 minutes on the development desktop; runtime is machine-dependent.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = ".matplotlib-cache"

.\.venv\Scripts\python.exe -m coverage run -m pytest
.\.venv\Scripts\python.exe -m coverage report
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_artifacts
.\.venv\Scripts\python.exe -m src.run_with_log src.regenerate_readout_affected
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_paper_assets
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_runtime_report
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_audit_manifest
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_nontrivial_benchmark
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_adaptive_analysis
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_zne_probability_diagnostics
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_zne_falsification
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_zne_validity_gate
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_combined_qem
.\.venv\Scripts\python.exe -m src.run_with_log src.readout_scalability_metrics
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_readout_tensor_validation
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_readout_subspace_validation
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_readout_matrix_free_validation
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_readout_scalability_benchmark
.\.venv\Scripts\python.exe -m src.run_with_log src.sequential_decision
.\.venv\Scripts\python.exe -m src.run_with_log src.calibration_drift
.\.venv\Scripts\python.exe -m src.run_with_log src.regime_labels_v2
.\.venv\Scripts\python.exe -m src.run_with_log src.selector_audit
.\.venv\Scripts\python.exe -m src.run_with_log src.reliability_envelopes
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_selector_holdout_v2
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_audit_paper_assets
.\.venv\Scripts\python.exe -m src.run_with_log src.hardware_readiness
.\.venv\Scripts\python.exe -m src.run_with_log src.final_synthesis
.\.venv\Scripts\python.exe -m src.run_with_log src.generate_paper_v2_assets
cd paper; pdflatex -jobname="main(2)" main_v2.tex; bibtex "main(2)"; pdflatex -jobname="main(2)" main_v2.tex; pdflatex -jobname="main(2)" main_v2.tex; cd ..
.\.venv\Scripts\python.exe -m src.run_with_log src.verify_final_project
.\.venv\Scripts\python.exe -m src.run_with_log src.claim_ledger
```

Verify committed deterministic outputs:

```powershell
python -m src.verify_generated_artifacts --current
python -m src.verify_generated_artifacts --phase-f
python -m src.verify_generated_artifacts --include-expensive
python -m src.verify_final_project
python -m src.claim_ledger
```

[The QAOA mechanism analysis](docs/qaoa_mechanism.md) is an exploratory look at why lower energy RMSE can
come with worse candidate selection.

[Independent cross-validation](docs/cross_validation.md) recalculates exact values, one readout cell, and
QAOA selections with enumerated code that shares nothing with the benchmark implementation.

To rebuild expensive results from a clean clone, follow [the release checklist](docs/release_checklist.md).
`gh workflow run rebuild-generators.yml` rebuilds each long generator as its own job.
`python -m src.artifact_provenance` checks the hash, generator, and row count of every generated
artifact without running a simulation.

`src.verify_final_project` accounts for every manifest, recomputes the main summaries from raw rows,
checks figures and the PDF, and writes `results/processed/final_verification_v1.csv`.
`src.claim_ledger` recomputes the paper's printed values without importing project code. It writes
[the claim ledger](results/processed/claim_ledger_v1.csv), which separates computed results from
review judgments, manual interpretation, hypotheses, and limitations.

The second command includes the approximately 35-minute width-scalability benchmark.
Routine CI validates its committed schema and coverage through the test suite without
rerunning that expensive grid on every push.

`results/processed/day08_12_audit_manifest.csv` is generated from the repository
artifacts. It records deterministic paths, byte sizes, SHA-256 hashes, and parsed CSV
data-row counts; values are not maintained by hand.

## Generation run logs

Each wrapped generation command writes a compact JSON record under `results/runs/`.
The record contains timing, environment and Git metadata, status, and SHA-256 hashes
for files written under `results/` and `figures/`; it does not embed result data.

```powershell
ls results/runs/
```

## Limitations

This is a simulator-only benchmark. It does not model measured device drift, leakage,
relaxation, or calibrated hardware noise. Further detail is in
[the limitations](docs/limitations.md), including 960 planned correlated-readout repeats at widths
8-12 that were not run. The structured
noise-shift study adds asymmetric readout, coherent over-rotation, and correlated-ZZ
controls, but these remain synthetic models. Declared GHZ depths 3 and 6 remain
`not_applicable`; hardware-efficient circuits supply the nontrivial depth grid.
Full-matrix calibration scales exponentially with width. Aggregate metrics are
conditional on completed rows, while operating-cell labels require all methods to
complete all 50 repeats.

The scalable-readout track contains a frozen width-2--12 protocol, analytic resource
estimates, width-2/4 full-matrix feasibility references, and a 20-repeat tensor-product
validation at widths 2, 4, and 6. Under independent and asymmetric per-qubit readout
models, tensor correction had lower RMSE in 19 of 24 fixed-budget cells while using
`2*w` rather than `2**w` calibration circuits. This validates the implementation in
its assumed model; it is not evidence of robustness to correlated readout.

The observed-subspace study adds 50-repeat full-matrix comparisons at widths 2, 4,
and 6 and 10-repeat sparse pilots at widths 8 and 10. Support is selected only from
noisy target counts with a frozen minimum count of two. All 2,480 rows completed under
their declared shot budgets. Mean retained support fell to 22.6% at width 8 and 8.3%
at width 10 for sparse outputs, while expectation RMSE increased with width. The
result demonstrates a classical memory reduction, not a quantum-shot saving, and
does not establish robustness to broad or correlated-error cases.

The matrix-free validation applies tensor-factorized assignment operators without
forming a support-square matrix. All 1,280 planned rows converged with exact shot
accounting and zero materialized matrix entries; observed iterations rose from a
width-2 mean of 23.3 to a width-10 mean of 94.6. This validates deterministic solver
behavior for independent and asymmetric local readout models. It does not establish
an advantage under correlated errors, and the length-`2**w` working vector remains
exponential.

The unified width-2--12 scalability benchmark contains 5,760 repeat-level rows. Full
matrix correction completes 1,242 rows, fails 38 width-8 repeats as ill-conditioned,
and is resource-limited for 160 width-10/12 rows. Dense observed-subspace correction
is resource-limited for 140 rows whose count-derived support exceeds 128 and fails
two additional width-12 rows. Tensor and matrix-free correction complete all 1,440
rows each. At width 8, mean measured full-matrix solve time is 10.9 seconds versus
0.057 seconds for matrix-free correction. Every method amplifies sampling standard
deviation on average; lower classical storage therefore does not imply lower
statistical uncertainty. These conclusions remain limited to synthetic independent
and asymmetric local readout errors.

The correlated-mismatch stress test adds 9,600 paired rows at widths 2, 4, and 6.
It compares full, tensor-product, observed-subspace, and matrix-free correction under
matched independent, asymmetric, pair-correlated, and deliberately mismatched
calibration channels. All methods use the same 20/80 calibration/target budget split.
For the representative width-6 broad-output, 16,384-shot cell, calibration mismatch
raises expectation RMSE to 0.074--0.078 and the error probability of a documented
0.05-margin threshold decision to 0.88--0.94. This is simulated evidence that reduced
classical storage cannot compensate for a wrong calibration model; it is not a
hardware-robustness claim.

The MaxCut/QAOA case study is also simulator-only and deliberately small: one fixed
ring, sparse, and medium/dense graph at widths 4, 6, and 8, five curated candidates
per graph, $p=1$, two budgets, and synthetic noise. Of 900 selection groups, 72
medium/dense readout groups failed assignment conditioning and remain explicit.
Only two groups certified a winner; all other completed groups remained unresolved
at the maximum budget. The exact solver certifies sampled cuts only because these
instances are classically tractable. The results do not establish behavior for
larger graphs, optimized QAOA parameters, deeper circuits, device-calibrated noise,
or quantum hardware.

## Repository structure

```text
src/                         Benchmark and mitigation implementations
tests/                       Analytic, statistical, and reproducibility invariants
experiments/                 Frozen configurations, manifests, and proposed hardware cells
results/raw/                 Immutable repeat-level benchmark rows
results/processed/           Coverage and runtime summaries
results/runs/                Per-run generation metadata and output hashes
figures/                     Generated project figures
paper/main_v2.tex            Current paper source
paper/main(2).pdf            Current compiled paper
paper/main.tex, main.pdf     Earlier paper, kept unchanged
paper/generated/             Tables and macros derived from authoritative CSV data
paper/figures/               Paper figures generated from repository outputs
```

## License and citation

Source code and tests are under the [MIT License](LICENSE). The paper, documentation, figures,
configurations, manifests, and results are under
[CC BY 4.0](LICENSE-CC-BY-4.0.md). Citation metadata is in [CITATION.cff](CITATION.cff).
