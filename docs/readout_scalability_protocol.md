# Readout Scalability Protocol v1

## Objective

Determine when full assignment-matrix correction becomes computationally or
statistically impractical, without treating classical memory savings as quantum-shot
savings. All evidence in this track is analytic or simulator-derived; no hardware
claim is authorized.

## Frozen grid

- Widths: 2, 4, 6, 8, 10, 12.
- Output distributions: sparse and broad.
- Readout models: independent, asymmetric, pair-correlated, and calibration mismatch.
- Methods: full matrix, tensor product, observed subspace, and matrix-free subspace.
- Total shot budgets: 4,096 and 16,384.
- Repeats: 50 for widths 2--6; 10-repeat pilots for widths 8--12.
- Widths 2 and 4 are exact-reference cells. Width 6 is a small-width validation cell.

The manifest contains one row per cell rather than one row per repeat. Repeat rows are
created only when the corresponding execution phase begins.

The E-A feasibility reference applies the existing deterministic full-matrix solver
to analytic independent-readout probability vectors at widths 2 and 4 for both
distribution families. It contains no finite-shot result and is labeled `analytic`.

## Analytic resource model

For width `w`, the assignment dimension is `2**w`, a dense assignment matrix has
`4**w` float64 entries, and its stored matrix alone requires `8 * 4**w` bytes. Full
calibration uses `2**w` prepared basis states. The tensor-product protocol uses two
single-qubit preparation states per qubit, or `2*w` calibration circuits. These
counts do not include solver workspaces, Python object overhead, transpilation, or
stored repeat data; measured peak memory will therefore be reported separately.

## Pre-execution gates

Full-matrix execution is marked `resource_limited` before outcomes are observed if
any of these frozen limits is exceeded:

- dense matrix storage greater than 512 MiB;
- more than 1,024 calibration circuits;
- assignment dimension greater than 256 for a dense solve.

Consequently, full-matrix width 10 and 12 cells are resource-limited in v1. This is a
protocol limit, not a claim that allocation is mathematically impossible. Reduced
methods remain planned and receive their own convergence and support gates later.

## Measurement contract

Future repeat-level rows must keep these quantities distinct:

1. Classical resources: matrix dimension, entries, estimated and measured memory,
   solve time, iteration count, convergence, and condition estimate.
2. Quantum sampling resources: calibration circuits, calibration shots, target
   circuits, and target shots.
3. Statistical behavior: bias, variance, RMSE, uncertainty amplification, and
   decision-error probability.
4. Approximation behavior: retained support, discarded probability mass, and error
   relative to the exact small-width correction.

Every planned cell must finish as `complete`, `failed`, `not_applicable`, or
`resource_limited`. Failures and non-convergence are data and may not be dropped.

## Boundaries for later phases

- Tensor correction must first agree with full correction for independent readout at
  widths 2 and 4.
- Observed support may not be chosen using the exact expectation or decision label.
- Low-probability, decision-critical states require a dedicated regression test.
- A matrix-free implementation must not materialize the matrix internally.
- Sparse and broad distributions are summarized separately.
- Correlated-error mismatch must remain an explicit experimental factor.

## Observed-subspace implementation

The observed-subspace validation selects computational-basis states from noisy
target counts using the frozen rule `count >= 2`. The selector never receives the
exact expectation or reference distribution. A preregistered protected-state input
exists for decision-critical states and is covered by a regression test, but the v1
validation does not add protected states. One calibration circuit is allocated per
retained state within the same 20% calibration fraction used by full correction.

The reduced assignment matrix contains only retained observed rows and prepared
columns. The solver normalizes its non-negative solution, so retained and discarded
observed probability mass are reported separately to expose truncation. Reduced
matrix bytes are a classical storage metric; calibration shots remain fixed at 20%
of the total budget and are not reported as saved shots.

## Matrix-free implementation

The matrix-free method uses the same count-derived support rule and fixed 20% local
calibration budget. A tensor-factorized operator embeds a support vector in a
length-`2**w` workspace, applies one sampled 2-by-2 assignment map per qubit, and
projects back to the support. Its transpose is applied in reverse through transposed
local maps. No support-square or full assignment matrix is constructed.

Non-negative least squares is solved by deterministic projected FISTA with a
predeclared limit of 1,000 iterations and relative update tolerance `1e-10`.
Iteration count, convergence, residual norm, operator storage, and working-vector
storage are recorded separately. A converged update does not imply a small residual:
support truncation can leave probability outside the modeled rows, so residual norm
remains a reported diagnostic rather than a hidden acceptance adjustment.

## Unified scalability execution

The unified benchmark executes all four methods with paired target seeds. Widths
2--6 use 50 repeats; widths 8--12 use 10. Full matrix follows the original resource
gates. A preflight found that dense observed-subspace solves near dimension 256 were
both slow and sometimes ill-conditioned, so its method-specific gate is frozen at a
count-derived support of 128 before the full run. Rows beyond that gate are retained
as `resource_limited`; their target sampling is preserved, but calibration and solve
work are not claimed as executed.

Classical solve times are measured by replaying only the correction from preserved
counts and calibration data. They are explicitly labeled `measured` and are not
expected to be byte- or value-identical across hosts. Reproducibility checks require
the same missing-value pattern and finite non-negative timings while continuing to
compare all scientific CSV columns numerically. Uncertainty amplification is the
repeat standard deviation of corrected errors divided by the paired raw-error
standard deviation; it is distinct from bias and RMSE.

## Correlated readout and calibration mismatch

The mismatch benchmark uses widths 2, 4, and 6, sparse and broad output families,
budgets 4,096 and 16,384, and 50 paired repeats. Target counts are shared by all four
correction methods within a paired cell. Calibration always assumes independent
single-qubit assignment error. Targets are matched independent, asymmetric,
pair-correlated (disjoint simultaneous bit flips), or deliberately mismatched
asymmetric channels. The single-qubit-sensitive observable avoids the parity blind
spot of global-Z and neighboring-ZZ observables under simultaneous pair flips.

Approximation bias is reported separately from repeat-level sampling standard
deviation. Downstream reliability uses a threshold fixed exactly 0.05 from the
analytic expectation; its exact decision is therefore known before sampling. Wilson
intervals accompany the observed decision-error probability. These synthetic models
test estimator failure under misspecification; they do not approximate a calibrated
device or establish robustness to drift, leakage, or general crosstalk.
