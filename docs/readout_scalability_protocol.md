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
