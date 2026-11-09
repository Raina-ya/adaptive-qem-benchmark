# Readout, Mismatch, and Drift Boundaries Protocol

**Status: frozen before the full runs.** The settings are in `experiments/configs/readout_boundaries_v1.yaml`,
and its hash is in `readout_boundaries_v1.sha256`. Every generator refuses to run if the config
changed. Everything here is a simulator sensitivity analysis on synthetic readout channels, labeled
`noisy_simulated`. None of it is hardware evidence.

## Disclosure: one design change after a smoke run

A 20-repeat smoke run on one mismatch cell showed that pair-correlated mismatch left every method
unchanged. The cause is structural: pair flips change two bits at once, so an all-qubit parity is
unchanged when the pairs tile every qubit. The observable was then changed, before freezing, to the
parity of qubits 0 to w-2. All configured widths are even, so the last pair now straddles the
observable and pair flips change it. No hypothesis, threshold, or grid value was changed after that
run, and the smoke rows were not kept. Timing runs of 2 to 30 repeats on a few cells (two feasibility
cells, one mismatch cell, one off-grid drift scenario) were also discarded.

## Shared setup

- **Observable:** Z parity of qubits 0 to w-2, the worst case for readout among low-cost observables.
- **Candidates:** a decision between two states with parity 0.30 and 0.20 (margin 0.10), or 0.25 and
  0.20 (margin 0.05). Ideal values are analytic.
- **Distributions:** `product` (identical RY rotations, broad support) and `sparse` (two outcomes).
- **Failure:** a failed correction counts as a wrong decision, and is reported with its reason.

## Width feasibility

Widths 4, 6, 8, 10, with 200 repeats. Readout models are independent, asymmetric, and pair-correlated.

| Method | Calibration circuits | Notes |
| --- | --- | --- |
| none | 0 | Raw parity |
| full_matrix | 2^w | Fails on no shots, a singular estimate, or condition above 1e8 |
| tensor_product | 2 | All zeros and all ones |
| pairwise_local | 4 | 4x4 blocks on pairs, estimated together |
| observed_subspace | 2 | Tensor model on the observed outcomes, renormalized |
| matrix_free | 2 | The same system by GMRES, rows built in chunks |

Two regimes are reported separately:

- **Fixed total budget:** 4,096 or 16,384 shots, 20% of it calibration. At width 8, each full-matrix
  state gets 3 shots at 4,096 and 12 at 16,384.
- **Fixed calibration quality:** 256 or 1,024 shots per calibration circuit, and 16,384 target shots.
  The report gives calibration share, so full matrix's cost is visible.

Reported: failure probability and reasons, bias, SD, RMSE, decision error, median condition number,
zero-diagonal fraction, variance amplification against raw, calibration share, support, iterations.
Width-10 full-matrix condition numbers use the first 20 repeats only.

**The 960 v1 repeats stay unexecuted.** This is a separate protocol version. Its rows never enter a
v1 table, and v1's failed full-matrix rows stay as committed.

## Mismatch dose-response

Calibrate once per repeat on `A_cal`, then measure under `A_eval(lambda) = (1 - lambda) A_cal +
lambda A_mismatch`, at 21 values of lambda from 0 to 1. Widths 4, 6, 8, with 150 repeats; full
matrix only up to width 6.

| Scenario | `A_cal` (p01, p10, pair) | `A_mismatch` |
| --- | --- | --- |
| asymmetric_drift | 0.02, 0.02, 0 | 0.01, 0.08, 0 |
| pair_correlated | 0.02, 0.02, 0 | 0.02, 0.02, 0.05 |
| calibration_age | 0.02, 0.03, 0 | 0.035, 0.06, 0 |
| abrupt_change | 0.02, 0.02, 0 | 0.06, 0.12, 0 |
| independent_to_correlated | 0.03, 0.03, 0 | 0, 0, 0.03 |

For `abrupt_change`, lambda is the fraction of shots after the change. For
`independent_to_correlated`, both ends flip each qubit with probability 0.03, so marginal
calibration cannot tell them apart.

**Rules:**

- **Loses to raw:** the first lambda where mitigated RMSE exceeds raw RMSE, and separately the first
  where a paired-repeat bootstrap interval for the difference lies above zero.
- **Steep rise:** the first lambda where decision error is at least its lambda-zero value plus 0.10.
- **Warnings:** condition number above twice its lambda-zero median; full-matrix negative mass
  above 0.05; a check circuit (1,024 shots each on all zeros and all ones, charged as diagnostic) with
  z above 3 in at least half the runs; and interval coverage under 0.90. Coverage needs the ideal value,
  so it is reported but is not a runtime warning.

## Drift policy

One qubit, one threshold decision per step: is the Z expectation above 0.2? The true value sits one
margin above or below. Grid: drift amplitude 0.01, 0.03, 0.06 on p10 (a third of it on p01);
timescale 25, 100; shape ramp, abrupt, sinusoid; run length 100, 300; margin 0.02, 0.05; calibration
cost 1,024, 4,096. 100 repeats.

Policies: never; every decision; fixed interval 10 or 50; triggered at z of 2, 3, or 4. The trigger reads
128 check shots on each basis state every step, and they are charged.

Reported: total and per-decision shots, decision error, stale bias, recalibrations, false
recalibrations (true rates still within 0.01 of the calibration), detection delay, missed drift
(violation for 20 steps with no recalibration), and violation fraction.

## Hypotheses fixed before the run

- **H8:** at fixed total budget and width 8, full-matrix correction has higher candidate RMSE than
  tensor-product for every model, with failures counted as not better. At fixed quality, it completes
  at width 8 with calibration shots above the target budget.
- **H9:** condition number cannot warn of mismatch, because it comes from the stale calibration. The
  check circuit can.
- **H10:** tensor-product correction loses to raw at a smaller lambda under pair-correlated mismatch
  than under asymmetric drift.
- **H11:** a triggered policy matches every-decision recalibration in decision error with fewer shots
  under slow drift, and loses that advantage for abrupt drift of amplitude 0.01, which sits near the
  check's noise.

## Limits

Channels are synthetic and time-independent within a decision. One observable family. Tensor-based
methods use only two calibration circuits. Drift uses one qubit, so it says nothing about correlated
drift across qubits.
