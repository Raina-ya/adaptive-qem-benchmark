# Limitations and Non-goals

## Current evidence boundary

This project uses analytic calculations, exact statevector simulation, and Qiskit Aer noisy simulation. It contains no hardware-derived evidence. The hardware readiness review ended in a no-go and submitted no job; see [the review](hardware_readiness_review.md).

## Explicit non-goals for this batch

- No machine-learning selector.
- No IBM Quantum or other real-device execution.
- No claims about a particular device, calibration cycle, or physical error rate.
- No probabilistic error cancellation, dynamical decoupling, measured device drift,
  leakage, or non-Markovian noise.
- Calibration drift is a synthetic stress model, not a measured device lifetime.

## Model limitations

The main volumetric grid assumes independent per-qubit assignment errors and idealized
depolarizing channels attached to declared simulator instructions. A separate
noise-shift holdout includes asymmetric readout, coherent over-rotation, and
correlated-ZZ controls. These are diagnostic stress tests, not calibrated device
models or evidence of hardware crosstalk.

Full assignment-matrix mitigation scales exponentially with qubit count and is included only for small benchmark circuits. An ill-conditioned calibration matrix is reported explicitly instead of being silently inverted.

Linear ZNE is limited to global folding factors 1, 3, and 5 with a preregistered
ordinary-least-squares extrapolation. The fixed total shot budget is split across
factors. No alternative noise scaling, fit order, or adaptive allocation is evaluated.

The combined-QEM study applies assignment-matrix correction separately at each noise
factor before linear extrapolation. It evaluates only two frozen allocations and six
preregistered simulator cells. Every combined validity assessment failed at least one
frozen negative-control criterion, so lower RMSE in selected cells is not treated as
evidence that the extrapolation mechanism is credible or broadly transferable.

The rule selector is a frozen, explainable baseline. Its modest holdout compatibility
and nonzero regret do not support deployment or universal method-selection claims.
A preregistered runtime selector also failed all three frozen criteria on unseen circuits.

## Later tracks

Sequential certification covers two-qubit circuits and one observable, and it cannot detect
estimator bias. Calibration reuse is a one-qubit threshold task. The QAOA study has three graphs,
five frozen candidates per graph, and $p=1$; its candidate pool is a frozen input because exact
energy ties are ordered by platform-dependent floating-point noise.

## Known deviation and verification scope

The scalable-readout manifest planned 960 repeats of pair-correlated and calibration-mismatch
readout at widths 8-12, and these were never run. Those models ran only at widths 2-6, in the
separate mismatch study. `src/verify_final_project.py` reports this explicitly.

The same script directly recomputes the main summaries from raw rows. The remaining derived tables
are checked by regenerating them. Routine CI skips the long-running benchmarks listed in
`EXPENSIVE_GENERATORS`, so those need `--include-expensive`. The combined-QEM and decision
benchmarks are in neither generator list, so their raw rows are checked only through summaries and
tests.

Hardware predictions depend on simulator noise and are hypotheses. Their required-repeat floors
come from simulated effect sizes and are likely optimistic for a device.
