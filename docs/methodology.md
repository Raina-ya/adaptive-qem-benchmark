# Methodology

## Scope

This simulator-only benchmark compares raw estimation, full assignment-matrix readout
mitigation, and preregistered linear ZNE under the same total shot budget. Evidence is
analytic or simulated; no result is hardware-derived.

## Circuit families and references

The structured family is GHZ with the analytic `X^w` reference. The nontrivial family
uses fixed-seed hardware-efficient circuits across widths 2, 4, and 6 and depths 1, 3,
and 6. Each hardware-efficient circuit is linked to a circuit ID and an exact noiseless
statevector reference. Conclusions are stratified by circuit family and observable.

## Repeats, uncertainty, and cost

Every applicable operating cell uses 50 deterministic repeat seeds. Bias, variance,
RMSE, execution overhead, and folded two-qubit gate counts are derived from immutable
repeat rows. Mitigation gains use matched repeat IDs and 1,000 paired BCa bootstrap
resamples. Decision-error probabilities use Wilson intervals. Negative gains remain
visible.

The cost-aware oracle minimizes RMSE plus a preregistered normalized execution penalty.
Cells may be practical ties or inconclusive. The selector may abstain and cannot inspect
exact-reference error at inference time.

## Holdouts

At least 20% of in-distribution operating conditions are reserved by deterministic hash
ranking before rule evaluation. A separate out-of-distribution holdout introduces an
intermediate mixed channel, asymmetric readout, coherent over-rotation, and correlated
`ZZ` noise. Selector rules remain unchanged after either holdout is evaluated.

## Limitations

The channels are controlled simulator models, not device calibrations. The benchmark
does not establish hardware performance, temporal drift robustness, leakage behavior,
or universality across algorithms and observables. Full readout calibration scales
exponentially with width. Statistical intervals quantify finite-repeat uncertainty but
do not remove model misspecification.
