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

## Later tracks

- **Combined mitigation:** readout correction at each noise factor, then linear extrapolation,
  charged in one ledger under two frozen allocations.
- **ZNE validity:** a frozen gate on signal retention, physical bounds, negative mass, gain
  uncertainty, trend, and a matched-cost garbage-folding control.
- **Decision reliability:** controlled margins scored by decision error, with Wilson intervals.
- **Readout scalability:** tensor, observed-subspace, and matrix-free correction, including
  correlated and mismatched channels.
- **Application:** top-1 selection among five frozen QAOA candidates, with exact values used only
  for evaluation.
- **Sequential certification:** Bonferroni-corrected looks with Wilson, Hoeffding, and empirical
  Bernstein bounds. See [the protocol](sequential_decision_protocol.md).
- **Calibration reuse:** recalibration policies under synthetic drift.
- **Selector audit:** consequences of selector choices, then a preregistered runtime selector on a
  fresh grid. See [the protocol](selector_audit_protocol.md).

## Hardware readiness

Six candidate cells have frozen simulator predictions, uncertainty, a hypothesis, and a falsifier.
Required shots use a two-sided normal approximation at alpha 0.05 and 80% power. Seven go
requirements and six no-go triggers decide the outcome. The result is a no-go, and no job was
submitted. See [the protocol](hardware_protocol.md) and [the review](hardware_readiness_review.md).

## Reported distinctions

- Expectation RMSE is reported separately from decision error.
- Apparent ZNE gain is reported separately from gain that passes the validity gate.
- Single-method and combined-method overhead use the same ledger.
- Measured results are reported separately from hypotheses.
- Simulator evidence is labeled separately from hardware evidence, of which there is none.

## Verification

`src/verify_final_project.py` accounts for every manifest, recomputes the main summaries from raw
rows at tolerance 1e-9, and checks figures and the PDF. `src/verify_generated_artifacts.py` compares
regenerated artifacts with committed files at relative tolerance 1e-9 and absolute tolerance 1e-12.

## Limitations

The channels are controlled simulator models, not device calibrations. The benchmark
does not establish hardware performance, physical drift robustness, leakage behavior,
or universality across algorithms and observables. Full readout calibration scales
exponentially with width. Statistical intervals quantify finite-repeat uncertainty but
do not remove model misspecification.
