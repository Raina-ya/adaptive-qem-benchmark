# Methodology

## Scope

This simulator-only benchmark compares raw estimation, full assignment-matrix readout
mitigation, and repository-precommitted linear ZNE under the same total shot budget. Evidence is
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

The cost-aware oracle minimizes RMSE plus a documented normalized execution penalty.
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
- **Selector audit:** consequences of selector choices, then a repository-precommitted runtime selector on a
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

## Claim types

- `computed`: obtained deterministically from identified data and a documented calculation.
- `manual_interpretation`: a human-written interpretation of computed evidence.
- `hypothesis`: a possible explanation not established by the current evidence.
- `limitation`: a documented boundary of the evidence.
- `review_judgment`: a go/no-go, readiness, or scientific-review decision requiring human judgment.

A manual interpretation explains results, whereas a review judgment makes a decision from evidence
and stated criteria.

## Evidence origin

- `raw_recomputed`: the ledger calculates the metric from repeat rows. Any stored upstream field it
  consumes is listed, and the claimed metric does not merely reuse a derived column.
- `processed_recomputed`: the calculation reuses a processed artifact or a derived column it did not
  reconstruct, even if that column is stored under `results/raw/`.
- `manifest_derived`: the value depends on a manifest. This is valid for design, coverage, seed, and
  planned-grid claims, but not for empirical performance.
- `processed_only`: every input is a processed artifact.
- `source_only_not_recomputed`: the source is located, but the value is not recomputed.

A file under `results/raw/` is not necessarily a primitive observation. Fields such as `true_gain`,
`decision_error`, `wrong_certified`, and `estimate` are derived. The ledger lists each one and states
whether it was reconstructed, in `results/processed/derived_field_reconstruction_v1.csv`. Historical
schemas stay as they are; future generators should separate primitive observations from derived
metrics.

## Terms for commitments

- **Preregistered in the repository (repository-precommitted):** the protocol's commit precedes the
  result artifacts. This covers the combined, QAOA, scalable-readout, runtime-selector, and volumetric
  protocols and the decision margins. The 0.05 mismatch threshold first appears with its results.
- **ZNE validity gate:** The validity checks were named in advance, but the numerical thresholds and the reported results first appeared together in the same repository state; therefore this is not an independently timestamped preregistration.
- **Frozen repository rule:** a rule stored with its results and not changed afterward, without evidence
  that it came first. This covers sequential analysis, calibration policy, and the first selector.
- **Documented evaluation rule:** a rule written down without a claim about timing, such as the 0.005
  practical threshold and cost penalty.

"Frozen" alone means fixed in the repository, not timestamped before results. Commit order proves
repository publication order only. It does not prove the wall-clock order of design or execution.

## Verification

`src/verify_final_project.py` accounts for every manifest, recomputes the main summaries from raw
rows at tolerance 1e-9, and checks figures and the PDF. `src/verify_generated_artifacts.py` compares
regenerated artifacts with committed files at relative tolerance 1e-9 and absolute tolerance 1e-12.
`src/claim_ledger.py` recomputes the printed values of computed claims without importing project
code and labels each claim by type, source, scope, and verification tier. Definitions are in
`experiments/configs/claim_ledger_v1.yaml`; results are in `results/processed/claim_ledger_v1.csv`,
and `paper_number_provenance_v1.csv` traces every macro, generated table, and figure.

Replacing the original BCa interval with a deterministic percentile-bootstrap sensitivity calculation
did not change the 144 cell labels. The BCa result remains authoritative, no threshold or claim changed
after this comparison, and it does not show independence from all statistical choices. The interval
choice does change the raw-win basis (accuracy or cost only) in the two cells listed there, although
the 40 and 37 totals stay the same. The comparison is in `results/processed/label_bootstrap_sensitivity_v1.csv`.

## Limitations

The channels are controlled simulator models, not device calibrations. The benchmark
does not establish hardware performance, physical drift robustness, leakage behavior,
or universality across algorithms and observables. Full readout calibration scales
exponentially with width. Statistical intervals quantify finite-repeat uncertainty but
do not remove model misspecification.
