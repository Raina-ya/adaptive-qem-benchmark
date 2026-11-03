# Extended Experiment Contract

## Scope

This contract governs the decision-reliability, ZNE-validity, combined-mitigation,
readout-scalability, and hardware-readiness extensions. It does not retroactively
change existing raw data or authorize hardware execution.

## Preregistration boundary

Experiment grids, exact-reference construction, tie policies, uncertainty methods,
practical-effect thresholds, cost penalties, validity gates, and holdout partitions
must be committed before the corresponding outcomes are inspected. Unfavorable
results may trigger implementation fixes but not post-hoc scientific threshold
changes.

## Row identity and status

Every applicable manifest row maps to one immutable repeat-level row. Identity
columns must include the circuit or candidate identifier, method, noise condition,
total shots, repeat, and seed. Status is explicit and belongs to the configured set:
`pending`, `complete`, `failed`, `unresolved`, `abstained`, or `not_applicable`.
Aggregation never deletes failed, unresolved, abstained, or negative-gain rows.

## Exact references and oracle isolation

Analytic or noiseless statevector references establish exact margins and evaluate
outcomes. They may construct a controlled benchmark but may not appear in selector
features at inference time. Candidate order, exact tie handling, and deterministic
tie breaking are fixed before sampling. Reference agreement is tested at absolute
tolerance `1e-12` where exact statevector comparison applies.

## Fixed total shot budget

Every comparison charges all target and auxiliary shots to the same declared budget.
The ledger includes readout calibration, every ZNE factor, combined calibration and
folding, training or method-selection records if later authorized, and repeated
sampling. Circuit executions, shots, classical runtime, and memory are distinct
resource columns; one cannot substitute silently for another.

## Statistical requirements

- Decision-error probabilities use two-sided 95% Wilson intervals.
- RMSE and decision-error results are reported separately.
- Paired methods use matched repeat identities where the design permits.
- Candidate covariance is measured for shared-record experiments.
- A decision is `unresolved` when its frozen gap interval crosses zero.
- A unique exact optimum with an estimated zero margin follows the configured error
  tie policy.
- Practical RMSE differences below `0.005` remain ties unless a later preregistered
  metric defines a separate practical threshold.

Intervals quantify finite sampling or finite repeats; they do not correct model
misspecification.

## ZNE evidence gate

Factor-level observations at every noise scale remain available. Apparent gain and
validated gain are separate outputs. No ZNE result may support a mitigation-success
claim until an observable-specific floor and signal-retention assessment exist and
the result passes the preregistered physical-bound, uncertainty, and matched-cost
negative-control checks. Invalid, unstable, or unphysical extrapolates remain in raw
data with an explicit status.

## Combined-method accounting

Combined readout mitigation and ZNE must fit calibration shots and all factor-level
target shots inside the same total budget used by the comparison methods. Intermediate
raw, corrected, and extrapolated estimates remain available. Worse and inconclusive
combined results are first-class outcomes.

## Holdouts and selectors

Rules and thresholds are frozen before holdout evaluation. A selector may use only
declared observable features, circuit/resource metadata, and diagnostics available
without the exact reference. It may return a method, `abstained`, `unresolved`, or a
request for additional information. Accuracy, decision compatibility, regret,
catastrophic errors, and abstention consequences are reported separately.

## Hardware boundary

This version is simulator-only and sets `hardware_execution_authorized` to false.
Hardware readiness requires frozen simulator predictions, a falsifiable question, a
minimal declared subset and budget, backend/job/calibration/transpilation metadata,
and a reason the run can distinguish competing explanations. A justified `no_go`
decision satisfies the review. No result is hardware-derived without an actual job
record.

## Reproducibility

Every generator uses root seed `20261102` and stable namespaced child seeds. Raw rows,
manifests, processed summaries, figures, and run logs are separate artifacts.
Documented generation commands, tests, lint, coverage audits, and artifact comparison
must pass before a result is presented as reproducible.

## Definition of Done

- Every primary question has a stated falsifier.
- Every primary metric has a mathematical or operational definition.
- Resource accounting includes all auxiliary executions and shots.
- Evidence labels and row statuses are machine checked.
- Oracle isolation and the hardware boundary are explicit.
- The contract configuration parses deterministically and its tests pass.
