# Sequential Decision Certification Protocol

## Purpose

This protocol asks whether a two-candidate decision can stop before the maximum shot
budget without losing error control. Evidence is `noisy_simulated`: shots are sampled
from exact channel probabilities with deterministic seeds. It makes no hardware claim.
All frozen values live in `experiments/configs/sequential_decision_v1.yaml`.

## Decision task

Candidates A and B have exact expectations `base + margin/2` and `base - margin/2` for
`Z` on qubit 0. Each circuit is `RY(acos(expectation))` followed by `depth` CX gates.
The task is to decide which candidate is larger.

| Task | Margins | Base | Depth A/B | Readout | Calibration |
|---|---|---:|---:|---|---|
| `symmetric_gap` | 0, 0.02, 0.05, 0.10, 0.20 | 0.0 | 3/3 | equal | shared |
| `gate_imbalanced` | 0.02, 0.05, 0.10, 0.20 | 0.6 | 6/1 | equal | shared |
| `readout_imbalanced` | 0.02, 0.05, 0.10, 0.20 | 0.6 | 3/3 | A is 4x B | per candidate |

`symmetric_gap` with margin 0 has no correct winner, so any certification is false.
This cell measures optional-stopping validity directly. The two imbalanced tasks let
systematic bias reverse the noisy ordering, which sampling intervals cannot detect.

## Resource accounting

Every row spends at most the declared maximum total budget (1,024, 4,096, or 16,384).

- `raw` samples A and B equally in batches of 256 target shots.
- `readout` first spends 20% of the maximum budget on one-qubit assignment calibration.
  This cost is charged whether or not the decision stops early. Calibration is shared
  when candidates use the same qubit and separate when they do not.
- `zne` splits each candidate's batch across folds 1, 3, and 5 and uses the frozen OLS
  intercept weights.
- A final partial batch uses any remainder, so `fixed_budget` always spends the full
  budget.

Rows also record circuit executions and two-qubit-gate shots, which charge ZNE for its
longer folded circuits.

Sampling probabilities are rounded to 12 decimals immediately before each binomial draw. Without this, a probability
of exactly 0.5 can land just above or below 0.5 on different platforms and take a
different NumPy binomial branch, which changes individual rows. This is an implementation
fix made after a cross-platform comparison, not a change to any scientific value.

## Stopping policies

Each policy computes an interval for the estimated gap after every look and certifies
when the interval excludes zero. If the budget ends first, the row is `unresolved`.
Every sequential policy tests at `alpha / K`, where `alpha = 0.05` and `K` is the
number of available looks. This union bound is what makes repeated inspection valid.

| Policy | Interval | Looks | Basis |
|---|---|---|---|
| `fixed_budget` | Wilson components combined by MOVER | final only | approximate |
| `wilson_stopping` | same, at `alpha / K` | every batch | approximate |
| `gap_confidence_stopping` | Hoeffding bound on the gap | every batch | finite-sample for linear estimators |
| `variance_aware_stopping` | empirical-Bernstein per component, summed | every batch | finite-sample for linear estimators, conservative |
| `naive_wald_peeking` | Wald at 95% with no look correction | every batch | negative control, never recommended |

Readout estimators are linearized by the delta method, so their bounds are approximate
for every policy. Calibration uncertainty is included and does not shrink as target
shots accumulate.

## Metrics

- `total_shots_used`: target plus calibration shots at the stopping point.
- `unresolved_rate`: fraction of repeats that never certify.
- `wrong_certified_rate`: fraction of repeats that certify the wrong candidate, or
  certify anything at margin 0.
- `wrong_early_stop_rate`: wrong certifications before the final look.
- `forced_decision_error`: error if an unresolved repeat must still choose by the sign
  of its last estimate. An exact zero estimate counts as an error.
- `abstain_regret` and `forced_regret`: mean exact regret when unresolved repeats make
  no decision, or are forced to choose.
- `abstention_break_even_penalty`: `(forced_regret - abstain_regret) / unresolved_rate`.
  Abstaining has lower total loss whenever the cost of one abstention is below this value.

## Threshold policy

No threshold is tuned. Alpha, batch size, grids, and bounds were written before any
outcome was generated, so there is nothing to select and no holdout split. Implementation
fixes may follow outcomes; scientific values may not.

## Limits

Intervals quantify sampling only. They do not correct model misspecification, and a
biased estimator can certify the wrong candidate at any confidence level. The study
uses two-qubit circuits, one observable, synthetic noise, and no drift.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log src.sequential_decision
.\.venv\Scripts\python.exe -m pytest tests/test_sequential_decision.py -q
```

## Observed results

These results are `noisy_simulated`, with 100 repeats per cell and 198,000 policy rows.
The raw data has 138,687 unresolved rows, 59,313 certified rows, and no failed rows.

### Cost on the symmetric task

This table covers margins above 0 and nonzero noise at a 16,384-shot budget. Shots
include calibration.

| Method | Policy | Mean shots | Median shots | Unresolved | Wrong early stop | Forced error | Two-qubit-gate shots |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | fixed budget | 16,384 | 16,384 | 27.1% | 0.00% | 3.25% | 49,152 |
| raw | Wilson | 9,583 | 9,984 | 40.7% | 0.00% | 3.25% | 28,749 |
| raw | gap confidence | 10,653 | 14,720 | 48.2% | 0.00% | 3.25% | 31,959 |
| raw | variance aware | 14,136 | 16,384 | 70.8% | 0.00% | 3.25% | 42,407 |
| raw | naive peeking (control) | 5,067 | 1,792 | 16.2% | 3.25% | 5.58% | 15,202 |
| readout | Wilson | 11,429 | 13,004 | 43.7% | 0.08% | 4.67% | 24,459 |
| zne | Wilson | 13,045 | 16,384 | 63.6% | 0.08% | 9.75% | 116,791 |

Early stopping saved shots, but it also left more decisions unresolved than one test
at the full budget, because the corrected test is stricter at every look.

### Review questions

**Is optional stopping handled correctly?** Yes for sampling error. Each sequential
policy divides alpha across its available looks. `naive_wald_peeking` is retained as
the control that shows what happens without this correction.

**Are intervals still valid under repeated inspection?** Margin 0 tests this directly,
pooling 1,200 repeats per method and policy:

| Policy | raw | readout | zne |
|---|---:|---:|---:|
| fixed budget | 5.75% | 6.08% | 4.67% |
| Wilson | 2.67% | 1.67% | 2.08% |
| gap confidence | 0.42% | 0.17% | 0.25% |
| variance aware | 0.00% | 0.00% | 0.00% |
| naive peeking (control) | 24.9% | 22.4% | 22.9% |

The corrected sequential rates stayed below 5%, including their 95% Wilson upper
bounds. The fixed-budget intervals include 5%. Naive peeking reached 36.0% for raw at
16,384 shots. Margin 0 exists only with shared calibration, so the per-candidate
readout interval has no direct null test.

**Does mitigation delay or accelerate certification?** It delays certification when
the estimator already has the right sign. On the symmetric task with Wilson stopping,
mean shots were 9,583 for raw, 11,429 for readout, and 13,045 for ZNE. At margin 0.20
and medium noise they were 1,690, 4,804, and 5,266. With shared calibration, readout
correction cannot change the sign of the gap, so it only removes target shots.

Mitigation helps when it removes the bias that hides or reverses the answer. In
`readout_imbalanced` at medium noise and margin 0.10, raw stayed 99% unresolved after
16,320 mean shots, while readout was 26% unresolved after 11,065.

**Do near ties consume nearly the entire budget?** Yes. At margin 0.02, 16,384 shots,
and nonzero noise, Wilson stopping used 98.1% to 99.8% of the budget and left 96.0% to
99.7% unresolved.

**Can abstention outperform forced decisions?** It depends on what abstaining costs.
Pooled over margins above 0 with Wilson stopping:

| Task | Forced regret | Abstain regret | Unresolved | Break-even penalty |
|---|---:|---:|---:|---:|
| symmetric | 0.00563 | 0.00009 | 65.3% | 0.0085 |
| gate imbalanced | 0.0359 | 0.0105 | 63.2% | 0.0403 |
| readout imbalanced | 0.0275 | 0.0066 | 69.6% | 0.0300 |

On the symmetric task, forcing unresolved repeats to choose was wrong 31.3% of the time
at margin 0.02, so abstaining is a real reliability gain. Under imbalance, abstaining
only avoids decisions. Certified decisions were as wrong as forced ones: 54.6% versus
53.5% with gate imbalance, and 40.3% versus 41.6% with readout imbalance.

**Is the stopping threshold selected without holdout leakage?** No threshold is
selected. Alpha, batch size, grids, and bounds were written to the configuration before
the full grid was generated, and none changed afterward. Only figure styling changed
after results were seen, and the raw file hash stayed the same. Git history alone cannot
prove this order.

### Main limit: certification does not detect bias

Stopping early makes a biased estimator wrong faster. With Wilson stopping at 16,384
shots:

| Task | Noise | Margin | Method | Wrong certified | Unresolved | Mean shots |
|---|---|---:|---|---:|---:|---:|
| gate imbalanced | high | 0.05 | raw | 100% | 0% | 2,022 |
| gate imbalanced | high | 0.05 | readout | 100% | 0% | 5,201 |
| gate imbalanced | high | 0.05 | zne | 77% | 23% | 10,028 |
| gate imbalanced | medium | 0.05 | raw | 28% | 72% | 14,602 |
| gate imbalanced | medium | 0.05 | zne | 0% | 100% | 16,384 |
| readout imbalanced | high | 0.05 | raw | 100% | 0% | 2,691 |
| readout imbalanced | high | 0.05 | readout | 0% | 98% | 16,258 |

A certificate therefore means the estimator's own limit is probably on one side of
zero. It does not mean the exact winner has been found.
