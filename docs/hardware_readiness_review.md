# Hardware Readiness Review

**Decision: no-go. No job was submitted.**

The decision follows `results/processed/hardware_readiness_criteria_v1.csv`, which
`src/final_synthesis.py` generates. Only one of seven go requirements is fully met, and all six
no-go triggers apply. Any one trigger is enough.

## Go requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Stable local tests and CI | Met | Local tests and ruff pass, and CI has been green since the QAOA ranking fix |
| Frozen simulator predictions | Partial | Frozen for named simulator noise, not for a backend |
| Falsifiable hardware hypothesis | Partial | Five cells qualify; calibration reuse depends on an unknown drift direction |
| Declared execution budget | Not met | No budget or access is declared; the plan needs about 72.1 million shots |
| Complete metadata capture | Not met | Fields are specified, but no provider dependency or capture code exists |
| No unresolved ZNE issue in the positive control | Not met | Credible at 16,384 shots, rejected at 4,096 on the physical-bounds check |
| Hardware can discriminate between explanations | Not met | Device noise cannot be set to the predicted conditions, and one backend confounds method, device, and drift |

## No-go triggers

| Trigger | Evidence |
| --- | --- |
| Hardware would only decorate the project | Without a budget and controls, a result could not change a stated conclusion |
| Prediction is not falsifiable | The calibration-reuse prediction assumes a drift direction that a device may not have |
| Shot budget is unavailable | No budget is declared |
| Calibration or drift cannot be documented | No code records backend properties, calibration time, or queue time |
| Selected cells are already inconclusive | `decision_near_tie` has a decision-error upper bound of 0.458 at 16,384 shots |
| ZNE validity controls are incomplete | Three of five relevant cells have no single-method assessment, and every combined assessment is rejected |

## Critical questions

**What result would falsify the simulator conclusion?** Each cell has a stated falsifier in the
manifest. However, the conclusions are conditional on synthetic channels, so a mismatch would first
falsify the transfer of those channels to a device, not the simulated result.

**Can device drift be separated from method failure?** Not with the current tools. That needs
interleaved blocks, repeated reference circuits, and calibration records around each block. The
protocol specifies these, but none is implemented.

**Is one backend enough?** Only for a time-bound claim about that backend. It cannot support a
general claim about hardware or about how a method transfers across devices.

**Will queue and calibration timing invalidate paired comparison?** They can. Methods in separate
jobs may run under different calibrations, so pairing holds only within a block that does not cross
an update. Queue behavior is unknown until a provider and backend are chosen.

**Is the subset statistically powered?** Not as a whole. Resolving the near-tie gap in one repeat
takes 205,218 shots, about 12.5 times the 16,384-shot budget. The three-repeat floor for the
readout and ZNE cells comes from simulator effect sizes and is likely optimistic on a device. The
calibration-reuse cell cannot be powered without a drift estimate.

**Research hypothesis or demonstration screenshot?** Today, mainly a demonstration. The hypotheses
are written down, but the controls needed to interpret a surprising result are missing.

**Would a no-go be more honest?** Yes. It keeps a clear line between simulated evidence and hardware
evidence, and it lists exactly what is missing.

## Limits of this review

The shot total assumes 50 repeats per method for each cell, plus a garbage-folding control for the
two ZNE-gate cells and the simulated workflow cost for calibration reuse. It excludes cost, queue
time, and failed jobs. This review evaluates the project's readiness and says nothing about how any
device performs.

## What would reopen the review

1. A declared budget and access to a named backend.
2. Backend-specific predictions and thresholds, committed before any job.
3. Tested metadata capture, including calibration and queue records.
4. A single-method validity assessment for every ZNE-relevant cell, with a positive control stable
   at the chosen budget.
5. A near-tie cell that is either removed or funded, decided before any job.
6. A drift estimate that makes the calibration-reuse test falsifiable and powerable.
