# Hardware Protocol

**Status: not authorized and not run.** No job has been submitted. This file states what a future
run would have to do. The current decision is in
[the readiness review](hardware_readiness_review.md).

## Proposed cells

The frozen predictions are in `experiments/hardware/hardware_manifest_v1.csv`, which
`src/hardware_readiness.py` derives from committed simulator results.

| Cell | Question it could answer | Falsifier |
| --- | --- | --- |
| `readout_stable` | Does readout correction help on a readout-dominated circuit? | The 95% paired interval for the absolute-error gain is at or below zero |
| `zne_positive_control` | Does ZNE pass the validity gate on a shallow stabilizer circuit? | Any frozen gate check fails |
| `zne_signal_risk` | Does apparent ZNE gain fail the gate on a deeper circuit? | Every frozen gate check passes |
| `decision_near_tie` | Does a 0.02-margin decision stay unresolved at this budget? | Any method's decision-error Wilson upper bound is at or below 0.20 |
| `decision_clear` | Does a 0.20-margin decision resolve at this budget? | Any method's decision-error Wilson lower bound is above 0.20 |
| `calibration_reuse` | Does decision error rise with calibration age? | Error does not rise while measured drift is present |

For `calibration_reuse`, `predicted_raw` is the never-recalibrated observed probability and
`predicted_readout` is the corrected probability when recalibrating at every decision.
`required_repeats` is a floor from simulator effect sizes, so a run would use `planned_repeats`.

## Requirements before a run

1. Reopen the review and reach a go decision.
2. Name one backend and date, then rerun predictions using a noise model from that backend's
   published calibration. Commit them, along with thresholds and repeats, before any job.
3. Declare a shot and cost budget that covers the whole plan. Dropping a cell for budget reasons
   must also happen before any job.
4. Add the provider dependency and tested metadata capture. Reject any record with a missing
   required field.

## Execution rules

- **Blocks:** keep all methods of a repeat in one job or session, with circuit order randomized by a
  committed seed.
- **Calibration boundary:** record the calibration timestamp before and after each block. A block
  crossing a calibration update is kept and labeled, and is excluded from paired comparison.
- **Drift controls:** repeat a fixed reference circuit and the readout calibration through the run,
  so that a change can be tracked separately from a method comparison.
- **Transpilation:** fix the qubit mapping, seed, and optimization level. ZNE folding must survive
  transpilation, and physical gate counts are recorded for each noise factor.
- **Budget:** every method, calibration circuit, and garbage-folding control is charged.
- **Failures:** failed, cancelled, and partial jobs stay as rows, and nothing is resubmitted without
  a new row.
- **Evidence labels:** device rows use `hardware_derived`, are stored apart from simulated rows, and
  never overwrite predictions.

## Required metadata

Each row records provider, backend, job ID, submission time, execution time, calibration timestamp,
backend properties, qubit mapping, transpiler version, transpiler seed, optimization level, physical
circuit depth, one-qubit gate count, two-qubit gate count, shots, noise factors, mitigation method,
calibration circuits, queue time, and failure status.

## How outcomes would be read

A mismatch with a prediction first challenges whether the synthetic noise model transfers to the
device. It counts against a method only if the positive control passes, reference controls stay
stable, and the comparison stays inside one calibration window. Any result describes one backend
during one period and does not establish general hardware performance.
