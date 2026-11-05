# Selector Audit Protocol

## Status

This is a post hoc reanalysis of inspected v1 results, with `noisy_simulated` evidence.
v2 metrics are diagnostic and are not an independent holdout. All v1 files are unchanged.
Thresholds are in `experiments/configs/selector_audit_v1.yaml`.

## Why v2 exists

The v1 oracle in `src/regime_map.py` could never assign the label `raw`. When raw won,
its gain over itself was zero, so the cell always fell into the `practical_tie` branch.
That mislabeled 77 cells in which raw won by at least the practical threshold.

v1 regret also compared RMSE with a winner chosen by cost-aware utility. Regret could
therefore be negative, and abstaining cells were dropped from the mean without being
reported.

## What changed

- A raw winner with a utility gap of at least 0.005 is labeled `raw`.
- `raw_win_basis` is `accuracy` when the runner-up's RMSE-gain interval is below zero, and
  `cost_only` otherwise.
- `rmse_regret` is selected RMSE minus the lowest RMSE among methods with results in the
  cell, so it cannot be negative.
- Abstaining cells keep a missing regret, and their count is reported.

The selector, its selections, the holdout split, and every other label rule are unchanged.

## Observed results

| Metric | v1 | v2 |
|---|---:|---:|
| Practical ties | 103 | 26 |
| Raw winners | 0 | 77 |
| Readout winners | 18 | 18 |
| ZNE winners | 14 | 14 |
| Inconclusive | 9 | 9 |
| Holdout decision compatibility | 41.4% | 41.4% |
| Holdout strict label match | 3.4% | 31.0% |
| Holdout mean regret | 0.0207 | 0.0210 |
| Holdout negative-regret cells | 2 | 0 |
| Noise-shift decision compatibility | 50.0% | 50.0% |
| Noise-shift strict label match | 8.3% | 47.2% |
| Noise-shift mean regret | 0.0077 | 0.0081 |
| Noise-shift median regret | 0 | 0.0007 |
| Noise-shift negative-regret cells | 1 | 0 |

Of the 77 raw winners, 40 won on accuracy and 37 won on cost alone. Compatibility did not
change because those cells already accepted only `raw`. What changes is the interpretation:
of the 12 compatible holdout cells, 8 are raw wins and only 2 are true ties.

## Limits

Most raw wins are in hardware-efficient cells whose reference is close to zero, where
noise pushes estimates toward the reference. They do not show that raw recovers a strong
signal. Cells with incomplete readout runs stay `inconclusive`, and their RMSE values use
completed repeats only.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log src.regime_labels_v2
.\.venv\Scripts\python.exe -m pytest tests/test_regime_labels_v2.py -q
```
