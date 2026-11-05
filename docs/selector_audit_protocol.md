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

## Consequence audit (I1)

Every selection falls into exactly one category, using definitions frozen in the
configuration before the categories were computed. The selector and three fixed policies
are scored with the same rules on 180 cells. No policy is charged a diagnostic or learning
cost here, which favors the selector.

| Category | Selector | Always raw | Always readout | Always ZNE |
|---|---:|---:|---:|---:|
| Correct, useful | 63 | 106 | 19 | 18 |
| Correct, tied | 20 | 23 | 11 | 24 |
| Incorrect, negligible | 29 | 0 | 71 | 3 |
| Incorrect, material | 42 | 26 | 45 | 78 |
| Catastrophic | 17 | 16 | 25 | 48 |
| Abstained, avoided failure | 1 | 0 | 0 | 0 |
| Abstained, unnecessary | 5 | 0 | 0 | 0 |
| Inconclusive | 3 | 9 | 9 | 9 |

Mean RMSE regret by evaluation set:

| Set | Selector | Always raw | Always readout | Always ZNE |
|---|---:|---:|---:|---:|
| Rule design | 0.0164 | 0.0234 | 0.0116 | 0.0217 |
| Holdout | 0.0210 | 0.0219 | 0.0077 | 0.0266 |
| Noise shift | 0.0081 | 0.0051 | 0.0101 | 0.0135 |

### Review questions answered so far

**Does accuracy reward correct guesses inside practical ties?** Only slightly. 20 of the
selector's 83 correct cells are ties. A larger concern is that 26 of its 52 useful raw picks
won on cost alone.

**Are catastrophic errors hidden by average regret?** Yes. Holdout median regret is 0.0004,
but its maximum is 0.2555. Eight of the selector's 17 catastrophic cells are GHZ cells at
high noise where it chose raw and readout was far better. The other nine meet only the
relative rule, at references with magnitude below 0.06.

**Does abstention improve reliability?** The evidence is too thin to say so. The selector
abstained in 6 of 180 cells, all at width 6 and 1,024 shots, and never under noise shift.
With its gates removed it would have chosen raw every time, and that was already the
lowest-RMSE method in 5 of the 6 cells. In the remaining cell, GHZ width 6 at high noise,
ZNE beat raw by 0.075 and readout completed only 25 of 50 repeats.

**Would always-raw outperform the selector?** On harm counts and under noise shift, yes,
even before any learning cost: 42 material-or-catastrophic cells versus 59, and mean regret
0.0051 versus 0.0081. This does not make raw a good general policy. All of raw's harmful
cells have a reference magnitude of at least 0.06. In those 84 cells, raw has 42 harmful
outcomes, the selector 28, and always-readout 19, with no catastrophic cell. The 0.06 split
is descriptive and was chosen after seeing the data.

Reliability envelopes, bootstrap threshold stability, and charged learning costs are not
evaluated here.

### Limits

Selector inputs are still 50-repeat cell means, which a runtime selector would not have.
The relative catastrophic rule is strict when the reference is near zero, so
`catastrophic_absolute` and `catastrophic_relative` are reported separately. Category counts
use point estimates of RMSE and ignore their sampling uncertainty.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log src.regime_labels_v2
.\.venv\Scripts\python.exe -m src.run_with_log src.selector_audit
.\.venv\Scripts\python.exe -m pytest tests/test_regime_labels_v2.py tests/test_selector_audit.py -q
```
