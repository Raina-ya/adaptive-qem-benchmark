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

Charged learning costs are not evaluated here.

### Limits

Selector inputs are still 50-repeat cell means, which a runtime selector would not have.
The relative catastrophic rule is strict when the reference is near zero, so
`catastrophic_absolute` and `catastrophic_relative` are reported separately. Category counts
use point estimates of RMSE and ignore their sampling uncertainty.

## Reliability envelopes (I2)

Definitions are in `experiments/configs/method_envelopes_v1.yaml` and were written before
these results were computed, but after the I1 results had been seen. A cell is reliable for
a method when all 50 repeats completed and its RMSE regret is at most 0.005. A level is
supported when it has at least 10 rule-design cells, a reliable rate of at least 80%, and a
Wilson lower bound of at least 50%.

### Supported levels

| Method | Level | Reliable rate | Supported in bootstrap |
|---|---|---:|---:|
| raw | zero noise | 32/32 | 100% |
| raw | `|reference|` below 0.06 | 56/56 | 100% |
| raw | decision margin 0.20 | 11/12 | 93% |
| raw | width 6 | 36/44 | 55% (unstable) |
| readout | decision margin 0.20 | 11/12 | 94% |
| readout | zero noise | 29/32 | 89% (unstable) |
| readout | `|reference|` at least 0.5 | 26/30 | 88% (unstable) |
| zne | none | | |

No method has a supported level on every feature; depth has none at all. The frozen joint
envelope therefore contains no cells, so its inside-versus-outside validation has nothing
to compare. A strict "inside on every feature" rule would always report no safe method on
this grid.

### Single levels on cells that did not define them

| Method and level | Holdout | Noise shift |
|---|---:|---:|
| raw, `|reference|` below 0.06 | 16/16 | 23/24 |
| raw, width 6 | 3/4 | 12/12 |
| raw, zero noise | 4/4 | no cells |
| readout, `|reference|` at least 0.5 | 5/6 | no cells |
| readout, zero noise | 4/4 | no cells |

The supported raw levels mostly reflect low signal: most width-6 cells have references near
zero. They do not show that raw recovers a strong signal.

### Do the diagnostics predict reliability?

- **Condition number:** partly, as a warning. Readout was reliable in 75% of cells below 1.5,
  50% from 1.5 to 3, 10% from 3 to 8, and 0 of 4 cells at 8 or above. Low values alone do not
  reach the 80% support rate.
- **ZNE fit residual:** no. Rates are 24-33% in every bin with at least 10 cells, so the
  selector's residual limits add no evidence.
- **Signal retention:** not clearly. Rates are 56%, 27%, and 46% across the three populated
  identified bins, and 11% when the baseline is unidentifiable.

### Review questions

**Is the envelope based on enough cells?** Only for single features. Levels have 10-115 cells,
but there is one circuit per width and depth, and interactions are not estimated. Decision
margin has 12 cells per level from a separate width-2, depth-3 experiment.

**Are thresholds stable under bootstrap?** Mixed. Of 7 supported levels, 3 are unstable.
Resampling repeats also moves selector gates in cells near the limit: 3 cells are unstable at
the condition-number gate of 8 (means 7.87, 8.10, and 8.16), and 7 cells are unstable at the
limit of 3. The residual limits of 0.01 and 0.03 are never reached.

**Does the selector generalize beyond its noise model?** This evidence cannot show that. Most
supported levels have few or no noise-shift cells, and the raw near-zero result is expected
because noise moves estimates toward zero whichever noise model acts.

### Limits

Ranges are marginal and describe only observed grid values, so they must not be interpolated
or extrapolated. The 0.06 reference edge was chosen after seeing the data. Rates use the same
evidence as earlier inspections, so holdout and noise-shift results are diagnostic.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m src.run_with_log src.regime_labels_v2
.\.venv\Scripts\python.exe -m src.run_with_log src.selector_audit
.\.venv\Scripts\python.exe -m src.run_with_log src.reliability_envelopes
.\.venv\Scripts\python.exe -m pytest tests/test_regime_labels_v2.py tests/test_selector_audit.py tests/test_reliability_envelopes.py -q
```
