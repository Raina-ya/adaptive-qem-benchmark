# Signal v2 Results

The rules were frozen in commit `c53d99f` before the benchmark ran. Everything here is
`noisy_simulated`, on one ansatz with synthetic noise. Nothing here is a hardware result. Section
"Post-hoc diagnostics" was written after the holdout was read, and it changes no choice.

These files come from the project environment, numpy 2.3.4 and pandas 2.3.3 on Windows. NumPy's
binomial stream changes between versions: an earlier run on numpy 2.5.3 matched exact values to
1e-12, gave a few changed count rows, and gave the same gate selection and holdout scores. A rerun
with a different NumPy version should therefore be expected to differ slightly.

## Coverage

126 independent circuits, 7 noise conditions, 2 budgets, and 50 repeats give 88,200 repeat rows, all
complete. Holdout has 30 circuits, 6 per band.

## Does raw reliably win?

Raw's advantage on small references comes mostly from ZNE's variance having nothing to remove. It is
not evidence that raw has recovered the value.

| Band, 4,096 shots | Circuits | Fit better | Raw better | Median raw bias / reference |
| --- | --- | --- | --- | --- |
| near_zero | 27 | 1.9% | 95.1% | 0.43 |
| weak | 27 | 16.7% | 59.9% | 0.24 |
| medium | 27 | 50.6% | 35.2% | 0.16 |
| strong | 24 | 66.7% | 16.0% | 0.11 |
| boundary_adjacent | 21 | 63.5% | 1.6% | 0.10 |

- **H1 supported:** the fit-better share is 0.611 higher at the boundary than near zero (circuit
  cluster interval 0.545 to 0.674). This covers noisy conditions and both budgets.
- **Budget matters near zero:** raw-better falls from 95.1% to 69.1% at 16,384 shots, as the variance
  penalty shrinks.
- **H2 supported descriptively:** within bands, ZNE gain follows raw attenuation (Spearman 0.55 to
  0.92) more closely than the size of the reference (-0.12 to 0.39). Below 0.01 attenuation, raw
  wins in nearly every condition in every band with such conditions, including strong (8 conditions,
  3 circuits). This is not a usable rule by itself, because attenuation needs the exact value.
- **Circuit variation:** per-circuit mean gain has SD 0.003 near zero, 0.006 weak, 0.007 medium,
  0.016 strong, 0.018 boundary. Medium ranges from -0.006 to 0.018, and strong from -0.007 to 0.051.
  Positive-gain circuits by band: 1/27, 11/27, 21/27, 22/24, 21/21.

## When does ZNE help?

| Noise, global 1,3,5 linear | Mean gain | Fit better | Raw better |
| --- | --- | --- | --- |
| depolarizing | 0.020 | 59.9% | 26.2% |
| amplitude_damping | 0.020 | 58.3% | 27.0% |
| mixed | 0.019 | 57.1% | 27.4% |
| dephasing | 0.008 | 38.5% | 31.0% |
| coherent_detuning | 0.001 | 37.7% | 39.7% |
| coherent_overrotation | -0.006 | 0% | 50.8% |
| none | -0.010 | 0% | 86.5% |

- **H3 supported:** under over-rotation, inverse gates cancel the error, so folding has nothing to
  amplify and the fit is never better. Detuning does help on some circuits, though its mean gain is
  near zero.
- **Global and local folding:** at equal budget, mean gain matches to three decimals in every
  noise condition, and fit-better shares differ by at most 2.4 points. Exact fold values do differ
  (up to 0.098), so this is no difference in net benefit on this grid rather than identical physics.

## Gate on holdout

The frozen rule selected `physical_constrained+trend_slope_sign`. In effect this is one check: four
physical variants tied, and the tie rule favours fewer checks.

| Gate | True accept | False accept | True reject | False reject | FA rate | FR rate |
| --- | --- | --- | --- | --- | --- | --- |
| v1_full (reads exact reference) | 77 | 5 | 274 | 64 | 0.018 | 0.454 |
| v1_runtime | 101 | 145 | 134 | 40 | 0.520 | 0.284 |
| v2 selected | 141 | 49 | 230 | 0 | 0.176 | 0.000 |

Cluster intervals, which treat circuits as the independent unit: v1_full FA 0.004 to 0.034 and FR
0.285 to 0.616; v1_runtime FA 0.372 to 0.698 and FR 0.138 to 0.440; v2 FA 0.123 to 0.223, with FR 0
in every resample.

- **H4 supported against the frozen comparator:** v2 beats `v1_runtime` on both error types, and the
  adoption rule passes.
- **v2 is worse than v1_full on false acceptance**, 0.176 against 0.018. `v1_full` cannot run without
  the exact value, but it is the stricter gate.
- **Every holdout rejection names a mechanism:** `zne_gate_v2_rejections.csv` has 230 rows, all
  `fitted_slope_sign_unstable_across_repeats`.

## Why the selected gate should not be over-read

It tests whether the decay across factors is resolved, not whether the extrapolation is right.
"Really improved" means the shift is larger than the variance cost, which is close to the same
quantity, so a high score is partly built in.

- False acceptance grows with shots: 0.131 at 4,096 and 0.224 at 16,384.
- It is noise dependent: FA is 0.414 under amplitude damping and 0.382 under detuning.
- It cannot detect a resolved but wrong extrapolation, and it would not catch a signal that rises.
- It cannot run on one run. It needs several repeats, whose cost the policy comparison does not charge.

## Checks on their own (holdout)

| Check | Balanced accuracy | FA rate | FR rate |
| --- | --- | --- | --- |
| trend_slope_sign | 0.912 | 0.176 | 0.000 |
| estimated_gain | 0.824 | 0.004 | 0.348 |
| garbage | 0.620 | 0.419 | 0.340 |
| trend_v1 | 0.595 | 0.527 | 0.284 |
| retention | 0.549 | 0.810 | 0.092 |
| trend_monotonic | 0.519 | 0.849 | 0.113 |
| trend_corrected_monotonic | 0.513 | 0.867 | 0.106 |
| trend_weighted_residual | 0.440 | 1.000 | 0.121 |
| trend_quadratic | 0.440 | 1.000 | 0.121 |
| trend_leave_one_out | 0.309 | 0.993 | 0.390 |

- **Leave-one-out, residual, and quadratic stability score below chance.** They pass when there is
  no decay, which is when nothing can be gained, and they reject some curved decays that do help.
- **Physical bounds could not be compared.** Strict bounds passed 99.7% of conditions, because only 2
  of 126 circuits have a reference above 0.99, and clipping changed mean RMSE by under 0.001. The v1
  GHZ rejections came from references at exactly 1, a case this grid barely covers.
- **Conditioning and leverage could not discriminate.** Readout conditioning never fails, because
  readout error is one fixed setting, and leverage is fixed by design at 4.375.

## Model selection

The frozen rule chose `local_135` with an exponential fit. On holdout this did **not** beat the
default: mean gain 0.006 (cluster interval -0.004 to 0.019), against 0.010 (0.003 to 0.017) for
`global_135` linear. The top training scores were within 0.001 of each other, so the rule was picking
among near-ties. The exponential fit was undefined in 18.5% of draws on this design, and fell back to
its factor-one estimate. Four-factor Richardson lost 0.019 on training, which is its variance cost.

| Holdout policy | Mean RMSE | Gain vs raw | Cluster interval |
| --- | --- | --- | --- |
| Always raw | 0.062 | 0 | |
| Always global 1,3,5 linear | 0.052 | 0.010 | 0.003 to 0.017 |
| Always selected model | 0.056 | 0.006 | -0.004 to 0.019 |
| v1_runtime gate, else raw | 0.052 | 0.010 | 0.005 to 0.016 |
| v2 gate, else raw | 0.048 | 0.014 | 0.008 to 0.020 |
| v2 gate with selected model, else raw | 0.050 | 0.012 | 0.002 to 0.023 |
| Oracle upper bound | 0.037 | 0.025 | 0.014 to 0.036 |

These RMSE values do not charge for the gate's repeats and garbage runs, so the gated rows flatter a
real single-budget run.

## Post-hoc diagnostics

- **Fewer repeats:** the same check scores balanced accuracy 0.708 with 2 repeats, 0.880 with 5,
  0.882 with 10, 0.901 with 20, and 0.912 with 50. False acceptance is 0.577 with 2 repeats. Reading
  N repeats costs N budgets, and splitting one budget into N runs was not tested.
- **Other truth thresholds:** v2's false acceptance rises from 0.093 with threshold 0 to 0.299 with
  threshold 0.02. At threshold 0.02, `estimated_gain` alone has higher balanced accuracy (0.943 against
  0.851). So the ranking of these two gates depends on what counts as worthwhile.
- **Paired contrasts:** v2 gated RMSE beats always-linear by 0.004 (0.003 to 0.006) and v1_runtime by
  0.004 (0.002 to 0.006). Clipping adds less than 0.001.

## Bounded conclusions

- ZNE's net benefit in this grid tracks how much attenuation exists to remove, compared with its
  roughly fourfold variance cost.
- Small-reference raw wins are not evidence of raw accuracy, because relative bias stays large.
- Folding-based ZNE gives no benefit under pulse-proportional coherent error.
- A resolved-decay check is a useful cheap filter against pointless extrapolation, not a certificate
  that the extrapolated value is right.

## Limits

Holdout has 30 circuits from one ansatz, and conditions within a circuit are correlated. Strong and
boundary circuits lean on low-weight observables. Readout error has one setting. Density-matrix
simulation is limited to width 6. None of this says how a device would behave.
