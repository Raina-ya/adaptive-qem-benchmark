# Signal v2 Protocol

**Status: frozen before any noisy run on manifest circuits.** Tests use only synthetic circuits.
The manifest hash, this protocol, and the analysis code should be committed before
`src.generate_signal_v2` runs. Commit order shows publication order only. It does not prove that
nobody ran anything privately.

## Questions

1. Is raw reliable, or does shrinkage just place it near a small reference?
2. When does a ZNE gate misjudge, and can a runtime-only gate do better on unseen circuits?
3. Over which circuits and noise does ZNE help after its variance cost?

## Circuits

- **Structure:** the benchmark's layered ansatz, at widths 2, 4, 6 and depths 1, 3, 6.
- **Parameters:** `derive_seed("signal-v2", width, depth, index)`, scanned from index 0.
- **Selection:** for each seed, observables are tried in a fixed order: alternating full-support,
  `ZZ`, then `Z`. A seed fills the first band still short of three circuits in that width and depth.
  Only noiseless values are read.
- **Bands by |ideal value|:** near_zero below 0.06, weak below 0.2, medium below 0.5, strong below
  0.9, boundary_adjacent from 0.9.
- **Shortfalls:** the scan stops at 1,500 seeds. Boundary circuits are scarce at large width and
  depth, so those cells stay short rather than being padded. `signal_v2_coverage.csv` lists them.
- **Confounding:** strong and boundary circuits use lower-weight observables more often, and
  deep cells contain fewer of them. Band effects are not free of observable weight or depth.
- **Splits:** by hash rank within each band, 50% train, 25% validation, 25% holdout, per circuit.

## Simulation

`src/density_sim.py` computes exact density matrices with NumPy. Shots are sampled from the exact
probabilities. It is tested against analytic cases and Aer, and results are labeled
`noisy_simulated`. Gate-noise values are synthetic, not taken from a device.

| Condition | Content | What it tests |
| --- | --- | --- |
| none | No noise | Negative control: nothing can truly improve |
| depolarizing | Per-gate depolarizing | Near-ideal case for extrapolation |
| amplitude_damping | Per-gate damping | Non-unital decay, with a nonzero floor |
| dephasing | Per-gate phase flips | Basis-dependent decay |
| coherent_overrotation | Angle error scaling with the pulse | Inverse gates cancel it, so folding should not amplify it |
| coherent_detuning | Angle error of fixed sign | Folding accumulates it |
| mixed | All of these at lower strength | Less idealized mix |

All noisy conditions also have asymmetric readout error. `coherent_detuning` is an addition to the
requested list, because over-rotation alone would test only the case folding cannot amplify.

Budgets are 4,096 and 16,384 shots, with 50 repeats. Every design splits one budget evenly among
its factors: global 1,3,5; global 1,3,5,7; local 1,3,5. Readout-corrected estimates give 20% of the
same budget to calibration.

## What "really improved" means

For one condition, `raw_rmse - fit_rmse > 0.005`. Raw RMSE is analytic. Fit RMSE uses 5,000
independent draws from the exact fold means, so it shares no samples with the repeats a gate reads.
Each fit's RMSE includes its variance, so no separate variance-cost adjustment is applied.

## Models

Linear, quadratic, Richardson, and exponential-like fits. With three factors, quadratic and
Richardson are the same polynomial, so Richardson appears only for the four-factor design. The
exponential fit is undefined when signs differ across factors: it then falls back to the factor-one
estimate, and `undefined_rate` records how often.

Design and model selection is frozen as follows: take the top three pairs by mean training gain,
then the best on validation. Ties sort by name. Holdout is not read.

## Gate candidates

The gate is scored on `global_135` with a linear fit. The target is condition-level, using all 50
repeats, as in v1. A candidate is one physical rule and one trend rule, with garbage, retention, and
estimated-gain checks each on or off, giving 256 candidates.

| Check | Rule | Failure mechanism named |
| --- | --- | --- |
| physical_strict | All point estimates within bounds | `point_estimate_outside_physical_bounds` |
| physical_ci | At most 5% of runs outside bounds by more than 1.96 SE | `physical_bound_exceeded_beyond_sampling_interval` |
| physical_tolerance | Largest overshoot at most 3 median SE | `overshoot_larger_than_sampling_tolerance` |
| physical_constrained | Never rejects; scored on the clipped estimate, keeping the unclipped one | none |
| trend_v1 | The v1 destroyed and unstable rates | `v1_destroyed_or_unstable_rate_too_high` |
| trend_monotonic | No significant rise in raw signal | `signal_rises_with_noise_factor` |
| trend_corrected_monotonic | The same, on readout-corrected values | `readout_corrected_signal_rises_with_noise_factor` |
| trend_leave_one_out | Pairwise lines within 2 median SE of the full fit | `extrapolation_depends_on_one_noise_factor` |
| trend_weighted_residual | Median chi-square at most 3.84 | `weighted_linear_fit_residual_too_large` |
| trend_quadratic | Quadratic and linear agree within 2 median SE | `quadratic_and_linear_extrapolations_disagree` |
| trend_slope_sign | At least 90% of runs share a slope sign | `fitted_slope_sign_unstable_across_repeats` |
| garbage | Genuine signal exceeds garbage at the last factor | `garbage_folding_not_distinguished_from_genuine` |
| retention | Identifiable baseline, and retention interval above 0.10 | `signal_retention_too_low_or_baseline_unidentifiable` |
| estimated_gain | `(fit-raw)^2 - 2 var(fit)` interval above 0 | `estimated_gain_not_resolved_after_variance_cost` |

No v2 check reads the reference. `estimated_gain` assumes the fit is unbiased, and other checks must
carry that assumption. Readout conditioning, leverage, residual curvature, and variance ratio are
reported as metrics. Conditioning is also tabulated as a check, but at these readout errors it is
not expected to fire.

**v1 comparators:** `v1_full` is the frozen gate, which reads the reference. `v1_runtime` keeps its
retention, physical, and trend checks. Negative-mass cannot be applied because v2 stores parity
counts for ZNE designs, so it is fixed at passing. It never failed in v1's 32 conditions.

## Selection rule

Take the five best candidates by training balanced accuracy; ties favour fewer checks, then name.
Pick the best on validation with the same tie rule. Then score holdout once.

v2 is adopted over `v1_runtime` only if holdout balanced accuracy is higher and its false-acceptance
rate is at most 0.05 higher. Otherwise v1 stays, and that outcome is reported as it stands.

Holdout reports v1 and v2 false acceptance and rejection with Wilson intervals on conditions and
circuit-cluster bootstrap intervals. Conditions on one circuit are not independent, so the cluster
interval is the one to read. Every holdout rejection is listed with its mechanisms.

## Hypotheses fixed before the run

- **H1:** ZNE beats raw more often on boundary circuits than near-zero circuits. The circuit-cluster
  interval for that difference excludes zero.
- **H2:** Within a band, ZNE's gain rises with raw attenuation.
- **H3:** Folding gives no useful gain under `coherent_overrotation`, and does under
  `coherent_detuning` at least on some circuits.
- **H4:** A runtime gate cuts false rejection compared with `v1_runtime` without raising false
  acceptance by more than 0.05.

## Limits

- Condition-level gates read 50 repeats and garbage folds. Their cost is not charged to the per-run
  budget in the policy comparison, so a gate on one real run would be less informed.
- Readout-corrected values are stored as derived values with seeds, without their bitstring counts.
- Circuits come from one ansatz, so results do not extend to other families.
- Noise strengths are synthetic and do not represent a device.
