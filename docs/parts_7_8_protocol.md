# Power, Shots-to-Reliability, Selector, and Certification Protocol

**Status: frozen before any new noisy run.** `experiments/manifests/confirmatory_v1_manifest.csv` and
`selector_v3_manifest.csv` were built without any new noisy outcome. The power inputs are committed
v1 results, so they are exploratory. Commit order shows publication order only.

## Part 7: which comparisons earn more repeats

`src.power_analysis_v1` reads committed paired outcomes, and labels each comparison:

| Label | Meaning |
| --- | --- |
| `resolved_difference` | The 95% paired interval excludes zero |
| `close_within_margin` | The 90% paired interval lies inside the minimum difference |
| `underpowered` | Neither, so "close" and "not enough repeats" cannot yet be told apart |

Minimum differences are 0.10 in decision error and 0.15 in selection probability. Intervals use
Agresti-Min, so zero discordant pairs still have width. Required pairs use the McNemar formula at the
observed discordant rate (raised to the minimum difference if lower), power 0.80, with Bonferroni
across the family. Decision repeats in v1 were sampled independently per method, so pairing by repeat
index is a matching for analysis and not shared randomness.

| Group | Contents | Use |
| --- | --- | --- |
| Exploratory | All 376 v1 comparisons | Chose what to test; confirm nothing |
| Confirmatory | 34 comparisons in the manifest | New seeds, Holm across the family |
| Holdout | Two new task settings, 72 comparisons | New seeds, read once, choose nothing |

A comparison enters the confirmatory group only if it is `near_tie` (margin at most 0.05) and
underpowered, `rank_uncertain` (selection error between 0.1 and 0.9) and underpowered, or the
`core_counterexample` (`medium_dense8`, raw against ZNE). Caps are 4,000 pairs on the exact engine
and 400 on Aer. `combined_qem_v1` decision rows were not used for sizing.

After Holm, a confirmatory comparison is `resolved_difference` if it survives. Otherwise it is
`close_within_margin` if its interval at `1 - 2 * alpha_per_comparison` lies inside the minimum
difference, and otherwise `underpowered`. Those labels stand, whatever the result.

## Part 7: shots to reliability

`src.shots_to_reliability` uses the exact two-candidate engine, for all positive-margin cells.

- **Methods:** raw, readout, ZNE, and combined, with every shot charged.
- **Budget ladder:** 1,024 to 1,048,576, doubling. 400 repeats per cell.
- **Targets:** decision error 5%, 10%, 20%, using the 95% Wilson upper bound.
- **Reached at:** the smallest budget where it and every larger budget stay under the target.
- **Stopping:** at each look, stop if the delta-method gap interval excludes zero, at confidence
  `1 - alpha / looks`.
- **Unresolved:** no stop by that budget. Its forced choice is still scored, and it never counts as
  reaching a target.
- **Failure:** a singular calibration or an exact-zero gap counts as an error.
- **No target reached:** `blocked_by_bias` when the limiting gap has the wrong sign, otherwise
  `not_reached_by_maximum_budget`.

## Part 8: decision tasks

Tasks choose the largest of three candidate values. There are 72 tasks across four families
(`hardware_efficient`, `ghz_rotated`, `tfim_trotter`, `qaoa_ring`), widths 2, 4, 6, depths 1, 3, and
three gap classes. Each class
scans its own seeds, so no circuit appears twice. Splits are by task within family: 50%, 25%, 25%.

Noise is synthetic. Conditions are none, depolarizing, amplitude damping, dephasing, coherent
over-rotation (a known bias folding cannot amplify), mixed, and readout drift (calibration taken at a
stale readout rate). Budgets are 4,096 and 16,384, with 50 repeats.

## Part 8: selector

Same total budget for every policy. The selector spends 10% on calibration (split before and after)
and 20% on a pilot at factors 1, 3, 5, and the remaining 70% on one branch.

1. If the two pilot calibrations differ by z of at least 3, abstain.
2. Default to readout correction.
3. Use combined only if both pilot contenders show a resolved rise in magnitude after extrapolation,
   by at least `z_deviate` standard errors.
4. If raw is allowed, use it only if correction is unresolved on every candidate.
5. After the branch, abstain if the top gap is under `z_separate` combined standard errors.

No rule reads an ideal value, and there is no ML. ZNE-only is a baseline and not a branch.

**Grid:** `z_separate` 0, 1, 1.645, 1.96, 2.5; `z_deviate` 2, 3, 4; raw allowed or not. **Choice:** among
training rules with coverage at least 0.5, the five lowest covered regrets, then the lowest on
validation. Ties favour larger `z_deviate`, smaller `z_separate`, then no raw switch. Holdout is read once.

**Baselines:** always raw, readout, ZNE, combined; oracle (best baseline per condition, scored only);
and `abstaining_readout`, the same abstention test applied to a readout branch.

**The selector passes if any one route holds on holdout, against always-readout:**

- lower forced regret, with task-cluster interval above zero;
- similar regret (interval low at least -0.002) with lower gate overhead;
- overhead within 10% and higher reliable coverage than `abstaining_readout`, with interval above
  zero.

If none holds, the report says adaptive selection did not beat the simple fixed strategy.
Leave-one-family, width, and noise-out refit the rule without that group and score it on its holdout.

## Part 8: certification

The certificate reads five independent batches, each with its own calibration (20%), base run (60%),
and folds at 3 and 5 (10% each).

**Layer 1, statistical separation.** At looks 3, 4, 5, the winner must beat every challenger by a
one-sided t test on batch differences. The level is `0.05 / (K-1) / 3`. Batch differences carry shared
calibration inside each batch, so independence between candidates is not assumed.

**Layer 2, bias alarms.** At that look, none may fire:

| Alarm | Fires when |
| --- | --- |
| `raw_mitigated_winner_differs` | Raw and corrected pick different winners |
| `fold_gap_reversal` | The gap at factor 5 reverses by over 2 standard errors |
| `calibration_age_drift` | First and latest batch calibration differ by z over 3 |
| `outside_physical_bound` | A corrected value exceeds the bound by over 3 standard errors |
| `variance_amplification` | Corrected spread exceeds 3 times the shot-noise value |
| `batch_sign_instability` | Fewer than 4 batches (or all, at look 3) favour the winner |

A run is certified at the first look passing both. The report compares it with Layer 1 alone and a naive
independent-z control. Higher certification with more false certification does not count as better.

## Hypotheses fixed before the run

- **H5:** both layers lower false certification against Layer 1 alone, especially under the bias
  conditions, at some cost in certification rate.
- **H6:** coherent over-rotation stays hard for Layer 2, because folding cannot reveal it.
- **H7:** the selector does not beat always-readout on regret by a resolved margin. This is the
  expected negative, because the pilot costs 30% of the budget.

## Limits

Everything is `noisy_simulated`. Noise strengths are synthetic and fold scaling assumes gate noise.
Readout-corrected values in the raw table are derived, reproducible from their seeds. A condition has
50 repeats, and holdout tasks are few per family and width, so worst-group numbers are coarse.
