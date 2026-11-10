# Power, Selector, Certification, and Readout Boundary Results

Rules were frozen in `e6ac8f3` (Parts 7 and 8) and `79d4492` (Part 9) before these runs. All results
come from the project environment (numpy 2.3.4, pandas 2.3.3, Windows) and are `noisy_simulated`,
with synthetic noise. Nothing here is hardware evidence.

## Confirmatory comparisons (Part 7)

34 confirmatory comparisons, Holm-corrected, all reaching 80% power at their planned repeats:

| Role | Resolved difference | Close within margin | Underpowered |
| --- | --- | --- | --- |
| core_counterexample | 1 | 1 | 0 |
| near_tie | 9 | 11 | 8 |
| rank_uncertain | 3 | 0 | 1 |

- **Core counterexample confirmed at 16,384 shots:** on `medium_dense8`, raw selects the right QAOA
  candidate more often than ZNE (error 0.271 against 0.379, 398 new pairs, p = 0.001). At 4,096 shots
  the two are within 0.10 of each other (0.397 against 0.414).
- **QAOA rank-uncertain cells:** ZNE selects worse than readout or raw in 3 of 4, by 0.16 to 0.19.
- **Near-tie cells:** 8 of 28 stay underpowered even at their planned repeats, so the v1-based power
  estimate was optimistic for them.
- **Holdout tasks** (two settings v1 never ran, 72 comparisons, read once): 35 resolved, 34 close
  within 0.10, 3 underpowered. So the null rule separated "close" from "underpowered" in 69 of 72.
  Under shifted readout imbalance, readout correction beats raw by up to 0.74 in decision error, and
  ZNE is worse than readout in every resolved comparison. Under the shifted gate setting at high noise
  and margin 0.03, ZNE beats both.

## Shots to reliability (Part 7)

At target 10% decision error, across 40 positive-margin cells:

| Method | Reached | Blocked by bias | Not reached by 2^20 |
| --- | --- | --- | --- |
| raw | 26 | 13 | 1 |
| readout | 33 | 7 | 0 |
| ZNE | 27 | 11 | 2 |
| combined | 34 | 5 | 1 |

- **Where nothing is biased** (`symmetric_gap`), raw or readout is cheapest in every cell, and ZNE or
  combined needs the same to 8 times more shots.
- **Where a bias exists, more shots cannot fix it.** Error rises toward 1 with budget for any method that
  keeps the bias. Only the method that removes that bias reaches the target.
- **`gate_imbalanced` at high noise blocks every method,** including ZNE, because the linear fit leaves
  enough residual bias to flip a margin of up to 0.20.
- **Sequential stopping is conservative:** at the budget that meets 10%, only about 16% of runs had
  stopped with a certified correct answer. The rest were still unresolved and decided by forced choice.

## Selector (Part 8)

The frozen rule chosen on training and validation was `sep1.96_dev4_raw`.

| Holdout policy (20 tasks, 14,000 runs) | Coverage | Forced regret | Covered regret | Reliable coverage | Gate shots |
| --- | --- | --- | --- | --- | --- |
| always_raw | 1.00 | 0.0024 | 0.0024 | 0.919 | 113,664 |
| always_readout | 1.00 | 0.0031 | 0.0031 | 0.908 | 90,931 |
| always_zne | 1.00 | 0.0059 | 0.0059 | 0.867 | 340,992 |
| always_combined | 1.00 | 0.0072 | 0.0072 | 0.855 | 272,794 |
| selector | 0.64 | 0.0031 | 0.0005 | 0.630 | 147,767 |
| abstaining_readout | 0.68 | 0.0033 | 0.0006 | 0.666 | 147,763 |
| oracle best baseline | 1.00 | 0.0022 | | | |

**No acceptance route passed.** Forced regret matched always-readout (difference 0.00005, interval
-0.0003 to 0.0004). The selector costs 1.63 times the gate shots, and it covers fewer runs reliably
than the same abstention rule on readout alone (-0.037, interval -0.047 to -0.025). So adaptive
selection did not beat a simple fixed strategy, as H7 expected.

- **Raw beat readout in every holdout group:** by family, width, noise, gap class, and budget. In these
  tasks candidates share one readout, so readout bias is mostly common-mode and cancels in the
  comparison, while calibration costs 20% of the budget.
- **Leave-one-group-out:** the refitted selector beats always-readout in some groups and loses in others,
  all within 0.0011, with no group where it is reliably better.

## Certification (Part 8)

| Rule (holdout) | Certified | False certification per run | Given certified |
| --- | --- | --- | --- |
| both layers | 0.502 | 0.0074 | 0.0148 |
| layer one only | 0.518 | 0.0076 | 0.0147 |
| naive independent z | 0.700 | 0.0175 | 0.0250 |

- **The naive control was worse on both counts:** 40% more certifications but more than twice the false
  certifications. So accounting for shared samples matters.
- **H5 not supported:** the bias layer removed 1.6 percentage points of certifications and almost no
  false ones. Its
  alarms fired in at most 2.4% of Layer-1 passes, and two alarms never fired.
- **Near-tie tasks are the failure case:** 36% of certified near-tie runs certify the wrong winner. Bias
  survives every layer, and statistical separation only controls sampling error.
- **H6 supported:** coherent over-rotation gives the highest false certification of any noise, 6.9% of
  certified runs.

## Readout feasibility (Part 9)

- **H8 supported.** At fixed total budget and width 8, full-matrix correction fails in 93 to 100% of runs
  at 4,096 shots (ill-conditioned with 3 shots per state). At 16,384 shots it completes, with RMSE 0.13
  to 0.24, worse than raw in every model, and 12 to 21 times raw spread. Tensor-product reaches 0.017
  and 0.019 under independent and asymmetric readout.
  At width 10 it fails always: no calibration shots at 4,096, singular at 16,384.
- **At fixed calibration quality** it completes at width 8 and 10 with RMSE 0.021 to 0.049, using 80 to
  98.5% of all shots for calibration. It trails tensor-product under independent and asymmetric
  readout, and is far ahead of it under pair-correlated readout.
- **Pair-correlated readout reverses the ranking.** Tensor-product, observed-subspace, and matrix-free
  correction all use a model without pair correlation, and they become worse than raw (width 8: about
  0.14 against 0.087). Pairwise-local correction stays at 0.014 to 0.043 on product states, below raw at
  every width and in both regimes.
- **Matrix-free and observed-subspace match tensor-product** wherever tensor-product is correct.

## Mismatch dose-response (Part 9)

- **Where mitigation loses to raw:** only in two scenarios. Under asymmetric drift it loses from lambda
  0.25 to 0.60, later at larger width. Under independent-to-correlated, it loses from 0.50 to 0.75. In
  abrupt change, calibration age, and pair-correlated mismatch, it stays better than raw up to lambda 1.
- **H10 not supported:** pair-correlated mismatch never made tensor correction lose to raw, because raw
  is hurt by the same flips.
- **No steep rise in decision error was found.** With margin 0.10 and 16,384 shots, decision error never
  rose by 0.10. That question is inconclusive for this design, not answered.
- **H9 mostly supported:** condition number never warned, since it comes from the stale calibration. The
  check circuit warned before or without a loss in 75 to 77% of cases, but it is blind to
  independent-to-correlated mismatch, where marginals do not change. That is exactly one of the two
  scenarios where mitigation loses.
- Coverage drops early (lambda 0 to 0.40) but needs the ideal value, so it is not a runtime warning.

## Drift policy (Part 9)

- **H11 partly supported.** A check-triggered policy at z = 2 matched or beat every-decision
  recalibration in mean decision error in every shape and amplitude except sinusoidal drift at 0.06
  (+0.004), using about 60% of the shots. Every-decision
  recalibration also adds fresh calibration noise each step.
- **The trigger loses under fast sinusoidal drift:** at amplitude 0.06, z = 3 and 4 are worse than
  every-decision in 75% and 94% of scenarios.
- **The abrupt 0.01 part of H11 was not supported:** drift of 0.01 changed no policy's decision error,
  because it barely moves a decision at margin 0.02 or more. The trigger missed it often (missed drift
  0.48 at z = 3), harmlessly.
- **Never recalibrating** costs up to 0.06 extra decision error at amplitude 0.06.

## Bounded conclusions

- Fixed-budget accounting reverses rankings: calibration and folding costs make raw the cheapest
  reliable choice whenever no decision-relevant bias exists.
- More shots cannot fix a bias. Shots-to-reliability is infinite for a biased method, and finite only for
  a method that removes that bias.
- Full-matrix calibration at width 8 is a resource failure under a shared budget, not a method failure.
  A correctly modeled local method (pairwise) succeeds where a mismodeled one (tensor under pair
  correlation) does worse than nothing.
- Runtime diagnostics catch drift in marginals and miss correlation changes.

## Limits

Synthetic channels and noise. Selector and certification holdout has 20 tasks. Drift uses one qubit.
The decision task in the mismatch study was too easy to show a steep rise.
