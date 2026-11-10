# Clean Rebuild of the Long Generators

`rebuild-generators.yml` rebuilds each of the 16 long generators as its own job on `ubuntu-latest`
with the locked dependencies (`requirements-ci.txt`, NumPy 2.3.4). Each job deletes that generator's
committed outputs in a clean clone, regenerates them, and compares them with the committed files,
which were generated on Windows. The comparison uses relative tolerance 1e-9 and absolute tolerance
1e-12. No tolerance was changed after the results were seen. The committed results remain the
Windows results.

## Runs

| Run | Commit | Result |
| --- | --- | --- |
| 42578064990 | `a6a81c8` | 11 reproduced, 4 differ, 1 registry error |
| 42588956790 | second run, after the registry fix and the more detailed comparison report | 12 reproduced, 4 differ |

Reproduced in both runs: `confirmatory_v1`, `confirmatory_qaoa_v1`, `shots_to_reliability`,
`generate_qaoa_selection`, `generate_decision_benchmark`, `generate_combined_qem`,
`generate_selector_holdout_v2`, `generate_readout_mismatch_benchmark`,
`generate_readout_mismatch_dose`, `generate_drift_policy`, and `generate_runtime_report`.
`generate_days08_12` also reproduced in the second run.

### Registry error in the first run

`figures/rmse_heatmap.png` was declared as an output of `src.generate_days08_12`, but it is written by
`src.generate_adaptive_analysis`. The registry now assigns it to the right generator. This was a
bookkeeping error, not a reproducibility failure.

## The four differences, measured in the second run

| Generator | What differs | Size | Effect on `main(2).pdf` |
| --- | --- | --- | --- |
| `generate_selector_v3` | `base_combined` in 1 of 50,400 rows, `pilot_fit` in 4 | Only the sign of zero: `0.000000` against `-0.000000` | None. The values are equal, so every selection, regret, and certificate is unchanged (claims G05, G06). |
| `generate_readout_feasibility_v2` | `condition_number` in 1,664 of 230,400 rows | One platform reports an infinite condition number where the other reports a large finite one | None. Run status is identical, and claim G07 uses status and shot counts. |
| `generate_signal_v2` | `out_of_bounds_rate` in 9 of 17,640 rows | Up to 0.97 in a rate | None. No analysis reads this column; the fit RMSE used by claims G01, G04, and G09 is identical. |
| `generate_readout_scalability_benchmark` | Run status in 29 of 5,760 runs (`complete` against `failed` as ill-conditioned); 5 of 192 summary cells | Corrected estimates otherwise agree within 4.2e-8; in the 5 cells, failure probability changes by up to 1, RMSE by up to 0.071, and decision-error probability by up to 0.1 | None in `main(2).pdf`, which prints only this study's planned repeat count (claim C09). The scalable-readout results of the earlier paper (`main.pdf`) are affected in these 5 cells. |

The first three are differences in how the same number is written or in which side of a numerical
edge a value lands, and they change no reported result. The fourth is substantive: whether an
iterative or ill-conditioned correction counts as converged depends on the platform's numerical
libraries in 0.5% of runs, and that decides 5 summary cells. The scalable-readout study should
therefore be read as platform dependent near its conditioning limit.

## Rerun

```powershell
gh workflow run rebuild-generators.yml
gh run list --workflow rebuild-generators.yml --limit 1
```
