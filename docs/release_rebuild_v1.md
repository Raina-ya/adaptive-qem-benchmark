# Clean Rebuild of the Long Generators

`rebuild-generators.yml` ran on 2026-11-09 at commit `a6a81c8` (GitHub run 42578064990). Each of the
16 long generators ran as its own job on `ubuntu-latest` with the locked dependencies
(`requirements-ci.txt`, NumPy 2.3.4). Each job deleted that generator's committed outputs in a clean
clone, regenerated them, and compared them with the committed files, which were generated on Windows.
The comparison uses relative tolerance 1e-9 and absolute tolerance 1e-12. No tolerance was changed
after the results were seen.

## Result

| Outcome | Generators |
| --- | --- |
| Reproduced within tolerance (11) | `confirmatory_v1`, `confirmatory_qaoa_v1`, `shots_to_reliability`, `generate_qaoa_selection`, `generate_decision_benchmark`, `generate_combined_qem`, `generate_selector_holdout_v2`, `generate_readout_mismatch_benchmark`, `generate_readout_mismatch_dose`, `generate_drift_policy`, `generate_runtime_report` |
| Registry error (1) | `generate_days08_12` |
| Differ from the Windows files (4) | `generate_selector_v3`, `generate_readout_feasibility_v2`, `generate_signal_v2`, `generate_readout_scalability_benchmark` |

### Registry error

`figures/rmse_heatmap.png` was declared as an output of `src.generate_days08_12`, but it is written by
`src.generate_adaptive_analysis`. The registry now assigns it to the right generator. This was a
bookkeeping error, not a reproducibility failure.

### Differences

| Generator | Columns that differ | Used by a printed number in `main(2).pdf`? |
| --- | --- | --- |
| `generate_selector_v3` | `base_combined`, `pilot_fit` in the raw repeat file | Possibly. `pilot_fit` decides when the selector switches to combined mitigation (claim G05), and `base_combined` gives the always-combined row of the selector table. The certification columns (claim G06) did not differ. |
| `generate_readout_feasibility_v2` | `condition_number` in the raw file | No. Run status did not differ, and claim G07 uses status and shot counts, not the condition number. |
| `generate_signal_v2` | `out_of_bounds_rate` in the truth table | No. No analysis reads this column; the fit RMSE used by claims G01, G04, and G09 did not differ. |
| `generate_readout_scalability_benchmark` | Run status, convergence, corrected estimates, and every summary column that depends on them | Not in `main(2).pdf`, which prints only the planned repeat count from this study (claim C09). The scalable-readout results of the earlier paper (`main.pdf`) depend on these files. |

All four involve a matrix inverse, an iterative solver, a fit, or a bound check. These can give
different floating-point results with different numerical libraries, and a small difference can
cross a threshold such as a convergence test. This is a likely explanation, not a verified one.

## What this means

- The committed results remain the Windows results. They are not replaced by the Linux results.
- The sampled pipelines whose outputs reproduced on a second operating system include every study
  behind the confirmatory, QAOA, decision, combined-QEM, shots-to-reliability, mismatch, and drift
  claims.
- The scalable-readout study is platform dependent: its run status changes between platforms. Its
  results should be read as Windows results only.
- Whether the selector-v3 numbers change has not been measured. The run reported only which columns
  differ, so the comparison tool now also reports how many rows differ and by how much. The next run
  will show the size of each difference.

## Rerun

```powershell
gh workflow run rebuild-generators.yml
gh run list --workflow rebuild-generators.yml --limit 1
```
