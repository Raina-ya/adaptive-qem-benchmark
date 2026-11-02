# Day 8-12 Runtime Summary

| Stage | Day | Wall seconds | Cells | Simulations |
|---|---:|---:|---:|---:|
| day08_artifacts | 8 | 4.950 | 9 | 0 |
| day09_artifacts | 9 | 15.343 | 60 | 180 |
| day10_manifest | 10 | 2.508 | 16200 | 0 |
| day11_first_run | 11 | 103.800 | 240 | 3120 |
| readout_optimization | 11 | 3.700 | 0 | 0 |
| day11_rerun | 11 | 28.000 | 240 | 3120 |
| day12_aborted | 12 | 306.000 |  |  |
| day12_final | 12 | 942.000 | 5400 | 59400 |
| pytest | all | 54.370 |  |  |
| ruff | all | 6.600 |  |  |

- The final Day 12 GHZ run dominated measured wall time at 942.0 seconds.
- Readout calibration optimization reduced the comparable Day 11 pilot from 103.8 to 28.0 seconds (73.0% lower wall time).
- Known completed workloads account for 65,820 logical sampling executions; the aborted Day 12 attempt is excluded because it wrote no checkpoint.
- Recorded wall time was 95.1% simulation, 0.8% artifact/code preparation, and 4.2% testing and linting.
- ZNE charges three folded-circuit samples per result while preserving the same total shot budget.
- The final benchmark retained 5,372 complete rows and 28 explicit failures; all failures were width-6, 1,024-shot readout corrections with ill-conditioned sampled assignment matrices (25 high-noise and 3 medium-noise).
- Calibration acceleration changed execution strategy, not the declared readout channel, seeds, shot accounting, or failure policy.
