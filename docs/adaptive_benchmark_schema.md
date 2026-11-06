# Adaptive Benchmark Schema

## Repeat-level rows

Each applicable manifest row maps to exactly one immutable raw row. The identity key
is `width, depth, noise, shots, method, repeat, seed`. Status is always explicit:
`complete`, `failed`, or `not_applicable`.

The hardware-efficient benchmark stores circuit family, circuit identifier,
observable, statevector reference, estimate, raw payload, mitigation diagnostics,
logical/transpiled depth, folded two-qubit count, and execution totals. Noise-shift
rows use the same schema.

## Processed rows

`benchmark_summary_v1.csv` has one row per operating cell and method. It records bias,
sample variance, RMSE, execution overhead, mitigation diagnostics, and paired BCa
RMSE-gain intervals. Negative gains are retained.

`regime_labels_v1.csv` stores cost-aware oracle labels. `practical_tie` means method
differences are below the documented practical threshold; `inconclusive` means the
50-repeat completeness requirement was not met.

Selector evaluation rows distinguish strict label agreement from decision
compatibility. A selected method is compatible with a practical tie when it belongs
to the recorded equivalent-method set. Regret is selected-method RMSE minus oracle
RMSE and is never clipped.

## Evidence labels

- `analytic`: closed-form reference.
- `noiseless_simulated`: exact statevector computation.
- `noisy_simulated`: finite-shot simulator output.
- `hardware_derived`: reserved; no current row uses this label.
