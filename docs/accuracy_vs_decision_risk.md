# Marginal accuracy is not decision reliability

For a two-candidate decision, marginal expectation RMSE does not determine the
error of the estimated gap. The relevant variance is
`Var(A - B) = Var(A) + Var(B) - 2 Cov(A, B)`, and bias can move the two
candidates in opposite directions.

The regression test uses an exact gap of `0.02`. Estimator X gives each candidate
an absolute error of `0.03`, but the errors point toward each other and reverse
the winner. Estimator Y gives each candidate an absolute error of `0.06`, but the
common directional shift preserves the winner. Thus X has lower marginal RMSE
and higher decision-error probability. This is a deterministic mathematical
counterexample, not simulator evidence.

For the simulator benchmark, a repeat is a practical tie when the absolute
estimated gap is at most half the exact margin. A cell is labelled unresolved
when its empirical 95% gap interval crosses zero or the upper Wilson bound for
decision-error probability exceeds `0.20`. These thresholds were fixed before
the benchmark was run.

`shared_records` uses the same derived random-number seed for both candidates.
For readout mitigation this also produces identical calibration records; for ZNE
it pairs the random streams at each folding factor. `independent_records` derives
candidate-specific seeds. This is a common-random-number simulator design, not a
claim that two incompatible hardware observables can always share physical shots.

Regenerate the complete preregistered benchmark with:

```powershell
python -m src.run_with_log src.generate_decision_benchmark
```
