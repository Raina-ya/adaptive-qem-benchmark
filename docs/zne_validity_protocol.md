# ZNE Validity and Falsification Protocol

## Purpose

This protocol separates an apparent numerical correction from ZNE evidence that
survives the declared falsification checks. It applies only to the declared
simulator cells. Passing it does not establish hardware validity.

## Evidence flow

The raw C4 table preserves raw, genuine-folding, and matched-cost garbage-folding
repeat records. Every genuine result retains factor estimates, factor
uncertainties, factor counts, fit residuals, signed probability diagnostics, and
signal-retention ratios. C5 groups only matched repeats with the same circuit,
noise condition, total shot budget, and repeat identifier.

## Frozen checks

For each condition, `credible` requires all of the following:

1. The 95% normal interval lower bound of the lambda-five signal-retention mean is
   greater than `0.10`.
2. Every extrapolated Pauli expectation lies within `[-1, 1]`, allowing only
   `1e-12` numerical tolerance.
3. Maximum total negative probability mass is at most `0.05`.
4. The 95% interval lower bound of paired genuine-minus-garbage true gain is
   positive.
5. The 95% interval lower bound of genuine true gain is positive.
6. No repeat is classified `signal_destroyed`, and no more than 5% are classified
   `extrapolation_unstable`.

The normal interval is `mean +/- 1.959963984540054 * sample_standard_error` over
the 50 declared repeats. The thresholds have not changed since they were first committed.

The validity checks were named in advance, but the numerical thresholds and the reported results first appeared together in the same repository state; therefore this is not an independently timestamped preregistration.

## Outputs and interpretation

`apparent_gain_mean` records the average magnitude of the extrapolation correction.
`true_gain_mean` uses the independently known noiseless reference.
`validated_gain` equals true gain only for a `credible` cell and is missing for a
rejected cell. `failure_reasons_json` preserves every failed check; one favorable
metric cannot override another failure.

Rejected, negative-gain, unphysical, signal-destroyed, and inconclusive repeat
evidence remains in the committed raw table. Garbage folding is a negative control
and can never be selected as a mitigation method.

## Reproduction

```powershell
python -m src.run_with_log src.generate_zne_validity_gate
python -m pytest tests/test_zne_validity_gate.py -q
```
