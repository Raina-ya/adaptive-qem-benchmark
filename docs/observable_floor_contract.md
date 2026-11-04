# Observable Floor Contract

## Scope

The current benchmark supports q0-first Pauli strings containing `I`, `X`, `Y`,
and `Z`. This contract defines the analytic floor used by later ZNE
signal-retention checks. It does not infer a floor from finite-shot results.

## Definitions

For observable (O), reference (r), and declared floor (f):

- **Physical range:** the set of expectation values allowed by (O).
- **Expected noise floor:** the expectation under the declared limiting noise
  model.
- **Maximum physically attainable improvement:**
  \[
  I_{\max}=|r-f|,
  \]
  the error removable when moving from the floor back to the reference.

For an (n)-qubit Pauli string, the declared floor model is
`maximally_mixed_state_before_measurement`. Since

\[
\operatorname{Tr}(O I/2^n)=\operatorname{Tr}(O)/2^n,
\]

every non-identity Pauli has floor `0` and physical range `[-1, 1]`. The
all-identity observable has floor `1` and the singleton physical range `[1, 1]`.
This identity case prevents downstream code from assuming that all floors are
zero.

## Evidence and limitations

The physical range and maximally mixed Pauli floors are `analytic`. A supplied
circuit reference remains labelled by its actual source, normally
`noiseless_simulated` or `analytic`.

This floor is a state-level limit before measurement. Asymmetric readout error can
shift the observed measurement distribution and therefore its empirical floor.
Coherent or non-unital noise may also approach a different state. Later validity
analysis must either use this declared model or register a separate, justified
floor model; it must not silently reinterpret this value as universal.

Projectors, success probabilities, shifted observables, and hardware-derived
floors are not currently registered. They must receive explicit physical ranges
and floor assumptions before entering the ZNE validity gate.

## Reproduction

```powershell
python -m pytest tests/test_observable_floors.py -q
```
