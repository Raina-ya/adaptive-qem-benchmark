# QAOA Candidate-Selection Application Protocol

## Scope

This track is a simulator-only MaxCut/QAOA application case study. It does not use
machine learning, quantum hardware, hardware calibration data, or noisy outcomes to
construct application instances. Phase F-A defines only exact references and ideal
circuit primitives; it does not execute the QEM benchmark.

## Frozen graph foundations

The application starts with three deterministic simple graphs:

- `ring4`: a four-vertex cycle;
- `sparse6`: a connected six-vertex graph whose additional edges are selected by
  root-derived deterministic edge scores;
- `medium_dense8`: a connected eight-vertex graph with a deterministic medium-density
  edge set.

Edges are immutable, unique, sorted, and stored as `(lower_vertex, upper_vertex)`.
The graph seed is derived from root seed `20261102`; Python's randomized `hash()` is
never used.

## Independent classical reference

For a computational-basis state, one shot yields one complete bitstring. Its cut
value is the number of graph edges whose endpoint bits differ. All edge contributions
come from that same shot; edges do not receive independent or uncharged shots.

Exact MaxCut enumerates only states with the highest-index qubit fixed to zero, which
removes the global complement symmetry. After finding the optimum, the implementation
adds each complementary state back and returns every degenerate optimum. The exact
classical optimum is distinct from every QAOA expectation value.

## Qiskit ordering and QAOA convention

Basis integers use Qiskit's little-endian qubit convention. Displayed and sampled
bitstrings are `q[n-1]...q[0]`, so the rightmost character is `q0`.

The QAOA cost function is

`C = sum_(i,j in E) (I - Z_i Z_j) / 2`.

Each layer applies `rzz(-gamma)` on every canonical edge and `rx(2*beta)` on every
qubit after an initial uniform superposition. The interface accepts equal-length
beta and gamma sequences, although Phase F-A validates the `p=1` foundation.

## Evidence labels

- `analytic`: exhaustive classical MaxCut optimum and optimal bitstrings;
- `noiseless_simulated`: exact statevector QAOA probabilities and expected cut;
- `noisy_simulated`: reserved for later finite-shot simulator runs;
- `hardware_derived`: absent from this project.

Expected cut from a probability vector is an exact statevector-derived quantity.
Expected cut from counts is a finite-shot estimator. These values must not be
described interchangeably.

## Current limitations

The instances have at most eight vertices and remain classically trivial. They test
accounting, ranking, and mitigation behavior rather than quantum advantage. Phase
F-A does not define candidate parameters, selection metrics, confidence rules, noise
models, or mitigation budgets; those decisions must be frozen before noisy outcomes
are inspected.
