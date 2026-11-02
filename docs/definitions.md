# Definitions and Experiment Contract

## Core terms

- **Shot:** One sampled execution of one measured circuit.
- **Repeat:** One independent estimator run with a seed derived from the root seed.
- **Benchmark cell:** One fixed combination of circuit, observable, noise, mitigation method, and total shot budget.
- **Total shot budget:** Every target, calibration, folded, or otherwise auxiliary shot charged to one benchmark cell.
- **Logical depth:** Circuit depth before backend transpilation.
- **Transpiled depth:** Depth of the circuit actually submitted to the simulator.

## Result labels

- `analytic`: obtained from a closed-form relation.
- `noiseless_simulated`: obtained from exact statevector simulation.
- `noisy_simulated`: sampled from an explicitly recorded simulator noise model.
- `hardware_derived`: reserved for actual quantum-device results; absent in the current project.

## Seed policy

The root seed is `20261102`. Every stochastic operation receives a deterministic child seed derived from a stable namespace and integer indices. Python's process-randomized `hash()` is never used for seed derivation.

## Fixed-budget policy

Unmitigated estimation spends the entire budget on the target circuit. Readout mitigation spends 20% on calibration and 80% on the target circuit. Calibration shots are divided deterministically across all computational-basis calibration states.

## Readout-correction algorithm

Readout mitigation solves the non-negative least-squares problem
`min ||A p - p_observed||_2` with `p >= 0`, then normalizes `p` to unit sum. The
implementation uses fixed-order Lawson-Hanson active-set updates, `math.fsum`
inner products, and deterministic partial-pivot Gaussian elimination. It does
not call a BLAS/LAPACK optimizer. Ill-conditioning is classified with a
fixed-order infinity-norm condition number computed from deterministic
Gauss-Jordan inversion; matrices above `1e8` return `ill_conditioned` without a
corrected estimate.

## Data policy

Raw counts, raw estimates, corrected estimates, resource use, seed, circuit metadata, and method status are retained. Aggregation creates new files and never overwrites raw observations.
