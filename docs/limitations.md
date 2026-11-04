# Limitations and Non-goals

## Current evidence boundary

This project uses analytic calculations, exact statevector simulation, and Qiskit Aer noisy simulation. It contains no hardware-derived evidence.

## Explicit non-goals for this batch

- No machine-learning selector.
- No IBM Quantum or other real-device execution.
- No claims about a particular device, calibration cycle, or physical error rate.
- No probabilistic error cancellation, dynamical decoupling, measured device drift,
  leakage, or non-Markovian noise.

## Model limitations

The main volumetric grid assumes independent per-qubit assignment errors and idealized
depolarizing channels attached to declared simulator instructions. A separate
noise-shift holdout includes asymmetric readout, coherent over-rotation, and
correlated-ZZ controls. These are diagnostic stress tests, not calibrated device
models or evidence of hardware crosstalk.

Full assignment-matrix mitigation scales exponentially with qubit count and is included only for small benchmark circuits. An ill-conditioned calibration matrix is reported explicitly instead of being silently inverted.

Linear ZNE is limited to global folding factors 1, 3, and 5 with a preregistered
ordinary-least-squares extrapolation. The fixed total shot budget is split across
factors. No alternative noise scaling, fit order, or adaptive allocation is evaluated.

The combined-QEM study applies assignment-matrix correction separately at each noise
factor before linear extrapolation. It evaluates only two frozen allocations and six
preregistered simulator cells. Every combined validity assessment failed at least one
frozen negative-control criterion, so lower RMSE in selected cells is not treated as
evidence that the extrapolation mechanism is credible or broadly transferable.

The rule selector is a frozen, explainable baseline. Its modest holdout compatibility
and nonzero regret do not support deployment or universal method-selection claims.
