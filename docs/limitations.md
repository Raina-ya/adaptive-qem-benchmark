# Limitations and Non-goals

## Current evidence boundary

This project uses analytic calculations, exact statevector simulation, and Qiskit Aer noisy simulation. It contains no hardware-derived evidence.

## Explicit non-goals for this batch

- No machine-learning selector.
- No IBM Quantum or other real-device execution.
- No claims about a particular device, calibration cycle, or physical error rate.
- No zero-noise extrapolation yet.
- No probabilistic error cancellation, dynamical decoupling, crosstalk, drift, leakage, or non-Markovian noise.

## Model limitations

The readout model assumes independent per-qubit assignment errors. Gate noise uses idealized depolarizing channels attached to declared simulator instructions. These models are diagnostic controls, not calibrated device models.

Full assignment-matrix mitigation scales exponentially with qubit count and is included only for small benchmark circuits. An ill-conditioned calibration matrix is reported explicitly instead of being silently inverted.
