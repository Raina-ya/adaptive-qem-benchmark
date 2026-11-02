"""Unmitigated Pauli estimation with analytic shot-noise uncertainty."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile

from src.noise_models import NoiseSpec, aer_simulator
from src.observables import expectation_from_counts, measurement_circuit


@dataclass(frozen=True)
class EstimationResult:
    """One finite-shot unmitigated estimate and its execution metadata."""

    estimate: float
    analytic_standard_error: float
    shots: int
    seed: int
    raw_counts_json: str
    logical_depth: int
    transpiled_depth: int
    transpiled_gate_count: int
    result_label: str


def analytic_pauli_standard_error(estimate: float, shots: int) -> float:
    """Plug-in standard error for the mean of a ±1 Pauli observation."""

    if shots <= 0:
        raise ValueError("Shots must be positive")
    if not -1.0 <= estimate <= 1.0:
        raise ValueError("Pauli expectation must lie in [-1, 1]")
    return math.sqrt(max(0.0, 1.0 - estimate**2) / shots)


def estimate_pauli(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    shots: int,
    seed: int,
    noise_spec: NoiseSpec | None = None,
) -> EstimationResult:
    """Estimate one Pauli observable and preserve its raw counts."""

    if shots <= 0:
        raise ValueError("Shots must be positive")
    measured = measurement_circuit(circuit, pauli_q0_first)
    simulator = aer_simulator(noise_spec)
    compiled = transpile(measured, simulator, optimization_level=0, seed_transpiler=seed)
    counts = simulator.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    normalized_counts = {str(key): int(value) for key, value in sorted(counts.items())}
    estimate = expectation_from_counts(normalized_counts, pauli_q0_first)
    return EstimationResult(
        estimate=estimate,
        analytic_standard_error=analytic_pauli_standard_error(estimate, shots),
        shots=shots,
        seed=seed,
        raw_counts_json=json.dumps(normalized_counts, sort_keys=True, separators=(",", ":")),
        logical_depth=measured.depth(),
        transpiled_depth=compiled.depth(),
        transpiled_gate_count=sum(compiled.count_ops().values()),
        result_label="noisy_simulated" if noise_spec else "noiseless_simulated",
    )
