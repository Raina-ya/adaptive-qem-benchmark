"""Minimal Bell-state exact and finite-shot experiment."""

from __future__ import annotations

from dataclasses import asdict

from src.circuits import bell_circuit
from src.config import DEFAULT_SHOTS, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.observables import exact_expectation


def run_minimal_example(shots: int = DEFAULT_SHOTS, repeat: int = 0) -> dict[str, object]:
    """Return one reproducible Bell-state XX experiment row."""

    circuit = bell_circuit()
    observable = "XX"
    result = estimate_pauli(
        circuit,
        observable,
        shots=shots,
        seed=derive_seed("minimal-bell-xx", repeat),
    )
    return {
        "repeat": repeat,
        "circuit_id": "bell_phi_plus",
        "observable_q0_first": observable,
        "analytic_reference": 1.0,
        "noiseless_statevector_reference": exact_expectation(circuit, observable),
        "result_label": "noiseless_simulated",
        **asdict(result),
    }
