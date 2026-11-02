import json

import numpy as np
from qiskit import QuantumCircuit

from src.config import derive_seed
from src.estimators.unmitigated import analytic_pauli_standard_error, estimate_pauli


def test_pauli_estimate_range_and_raw_count_preservation() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    result = estimate_pauli(circuit, "Z", shots=512, seed=derive_seed("range"))
    assert -1.0 <= result.estimate <= 1.0
    assert sum(json.loads(result.raw_counts_json).values()) == 512


def test_analytic_standard_error_matches_repeated_sampling() -> None:
    circuit = QuantumCircuit(1)
    circuit.h(0)
    shots = 256
    estimates = np.array(
        [
            estimate_pauli(
                circuit, "Z", shots=shots, seed=derive_seed("se-repeat", repeat)
            ).estimate
            for repeat in range(120)
        ]
    )
    empirical = float(estimates.std(ddof=1))
    analytic = analytic_pauli_standard_error(0.0, shots)
    assert abs(empirical - analytic) < 0.012
