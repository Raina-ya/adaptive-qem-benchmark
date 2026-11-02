import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from src.circuits import bell_circuit
from src.mitigation.folding import fold_global, folded_metadata


@pytest.mark.parametrize("factor", [1, 3, 5])
def test_global_folding_preserves_ideal_unitary(factor: int) -> None:
    original = Operator(bell_circuit()).data
    folded = Operator(fold_global(bell_circuit(), factor)).data
    overlap = np.vdot(original.ravel(), folded.ravel())
    phase = overlap / abs(overlap)
    np.testing.assert_allclose(folded, phase * original, atol=1e-12)


def test_folded_resources_grow_with_factor() -> None:
    rows = [folded_metadata(bell_circuit(), factor, seed=17) for factor in (1, 3, 5)]
    assert [row["logical_depth"] for row in rows] == sorted(row["logical_depth"] for row in rows)
    assert [row["transpiled_two_qubit_gate_count"] for row in rows] == [1, 3, 5]


@pytest.mark.parametrize("factor", [0, 2, 4, -1])
def test_unsupported_factor_fails_explicitly(factor: int) -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        fold_global(bell_circuit(), factor)


def test_measurement_folding_fails_explicitly() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.raises(ValueError, match="unitary circuits"):
        fold_global(circuit, 3)
