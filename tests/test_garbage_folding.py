import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from src.hardware_efficient_circuits import hardware_efficient_circuit
from src.mitigation.folding import fold_global
from src.mitigation.garbage_folding import garbage_fold


@pytest.mark.parametrize("factor", [1, 3, 5])
def test_garbage_and_genuine_folding_use_equal_copy_cost(factor: int) -> None:
    circuit = hardware_efficient_circuit(2, 3)
    genuine = fold_global(circuit, factor)
    garbage = garbage_fold(circuit, factor)
    assert len(genuine.data) == len(garbage.data) == factor * len(circuit.data)


def test_factor_one_is_the_common_ideal_baseline() -> None:
    circuit = hardware_efficient_circuit(2, 3)
    assert np.allclose(Operator(garbage_fold(circuit, 1)), Operator(circuit), atol=1e-12)


@pytest.mark.parametrize("factor", [3, 5])
def test_garbage_folding_does_not_preserve_target_unitary(factor: int) -> None:
    circuit = hardware_efficient_circuit(2, 3)
    assert not np.allclose(
        Operator(garbage_fold(circuit, factor)).data,
        Operator(circuit).data,
        atol=1e-12,
    )


@pytest.mark.parametrize("factor", [0, 2, -1, 1.5])
def test_invalid_factors_fail_explicitly(factor) -> None:
    with pytest.raises(ValueError, match="positive odd integer"):
        garbage_fold(hardware_efficient_circuit(2, 3), factor)


def test_measurement_circuit_is_rejected() -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    with pytest.raises(ValueError, match="unitary"):
        garbage_fold(circuit, 3)
