"""Pauli-observable conventions and count-based estimators."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

VALID_PAULIS = frozenset("IXYZ")


def validate_pauli(pauli: str, width: int | None = None) -> None:
    """Validate a q0-first Pauli string."""

    if not pauli or any(symbol not in VALID_PAULIS for symbol in pauli):
        raise ValueError("Pauli strings must contain only I, X, Y, and Z")
    if width is not None and len(pauli) != width:
        raise ValueError(f"Pauli width {len(pauli)} does not match circuit width {width}")


def qiskit_pauli(pauli_q0_first: str) -> SparsePauliOp:
    """Convert the project q0-first convention to Qiskit's qN...q0 label."""

    validate_pauli(pauli_q0_first)
    return SparsePauliOp.from_list([(pauli_q0_first[::-1], 1.0)])


def exact_expectation(circuit: QuantumCircuit, pauli_q0_first: str) -> float:
    """Calculate a noiseless statevector expectation value."""

    validate_pauli(pauli_q0_first, circuit.num_qubits)
    value = Statevector.from_instruction(circuit).expectation_value(
        qiskit_pauli(pauli_q0_first)
    )
    return float(np.real_if_close(value).real)


def measurement_circuit(circuit: QuantumCircuit, pauli_q0_first: str) -> QuantumCircuit:
    """Rotate into a Pauli basis and measure every qubit."""

    validate_pauli(pauli_q0_first, circuit.num_qubits)
    measured = circuit.copy()
    for qubit, symbol in enumerate(pauli_q0_first):
        if symbol == "X":
            measured.h(qubit)
        elif symbol == "Y":
            measured.sdg(qubit)
            measured.h(qubit)
    measured.measure_all()
    return measured


def _clean_bitstring(bitstring: str) -> str:
    return bitstring.replace(" ", "")


def parity_eigenvalue(bitstring: str, pauli_q0_first: str) -> int:
    """Return the ±1 eigenvalue for a measured computational-basis outcome."""

    bits = _clean_bitstring(bitstring)
    validate_pauli(pauli_q0_first, len(bits))
    parity = 0
    for qubit, symbol in enumerate(pauli_q0_first):
        if symbol != "I":
            parity ^= int(bits[-1 - qubit])
    return 1 if parity == 0 else -1


def expectation_from_counts(counts: Mapping[str, int], pauli_q0_first: str) -> float:
    """Estimate a Pauli expectation from measurement-basis counts."""

    shots = sum(counts.values())
    if shots <= 0:
        raise ValueError("Counts must contain at least one shot")
    total = sum(
        parity_eigenvalue(bitstring, pauli_q0_first) * count
        for bitstring, count in counts.items()
    )
    return total / shots


def expectation_from_probabilities(
    probabilities: np.ndarray, pauli_q0_first: str
) -> float:
    """Evaluate a Pauli expectation from probabilities indexed by basis integer."""

    probabilities = np.asarray(probabilities, dtype=float)
    width = int(np.log2(probabilities.size))
    if 2**width != probabilities.size:
        raise ValueError("Probability vector length must be a power of two")
    validate_pauli(pauli_q0_first, width)
    eigenvalues = np.empty(probabilities.size)
    for state in range(probabilities.size):
        parity = 0
        for qubit, symbol in enumerate(pauli_q0_first):
            if symbol != "I":
                parity ^= (state >> qubit) & 1
        eigenvalues[state] = 1.0 if parity == 0 else -1.0
    return float(eigenvalues @ probabilities)
