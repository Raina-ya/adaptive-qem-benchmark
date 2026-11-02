"""Deterministic hardware-efficient benchmark circuits."""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

from src.config import derive_seed


def hardware_efficient_circuit(width: int, depth: int) -> QuantumCircuit:
    """Create a fixed-seed rotation-and-entanglement circuit."""

    if width < 2:
        raise ValueError("Circuit width must be at least two")
    if depth < 1:
        raise ValueError("Circuit depth parameter must be positive")
    rng = np.random.default_rng(derive_seed("hardware-efficient", width, depth))
    angles = rng.uniform(-np.pi, np.pi, size=(depth, width, 2))
    circuit = QuantumCircuit(width, name=f"he_w{width}_d{depth}")
    for layer in range(depth):
        for qubit in range(width):
            circuit.ry(float(angles[layer, qubit, 0]), qubit)
            circuit.rz(float(angles[layer, qubit, 1]), qubit)
        for qubit in range(width - 1):
            circuit.cx(qubit, qubit + 1)
        if width > 2 and layer % 2 == 1:
            circuit.cx(width - 1, 0)
    return circuit


def hardware_efficient_observable(width: int) -> str:
    """Return a fixed q0-first Pauli observable with X and Z support."""

    if width < 2:
        raise ValueError("Observable width must be at least two")
    return "".join("X" if qubit % 2 == 0 else "Z" for qubit in range(width))


def parameter_signature(circuit: QuantumCircuit) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Expose numeric gate parameters for deterministic-regeneration tests."""

    signature = []
    for instruction in circuit.data:
        values = tuple(float(value) for value in instruction.operation.params)
        signature.append((instruction.operation.name, values))
    return tuple(signature)
