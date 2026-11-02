"""Small benchmark circuit families with analytic structure."""

from __future__ import annotations

from qiskit import QuantumCircuit


def bell_circuit() -> QuantumCircuit:
    """Return a two-qubit circuit preparing |Phi+>."""

    circuit = QuantumCircuit(2, name="bell_phi_plus")
    circuit.h(0)
    circuit.cx(0, 1)
    return circuit


def ghz_circuit(width: int) -> QuantumCircuit:
    """Return a width-qubit GHZ preparation circuit."""

    if width < 2:
        raise ValueError("GHZ width must be at least two")
    circuit = QuantumCircuit(width, name=f"ghz_{width}")
    circuit.h(0)
    for target in range(1, width):
        circuit.cx(0, target)
    return circuit


def ghz_observables(width: int) -> tuple[str, ...]:
    """Return q0-first stabilizer observables for the GHZ state."""

    if width < 2:
        raise ValueError("GHZ width must be at least two")
    x_global = "X" * width
    neighboring_zz = tuple(
        "".join("Z" if qubit in (left, left + 1) else "I" for qubit in range(width))
        for left in range(width - 1)
    )
    return (x_global, *neighboring_zz)


def circuit_metadata(circuit: QuantumCircuit) -> dict[str, int]:
    """Return stable logical circuit metadata."""

    operations = circuit.count_ops()
    return {
        "width": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "logical_gate_count": sum(operations.values()),
        "logical_two_qubit_gate_count": sum(
            1 for instruction in circuit.data if len(instruction.qubits) == 2
        ),
    }
