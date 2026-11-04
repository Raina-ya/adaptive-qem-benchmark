"""Matched-copy negative control for genuine global folding."""

from __future__ import annotations

from qiskit import QuantumCircuit


def garbage_fold(circuit: QuantumCircuit, noise_factor: int) -> QuantumCircuit:
    """Append inverse copies without restoring the target unitary.

    The result contains the same number of full circuit copies as genuine global
    folding, but deliberately does not implement ``U(U†U)^n``.
    """

    if not isinstance(noise_factor, int) or noise_factor < 1 or noise_factor % 2 == 0:
        raise ValueError("Garbage folding requires a positive odd integer noise factor")
    if circuit.num_clbits or any(
        instruction.operation.name in {"measure", "reset"} for instruction in circuit.data
    ):
        raise ValueError("Only unitary circuits without classical bits are supported")
    try:
        inverse = circuit.inverse()
    except Exception as error:
        raise ValueError("Circuit contains an unsupported non-unitary instruction") from error

    folded = circuit.copy()
    for _ in range(noise_factor - 1):
        folded.compose(inverse, inplace=True)
    folded.name = f"{circuit.name}_garbage_fold{noise_factor}"
    return folded
