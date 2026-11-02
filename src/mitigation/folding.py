"""Deterministic global unitary folding for odd noise factors."""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def fold_global(circuit: QuantumCircuit, noise_factor: int) -> QuantumCircuit:
    """Return U(U†U)^n for an odd positive integer noise factor."""

    if not isinstance(noise_factor, int) or noise_factor < 1 or noise_factor % 2 == 0:
        raise ValueError("Global folding requires a positive odd integer noise factor")
    if circuit.num_clbits or any(
        instruction.operation.name in {"measure", "reset"} for instruction in circuit.data
    ):
        raise ValueError("Only unitary circuits without classical bits are supported")
    try:
        inverse = circuit.inverse()
    except Exception as error:
        raise ValueError("Circuit contains an unsupported non-unitary instruction") from error

    folded = circuit.copy()
    for _ in range((noise_factor - 1) // 2):
        folded.compose(inverse, inplace=True)
        folded.compose(circuit, inplace=True)
    folded.name = f"{circuit.name}_fold{noise_factor}"
    return folded


def folded_metadata(circuit: QuantumCircuit, noise_factor: int, *, seed: int) -> dict[str, int]:
    """Return actual simulator-transpiled resource counts for one fold."""

    folded = fold_global(circuit, noise_factor)
    compiled = transpile(folded, AerSimulator(), optimization_level=0, seed_transpiler=seed)
    return {
        "noise_factor": noise_factor,
        "logical_depth": folded.depth(),
        "transpiled_depth": compiled.depth(),
        "transpiled_two_qubit_gate_count": sum(
            count
            for gate, count in compiled.count_ops().items()
            if gate in {"cx", "cz", "ecr", "swap"}
        ),
    }
