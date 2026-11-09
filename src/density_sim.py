"""A small density-matrix simulator for the v2 grids, using enumerated NumPy operations.

It imports no project code and no Qiskit, and tests compare it with Aer. Basis integer bit i is
qubit i. Unlike the Aer runs, it supplies exact outcome probabilities, so repeats can be sampled
from them directly and a large grid stays affordable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ROOT_SEED = 20261102
IDENTITY = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
S_DAGGER = np.array([[1, 0], [0, -1j]], dtype=complex)
CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)  # Control is the first listed qubit.


def derive_seed(namespace: str, *indices: object) -> int:
    text = ":".join([str(ROOT_SEED), namespace, *(str(index) for index in indices)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def ry(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def rx(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]], dtype=complex)


@dataclass(frozen=True)
class Noise:
    """Per-gate channel strengths and readout errors for one synthetic condition."""

    name: str
    depolarizing_1q: float = 0.0
    depolarizing_2q: float = 0.0
    damping_1q: float = 0.0
    damping_2q: float = 0.0
    dephasing_1q: float = 0.0
    dephasing_2q: float = 0.0
    overrotation: float = 0.0
    detuning: float = 0.0
    readout_p01: float = 0.0
    readout_p10: float = 0.0


@dataclass(frozen=True)
class Gate:
    """One logical gate: name, qubits, and an angle where it has one."""

    name: str
    qubits: tuple[int, ...]
    angle: float = 0.0

    def inverse(self) -> Gate:
        return Gate(self.name, self.qubits, -self.angle)

    def matrix(self, overrotation: float = 0.0, detuning: float = 0.0) -> np.ndarray:
        """Return the unitary with two coherent errors on rotations.

        An over-rotation scales with the pulse, so it flips with the angle and an inverse gate
        cancels it: folding does not amplify it. A detuning keeps its sign, so it accumulates.
        """

        if self.name in ("rx", "ry", "rz"):
            angle = self.angle + math.copysign(overrotation, self.angle) + detuning
            return {"rx": rx, "ry": ry, "rz": rz}[self.name](angle)
        if self.name == "cx":
            return CNOT
        if self.name == "h":
            return HADAMARD
        if self.name == "sdg":
            return S_DAGGER
        raise ValueError(f"Unsupported gate: {self.name}")


def hardware_efficient_gates(width: int, depth: int, seed: int) -> list[Gate]:
    """Build the benchmark's layered ansatz with angles drawn from one parameter seed."""

    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, size=(depth, width, 2))
    gates: list[Gate] = []
    for layer in range(depth):
        for qubit in range(width):
            gates.append(Gate("ry", (qubit,), float(angles[layer, qubit, 0])))
            gates.append(Gate("rz", (qubit,), float(angles[layer, qubit, 1])))
        gates.extend(Gate("cx", (qubit, qubit + 1)) for qubit in range(width - 1))
        if width > 2 and layer % 2 == 1:
            gates.append(Gate("cx", (width - 1, 0)))
    return gates


def fold_global(gates: Sequence[Gate], factor: int) -> list[Gate]:
    """Return U (U^dagger U)^k for an odd noise factor 2k + 1."""

    _check_factor(factor)
    inverse = [gate.inverse() for gate in reversed(gates)]
    return [*gates, *(inverse + list(gates)) * ((factor - 1) // 2)]


def fold_local(gates: Sequence[Gate], factor: int) -> list[Gate]:
    """Return each gate G as G (G^dagger G)^k, in the original gate order."""

    _check_factor(factor)
    output: list[Gate] = []
    for gate in gates:
        output.append(gate)
        output.extend([gate.inverse(), gate] * ((factor - 1) // 2))
    return output


def garbage_fold(gates: Sequence[Gate], factor: int, seed: int) -> list[Gate]:
    """Replace each inverse layer with random angles, a control that is not unitary-equivalent."""

    _check_factor(factor)
    rng = np.random.default_rng(seed)
    output = list(gates)
    for _ in range((factor - 1) // 2):
        for gate in reversed(gates):
            if gate.name in ("rx", "ry", "rz"):
                output.append(Gate(gate.name, gate.qubits, float(rng.uniform(-np.pi, np.pi))))
            else:
                output.append(gate)
        output.extend(gates)
    return output


def _check_factor(factor: int) -> None:
    if factor < 1 or factor % 2 == 0:
        raise ValueError("Noise factors must be positive odd integers")


def basis_rotation(pauli: str) -> list[Gate]:
    """Rotate each measured qubit into its Pauli basis; pauli lists qubit 0 first."""

    gates: list[Gate] = []
    for qubit, symbol in enumerate(pauli):
        if symbol == "X":
            gates.append(Gate("h", (qubit,)))
        elif symbol == "Y":
            gates.extend([Gate("sdg", (qubit,)), Gate("h", (qubit,))])
    return gates


def _apply_operator(rho: np.ndarray, operator: np.ndarray, qubits: tuple[int, ...], width: int):
    """Compute K rho K^dagger, with qubit i on tensor axes width-1-i and 2*width-1-i."""

    count = len(qubits)
    matrix = operator.reshape((2,) * (2 * count))
    # The first listed qubit is the operator's most significant index.
    kets = [width - 1 - qubit for qubit in qubits]
    bras = [2 * width - 1 - qubit for qubit in qubits]
    output = np.tensordot(matrix, rho, axes=(range(count, 2 * count), kets))
    output = np.moveaxis(output, range(count), kets)
    conjugate = matrix.conj()
    output = np.tensordot(conjugate, output, axes=(range(count, 2 * count), bras))
    return np.moveaxis(output, range(count), bras)


def _kraus(rho: np.ndarray, operators: Sequence[np.ndarray], qubit: int, width: int) -> np.ndarray:
    return sum(_apply_operator(rho, item, (qubit,), width) for item in operators)


def _depolarize_1q(rho: np.ndarray, p: float, qubit: int, width: int) -> np.ndarray:
    """Apply (1-p) rho + (p/4) sum over all four Paulis of P rho P."""

    if p == 0.0:
        return rho
    terms = sum(_apply_operator(rho, item, (qubit,), width) for item in (PAULI_X, PAULI_Y, PAULI_Z))
    return (1.0 - 3.0 * p / 4.0) * rho + (p / 4.0) * terms


def _depolarize_2q(rho: np.ndarray, p: float, qubits: tuple[int, int], width: int) -> np.ndarray:
    """Apply (1-p) rho + (p/16) sum over all sixteen two-qubit Paulis of P rho P."""

    if p == 0.0:
        return rho
    total = np.zeros_like(rho)
    paulis = (IDENTITY, PAULI_X, PAULI_Y, PAULI_Z)
    for left in paulis:
        applied = _apply_operator(rho, left, (qubits[0],), width)
        for right in paulis:
            total = total + _apply_operator(applied, right, (qubits[1],), width)
    return (1.0 - p) * rho + (p / 16.0) * total


def _damp(rho: np.ndarray, gamma: float, qubit: int, width: int) -> np.ndarray:
    if gamma == 0.0:
        return rho
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - gamma)]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return _kraus(rho, (k0, k1), qubit, width)


def _dephase(rho: np.ndarray, p: float, qubit: int, width: int) -> np.ndarray:
    """Apply a phase flip with probability p/2."""

    if p == 0.0:
        return rho
    flipped = _apply_operator(rho, PAULI_Z, (qubit,), width)
    return (1.0 - p / 2.0) * rho + (p / 2.0) * flipped


def run(gates: Sequence[Gate], width: int, noise: Noise) -> np.ndarray:
    """Run gates from |0...0>, applying the noise channels after each gate."""

    rho = np.zeros((2,) * (2 * width), dtype=complex)
    rho[(0,) * (2 * width)] = 1.0
    for gate in gates:
        rho = _apply_operator(
            rho, gate.matrix(noise.overrotation, noise.detuning), gate.qubits, width
        )
        if len(gate.qubits) == 1:
            qubit = gate.qubits[0]
            rho = _depolarize_1q(rho, noise.depolarizing_1q, qubit, width)
            rho = _damp(rho, noise.damping_1q, qubit, width)
            rho = _dephase(rho, noise.dephasing_1q, qubit, width)
        else:
            rho = _depolarize_2q(rho, noise.depolarizing_2q, gate.qubits, width)
            for qubit in gate.qubits:
                rho = _damp(rho, noise.damping_2q, qubit, width)
                rho = _dephase(rho, noise.dephasing_2q, qubit, width)
    return rho


def diagonal(rho: np.ndarray, width: int) -> np.ndarray:
    """Return outcome probabilities indexed by basis integer."""

    size = 2**width
    probabilities = np.real(np.diag(rho.reshape(size, size))).copy()
    probabilities[probabilities < 0.0] = 0.0
    return probabilities / probabilities.sum()


def assignment_matrix(width: int, p01: float, p10: float) -> np.ndarray:
    """Return P(observed | prepared) for independent, identical readout on every qubit."""

    single = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]])
    matrix = np.array([[1.0]])
    for _ in range(width):
        matrix = np.kron(single, matrix)
    return matrix


def outcome_probabilities(
    gates: Sequence[Gate], pauli: str, noise: Noise, *, apply_readout: bool = True
) -> np.ndarray:
    """Return measured-basis probabilities after noisy rotation and readout."""

    width = len(pauli)
    probabilities = diagonal(run([*gates, *basis_rotation(pauli)], width, noise), width)
    if not apply_readout or (noise.readout_p01 == 0.0 and noise.readout_p10 == 0.0):
        return probabilities
    return assignment_matrix(width, noise.readout_p01, noise.readout_p10) @ probabilities


def parity_signs(pauli: str) -> np.ndarray:
    """Return the +1 or -1 eigenvalue of the Pauli's support for each basis integer."""

    return np.array(
        [
            1.0
            if sum((index >> qubit) & 1 for qubit, s in enumerate(pauli) if s != "I") % 2 == 0
            else -1.0
            for index in range(2 ** len(pauli))
        ]
    )


def ideal_expectation(gates: Sequence[Gate], pauli: str) -> float:
    """Return the noiseless value by statevector, separate from the density-matrix path."""

    width = len(pauli)
    state = np.zeros((2,) * width, dtype=complex)
    state[(0,) * width] = 1.0
    for gate in [*gates, *basis_rotation(pauli)]:
        axes = [width - 1 - qubit for qubit in gate.qubits]
        count = len(gate.qubits)
        matrix = gate.matrix().reshape((2,) * (2 * count))
        state = np.moveaxis(
            np.tensordot(matrix, state, axes=(range(count, 2 * count), axes)), range(count), axes
        )
    return float(np.abs(state.reshape(-1)) ** 2 @ parity_signs(pauli))
