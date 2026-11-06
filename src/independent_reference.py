"""A second, deliberately plain implementation of the benchmark's core calculations.

Nothing here imports project code or Qiskit. Everything is written out by enumeration, so a
shared helper or a misread library convention cannot make both paths agree by construction.

Conventions restated once: basis integer bit i is qubit i, a measured bitstring is written with
qubit 0 rightmost, and Pauli strings list qubit 0 first. `reverse_bits=True` exists only so the
tests can break this convention on purpose.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import nnls

ROOT_SEED = 20261102
SQRT_HALF = 1.0 / math.sqrt(2.0)
NNLS_TOLERANCE = 1e-13
NNLS_ENTRY_STEP = 1e-12


def derive_seed(namespace: str, *indices: int) -> int:
    """Repeat the published rule: first four SHA-256 bytes of the colon-joined parts."""

    text = ":".join([str(ROOT_SEED), namespace, *(str(index) for index in indices)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def zero_state(width: int) -> np.ndarray:
    state = np.zeros(2**width, dtype=complex)
    state[0] = 1.0
    return state


def apply_one_qubit(state: np.ndarray, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2x2 gate by visiting every amplitude pair."""

    output = state.copy()
    mask = 1 << qubit
    for index in range(state.size):
        if index & mask:
            continue
        low, high = state[index], state[index | mask]
        output[index] = gate[0, 0] * low + gate[0, 1] * high
        output[index | mask] = gate[1, 0] * low + gate[1, 1] * high
    return output


def apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    output = state.copy()
    for index in range(state.size):
        if (index >> control) & 1:
            output[index] = state[index ^ (1 << target)]
    return output


HADAMARD = np.array([[SQRT_HALF, SQRT_HALF], [SQRT_HALF, -SQRT_HALF]], dtype=complex)


def ry(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)


def rz(angle: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]], dtype=complex)


def rx(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle / 2.0), math.sin(angle / 2.0)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=complex)


def ghz_state(width: int) -> np.ndarray:
    """Write the GHZ state directly instead of simulating its circuit."""

    state = np.zeros(2**width, dtype=complex)
    state[0] = state[-1] = SQRT_HALF
    return state


def ghz_state_from_gates(width: int) -> np.ndarray:
    state = apply_one_qubit(zero_state(width), HADAMARD, 0)
    for target in range(1, width):
        state = apply_cnot(state, 0, target)
    return state


def hardware_efficient_state(width: int, depth: int) -> np.ndarray:
    """Simulate the fixed-seed layered circuit gate by gate."""

    rng = np.random.default_rng(derive_seed("hardware-efficient", width, depth))
    angles = rng.uniform(-np.pi, np.pi, size=(depth, width, 2))
    state = zero_state(width)
    for layer in range(depth):
        for qubit in range(width):
            state = apply_one_qubit(state, ry(float(angles[layer, qubit, 0])), qubit)
            state = apply_one_qubit(state, rz(float(angles[layer, qubit, 1])), qubit)
        for qubit in range(width - 1):
            state = apply_cnot(state, qubit, qubit + 1)
        if width > 2 and layer % 2 == 1:
            state = apply_cnot(state, width - 1, 0)
    return state


def observable_for_hardware_efficient(width: int) -> str:
    return "".join("X" if qubit % 2 == 0 else "Z" for qubit in range(width))


def pauli_expectation(state: np.ndarray, pauli: str, *, reverse_bits: bool = False) -> float:
    """Evaluate <state|P|state> by applying each Pauli to every amplitude."""

    width = len(pauli)
    symbols = pauli[::-1] if reverse_bits else pauli
    applied = state.copy()
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    for qubit in range(width):
        if symbols[qubit] != "I":
            applied = apply_one_qubit(applied, paulis[symbols[qubit]], qubit)
    return float(np.real(np.vdot(state, applied)))


def _bit_value(bitstring: str, qubit: int, reverse_bits: bool) -> int:
    return int(bitstring[qubit] if reverse_bits else bitstring[-1 - qubit])


def expectation_from_counts(
    counts: Mapping[str, int], pauli: str, *, reverse_bits: bool = False
) -> float:
    """Average a Pauli parity over shots from fixed measurement-basis counts."""

    shots = sum(counts.values())
    total = 0
    for bitstring, count in counts.items():
        parity = sum(
            _bit_value(bitstring, qubit, reverse_bits)
            for qubit, symbol in enumerate(pauli)
            if symbol != "I"
        )
        total += count if parity % 2 == 0 else -count
    return total / shots


def probabilities_from_counts(
    counts: Mapping[str, int], width: int, *, reverse_bits: bool = False
) -> np.ndarray:
    probabilities = np.zeros(2**width)
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        index = int(bitstring[::-1] if reverse_bits else bitstring, 2)
        probabilities[index] += count / shots
    return probabilities


def expectation_from_probabilities(probabilities: np.ndarray, pauli: str) -> float:
    total = 0.0
    for index, probability in enumerate(probabilities):
        parity = sum((index >> qubit) & 1 for qubit, symbol in enumerate(pauli) if symbol != "I")
        total += probability if parity % 2 == 0 else -probability
    return total


def assignment_matrix(
    records: Sequence[Mapping[str, object]],
    width: int,
    *,
    transpose: bool = False,
    reverse_bits: bool = False,
) -> np.ndarray:
    """Build P(observed row | prepared column) by enumerating calibration counts."""

    size = 2**width
    matrix = np.zeros((size, size))
    for record in records:
        prepared = int(record["prepared_state"])
        counts = record["counts"]
        shots = sum(counts.values())
        for bitstring, count in counts.items():
            observed = int(bitstring[::-1] if reverse_bits else bitstring, 2)
            matrix[observed, prepared] += count / shots
    return matrix.T if transpose else matrix


def _optimality_gaps(
    matrix: np.ndarray, observed: np.ndarray, x: np.ndarray
) -> tuple[float, float]:
    """Return the largest KKT violations: off-support descent and on-support gradient."""

    gradient = matrix.T @ (matrix @ x - observed)
    support = x > 0.0
    off = float(-gradient[~support].min()) if (~support).any() else 0.0
    on = float(np.abs(gradient[support]).max()) if support.any() else 0.0
    return max(off, 0.0), on


def solve_nnls(matrix: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Solve non-negative least squares and refuse any answer that fails its KKT conditions.

    SciPy supplies the starting support only. It returned a non-optimal point on some committed
    inputs, so active-set steps continue from there until optimality is certified.
    """

    x, _ = nnls(matrix, observed, maxiter=100 * matrix.shape[1])
    gram, cross = matrix.T @ matrix, matrix.T @ observed
    for _ in range(50 * matrix.shape[1]):
        support = np.flatnonzero(x > 0.0)
        trial = np.zeros_like(x)
        trial[support] = np.linalg.solve(gram[np.ix_(support, support)], cross[support])
        if (trial[support] <= 0.0).any():
            shrinking = support[trial[support] <= 0.0]
            step = float(np.min(x[shrinking] / (x[shrinking] - trial[shrinking])))
            x = x + step * (trial - x)
            x[shrinking[x[shrinking] <= 1e-15]] = 0.0
            continue
        x = trial
        off, on = _optimality_gaps(matrix, observed, x)
        if off <= NNLS_TOLERANCE and on <= NNLS_TOLERANCE:
            return x
        gradient = cross - gram @ x
        x = x.copy()
        entering = int(np.argmax(np.where(x > 0.0, -np.inf, gradient)))
        x[entering] = NNLS_ENTRY_STEP
    raise RuntimeError("Reference NNLS did not reach certified optimality")


def correct_readout(matrix: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Solve the constrained correction, then normalize it to one."""

    solution = solve_nnls(matrix, observed)
    return solution / solution.sum()


def infinity_condition_number(matrix: np.ndarray) -> float:
    inverse = np.linalg.inv(matrix)
    return float(np.abs(matrix).sum(axis=1).max() * np.abs(inverse).sum(axis=1).max())


def linear_intercept(factors: Sequence[float], estimates: Sequence[float]) -> float:
    """Return the ordinary-least-squares value at factor zero from closed-form sums."""

    count = len(factors)
    mean_factor = math.fsum(factors) / count
    mean_estimate = math.fsum(estimates) / count
    numerator = math.fsum(
        (factor - mean_factor) * (estimate - mean_estimate)
        for factor, estimate in zip(factors, estimates, strict=True)
    )
    denominator = math.fsum((factor - mean_factor) ** 2 for factor in factors)
    return mean_estimate - (numerator / denominator) * mean_factor


def split_evenly(total: int, parts: int) -> tuple[int, ...]:
    """Split shots with the remainder going to the earliest parts."""

    base, extra = divmod(total, parts)
    return tuple(base + (1 if index < extra else 0) for index in range(parts))


def seeded_edges(
    width: int, family_code: int, density: float, mandatory: set[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    edges = set(mandatory)
    for left in range(width):
        for right in range(left + 1, width):
            score = derive_seed("maxcut-graph-edge", width, family_code, left, right) / 2**32
            if score < density:
                edges.add((left, right))
    return tuple(sorted(edges))


def benchmark_graphs() -> dict[str, tuple[int, tuple[tuple[int, int], ...]]]:
    return {
        "ring4": (4, ((0, 1), (0, 3), (1, 2), (2, 3))),
        "sparse6": (6, seeded_edges(6, 1, 0.12, {(v, v + 1) for v in range(5)})),
        "medium_dense8": (
            8,
            seeded_edges(8, 2, 0.50, {*((v, v + 1) for v in range(7)), (0, 7)}),
        ),
    }


def cut_of_index(index: int, edges: Sequence[tuple[int, int]]) -> int:
    return sum(((index >> left) & 1) != ((index >> right) & 1) for left, right in edges)


def cut_table(width: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    return np.array([cut_of_index(index, edges) for index in range(2**width)], dtype=float)


def exact_maxcut(width: int, edges: Sequence[tuple[int, int]]) -> tuple[int, tuple[int, ...]]:
    """Enumerate every assignment and return the optimum with all optimal states."""

    table = cut_table(width, edges)
    optimum = int(table.max())
    return optimum, tuple(int(index) for index in np.flatnonzero(table == optimum))


def qaoa_state(
    width: int,
    edges: Sequence[tuple[int, int]],
    beta: float,
    gamma: float,
    *,
    gamma_sign: float = 1.0,
) -> np.ndarray:
    """Apply e^{i gamma Z Z / 2} per edge as a diagonal, then an X mixer on each qubit."""

    state = np.full(2**width, 2.0 ** (-width / 2.0), dtype=complex)
    for index in range(state.size):
        phase = 0.0
        for left, right in edges:
            same = ((index >> left) & 1) == ((index >> right) & 1)
            phase += gamma_sign * gamma / 2.0 * (1.0 if same else -1.0)
        state[index] *= np.exp(1j * phase)
    mixer = rx(2.0 * beta)
    for qubit in range(width):
        state = apply_one_qubit(state, mixer, qubit)
    return state


def ideal_energy(
    width: int,
    edges: Sequence[tuple[int, int]],
    beta: float,
    gamma: float,
    *,
    gamma_sign: float = 1.0,
) -> float:
    state = qaoa_state(width, edges, beta, gamma, gamma_sign=gamma_sign)
    return float(np.abs(state) ** 2 @ cut_table(width, edges))


def energy_from_counts(
    counts: Mapping[str, int],
    edges: Sequence[tuple[int, int]],
    *,
    reverse_bits: bool = False,
) -> float:
    total, shots = 0, 0
    for bitstring, count in counts.items():
        index = int(bitstring[::-1] if reverse_bits else bitstring, 2)
        total += cut_of_index(index, edges) * count
        shots += count
    return total / shots


def selection_metrics(
    candidate_ids: Sequence[str],
    exact: Sequence[float],
    estimates: Sequence[float],
    *,
    top_k: int,
    tolerance: float = 1e-12,
) -> dict[str, object]:
    """Choose the highest estimate, with ties resolved by identifier, and score it."""

    exact_by_id = dict(zip(candidate_ids, exact, strict=True))
    noisy_by_id = dict(zip(candidate_ids, estimates, strict=True))
    noisy_best = max(estimates)
    noisy_winners = sorted(
        key for key, value in noisy_by_id.items() if noisy_best - value <= tolerance
    )
    selected = noisy_winners[0]
    exact_best = max(exact)
    exact_winners = sorted(
        key for key, value in exact_by_id.items() if exact_best - value <= tolerance
    )
    exact_ranked = sorted(candidate_ids, key=lambda key: (-exact_by_id[key], key))
    boundary = exact_by_id[exact_ranked[top_k - 1]]
    exact_top = {key for key, value in exact_by_id.items() if value >= boundary - tolerance}
    noisy_top = set(sorted(candidate_ids, key=lambda key: (-noisy_by_id[key], key))[:top_k])
    return {
        "selected": selected,
        "noisy_winners": tuple(noisy_winners),
        "exact_winners": tuple(exact_winners),
        "ranking": tuple(sorted(candidate_ids, key=lambda key: (-noisy_by_id[key], key))),
        "decision_error": int(selected not in exact_winners),
        "simple_regret": max(0.0, exact_best - exact_by_id[selected]),
        "top_k_recall": len(noisy_top & exact_top) / len(exact_top),
        "energy_rmse": math.sqrt(
            math.fsum((noisy_by_id[key] - exact_by_id[key]) ** 2 for key in candidate_ids)
            / len(candidate_ids)
        ),
    }
