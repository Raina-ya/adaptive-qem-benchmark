"""Readout channels, calibration designs, and correction methods for the Part 9 studies.

NumPy and SciPy only. Basis integer bit q is qubit q. Every study corrects the Z parity of qubits
0 to w-2. The last qubit is left out on purpose: an all-qubit parity is unchanged by pair flips
when the pairs tile every qubit, so it could not show correlated readout error at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

FULL_CONDITION_FAILURE = 1e8


@dataclass(frozen=True)
class Channel:
    """Independent per-qubit errors, then flips of disjoint pairs (0,1), (2,3), and so on."""

    p01: tuple[float, ...]
    p10: tuple[float, ...]
    pair_flip: float = 0.0

    @property
    def width(self) -> int:
        return len(self.p01)

    @staticmethod
    def uniform(width: int, p01: float, p10: float, pair_flip: float = 0.0) -> Channel:
        return Channel((p01,) * width, (p10,) * width, pair_flip)


@dataclass(frozen=True)
class Mixture:
    """Apply (1 - weight) * first + weight * second, the dose-response channel."""

    first: Channel
    second: Channel
    weight: float = field(default=0.0)

    @property
    def width(self) -> int:
        return self.first.width


def local(p01: float, p10: float) -> np.ndarray:
    return np.array([[1.0 - p01, p10], [p01, 1.0 - p10]])


def _axis(width: int, qubit: int) -> int:
    return width - 1 - qubit


def apply_channel(probabilities: np.ndarray, channel: Channel | Mixture) -> np.ndarray:
    """Return observed probabilities, without forming the 2^w by 2^w matrix."""

    if isinstance(channel, Mixture):
        first = apply_channel(probabilities, channel.first)
        second = apply_channel(probabilities, channel.second)
        return (1.0 - channel.weight) * first + channel.weight * second
    width = channel.width
    tensor = np.asarray(probabilities, dtype=float).reshape((2,) * width)
    for qubit in range(width):
        axis = _axis(width, qubit)
        matrix = local(channel.p01[qubit], channel.p10[qubit])
        tensor = np.moveaxis(np.tensordot(matrix, tensor, axes=([1], [axis])), 0, axis)
    if channel.pair_flip:
        for qubit in range(0, width - 1, 2):
            flipped = np.flip(tensor, axis=(_axis(width, qubit), _axis(width, qubit + 1)))
            tensor = (1.0 - channel.pair_flip) * tensor + channel.pair_flip * flipped
    return tensor.reshape(-1)


def basis(width: int, index: int) -> np.ndarray:
    vector = np.zeros(2**width)
    vector[index] = 1.0
    return vector


def in_support(width: int, qubit: int) -> bool:
    """Return whether a qubit is in the parity observable: every qubit but the last."""

    return qubit < width - 1


def parity_signs(width: int) -> np.ndarray:
    indices = np.arange(2**width)
    ones = np.zeros(2**width, dtype=int)
    for qubit in range(width - 1):
        ones += (indices >> qubit) & 1
    return np.where(ones % 2 == 0, 1.0, -1.0)


def bits(width: int, indices: np.ndarray) -> np.ndarray:
    return ((np.asarray(indices)[:, None] >> np.arange(width)[None, :]) & 1).astype(int)


def product_distribution(width: int, value: float) -> np.ndarray:
    """Return identical RY rotations whose parity over qubits 0 to w-2 equals value."""

    cosine = value ** (1.0 / (width - 1))
    one_qubit = np.array([(1.0 + cosine) / 2.0, (1.0 - cosine) / 2.0])
    distribution = np.array([1.0])
    for _ in range(width):
        distribution = np.kron(one_qubit, distribution)
    return distribution


def sparse_distribution(width: int, value: float) -> np.ndarray:
    """Return weight on all-zeros and on qubit zero flipped, with parity equal to value."""

    distribution = np.zeros(2**width)
    distribution[0] = (1.0 + value) / 2.0
    distribution[1] = (1.0 - value) / 2.0
    return distribution


def raw_parity(counts: np.ndarray, signs: np.ndarray) -> float:
    return float(counts @ signs) / float(counts.sum())


# Calibration designs. Each returns the counts a runtime would hold, never the true channel.


def calibrate_tensor(rng, channel, shots: int) -> tuple[np.ndarray, np.ndarray]:
    """Prepare all zeros and all ones, and return per-qubit flip-rate estimates."""

    width = channel.width
    zero, one = shots - shots // 2, shots // 2
    estimates = []
    for prepared, count in ((0, zero), (2**width - 1, one)):
        sampled = rng.multinomial(count, apply_channel(basis(width, prepared), channel))
        marginal = bits(width, np.arange(2**width)).T @ sampled / count
        estimates.append(marginal if prepared == 0 else 1.0 - marginal)
    return estimates[0], estimates[1]


def calibrate_pairs(rng, channel, shots: int) -> list[np.ndarray]:
    """Prepare four pair patterns on every pair at once, and return 4x4 block estimates."""

    width = channel.width
    pairs = [(q, q + 1) for q in range(0, width - 1, 2)]
    single = [width - 1] if width % 2 else []
    per_state = [shots // 4 + (index < shots % 4) for index in range(4)]
    blocks = [np.zeros((4, 4)) for _ in pairs]
    lone = np.zeros((2, 2))
    table = bits(width, np.arange(2**width))
    for pattern, count in enumerate(per_state):
        low, high = pattern & 1, (pattern >> 1) & 1
        prepared = 0
        for first, second in pairs:
            prepared |= (low << first) | (high << second)
        for qubit in single:
            prepared |= low << qubit
        sampled = rng.multinomial(count, apply_channel(basis(width, prepared), channel))
        for block, (first, second) in zip(blocks, pairs, strict=True):
            observed = table[:, first] + 2 * table[:, second]
            block[:, pattern] = np.bincount(observed, weights=sampled, minlength=4) / count
        for qubit in single:
            if pattern in (0, 1):
                lone[:, low] = np.bincount(table[:, qubit], weights=sampled, minlength=2) / count
    return [*blocks, *([lone] if single else [])]


def calibrate_full(rng, channel, shots_per_state: int) -> np.ndarray:
    """Prepare every basis state, and return the estimated assignment matrix by columns."""

    width = channel.width
    estimate = np.zeros((2**width, 2**width))
    if shots_per_state <= 0:
        return estimate
    for prepared in range(2**width):
        column = apply_channel(basis(width, prepared), channel)
        estimate[:, prepared] = rng.multinomial(shots_per_state, column) / shots_per_state
    return estimate


# Corrections. Each returns (estimate, status, diagnostics).


def tensor_weights(width: int, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    weights = np.array([1.0])
    for qubit in range(width):
        if not in_support(width, qubit):
            weights = np.kron(np.array([1.0, 1.0]), weights)
            continue
        scale = 1.0 - p01[qubit] - p10[qubit]
        factor = np.array([1.0 - p10[qubit] + p01[qubit], -(1.0 - p01[qubit] + p10[qubit])])
        weights = np.kron(factor / scale, weights)
    return weights


def correct_tensor(counts, p01, p10) -> tuple[float, str, dict]:
    width = len(p01)
    scales = 1.0 - p01 - p10
    if np.any(scales <= 0.0):
        return math.nan, "calibration_invalid", {"condition": math.inf}
    weights = tensor_weights(width, p01, p10)
    condition = float(np.prod([np.linalg.cond(local(a, b)) for a, b in zip(p01, p10, strict=True)]))
    total = float(counts.sum())
    estimate = float(counts @ weights) / total
    variance = max(float(counts @ weights**2) / total - estimate**2, 0.0) / total
    return estimate, "complete", {"condition": condition, "standard_error": math.sqrt(variance)}


def correct_pairs(counts, blocks: list[np.ndarray]) -> tuple[float, str, dict]:
    """Apply the inverse of each block to the parity, as a tensor of block row vectors."""

    width = int(math.log2(len(counts)))
    pairs = [(q, q + 1) for q in range(0, width - 1, 2)]
    groups = [*pairs, *([(width - 1,)] if width % 2 else [])]
    weights = np.array([1.0])
    condition = 1.0
    for block, group in zip(blocks, groups, strict=True):
        single = [np.array([1.0, -1.0]) if in_support(width, q) else np.ones(2) for q in group]
        signs = np.kron(single[1], single[0]) if len(group) == 2 else single[0]
        try:
            row = np.linalg.solve(block.T, signs)
        except np.linalg.LinAlgError:
            return math.nan, "calibration_singular", {"condition": math.inf}
        condition *= float(np.linalg.cond(block))
        weights = np.kron(row, weights)
    total = float(counts.sum())
    estimate = float(counts @ weights) / total
    variance = max(float(counts @ weights**2) / total - estimate**2, 0.0) / total
    return estimate, "complete", {"condition": condition, "standard_error": math.sqrt(variance)}


def correct_full(counts, estimate_matrix: np.ndarray, condition: float | None) -> tuple:
    """Solve the full estimated system. Zero columns or extreme conditioning count as failure."""

    width = int(math.log2(len(counts)))
    signs = parity_signs(width)
    if not np.any(estimate_matrix):
        return math.nan, "no_calibration_shots", {"condition": math.inf}
    observed = counts / counts.sum()
    if condition is not None and condition > FULL_CONDITION_FAILURE:
        return math.nan, "ill_conditioned", {"condition": condition}
    try:
        quasi = np.linalg.solve(estimate_matrix, observed)
    except np.linalg.LinAlgError:
        return math.nan, "calibration_singular", {"condition": math.inf}
    if not np.all(np.isfinite(quasi)):
        return math.nan, "calibration_singular", {"condition": math.inf}
    negative = float(-quasi[quasi < 0.0].sum())
    return float(quasi @ signs), "complete", {"condition": condition, "negative_mass": negative}


def _subspace_matrix(width: int, support: np.ndarray, p01, p10) -> np.ndarray:
    table = bits(width, support)
    matrix = np.ones((len(support), len(support)))
    for qubit in range(width):
        block = local(p01[qubit], p10[qubit])
        matrix *= block[table[:, qubit][:, None], table[:, qubit][None, :]]
    return matrix


def correct_subspace(counts, p01, p10, *, iterative: bool) -> tuple[float, str, dict]:
    """Solve the tensor model restricted to observed outcomes, then renormalize.

    The iterative path never forms the matrix: it rebuilds rows in chunks inside each product.
    """

    width = len(p01)
    support = np.flatnonzero(counts)
    observed = counts[support] / counts.sum()
    signs = parity_signs(width)[support]
    table = bits(width, support)
    if iterative:
        blocks = [local(p01[q], p10[q]) for q in range(width)]

        def matvec(vector):
            output = np.empty(len(support))
            for start in range(0, len(support), 256):
                rows = table[start : start + 256]
                chunk = np.ones((len(rows), len(support)))
                for qubit in range(width):
                    chunk *= blocks[qubit][rows[:, qubit][:, None], table[:, qubit][None, :]]
                output[start : start + 256] = chunk @ vector
            return output

        operator = LinearOperator((len(support), len(support)), matvec=matvec)
        iterations = [0]

        def count(_):
            iterations[0] += 1

        solution, info = gmres(
            operator, observed, rtol=1e-10, maxiter=200, callback=count, callback_type="pr_norm"
        )
        if info != 0:
            return math.nan, "not_converged", {"iterations": iterations[0]}
        extra = {"iterations": iterations[0]}
    else:
        matrix = _subspace_matrix(width, support, p01, p10)
        try:
            solution = np.linalg.solve(matrix, observed)
        except np.linalg.LinAlgError:
            return math.nan, "calibration_singular", {}
        extra = {}
    total = solution.sum()
    if total <= 0.0:
        return math.nan, "renormalization_failed", extra
    solution = solution / total
    return float(solution @ signs), "complete", {**extra, "support": len(support)}
