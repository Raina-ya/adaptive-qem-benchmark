"""Full assignment-matrix readout mitigation for small circuits."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import derive_seed
from src.estimators.unmitigated import EstimationResult, estimate_pauli
from src.noise_models import NoiseSpec
from src.observables import expectation_from_probabilities
from src.resource_accounting import readout_mitigation_ledger, split_shots


@dataclass(frozen=True)
class ReadoutMitigationResult:
    """Raw and corrected estimates plus calibration diagnostics."""

    raw_estimate: float
    corrected_estimate: float | None
    condition_number: float
    status: str
    total_shot_budget: int
    target_shots: int
    calibration_shots: int
    target_circuit_executions: int
    calibration_circuit_executions: int
    raw_counts_json: str
    calibration_counts_json: str
    corrected_probabilities_json: str | None
    assignment_matrix_json: str
    seed: int
    result_label: str = "noisy_simulated"


def calibration_circuit(width: int, prepared_state: int) -> QuantumCircuit:
    """Prepare one computational basis state and measure all qubits."""

    if not 0 <= prepared_state < 2**width:
        raise ValueError("Prepared state is outside the calibration basis")
    circuit = QuantumCircuit(width)
    for qubit in range(width):
        if (prepared_state >> qubit) & 1:
            circuit.x(qubit)
    circuit.measure_all()
    return circuit


def estimate_assignment_matrix(
    width: int,
    calibration_shots: int,
    noise_spec: NoiseSpec,
    *,
    seed: int,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    """Estimate P(observed row | prepared column) for every basis state."""

    state_count = 2**width
    shots_by_state = split_shots(calibration_shots, state_count)
    matrix = np.zeros((state_count, state_count), dtype=float)
    calibration_records: list[dict[str, object]] = []
    for prepared_state, shots in enumerate(shots_by_state):
        child_seed = derive_seed("readout-calibration", seed, prepared_state)
        probabilities = np.ones(state_count, dtype=float)
        for observed_state in range(state_count):
            for qubit in range(width):
                prepared_bit = (prepared_state >> qubit) & 1
                observed_bit = (observed_state >> qubit) & 1
                if prepared_bit == 0:
                    probabilities[observed_state] *= (
                        noise_spec.readout_p01
                        if observed_bit
                        else 1.0 - noise_spec.readout_p01
                    )
                else:
                    probabilities[observed_state] *= (
                        1.0 - noise_spec.readout_p10
                        if observed_bit
                        else noise_spec.readout_p10
                    )
        sampled = np.random.default_rng(child_seed).multinomial(shots, probabilities)
        counts = {
            format(state, f"0{width}b"): int(count)
            for state, count in enumerate(sampled)
            if count
        }
        normalized_counts = {str(key): int(value) for key, value in sorted(counts.items())}
        calibration_records.append(
            {
                "prepared_state": prepared_state,
                "shots": shots,
                "seed": child_seed,
                "counts": normalized_counts,
            }
        )
        for bitstring, count in counts.items():
            observed_state = int(bitstring.replace(" ", ""), 2)
            matrix[observed_state, prepared_state] += count / shots
    return (
        matrix,
        shots_by_state,
        json.dumps(calibration_records, sort_keys=True, separators=(",", ":")),
    )


def counts_probability_vector(raw_counts_json: str, width: int) -> np.ndarray:
    """Convert preserved raw counts into basis-index probabilities."""

    counts = json.loads(raw_counts_json)
    probabilities = np.zeros(2**width, dtype=float)
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        probabilities[int(bitstring.replace(" ", ""), 2)] += count / shots
    return probabilities


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Solve a small dense system with deterministic partial pivoting."""

    size = len(vector)
    augmented = [matrix[row].copy() + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: (abs(augmented[row][column]), -row))
        pivot_value = augmented[pivot][column]
        if abs(pivot_value) <= 1e-18:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for index in range(column, size + 1):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][-1] for row in range(size)]


def _inverse_and_condition_inf(matrix: np.ndarray) -> tuple[list[list[float]] | None, float]:
    """Invert a matrix and compute its infinity-norm condition number deterministically."""

    size = matrix.shape[0]
    augmented = [
        [float(value) for value in matrix[row]]
        + [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: (abs(augmented[row][column]), -row))
        pivot_value = augmented[pivot][column]
        if abs(pivot_value) <= 1e-18:
            return None, math.inf
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for index in range(column, 2 * size):
            augmented[column][index] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for index in range(column, 2 * size):
                augmented[row][index] -= factor * augmented[column][index]

    inverse = [row[size:] for row in augmented]
    matrix_norm = max(math.fsum(abs(float(value)) for value in row) for row in matrix)
    inverse_norm = max(math.fsum(abs(value) for value in row) for row in inverse)
    return inverse, matrix_norm * inverse_norm


def _diagonal_dominance_condition_bound(matrix: np.ndarray) -> float:
    """Return Varah's deterministic infinity-condition upper bound."""

    matrix_norm = max(math.fsum(abs(float(value)) for value in row) for row in matrix)
    margins = [
        abs(float(matrix[index, index]))
        - math.fsum(
            abs(float(matrix[index, column]))
            for column in range(matrix.shape[1])
            if column != index
        )
        for index in range(matrix.shape[0])
    ]
    minimum_margin = min(margins)
    return matrix_norm / minimum_margin if minimum_margin > 0.0 else math.inf


class _DenseAssignmentOperator:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.shape = matrix.shape

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix @ vector

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        return self.matrix.T @ vector


def _projected_fista_nnls(
    matrix: np.ndarray,
    observed: np.ndarray,
    *,
    max_iterations: int = 2000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray | None, str]:
    """Solve dense NNLS with the same fixed projected-FISTA update as the E track."""

    operator = _DenseAssignmentOperator(matrix)
    row_norm = float(np.max(operator.matvec(np.ones(matrix.shape[1]))))
    column_norm = float(np.max(operator.rmatvec(np.ones(matrix.shape[0]))))
    lipschitz = row_norm * column_norm
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        return None, "invalid_operator"
    total = math.fsum(float(value) for value in observed)
    if total <= 0.0:
        return None, "empty_observation"
    solution = np.maximum(observed / total, 0.0)
    accelerated = solution.copy()
    momentum = 1.0
    for _ in range(max_iterations):
        residual = operator.matvec(accelerated) - observed
        gradient = operator.rmatvec(residual)
        updated = np.maximum(accelerated - gradient / lipschitz, 0.0)
        delta = float(np.linalg.norm(updated - solution))
        scale = max(1.0, float(np.linalg.norm(solution)))
        if delta <= tolerance * scale:
            corrected_total = math.fsum(float(value) for value in updated)
            if corrected_total <= 0.0:
                return None, "solver_failed"
            return updated / corrected_total, "ok"
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (
            updated - solution
        )
        solution = updated
        momentum = next_momentum
    return None, "not_converged"


def _deterministic_nnls(
    assignment_matrix: np.ndarray, observed_probabilities: np.ndarray
) -> np.ndarray | None:
    """Solve non-negative least squares with fixed-order Lawson-Hanson updates."""

    size = assignment_matrix.shape[1]
    gram = [
        [
            math.fsum(
                float(assignment_matrix[row, left]) * float(assignment_matrix[row, right])
                for row in range(assignment_matrix.shape[0])
            )
            for right in range(size)
        ]
        for left in range(size)
    ]
    cross = [
        math.fsum(
            float(assignment_matrix[row, column]) * float(observed_probabilities[row])
            for row in range(assignment_matrix.shape[0])
        )
        for column in range(size)
    ]
    solution = [0.0] * size
    passive: list[int] = []
    tolerance = 1e-14

    for _ in range(4 * size + 1):
        gradient = [
            cross[index]
            - math.fsum(gram[index][column] * solution[column] for column in range(size))
            for index in range(size)
        ]
        candidates = [
            index for index in range(size) if index not in passive and gradient[index] > tolerance
        ]
        if not candidates:
            break
        passive.append(max(candidates, key=lambda index: (gradient[index], -index)))
        passive.sort()

        for _ in range(4 * size + 1):
            submatrix = [[gram[row][column] for column in passive] for row in passive]
            subvector = [cross[index] for index in passive]
            passive_solution = _solve_linear_system(submatrix, subvector)
            if passive_solution is None:
                return None
            trial = [0.0] * size
            for index, value in zip(passive, passive_solution, strict=True):
                trial[index] = value
            nonpositive = [index for index in passive if trial[index] <= tolerance]
            if not nonpositive:
                solution = trial
                break
            steps = [
                solution[index] / (solution[index] - trial[index])
                for index in nonpositive
                if solution[index] - trial[index] > 0.0
            ]
            if not steps:
                return None
            step = min(steps)
            solution = [
                solution[index] + step * (trial[index] - solution[index])
                for index in range(size)
            ]
            passive = [index for index in passive if solution[index] > tolerance]
        else:
            return None
    else:
        return None

    total = math.fsum(solution)
    if total <= 0.0:
        return None
    return np.array([max(value / total, 0.0) for value in solution], dtype=float)


def constrained_readout_correction(
    assignment_matrix: np.ndarray,
    observed_probabilities: np.ndarray,
    *,
    condition_threshold: float = 1e8,
    precomputed_condition_number: float | None = None,
) -> tuple[np.ndarray | None, float, str]:
    """Solve a fixed-order non-negative least-squares correction."""

    condition_number = precomputed_condition_number
    if condition_number is None:
        if assignment_matrix.shape[0] >= 128:
            condition_number = _diagonal_dominance_condition_bound(assignment_matrix)
        else:
            _, condition_number = _inverse_and_condition_inf(assignment_matrix)
    if not np.isfinite(condition_number) or condition_number > condition_threshold:
        return None, condition_number, "ill_conditioned"
    if assignment_matrix.shape[0] >= 128:
        corrected, solve_status = _projected_fista_nnls(
            assignment_matrix, observed_probabilities
        )
        if corrected is None:
            return None, condition_number, solve_status
    else:
        corrected = _deterministic_nnls(assignment_matrix, observed_probabilities)
    if corrected is None:
        return None, condition_number, "solver_failed"
    return corrected, condition_number, "ok"


def run_readout_mitigation(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    condition_threshold: float = 1e8,
) -> ReadoutMitigationResult:
    """Run target sampling and calibration within one fixed total budget."""

    ledger = readout_mitigation_ledger(total_shot_budget, circuit.num_qubits)
    raw: EstimationResult = estimate_pauli(
        circuit,
        pauli_q0_first,
        shots=ledger.target_shots,
        seed=derive_seed("readout-target", seed),
        noise_spec=noise_spec,
    )
    assignment, _, calibration_counts_json = estimate_assignment_matrix(
        circuit.num_qubits,
        ledger.calibration_shots,
        noise_spec,
        seed=seed,
    )
    observed = counts_probability_vector(raw.raw_counts_json, circuit.num_qubits)
    corrected, condition_number, status = constrained_readout_correction(
        assignment, observed, condition_threshold=condition_threshold
    )
    corrected_estimate = (
        expectation_from_probabilities(corrected, pauli_q0_first)
        if corrected is not None
        else None
    )
    return ReadoutMitigationResult(
        raw_estimate=raw.estimate,
        corrected_estimate=corrected_estimate,
        condition_number=condition_number,
        status=status,
        raw_counts_json=raw.raw_counts_json,
        calibration_counts_json=calibration_counts_json,
        corrected_probabilities_json=(
            json.dumps(corrected.tolist(), separators=(",", ":"))
            if corrected is not None
            else None
        ),
        assignment_matrix_json=json.dumps(assignment.tolist(), separators=(",", ":")),
        seed=seed,
        **asdict(ledger),
    )
