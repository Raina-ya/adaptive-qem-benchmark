"""Matrix-free observed-subspace readout mitigation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.readout_subspace import select_observed_support
from src.mitigation.readout_tensor import estimate_tensor_assignment_matrices
from src.noise_models import NoiseSpec
from src.observables import expectation_from_probabilities
from src.resource_accounting import ResourceLedger


@dataclass(frozen=True)
class MatrixFreeSolveResult:
    """A matrix-free non-negative least-squares result."""

    solution: np.ndarray | None
    iterations: int
    converged: bool
    residual_norm: float
    status: str


@dataclass(frozen=True)
class MatrixFreeReadoutResult:
    """Corrected estimate with resource, support, and convergence diagnostics."""

    raw_estimate: float
    corrected_estimate: float | None
    condition_estimate: float
    assignment_dimension: int
    observed_support_size: int
    operator_storage_bytes: int
    working_vector_bytes: int
    materialized_matrix_entries: int
    retained_probability_mass: float
    discarded_probability_mass: float
    negative_probability_mass: float | None
    iterations: int
    converged: bool
    residual_norm: float
    status: str
    total_shot_budget: int
    target_shots: int
    calibration_shots: int
    target_circuit_executions: int
    calibration_circuit_executions: int
    raw_counts_json: str
    calibration_counts_json: str
    support_states_json: str
    assignment_matrices_json: str
    corrected_probabilities_json: str | None
    seed: int
    result_label: str = "noisy_simulated"


class TensorSubspaceOperator:
    """Apply a tensor-product assignment map without forming its dense matrix."""

    def __init__(
        self,
        assignment_matrices: tuple[np.ndarray, ...],
        support_states: tuple[int, ...],
    ) -> None:
        if not assignment_matrices:
            raise ValueError("At least one local assignment matrix is required")
        if any(matrix.shape != (2, 2) for matrix in assignment_matrices):
            raise ValueError("Every local assignment matrix must be 2x2")
        dimension = 2 ** len(assignment_matrices)
        if not support_states or tuple(sorted(set(support_states))) != support_states:
            raise ValueError("Support states must be unique, sorted, and non-empty")
        if support_states[0] < 0 or support_states[-1] >= dimension:
            raise ValueError("Support state is outside the computational basis")
        self.assignment_matrices = tuple(
            np.asarray(matrix, dtype=float) for matrix in assignment_matrices
        )
        self.support_states = support_states
        self.full_dimension = dimension
        self.shape = (len(support_states), len(support_states))

    def _apply_full(self, vector: np.ndarray, *, transpose: bool) -> np.ndarray:
        width = len(self.assignment_matrices)
        transformed = np.asarray(vector, dtype=float).reshape((2,) * width)
        for qubit, matrix in enumerate(self.assignment_matrices):
            local = matrix.T if transpose else matrix
            axis = width - 1 - qubit
            moved = np.moveaxis(transformed, axis, 0)
            flat = moved.reshape(2, -1)
            output = np.empty_like(flat)
            output[0] = local[0, 0] * flat[0] + local[0, 1] * flat[1]
            output[1] = local[1, 0] * flat[0] + local[1, 1] * flat[1]
            transformed = np.moveaxis(output.reshape(moved.shape), 0, axis)
        return transformed.reshape(-1)

    def matvec(self, support_vector: np.ndarray) -> np.ndarray:
        """Apply the support-restricted forward assignment map."""

        vector = np.asarray(support_vector, dtype=float)
        if vector.shape != (self.shape[1],):
            raise ValueError("Input vector shape does not match the support")
        embedded = np.zeros(self.full_dimension, dtype=float)
        embedded[list(self.support_states)] = vector
        return self._apply_full(embedded, transpose=False)[list(self.support_states)]

    def rmatvec(self, support_vector: np.ndarray) -> np.ndarray:
        """Apply the transpose of the support-restricted assignment map."""

        vector = np.asarray(support_vector, dtype=float)
        if vector.shape != (self.shape[0],):
            raise ValueError("Input vector shape does not match the support")
        embedded = np.zeros(self.full_dimension, dtype=float)
        embedded[list(self.support_states)] = vector
        return self._apply_full(embedded, transpose=True)[list(self.support_states)]


def _local_condition_estimate(matrix: np.ndarray) -> float:
    determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if abs(determinant) <= 1e-18:
        return math.inf
    inverse = np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
        dtype=float,
    ) / determinant
    matrix_norm = max(math.fsum(abs(float(value)) for value in row) for row in matrix)
    inverse_norm = max(math.fsum(abs(float(value)) for value in row) for row in inverse)
    return matrix_norm * inverse_norm


def tensor_condition_estimate(assignment_matrices: tuple[np.ndarray, ...]) -> float:
    """Return the product infinity-norm condition estimate for local factors."""

    return math.prod(_local_condition_estimate(matrix) for matrix in assignment_matrices)


def solve_matrix_free_nnls(
    operator: TensorSubspaceOperator,
    observed_probabilities: np.ndarray,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> MatrixFreeSolveResult:
    """Solve non-negative least squares using deterministic projected FISTA."""

    observed = np.asarray(observed_probabilities, dtype=float)
    if observed.shape != (operator.shape[0],):
        raise ValueError("Observed vector shape does not match the operator")
    if max_iterations < 1 or tolerance <= 0.0:
        raise ValueError("Solver limits must be positive")
    row_norm = float(np.max(operator.matvec(np.ones(operator.shape[1]))))
    column_norm = float(np.max(operator.rmatvec(np.ones(operator.shape[0]))))
    lipschitz = row_norm * column_norm
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        return MatrixFreeSolveResult(None, 0, False, math.inf, "invalid_operator")

    initial_total = math.fsum(float(value) for value in observed)
    if initial_total <= 0.0:
        return MatrixFreeSolveResult(None, 0, False, math.inf, "empty_observation")
    solution = np.maximum(observed / initial_total, 0.0)
    accelerated = solution.copy()
    momentum = 1.0
    residual_norm = math.inf
    for iteration in range(1, max_iterations + 1):
        residual = operator.matvec(accelerated) - observed
        gradient = operator.rmatvec(residual)
        updated = np.maximum(accelerated - gradient / lipschitz, 0.0)
        delta = float(np.linalg.norm(updated - solution))
        scale = max(1.0, float(np.linalg.norm(solution)))
        residual_norm = float(np.linalg.norm(operator.matvec(updated) - observed))
        if delta <= tolerance * scale:
            total = math.fsum(float(value) for value in updated)
            if total <= 0.0:
                return MatrixFreeSolveResult(
                    None, iteration, False, residual_norm, "solver_failed"
                )
            return MatrixFreeSolveResult(
                updated / total,
                iteration,
                True,
                residual_norm,
                "ok",
            )
        next_momentum = (1.0 + math.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        accelerated = updated + ((momentum - 1.0) / next_momentum) * (updated - solution)
        solution = updated
        momentum = next_momentum
    return MatrixFreeSolveResult(
        None,
        max_iterations,
        False,
        residual_norm,
        "not_converged",
    )


def matrix_free_readout_ledger(total_shot_budget: int, width: int) -> ResourceLedger:
    """Allocate 20% calibration shots over two local states per qubit."""

    if width < 1:
        raise ValueError("Width must be positive")
    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    calibration_circuits = 2 * width
    if calibration_shots < calibration_circuits:
        raise ValueError("Budget is too small for local calibration")
    return ResourceLedger(
        total_shot_budget=total_shot_budget,
        target_shots=total_shot_budget - calibration_shots,
        calibration_shots=calibration_shots,
        calibration_circuit_executions=calibration_circuits,
    )


def run_matrix_free_readout_mitigation(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    min_count: int = 2,
    protected_states: tuple[int, ...] = (),
    condition_threshold: float = 1e8,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
) -> MatrixFreeReadoutResult:
    """Run matrix-free support correction under one fixed total shot budget."""

    ledger = matrix_free_readout_ledger(total_shot_budget, circuit.num_qubits)
    raw = estimate_pauli(
        circuit,
        pauli_q0_first,
        shots=ledger.target_shots,
        seed=derive_seed("readout-target", seed),
        noise_spec=noise_spec,
    )
    support = select_observed_support(
        raw.raw_counts_json,
        circuit.num_qubits,
        min_count=min_count,
        protected_states=protected_states,
    )
    matrices, _, calibration_json = estimate_tensor_assignment_matrices(
        circuit.num_qubits,
        ledger.calibration_shots,
        noise_spec,
        seed=seed,
    )
    condition = tensor_condition_estimate(matrices)
    counts = json.loads(raw.raw_counts_json)
    observed = np.array(
        [
            counts.get(format(state, f"0{circuit.num_qubits}b"), 0) / ledger.target_shots
            for state in support.states
        ],
        dtype=float,
    )
    operator = TensorSubspaceOperator(matrices, support.states)
    if not math.isfinite(condition) or condition > condition_threshold:
        solve = MatrixFreeSolveResult(None, 0, False, math.inf, "ill_conditioned")
    else:
        solve = solve_matrix_free_nnls(
            operator,
            observed,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
    corrected = None
    corrected_estimate = None
    corrected_json = None
    negative_mass = None
    if solve.solution is not None:
        corrected = np.zeros(operator.full_dimension, dtype=float)
        corrected[list(support.states)] = solve.solution
        corrected_estimate = expectation_from_probabilities(corrected, pauli_q0_first)
        corrected_json = json.dumps(corrected.tolist(), separators=(",", ":"))
        negative_mass = math.fsum(max(-float(value), 0.0) for value in corrected)
    return MatrixFreeReadoutResult(
        raw_estimate=raw.estimate,
        corrected_estimate=corrected_estimate,
        condition_estimate=condition,
        assignment_dimension=operator.full_dimension,
        observed_support_size=len(support.states),
        operator_storage_bytes=sum(matrix.nbytes for matrix in matrices)
        + 8 * len(support.states),
        working_vector_bytes=8 * operator.full_dimension,
        materialized_matrix_entries=0,
        retained_probability_mass=support.retained_probability_mass,
        discarded_probability_mass=support.discarded_probability_mass,
        negative_probability_mass=negative_mass,
        iterations=solve.iterations,
        converged=solve.converged,
        residual_norm=solve.residual_norm,
        status=solve.status,
        raw_counts_json=raw.raw_counts_json,
        calibration_counts_json=calibration_json,
        support_states_json=json.dumps(support.states, separators=(",", ":")),
        assignment_matrices_json=json.dumps(
            [matrix.tolist() for matrix in matrices], separators=(",", ":")
        ),
        corrected_probabilities_json=corrected_json,
        seed=seed,
        **ledger.__dict__,
    )
