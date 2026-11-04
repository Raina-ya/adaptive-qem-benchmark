"""Tensor-product readout mitigation with an explicit fixed-shot ledger."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.readout import counts_probability_vector
from src.noise_models import NoiseSpec
from src.observables import expectation_from_probabilities
from src.resource_accounting import ResourceLedger, split_shots


@dataclass(frozen=True)
class TensorReadoutResult:
    """Raw and tensor-corrected estimates with calibration diagnostics."""

    raw_estimate: float
    corrected_estimate: float | None
    condition_estimate: float
    negative_probability_mass: float | None
    physical_overshoot: float | None
    status: str
    total_shot_budget: int
    target_shots: int
    calibration_shots: int
    target_circuit_executions: int
    calibration_circuit_executions: int
    raw_counts_json: str
    calibration_counts_json: str
    corrected_probabilities_json: str | None
    assignment_matrices_json: str
    seed: int
    result_label: str = "noisy_simulated"


def tensor_readout_ledger(total_shot_budget: int, width: int) -> ResourceLedger:
    """Allocate 20% calibration shots over two states for each qubit."""

    if width < 1:
        raise ValueError("Width must be positive")
    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    calibration_circuits = 2 * width
    if calibration_shots < calibration_circuits:
        raise ValueError("Budget is too small to sample every tensor calibration circuit")
    return ResourceLedger(
        total_shot_budget=total_shot_budget,
        target_shots=total_shot_budget - calibration_shots,
        calibration_shots=calibration_shots,
        calibration_circuit_executions=calibration_circuits,
    )


def estimate_tensor_assignment_matrices(
    width: int,
    calibration_shots: int,
    noise_spec: NoiseSpec,
    *,
    seed: int,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...], str]:
    """Estimate one 2x2 assignment matrix per qubit from sampled calibration."""

    if width < 1:
        raise ValueError("Width must be positive")
    shots_by_circuit = split_shots(calibration_shots, 2 * width)
    matrices: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for qubit in range(width):
        matrix = np.zeros((2, 2), dtype=float)
        for prepared in (0, 1):
            index = 2 * qubit + prepared
            shots = shots_by_circuit[index]
            probability_one = (
                noise_spec.readout_p01 if prepared == 0 else 1.0 - noise_spec.readout_p10
            )
            child_seed = derive_seed("tensor-readout-calibration", seed, qubit, prepared)
            observed_one = int(
                np.random.default_rng(child_seed).binomial(shots, probability_one)
            )
            observed_zero = shots - observed_one
            matrix[:, prepared] = (observed_zero / shots, observed_one / shots)
            records.append(
                {
                    "qubit": qubit,
                    "prepared": prepared,
                    "shots": shots,
                    "seed": child_seed,
                    "counts": {"0": observed_zero, "1": observed_one},
                }
            )
        matrices.append(matrix)
    return (
        tuple(matrices),
        shots_by_circuit,
        json.dumps(records, sort_keys=True, separators=(",", ":")),
    )


def _inverse_and_condition(matrix: np.ndarray) -> tuple[np.ndarray | None, float]:
    determinant = float(matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0])
    if abs(determinant) <= 1e-18:
        return None, math.inf
    inverse = np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
        dtype=float,
    ) / determinant
    matrix_norm = max(math.fsum(abs(float(value)) for value in row) for row in matrix)
    inverse_norm = max(math.fsum(abs(float(value)) for value in row) for row in inverse)
    return inverse, matrix_norm * inverse_norm


def tensor_product_correction(
    assignment_matrices: tuple[np.ndarray, ...],
    observed_probabilities: np.ndarray,
    *,
    condition_threshold: float = 1e8,
) -> tuple[np.ndarray | None, float, str]:
    """Apply local inverses without materializing a global assignment matrix."""

    width = len(assignment_matrices)
    if width < 1 or observed_probabilities.shape != (2**width,):
        raise ValueError("Observed probabilities do not match the tensor width")
    inverses: list[np.ndarray] = []
    conditions: list[float] = []
    for matrix in assignment_matrices:
        if matrix.shape != (2, 2):
            raise ValueError("Every tensor assignment matrix must be 2x2")
        inverse, condition = _inverse_and_condition(matrix)
        if inverse is None:
            return None, math.inf, "ill_conditioned"
        inverses.append(inverse)
        conditions.append(condition)
    condition_estimate = math.prod(conditions)
    if not math.isfinite(condition_estimate) or condition_estimate > condition_threshold:
        return None, condition_estimate, "ill_conditioned"

    corrected = np.asarray(observed_probabilities, dtype=float).reshape((2,) * width)
    for qubit, inverse in enumerate(inverses):
        axis = width - 1 - qubit
        moved = np.moveaxis(corrected, axis, 0)
        flat = moved.reshape(2, -1)
        transformed = np.empty_like(flat)
        transformed[0] = inverse[0, 0] * flat[0] + inverse[0, 1] * flat[1]
        transformed[1] = inverse[1, 0] * flat[0] + inverse[1, 1] * flat[1]
        corrected = np.moveaxis(transformed.reshape(moved.shape), 0, axis)
    vector = corrected.reshape(-1)
    total = math.fsum(float(value) for value in vector)
    if not math.isfinite(total) or abs(total) <= 1e-18:
        return None, condition_estimate, "solver_failed"
    return vector / total, condition_estimate, "ok"


def run_tensor_readout_mitigation(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    condition_threshold: float = 1e8,
) -> TensorReadoutResult:
    """Run tensor-product correction under one declared total shot budget."""

    ledger = tensor_readout_ledger(total_shot_budget, circuit.num_qubits)
    raw = estimate_pauli(
        circuit,
        pauli_q0_first,
        shots=ledger.target_shots,
        seed=derive_seed("readout-target", seed),
        noise_spec=noise_spec,
    )
    matrices, _, calibration_counts_json = estimate_tensor_assignment_matrices(
        circuit.num_qubits,
        ledger.calibration_shots,
        noise_spec,
        seed=seed,
    )
    observed = counts_probability_vector(raw.raw_counts_json, circuit.num_qubits)
    corrected, condition, status = tensor_product_correction(
        matrices,
        observed,
        condition_threshold=condition_threshold,
    )
    corrected_estimate = None
    corrected_json = None
    negative_mass = None
    overshoot = None
    if corrected is not None:
        corrected_estimate = expectation_from_probabilities(corrected, pauli_q0_first)
        corrected_json = json.dumps(corrected.tolist(), separators=(",", ":"))
        negative_mass = math.fsum(max(-float(value), 0.0) for value in corrected)
        overshoot = max(max(float(value) - 1.0, 0.0) for value in corrected)
    matrices_json = json.dumps(
        [matrix.tolist() for matrix in matrices], separators=(",", ":")
    )
    return TensorReadoutResult(
        raw_estimate=raw.estimate,
        corrected_estimate=corrected_estimate,
        condition_estimate=condition,
        negative_probability_mass=negative_mass,
        physical_overshoot=overshoot,
        status=status,
        total_shot_budget=total_shot_budget,
        target_shots=ledger.target_shots,
        calibration_shots=ledger.calibration_shots,
        target_circuit_executions=ledger.target_circuit_executions,
        calibration_circuit_executions=ledger.calibration_circuit_executions,
        raw_counts_json=raw.raw_counts_json,
        calibration_counts_json=calibration_counts_json,
        corrected_probabilities_json=corrected_json,
        assignment_matrices_json=matrices_json,
        seed=seed,
    )
