"""Full assignment-matrix readout mitigation for small circuits."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit
from scipy.optimize import lsq_linear

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


def constrained_readout_correction(
    assignment_matrix: np.ndarray,
    observed_probabilities: np.ndarray,
    *,
    condition_threshold: float = 1e8,
) -> tuple[np.ndarray | None, float, str]:
    """Solve A p = p_obs with non-negative probabilities and diagnostics."""

    condition_number = float(np.linalg.cond(assignment_matrix))
    if not np.isfinite(condition_number) or condition_number > condition_threshold:
        return None, condition_number, "ill_conditioned"
    solution = lsq_linear(assignment_matrix, observed_probabilities, bounds=(0.0, np.inf))
    if not solution.success or solution.x.sum() <= 0:
        return None, condition_number, "solver_failed"
    corrected = solution.x / solution.x.sum()
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
