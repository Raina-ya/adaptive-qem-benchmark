"""Observed-subspace readout mitigation with explicit truncation diagnostics."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.readout import constrained_readout_correction
from src.noise_models import NoiseSpec
from src.observables import expectation_from_probabilities
from src.resource_accounting import ResourceLedger, split_shots


@dataclass(frozen=True)
class ObservedSupport:
    """A target-count support selected without reference information."""

    states: tuple[int, ...]
    retained_probability_mass: float
    discarded_probability_mass: float


@dataclass(frozen=True)
class SubspaceReadoutResult:
    """Corrected estimate and all support, budget, and solver diagnostics."""

    raw_estimate: float
    corrected_estimate: float | None
    condition_estimate: float
    assignment_dimension: int
    observed_support_size: int
    reduced_matrix_bytes: int
    retained_probability_mass: float
    discarded_probability_mass: float
    negative_probability_mass: float | None
    status: str
    total_shot_budget: int
    target_shots: int
    calibration_shots: int
    target_circuit_executions: int
    calibration_circuit_executions: int
    raw_counts_json: str
    calibration_counts_json: str
    support_states_json: str
    assignment_matrix_json: str
    corrected_probabilities_json: str | None
    seed: int
    result_label: str = "noisy_simulated"


def select_observed_support(
    raw_counts_json: str,
    width: int,
    *,
    min_count: int = 2,
    protected_states: tuple[int, ...] = (),
) -> ObservedSupport:
    """Select count-supported states plus explicitly preregistered protected states."""

    if width < 1:
        raise ValueError("Width must be positive")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    dimension = 2**width
    if any(state < 0 or state >= dimension for state in protected_states):
        raise ValueError("Protected state is outside the computational basis")
    counts = json.loads(raw_counts_json)
    shots = sum(int(count) for count in counts.values())
    if shots <= 0:
        raise ValueError("Counts must contain at least one shot")
    state_counts = {
        int(str(bitstring).replace(" ", ""), 2): int(count)
        for bitstring, count in counts.items()
    }
    selected = {state for state, count in state_counts.items() if count >= min_count}
    selected.update(protected_states)
    if not selected:
        selected.add(max(state_counts, key=lambda state: (state_counts[state], -state)))
    retained = math.fsum(state_counts.get(state, 0) for state in selected) / shots
    return ObservedSupport(
        states=tuple(sorted(selected)),
        retained_probability_mass=retained,
        discarded_probability_mass=max(0.0, 1.0 - retained),
    )


def subspace_readout_ledger(total_shot_budget: int, support_size: int) -> ResourceLedger:
    """Allocate the fixed calibration fraction over one circuit per support state."""

    if support_size < 1:
        raise ValueError("Support must contain at least one state")
    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    if calibration_shots < support_size:
        raise ValueError("Budget is too small to calibrate every support state")
    return ResourceLedger(
        total_shot_budget=total_shot_budget,
        target_shots=total_shot_budget - calibration_shots,
        calibration_shots=calibration_shots,
        calibration_circuit_executions=support_size,
    )


def _assignment_probabilities(width: int, prepared_state: int, noise_spec: NoiseSpec) -> np.ndarray:
    observed_states = np.arange(2**width, dtype=np.uint64)
    probabilities = np.ones(2**width, dtype=float)
    for qubit in range(width):
        observed_one = ((observed_states >> np.uint64(qubit)) & np.uint64(1)).astype(bool)
        if (prepared_state >> qubit) & 1:
            probabilities *= np.where(
                observed_one, 1.0 - noise_spec.readout_p10, noise_spec.readout_p10
            )
        else:
            probabilities *= np.where(
                observed_one, noise_spec.readout_p01, 1.0 - noise_spec.readout_p01
            )
    return probabilities


def estimate_subspace_assignment_matrix(
    width: int,
    support_states: tuple[int, ...],
    calibration_shots: int,
    noise_spec: NoiseSpec,
    *,
    seed: int,
) -> tuple[np.ndarray, str]:
    """Estimate the support-restricted assignment matrix from sampled calibration."""

    if not support_states or tuple(sorted(set(support_states))) != support_states:
        raise ValueError("Support states must be unique, sorted, and non-empty")
    if support_states[-1] >= 2**width or support_states[0] < 0:
        raise ValueError("Support state is outside the computational basis")
    shots_by_state = split_shots(calibration_shots, len(support_states))
    matrix = np.zeros((len(support_states), len(support_states)), dtype=float)
    records: list[dict[str, object]] = []
    for column, (prepared_state, shots) in enumerate(
        zip(support_states, shots_by_state, strict=True)
    ):
        child_seed = derive_seed("subspace-readout-calibration", seed, prepared_state)
        probabilities = _assignment_probabilities(width, prepared_state, noise_spec)
        sampled = np.random.default_rng(child_seed).multinomial(shots, probabilities)
        matrix[:, column] = sampled[list(support_states)] / shots
        records.append(
            {
                "prepared_state": prepared_state,
                "shots": shots,
                "seed": child_seed,
                "support_counts": {
                    str(state): int(sampled[state])
                    for state in support_states
                    if sampled[state]
                },
                "outside_support_count": int(shots - sampled[list(support_states)].sum()),
            }
        )
    return matrix, json.dumps(records, sort_keys=True, separators=(",", ":"))


def observed_subspace_correction(
    assignment_matrix: np.ndarray,
    observed_probabilities: np.ndarray,
    support_states: tuple[int, ...],
    *,
    full_dimension: int,
    condition_threshold: float = 1e8,
) -> tuple[np.ndarray | None, float, str]:
    """Correct on a reduced support, then embed the result in the full basis."""

    size = len(support_states)
    if assignment_matrix.shape != (size, size):
        raise ValueError("Assignment matrix shape does not match the support")
    if observed_probabilities.shape != (size,):
        raise ValueError("Observed vector shape does not match the support")
    if full_dimension < 1 or full_dimension & (full_dimension - 1):
        raise ValueError("Full dimension must be a positive power of two")
    if any(state < 0 or state >= full_dimension for state in support_states):
        raise ValueError("Support state is outside the full basis")
    corrected, condition, status = constrained_readout_correction(
        assignment_matrix,
        observed_probabilities,
        condition_threshold=condition_threshold,
    )
    if corrected is None:
        return None, condition, status
    embedded = np.zeros(full_dimension, dtype=float)
    embedded[list(support_states)] = corrected
    return embedded, condition, status


def run_observed_subspace_mitigation(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    min_count: int = 2,
    protected_states: tuple[int, ...] = (),
    condition_threshold: float = 1e8,
) -> SubspaceReadoutResult:
    """Run target sampling and support-restricted calibration under one budget."""

    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    target_shots = total_shot_budget - calibration_shots
    raw = estimate_pauli(
        circuit,
        pauli_q0_first,
        shots=target_shots,
        seed=derive_seed("readout-target", seed),
        noise_spec=noise_spec,
    )
    support = select_observed_support(
        raw.raw_counts_json,
        circuit.num_qubits,
        min_count=min_count,
        protected_states=protected_states,
    )
    ledger = subspace_readout_ledger(total_shot_budget, len(support.states))
    assignment, calibration_json = estimate_subspace_assignment_matrix(
        circuit.num_qubits,
        support.states,
        ledger.calibration_shots,
        noise_spec,
        seed=seed,
    )
    counts = json.loads(raw.raw_counts_json)
    observed = np.array(
        [
            counts.get(format(state, f"0{circuit.num_qubits}b"), 0) / target_shots
            for state in support.states
        ],
        dtype=float,
    )
    corrected, condition, status = observed_subspace_correction(
        assignment,
        observed,
        support.states,
        full_dimension=2**circuit.num_qubits,
        condition_threshold=condition_threshold,
    )
    corrected_estimate = None
    corrected_json = None
    negative_mass = None
    if corrected is not None:
        corrected_estimate = expectation_from_probabilities(corrected, pauli_q0_first)
        corrected_json = json.dumps(corrected.tolist(), separators=(",", ":"))
        negative_mass = math.fsum(max(-float(value), 0.0) for value in corrected)
    return SubspaceReadoutResult(
        raw_estimate=raw.estimate,
        corrected_estimate=corrected_estimate,
        condition_estimate=condition,
        assignment_dimension=2**circuit.num_qubits,
        observed_support_size=len(support.states),
        reduced_matrix_bytes=8 * len(support.states) ** 2,
        retained_probability_mass=support.retained_probability_mass,
        discarded_probability_mass=support.discarded_probability_mass,
        negative_probability_mass=negative_mass,
        status=status,
        raw_counts_json=raw.raw_counts_json,
        calibration_counts_json=calibration_json,
        support_states_json=json.dumps(support.states, separators=(",", ":")),
        assignment_matrix_json=json.dumps(assignment.tolist(), separators=(",", ":")),
        corrected_probabilities_json=corrected_json,
        seed=seed,
        **ledger.__dict__,
    )
