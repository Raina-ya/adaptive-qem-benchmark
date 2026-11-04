"""Readout mitigation followed by ZNE under one fixed shot budget."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.folding import fold_global
from src.mitigation.readout import (
    constrained_readout_correction,
    counts_probability_vector,
    estimate_assignment_matrix,
)
from src.mitigation.zne import linear_extrapolate
from src.noise_models import NoiseSpec
from src.observable_floors import pauli_observable_floor
from src.observables import exact_expectation, expectation_from_probabilities
from src.resource_accounting import split_shots
from src.zne_validity import assess_zne_validity, extrapolate_probability_distribution

FACTORS = (1, 3, 5)
ALLOCATION_POLICIES = ("balanced", "base_heavy")


@dataclass(frozen=True)
class CombinedBudget:
    """One calibration allocation plus all folded target allocations."""

    total_shot_budget: int
    calibration_shots: int
    factor_shots: tuple[int, ...]
    allocation_policy: str

    def __post_init__(self) -> None:
        if self.total_shot_budget <= 0:
            raise ValueError("Total shot budget must be positive")
        if self.calibration_shots <= 0 or any(shots <= 0 for shots in self.factor_shots):
            raise ValueError("Calibration and every factor must receive positive shots")
        if self.calibration_shots + sum(self.factor_shots) != self.total_shot_budget:
            raise ValueError("Calibration and factor shots must equal the total budget")
        if self.allocation_policy not in ALLOCATION_POLICIES:
            raise ValueError(f"Unknown allocation policy: {self.allocation_policy}")


@dataclass(frozen=True)
class CombinedMitigationResult:
    """Preserved intermediate values from readout correction followed by ZNE."""

    status: str
    error: str
    total_shot_budget: int
    calibration_shots: int
    factors: tuple[int, ...]
    factor_shots: tuple[int, ...]
    allocation_policy: str
    calibration_circuit_executions: int
    folded_circuit_executions: int
    condition_number: float
    factor_seeds: tuple[int, ...]
    factor_raw_estimates: tuple[float, ...]
    factor_corrected_estimates: tuple[float, ...]
    factor_corrected_standard_errors: tuple[float, ...]
    factor_raw_counts_json: tuple[str, ...]
    factor_corrected_probabilities_json: tuple[str, ...]
    calibration_counts_json: str
    assignment_matrix_json: str
    raw_extrapolated_estimate: float | None
    corrected_extrapolated_estimate: float | None
    corrected_slope: float | None
    corrected_fit_residual: float | None
    out_of_range: bool | None
    negative_mass: float | None
    physical_overshoot: float | None
    final_signal_retention: float | None
    validity_classification: str
    result_label: str = "noisy_simulated"


def _weighted_split(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    if total < len(weights) or any(weight <= 0.0 for weight in weights):
        raise ValueError("Every weighted bucket must receive positive shots")
    normalized = tuple(weight / math.fsum(weights) for weight in weights)
    exact = tuple(total * weight for weight in normalized)
    allocated = [math.floor(value) for value in exact]
    for index in sorted(
        range(len(weights)), key=lambda item: (-(exact[item] - allocated[item]), item)
    )[: total - sum(allocated)]:
        allocated[index] += 1
    return tuple(allocated)


def allocate_combined_budget(
    total_shot_budget: int, allocation_policy: str
) -> CombinedBudget:
    """Apply the frozen 20% calibration and factor-allocation policy."""

    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    target_shots = total_shot_budget - calibration_shots
    if allocation_policy == "balanced":
        factor_shots = split_shots(target_shots, len(FACTORS))
    elif allocation_policy == "base_heavy":
        factor_shots = _weighted_split(target_shots, (0.5, 0.3, 0.2))
    else:
        raise ValueError(f"Unknown allocation policy: {allocation_policy}")
    return CombinedBudget(
        total_shot_budget=total_shot_budget,
        calibration_shots=calibration_shots,
        factor_shots=factor_shots,
        allocation_policy=allocation_policy,
    )


def run_combined_mitigation(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    allocation_policy: str,
    folder: Callable[[QuantumCircuit, int], QuantumCircuit] = fold_global,
    condition_threshold: float = 1e8,
) -> CombinedMitigationResult:
    """Correct every folded distribution, then extrapolate without clipping."""

    budget = allocate_combined_budget(total_shot_budget, allocation_policy)
    assignment, _, calibration_counts = estimate_assignment_matrix(
        circuit.num_qubits,
        budget.calibration_shots,
        noise_spec,
        seed=derive_seed("combined-calibration", seed),
    )
    raw_estimates: list[float] = []
    corrected_estimates: list[float] = []
    corrected_uncertainties: list[float] = []
    raw_counts: list[str] = []
    corrected_probabilities: list[str] = []
    factor_seeds: list[int] = []
    condition_number = math.nan
    status = "ok"
    error = ""

    for factor, shots in zip(FACTORS, budget.factor_shots, strict=True):
        child_seed = derive_seed("combined-factor", seed, factor)
        sampled = estimate_pauli(
            folder(circuit, factor),
            pauli_q0_first,
            shots=shots,
            seed=child_seed,
            noise_spec=noise_spec,
        )
        factor_seeds.append(child_seed)
        raw_estimates.append(sampled.estimate)
        raw_counts.append(sampled.raw_counts_json)
        observed = counts_probability_vector(sampled.raw_counts_json, circuit.num_qubits)
        corrected, condition_number, correction_status = constrained_readout_correction(
            assignment, observed, condition_threshold=condition_threshold
        )
        if corrected is None:
            status = correction_status
            error = f"readout correction failed at factor {factor}: {correction_status}"
            break
        estimate = expectation_from_probabilities(corrected, pauli_q0_first)
        corrected_estimates.append(estimate)
        corrected_uncertainties.append(math.sqrt(max(1.0 - estimate**2, 0.0) / shots))
        corrected_probabilities.append(json.dumps(corrected.tolist(), separators=(",", ":")))

    assignment_json = json.dumps(assignment.tolist(), separators=(",", ":"))
    common = {
        "status": status,
        "error": error,
        "total_shot_budget": budget.total_shot_budget,
        "calibration_shots": budget.calibration_shots,
        "factors": FACTORS,
        "factor_shots": budget.factor_shots,
        "allocation_policy": budget.allocation_policy,
        "calibration_circuit_executions": 2**circuit.num_qubits,
        "folded_circuit_executions": len(raw_estimates),
        "condition_number": condition_number,
        "factor_seeds": tuple(factor_seeds),
        "factor_raw_estimates": tuple(raw_estimates),
        "factor_corrected_estimates": tuple(corrected_estimates),
        "factor_corrected_standard_errors": tuple(corrected_uncertainties),
        "factor_raw_counts_json": tuple(raw_counts),
        "factor_corrected_probabilities_json": tuple(corrected_probabilities),
        "calibration_counts_json": calibration_counts,
        "assignment_matrix_json": assignment_json,
    }
    if status != "ok":
        return CombinedMitigationResult(
            **common,
            raw_extrapolated_estimate=None,
            corrected_extrapolated_estimate=None,
            corrected_slope=None,
            corrected_fit_residual=None,
            out_of_range=None,
            negative_mass=None,
            physical_overshoot=None,
            final_signal_retention=None,
            validity_classification="not_evaluated",
        )

    raw_extrapolated, _, _ = linear_extrapolate(FACTORS, tuple(raw_estimates))
    corrected_extrapolated, slope, residual = linear_extrapolate(
        FACTORS, tuple(corrected_estimates)
    )
    probability_diagnostic = extrapolate_probability_distribution(
        FACTORS,
        tuple(np.asarray(json.loads(payload), dtype=float) for payload in corrected_probabilities),
    )
    reference = exact_expectation(circuit, pauli_q0_first)
    validity = assess_zne_validity(
        FACTORS,
        tuple(corrected_estimates),
        tuple(corrected_uncertainties),
        pauli_observable_floor(pauli_q0_first, reference),
    )
    return CombinedMitigationResult(
        **common,
        raw_extrapolated_estimate=raw_extrapolated,
        corrected_extrapolated_estimate=corrected_extrapolated,
        corrected_slope=slope,
        corrected_fit_residual=residual,
        out_of_range=not -1.0 <= corrected_extrapolated <= 1.0,
        negative_mass=probability_diagnostic.negative_mass,
        physical_overshoot=validity.physical_overshoot,
        final_signal_retention=validity.signal_retention_ratios[-1],
        validity_classification=validity.validity_classification,
    )
