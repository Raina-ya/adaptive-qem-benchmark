"""Synthetic target readout channels for calibration-mismatch experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from src.mitigation.readout_tensor import tensor_product_correction
from src.noise_models import NoiseSpec
from src.observables import expectation_from_probabilities, measurement_circuit


@dataclass(frozen=True)
class TargetReadoutModel:
    """Independent bit errors followed by disjoint correlated pair flips."""

    name: str
    readout_p01: float
    readout_p10: float
    pair_flip_probability: float = 0.0

    def __post_init__(self) -> None:
        for value in (
            self.readout_p01,
            self.readout_p10,
            self.pair_flip_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Readout probabilities must lie in [0, 1]")


def target_readout_model(name: str) -> TargetReadoutModel:
    """Return one preregistered target channel."""

    if name == "independent":
        return TargetReadoutModel(name, 0.03, 0.03)
    if name == "asymmetric":
        return TargetReadoutModel(name, 0.015, 0.06)
    if name == "pair_correlated":
        return TargetReadoutModel(name, 0.02, 0.02, pair_flip_probability=0.03)
    if name == "calibration_mismatch":
        return TargetReadoutModel(name, 0.015, 0.08)
    raise ValueError(f"Unknown target readout model: {name}")


def calibration_noise_spec(name: str) -> NoiseSpec:
    """Return the independent approximation used for calibration."""

    model = target_readout_model(name)
    if name == "calibration_mismatch":
        return NoiseSpec("calibration_independent", readout_p01=0.03, readout_p10=0.03)
    return NoiseSpec(
        f"calibration_{name}",
        readout_p01=model.readout_p01,
        readout_p10=model.readout_p10,
    )


def independent_assignment_matrix(width: int, p01: float, p10: float) -> np.ndarray:
    """Return P(observed row | prepared column) for independent readout."""

    if width < 1:
        raise ValueError("Width must be positive")
    local = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)
    matrix = local.copy()
    for _ in range(width - 1):
        matrix = np.kron(local, matrix)
    return matrix


def _pair_flip_matrix(width: int, probability: float) -> np.ndarray:
    dimension = 2**width
    pairs = tuple(range(0, width - 1, 2))
    matrix = np.zeros((dimension, dimension), dtype=float)
    for source in range(dimension):
        for pattern in range(2 ** len(pairs)):
            target = source
            weight = 1.0
            for index, first_qubit in enumerate(pairs):
                flipped = (pattern >> index) & 1
                weight *= probability if flipped else 1.0 - probability
                if flipped:
                    target ^= (1 << first_qubit) | (1 << (first_qubit + 1))
            matrix[target, source] += weight
    return matrix


def target_assignment_matrix(width: int, model: TargetReadoutModel) -> np.ndarray:
    """Compose independent errors with the declared correlated pair channel."""

    independent = independent_assignment_matrix(
        width, model.readout_p01, model.readout_p10
    )
    if model.pair_flip_probability == 0.0:
        return independent
    return _pair_flip_matrix(width, model.pair_flip_probability) @ independent


def ideal_measurement_probabilities(
    circuit: QuantumCircuit, pauli_q0_first: str
) -> np.ndarray:
    """Return noiseless probabilities in the observable measurement basis."""

    measured = measurement_circuit(circuit, pauli_q0_first)
    rotated = measured.remove_final_measurements(inplace=False)
    return np.asarray(Statevector.from_instruction(rotated).probabilities(), dtype=float)


def sample_target_counts(
    ideal_probabilities: np.ndarray,
    model: TargetReadoutModel,
    *,
    shots: int,
    seed: int,
) -> str:
    """Sample target observations and preserve them as sorted JSON counts."""

    probabilities = np.asarray(ideal_probabilities, dtype=float)
    width = int(np.log2(probabilities.size))
    if shots < 1 or probabilities.shape != (2**width,):
        raise ValueError("Shots and ideal probability dimension must be valid")
    observed = target_assignment_matrix(width, model) @ probabilities
    sampled = np.random.default_rng(seed).multinomial(shots, observed)
    counts = {
        format(state, f"0{width}b"): int(count)
        for state, count in enumerate(sampled)
        if count
    }
    return json.dumps(counts, sort_keys=True, separators=(",", ":"))


def exact_model_mismatch_error(
    ideal_probabilities: np.ndarray,
    pauli_q0_first: str,
    target_model: TargetReadoutModel,
    calibration_spec: NoiseSpec,
) -> float:
    """Return the infinite-shot full-support error from model mismatch alone."""

    width = len(pauli_q0_first)
    target_observed = target_assignment_matrix(width, target_model) @ ideal_probabilities
    local = np.array(
        [
            [1.0 - calibration_spec.readout_p01, calibration_spec.readout_p10],
            [calibration_spec.readout_p01, 1.0 - calibration_spec.readout_p10],
        ],
        dtype=float,
    )
    corrected, _, status = tensor_product_correction((local,) * width, target_observed)
    if corrected is None or status != "ok":
        raise ValueError("Exact calibration model is not invertible")
    reference = expectation_from_probabilities(ideal_probabilities, pauli_q0_first)
    return expectation_from_probabilities(corrected, pauli_q0_first) - reference
