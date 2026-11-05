import json

import numpy as np
import pytest
from qiskit import QuantumCircuit

from src.circuits import bell_circuit
from src.readout_mismatch import (
    calibration_noise_spec,
    exact_model_mismatch_error,
    ideal_measurement_probabilities,
    sample_target_counts,
    target_assignment_matrix,
    target_readout_model,
)


def test_target_assignment_columns_are_probabilities() -> None:
    for name in (
        "independent",
        "asymmetric",
        "pair_correlated",
        "calibration_mismatch",
    ):
        matrix = target_assignment_matrix(4, target_readout_model(name))
        assert (matrix >= 0).all()
        np.testing.assert_allclose(matrix.sum(axis=0), 1.0, atol=1e-15)


def test_pair_channel_creates_joint_flips() -> None:
    model = target_readout_model("pair_correlated")
    pair_only = type(model)("pair_only", 0.0, 0.0, pair_flip_probability=0.1)
    observed = target_assignment_matrix(2, pair_only)[:, 0]

    np.testing.assert_allclose(observed, [0.9, 0.0, 0.0, 0.1], atol=1e-15)


def test_target_sampling_is_seed_reproducible() -> None:
    probabilities = ideal_measurement_probabilities(bell_circuit(), "ZZ")
    model = target_readout_model("pair_correlated")

    first = sample_target_counts(probabilities, model, shots=1024, seed=1729)
    second = sample_target_counts(probabilities, model, shots=1024, seed=1729)

    assert first == second
    assert sum(json.loads(first).values()) == 1024


def test_matching_independent_calibration_has_zero_model_floor() -> None:
    probabilities = ideal_measurement_probabilities(bell_circuit(), "ZZ")
    model = target_readout_model("independent")
    error = exact_model_mismatch_error(
        probabilities, "ZZ", model, calibration_noise_spec("independent")
    )

    assert error == pytest.approx(0.0, abs=1e-14)


def test_correlated_target_has_nonzero_model_floor() -> None:
    circuit = QuantumCircuit(2)
    probabilities = ideal_measurement_probabilities(circuit, "ZI")
    model = target_readout_model("pair_correlated")
    error = exact_model_mismatch_error(
        probabilities, "ZI", model, calibration_noise_spec("pair_correlated")
    )

    assert abs(error) > 1e-3
