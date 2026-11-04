import json

import numpy as np
import pytest

from src.circuits import bell_circuit
from src.mitigation.readout_tensor import (
    estimate_tensor_assignment_matrices,
    run_tensor_readout_mitigation,
    tensor_product_correction,
    tensor_readout_ledger,
)
from src.noise_models import NoiseSpec


def test_tensor_ledger_counts_every_shot_and_calibration_circuit() -> None:
    ledger = tensor_readout_ledger(4096, 6)

    assert ledger.calibration_shots == 819
    assert ledger.target_shots == 3277
    assert ledger.calibration_circuit_executions == 12
    assert ledger.target_shots + ledger.calibration_shots == 4096


def test_exact_tensor_correction_preserves_qubit_ordering() -> None:
    q0 = np.array([[0.93, 0.04], [0.07, 0.96]])
    q1 = np.array([[0.89, 0.02], [0.11, 0.98]])
    exact = np.array([0.05, 0.15, 0.30, 0.50])
    observed = np.kron(q1, q0) @ exact

    corrected, _, status = tensor_product_correction((q0, q1), observed)

    assert status == "ok"
    assert corrected is not None
    np.testing.assert_allclose(corrected, exact, rtol=0.0, atol=1e-15)


def test_tensor_correction_does_not_clip_negative_probabilities() -> None:
    matrix = np.array([[0.9, 0.1], [0.1, 0.9]])
    corrected, _, status = tensor_product_correction(
        (matrix,), np.array([1.0, 0.0])
    )

    assert status == "ok"
    assert corrected is not None
    assert corrected[1] < 0.0


def test_singular_local_matrix_returns_explicit_failure() -> None:
    singular = np.ones((2, 2)) / 2
    corrected, condition, status = tensor_product_correction(
        (singular,), np.array([0.5, 0.5])
    )

    assert corrected is None
    assert np.isinf(condition)
    assert status == "ill_conditioned"


def test_sampled_tensor_calibration_is_seed_reproducible() -> None:
    spec = NoiseSpec("asymmetric", readout_p01=0.02, readout_p10=0.07)
    first = estimate_tensor_assignment_matrices(3, 600, spec, seed=1729)
    second = estimate_tensor_assignment_matrices(3, 600, spec, seed=1729)

    for left, right in zip(first[0], second[0], strict=True):
        np.testing.assert_array_equal(left, right)
    assert first[1:] == second[1:]


def test_tensor_integration_preserves_diagnostics_and_budget() -> None:
    result = run_tensor_readout_mitigation(
        bell_circuit(),
        "XX",
        total_shot_budget=4096,
        noise_spec=NoiseSpec("independent", readout_p01=0.03, readout_p10=0.03),
        seed=20261104,
    )

    assert result.status == "ok"
    assert result.corrected_estimate is not None
    assert result.total_shot_budget == result.target_shots + result.calibration_shots
    assert result.calibration_circuit_executions == 4
    assert len(json.loads(result.assignment_matrices_json)) == 2
    assert result.negative_probability_mass is not None
    assert result.corrected_probabilities_json is not None


def test_tensor_correction_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="tensor width"):
        tensor_product_correction((np.eye(2), np.eye(2)), np.array([0.5, 0.5]))
