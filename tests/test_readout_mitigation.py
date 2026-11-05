import os

import numpy as np

from src.circuits import bell_circuit
from src.config import derive_seed
from src.mitigation.readout import (
    constrained_readout_correction,
    estimate_assignment_matrix,
    run_readout_mitigation,
)
from src.noise_models import noise_spec


def test_zero_readout_error_yields_identity_assignment_matrix() -> None:
    matrix, shots_by_state, calibration_counts_json = estimate_assignment_matrix(
        2,
        400,
        noise_spec("readout", "zero"),
        seed=derive_seed("identity-calibration"),
    )
    np.testing.assert_allclose(matrix, np.eye(4), atol=0.0)
    assert sum(shots_by_state) == 400
    assert calibration_counts_json


def test_ill_conditioned_matrix_returns_explicit_status() -> None:
    singular = np.ones((2, 2)) / 2
    corrected, condition, status = constrained_readout_correction(
        singular, np.array([0.5, 0.5])
    )
    assert corrected is None
    assert np.isinf(condition) or condition > 1e8
    assert status == "ill_conditioned"


def test_readout_mitigation_preserves_raw_and_corrected_results() -> None:
    result = run_readout_mitigation(
        bell_circuit(),
        "XX",
        total_shot_budget=4096,
        noise_spec=noise_spec("readout", "medium"),
        seed=derive_seed("readout-integration"),
    )
    assert result.status == "ok"
    assert result.corrected_estimate is not None
    assert result.raw_counts_json
    assert result.calibration_counts_json
    assert result.corrected_probabilities_json
    assert result.assignment_matrix_json
    assert result.target_shots + result.calibration_shots == 4096
    assert abs(1.0 - result.corrected_estimate) <= abs(1.0 - result.raw_estimate)


def test_fixed_matrix_correction_is_bitwise_stable_across_thread_settings() -> None:
    matrix = np.array(
        [
            [0.91, 0.08, 0.07, 0.01],
            [0.04, 0.87, 0.01, 0.06],
            [0.04, 0.01, 0.88, 0.07],
            [0.01, 0.04, 0.04, 0.86],
        ]
    )
    observed = np.array([0.47, 0.03, 0.04, 0.46])
    original = os.environ.get("OPENBLAS_NUM_THREADS")
    try:
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        first = constrained_readout_correction(matrix, observed)
        os.environ["OPENBLAS_NUM_THREADS"] = "4"
        second = constrained_readout_correction(matrix, observed)
    finally:
        if original is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = original

    assert first[2] == second[2] == "ok"
    assert first[1].hex() == second[1].hex()
    assert first[0] is not None and second[0] is not None
    assert first[0].tobytes() == second[0].tobytes()


def test_large_full_matrix_uses_convergent_projected_solver() -> None:
    matrix = np.eye(128)
    observed = np.zeros(128)
    observed[[3, 17]] = (0.7, 0.3)

    corrected, condition, status = constrained_readout_correction(matrix, observed)

    assert status == "ok"
    assert condition == 1.0
    np.testing.assert_allclose(corrected, observed, rtol=0.0, atol=1e-12)
