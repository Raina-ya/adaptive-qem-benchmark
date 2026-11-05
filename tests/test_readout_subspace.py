import json

import numpy as np
import pytest

from src.circuits import bell_circuit
from src.mitigation.readout_subspace import (
    estimate_subspace_assignment_matrix,
    observed_subspace_correction,
    run_observed_subspace_mitigation,
    select_observed_support,
)
from src.noise_models import NoiseSpec


def test_full_support_matches_full_space_correction() -> None:
    assignment = np.array(
        [
            [0.82, 0.08, 0.08, 0.02],
            [0.08, 0.82, 0.02, 0.08],
            [0.08, 0.02, 0.82, 0.08],
            [0.02, 0.08, 0.08, 0.82],
        ]
    )
    exact = np.array([0.45, 0.05, 0.15, 0.35])
    observed = assignment @ exact

    corrected, _, status = observed_subspace_correction(
        assignment, observed, (0, 1, 2, 3), full_dimension=4
    )

    assert status == "ok"
    assert corrected is not None
    np.testing.assert_allclose(corrected, exact, rtol=0.0, atol=1e-14)


def test_support_reports_discarded_mass_without_reference_access() -> None:
    support = select_observed_support(
        json.dumps({"00": 90, "01": 8, "10": 1, "11": 1}),
        2,
        min_count=2,
    )

    assert support.states == (0, 1)
    assert support.retained_probability_mass == pytest.approx(0.98)
    assert support.discarded_probability_mass == pytest.approx(0.02)


def test_preregistered_decision_critical_state_is_not_discarded() -> None:
    support = select_observed_support(
        json.dumps({"000": 98, "001": 1, "111": 1}),
        3,
        min_count=2,
        protected_states=(7,),
    )

    assert support.states == (0, 7)
    assert support.discarded_probability_mass == pytest.approx(0.01)


def test_sampled_subspace_calibration_is_seed_reproducible() -> None:
    spec = NoiseSpec("asymmetric", readout_p01=0.02, readout_p10=0.07)
    first = estimate_subspace_assignment_matrix(3, (0, 3, 7), 600, spec, seed=1729)
    second = estimate_subspace_assignment_matrix(3, (0, 3, 7), 600, spec, seed=1729)

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1] == second[1]


def test_subspace_integration_preserves_budget_and_diagnostics() -> None:
    result = run_observed_subspace_mitigation(
        bell_circuit(),
        "ZZ",
        total_shot_budget=4096,
        noise_spec=NoiseSpec("independent", readout_p01=0.03, readout_p10=0.03),
        seed=20261104,
    )

    assert result.status == "ok"
    assert result.corrected_estimate is not None
    assert result.target_shots + result.calibration_shots == 4096
    assert result.calibration_circuit_executions == result.observed_support_size
    assert result.reduced_matrix_bytes == 8 * result.observed_support_size**2
    assert result.retained_probability_mass + result.discarded_probability_mass == pytest.approx(1)
    assert json.loads(result.support_states_json)


def test_subspace_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        observed_subspace_correction(
            np.eye(2), np.array([1.0]), (0, 1), full_dimension=4
        )
