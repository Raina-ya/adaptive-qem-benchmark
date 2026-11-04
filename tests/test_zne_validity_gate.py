import numpy as np
import pytest

from src.zne_validity_gate import evaluate_zne_gate


def credible_inputs() -> dict[str, np.ndarray]:
    return {
        "genuine_true_gains": np.array([0.08, 0.10, 0.12, 0.09]),
        "garbage_true_gains": np.array([-0.10, -0.08, -0.09, -0.11]),
        "final_retentions": np.array([0.50, 0.55, 0.60, 0.58]),
        "negative_masses": np.array([0.0, 0.001, 0.0, 0.002]),
        "physical_overshoots": np.zeros(4),
        "destroyed": np.zeros(4, dtype=bool),
        "unstable": np.zeros(4, dtype=bool),
    }


def test_all_checks_must_pass_for_credible_status() -> None:
    result = evaluate_zne_gate(**credible_inputs())
    assert result.validity_status == "credible"
    assert result.failure_reasons == ()
    assert result.true_gain_ci_low > 0.0
    assert result.garbage_advantage_ci_low > 0.0


@pytest.mark.parametrize(
    ("field", "replacement", "reason"),
    [
        ("final_retentions", np.array([0.01, 0.02, 0.03, 0.02]), "signal_retention"),
        ("negative_masses", np.array([0.0, 0.0, 0.06, 0.0]), "negative_mass"),
        ("physical_overshoots", np.array([0.0, 0.0, 1e-3, 0.0]), "physical_bounds"),
        ("genuine_true_gains", np.array([-0.01, 0.0, 0.01, 0.0]), "gain_uncertainty"),
        ("garbage_true_gains", np.array([0.20, 0.20, 0.20, 0.20]), "garbage_control"),
        ("destroyed", np.array([False, False, True, False]), "trend"),
        ("unstable", np.array([False, False, True, False]), "trend"),
    ],
)
def test_any_failed_check_rejects_evidence(field, replacement, reason: str) -> None:
    inputs = credible_inputs()
    inputs[field] = replacement
    result = evaluate_zne_gate(**inputs)
    assert result.validity_status == "rejected"
    assert reason in result.failure_reasons


def test_unmatched_or_nonfinite_inputs_fail() -> None:
    inputs = credible_inputs()
    inputs["negative_masses"] = np.array([0.0])
    with pytest.raises(ValueError, match="matched"):
        evaluate_zne_gate(**inputs)
    inputs = credible_inputs()
    inputs["final_retentions"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_zne_gate(**inputs)
