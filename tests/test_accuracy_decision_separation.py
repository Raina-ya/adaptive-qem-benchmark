import numpy as np

from src.decision_metrics import evaluate_decision


def test_lower_marginal_rmse_can_have_higher_decision_error() -> None:
    exact_a, exact_b = 0.01, -0.01
    estimator_x = np.array([-0.02, 0.02])
    estimator_y = np.array([0.07, 0.05])
    references = np.array([exact_a, exact_b])
    rmse_x = float(np.sqrt(np.mean((estimator_x - references) ** 2)))
    rmse_y = float(np.sqrt(np.mean((estimator_y - references) ** 2)))
    error_x = evaluate_decision(*estimator_x, exact_a, exact_b).decision_error
    error_y = evaluate_decision(*estimator_y, exact_a, exact_b).decision_error
    assert rmse_x < rmse_y
    assert error_x and not error_y
