import numpy as np
import pytest

from src.decision_metrics import (
    decision_error_interval,
    evaluate_decision,
    paired_gap_statistics,
)


def test_decision_error_tie_and_wrong_winner() -> None:
    assert evaluate_decision(0.0, 0.0, 0.01, -0.01).decision_error
    outcome = evaluate_decision(-0.1, 0.1, 0.01, -0.01)
    assert outcome.decision_error
    assert outcome.regret == pytest.approx(0.02)


def test_wilson_interval_contains_observed_rate() -> None:
    rate, low, high = decision_error_interval(np.array([True, False, False, True]))
    assert rate == 0.5
    assert low < rate < high


def test_paired_gap_statistics_include_covariance() -> None:
    left = np.array([0.1, 0.2, 0.3])
    right = np.array([0.0, 0.1, 0.2])
    covariance, gap_variance = paired_gap_statistics(left, right)
    assert covariance == pytest.approx(0.01)
    assert gap_variance == pytest.approx(0.0, abs=1e-15)


def test_exact_tie_is_rejected() -> None:
    with pytest.raises(ValueError, match="unique winner"):
        evaluate_decision(1.0, 0.0, 0.5, 0.5)
