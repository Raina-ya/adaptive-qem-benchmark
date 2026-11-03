"""Metrics for two-choice decisions and their uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.metrics import wilson_interval


@dataclass(frozen=True)
class DecisionOutcome:
    """One estimated two-choice decision."""

    estimated_gap: float
    decision_error: bool
    regret: float


def evaluate_decision(
    estimate_a: float,
    estimate_b: float,
    exact_a: float,
    exact_b: float,
) -> DecisionOutcome:
    """Evaluate argmax correctness; ties count as errors for a unique winner."""

    exact_gap = exact_a - exact_b
    if exact_gap == 0:
        raise ValueError("The exact decision must have a unique winner")
    estimated_gap = estimate_a - estimate_b
    error = estimated_gap * exact_gap <= 0
    regret = abs(exact_gap) if error else 0.0
    return DecisionOutcome(estimated_gap, error, regret)


def decision_error_interval(errors: np.ndarray) -> tuple[float, float, float]:
    """Return error probability and its 95% Wilson interval."""

    values = np.asarray(errors, dtype=bool)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("At least one decision outcome is required")
    count = int(values.sum())
    low, high = wilson_interval(count, len(values))
    return count / len(values), low, high


def paired_gap_statistics(estimates_a: np.ndarray, estimates_b: np.ndarray) -> tuple[float, float]:
    """Return paired covariance and variance of A-minus-B estimates."""

    left = np.asarray(estimates_a, dtype=float)
    right = np.asarray(estimates_b, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Matched one-dimensional estimates are required")
    covariance = float(np.cov(left, right, ddof=1)[0, 1])
    gap_variance = float(np.var(left - right, ddof=1))
    if not math.isfinite(covariance) or not math.isfinite(gap_variance):
        raise ValueError("Decision statistics must be finite")
    return covariance, gap_variance
