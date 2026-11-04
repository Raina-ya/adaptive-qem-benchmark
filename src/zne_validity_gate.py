"""Condition-level preregistered credibility gate for ZNE evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_SIGNAL_RETENTION = 0.10
MAX_NEGATIVE_MASS = 0.05
MAX_PHYSICAL_OVERSHOOT = 1e-12
MAX_DESTROYED_RATE = 0.0
MAX_UNSTABLE_RATE = 0.05
NORMAL_95 = 1.959963984540054


@dataclass(frozen=True)
class ZNEGateResult:
    """Aggregate evidence and pass/fail checks for one benchmark condition."""

    repeat_count: int
    true_gain_mean: float
    true_gain_ci_low: float
    true_gain_ci_high: float
    garbage_advantage_mean: float
    garbage_advantage_ci_low: float
    garbage_advantage_ci_high: float
    final_retention_mean: float
    final_retention_ci_low: float
    final_retention_ci_high: float
    maximum_negative_mass: float
    maximum_physical_overshoot: float
    destroyed_rate: float
    unstable_rate: float
    signal_retention_pass: bool
    physical_bounds_pass: bool
    negative_mass_pass: bool
    garbage_control_pass: bool
    gain_uncertainty_pass: bool
    trend_pass: bool
    validity_status: str
    failure_reasons: tuple[str, ...]


def _mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    standard_error = float(values.std(ddof=1) / math.sqrt(len(values)))
    return mean, mean - NORMAL_95 * standard_error, mean + NORMAL_95 * standard_error


def evaluate_zne_gate(
    genuine_true_gains: np.ndarray,
    garbage_true_gains: np.ndarray,
    final_retentions: np.ndarray,
    negative_masses: np.ndarray,
    physical_overshoots: np.ndarray,
    destroyed: np.ndarray,
    unstable: np.ndarray,
) -> ZNEGateResult:
    """Apply every frozen C5 check; no single metric can override a failure."""

    arrays = tuple(
        np.asarray(values)
        for values in (
            genuine_true_gains,
            garbage_true_gains,
            final_retentions,
            negative_masses,
            physical_overshoots,
            destroyed,
            unstable,
        )
    )
    lengths = {len(values) for values in arrays}
    if len(lengths) != 1 or next(iter(lengths), 0) < 2:
        raise ValueError("Gate inputs must contain at least two matched repeats")
    numeric = arrays[:5]
    if any(np.any(~np.isfinite(values.astype(float))) for values in numeric):
        raise ValueError("Gate metrics must be finite")
    gains, garbage, retention, mass, overshoot = (values.astype(float) for values in numeric)
    destroyed_flags = arrays[5].astype(bool)
    unstable_flags = arrays[6].astype(bool)

    gain_mean, gain_low, gain_high = _mean_interval(gains)
    advantage_mean, advantage_low, advantage_high = _mean_interval(gains - garbage)
    retention_mean, retention_low, retention_high = _mean_interval(retention)
    maximum_mass = float(mass.max())
    maximum_overshoot = float(overshoot.max())
    destroyed_rate = float(destroyed_flags.mean())
    unstable_rate = float(unstable_flags.mean())

    checks = {
        "signal_retention": retention_low > MIN_SIGNAL_RETENTION,
        "physical_bounds": maximum_overshoot <= MAX_PHYSICAL_OVERSHOOT,
        "negative_mass": maximum_mass <= MAX_NEGATIVE_MASS,
        "garbage_control": advantage_low > 0.0,
        "gain_uncertainty": gain_low > 0.0,
        "trend": destroyed_rate <= MAX_DESTROYED_RATE and unstable_rate <= MAX_UNSTABLE_RATE,
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    return ZNEGateResult(
        repeat_count=len(gains),
        true_gain_mean=gain_mean,
        true_gain_ci_low=gain_low,
        true_gain_ci_high=gain_high,
        garbage_advantage_mean=advantage_mean,
        garbage_advantage_ci_low=advantage_low,
        garbage_advantage_ci_high=advantage_high,
        final_retention_mean=retention_mean,
        final_retention_ci_low=retention_low,
        final_retention_ci_high=retention_high,
        maximum_negative_mass=maximum_mass,
        maximum_physical_overshoot=maximum_overshoot,
        destroyed_rate=destroyed_rate,
        unstable_rate=unstable_rate,
        signal_retention_pass=checks["signal_retention"],
        physical_bounds_pass=checks["physical_bounds"],
        negative_mass_pass=checks["negative_mass"],
        garbage_control_pass=checks["garbage_control"],
        gain_uncertainty_pass=checks["gain_uncertainty"],
        trend_pass=checks["trend"],
        validity_status="credible" if not failures else "rejected",
        failure_reasons=failures,
    )
