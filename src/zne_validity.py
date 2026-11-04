"""Preregistered factor-level signal-retention diagnostics for linear ZNE."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.mitigation.zne import linear_extrapolate
from src.observable_floors import ObservableFloor

SIGNIFICANCE_MULTIPLIER = 2.0
DESTROYED_RETENTION_THRESHOLD = 0.10
DECAYED_RETENTION_THRESHOLD = 0.50
MAX_NORMALIZED_FIT_RESIDUAL = 0.25
PHYSICAL_TOLERANCE = 1e-12

VALIDITY_CLASSIFICATIONS = frozenset(
    {
        "signal_retained",
        "signal_decayed",
        "signal_destroyed",
        "extrapolation_unstable",
        "artefact_suspected",
        "inconclusive",
    }
)


@dataclass(frozen=True)
class ZNEValidityResult:
    """Complete scalar diagnostics for one factor-level ZNE result."""

    factors: tuple[int, ...]
    factor_estimates: tuple[float, ...]
    factor_uncertainties: tuple[float, ...]
    signal_distances: tuple[float, ...]
    signal_retention_ratios: tuple[float, ...]
    monotonicity_status: str
    extrapolated_value: float
    physical_overshoot: float
    fit_residual: float
    normalized_fit_residual: float
    validity_classification: str


def _validate_inputs(
    factors: tuple[int, ...],
    estimates: tuple[float, ...],
    uncertainties: tuple[float, ...],
) -> None:
    if not (len(factors) == len(estimates) == len(uncertainties)) or len(factors) < 2:
        raise ValueError("Factors, estimates, and uncertainties must have matching lengths")
    if tuple(sorted(factors)) != factors or len(set(factors)) != len(factors):
        raise ValueError("Noise factors must be strictly increasing")
    if any(factor <= 0 for factor in factors):
        raise ValueError("Noise factors must be positive")
    if any(not math.isfinite(value) for value in (*estimates, *uncertainties)):
        raise ValueError("Estimates and uncertainties must be finite")
    if any(value < 0.0 for value in uncertainties):
        raise ValueError("Factor uncertainties must be non-negative")


def _monotonicity(distances: np.ndarray, uncertainties: np.ndarray) -> str:
    differences = np.diff(distances)
    tolerances = SIGNIFICANCE_MULTIPLIER * np.sqrt(uncertainties[:-1] ** 2 + uncertainties[1:] ** 2)
    if np.any(differences > tolerances):
        return "non_monotonic"
    if np.all(differences < -tolerances):
        return "monotonic_decay"
    return "indeterminate"


def assess_zne_validity(
    factors: tuple[int, ...],
    estimates: tuple[float, ...],
    uncertainties: tuple[float, ...],
    observable_floor: ObservableFloor,
) -> ZNEValidityResult:
    """Assess signal retention without clipping unphysical extrapolates."""

    _validate_inputs(factors, estimates, uncertainties)
    values = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(uncertainties, dtype=float)
    floor = observable_floor.expected_noise_floor
    distances = np.abs(values - floor)
    baseline_distance = float(distances[0])
    baseline_identifiable = baseline_distance > SIGNIFICANCE_MULTIPLIER * standard_errors[0]
    if baseline_distance == 0.0:
        retention = np.full(len(values), np.nan)
    else:
        retention = distances / baseline_distance

    extrapolated, _, fit_residual = linear_extrapolate(factors, estimates)
    physical_overshoot = max(
        observable_floor.physical_minimum - extrapolated,
        extrapolated - observable_floor.physical_maximum,
        0.0,
    )
    residual_scale = max(baseline_distance**2, np.finfo(float).eps)
    normalized_residual = fit_residual / residual_scale
    monotonicity = _monotonicity(distances, standard_errors)
    signed_signal = values - floor
    sign_reversal = any(
        value * signed_signal[0] < 0.0 and abs(value) > SIGNIFICANCE_MULTIPLIER * uncertainty
        for value, uncertainty in zip(signed_signal[1:], standard_errors[1:], strict=True)
    )

    if physical_overshoot > PHYSICAL_TOLERANCE or normalized_residual > MAX_NORMALIZED_FIT_RESIDUAL:
        classification = "extrapolation_unstable"
    elif not baseline_identifiable:
        classification = "inconclusive"
    elif sign_reversal or monotonicity == "non_monotonic":
        classification = "artefact_suspected"
    elif (
        retention[-1] <= DESTROYED_RETENTION_THRESHOLD
        and distances[-1] <= SIGNIFICANCE_MULTIPLIER * standard_errors[-1]
    ):
        classification = "signal_destroyed"
    elif retention[-1] <= DECAYED_RETENTION_THRESHOLD:
        classification = "signal_decayed"
    else:
        classification = "signal_retained"

    return ZNEValidityResult(
        factors=factors,
        factor_estimates=estimates,
        factor_uncertainties=uncertainties,
        signal_distances=tuple(float(value) for value in distances),
        signal_retention_ratios=tuple(float(value) for value in retention),
        monotonicity_status=monotonicity,
        extrapolated_value=extrapolated,
        physical_overshoot=physical_overshoot,
        fit_residual=fit_residual,
        normalized_fit_residual=normalized_residual,
        validity_classification=classification,
    )
