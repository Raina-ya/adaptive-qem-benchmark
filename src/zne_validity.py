"""Preregistered factor-level signal-retention diagnostics for linear ZNE."""

from __future__ import annotations

import json
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
UNPHYSICAL_MASS_THRESHOLD = 0.05


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


@dataclass(frozen=True)
class ProbabilityExtrapolationDiagnostic:
    """Unclipped signed-mass diagnostics for extrapolated probabilities."""

    factors: tuple[int, ...]
    extrapolation_weights: tuple[float, ...]
    extrapolated_probabilities: tuple[float, ...]
    negative_mass: float
    above_one_mass: float
    normalization_error: float
    unphysical_mass_flag: bool


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


def linear_intercept_weights(factors: tuple[int, ...]) -> tuple[float, ...]:
    """Return ordinary-least-squares weights for the intercept at factor zero."""

    if len(factors) < 2 or len(set(factors)) != len(factors):
        raise ValueError("At least two distinct factors are required")
    values = np.asarray(factors, dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("Factors must be finite")
    centered = values - values.mean()
    denominator = float(centered @ centered)
    weights = 1.0 / len(values) - values.mean() * centered / denominator
    return tuple(float(value) for value in weights)


def probability_vector_from_counts(counts_json: str, width: int) -> np.ndarray:
    """Convert preserved Qiskit counts to a computational-basis probability vector."""

    if width < 1:
        raise ValueError("Width must be positive")
    counts = json.loads(counts_json)
    if not isinstance(counts, dict) or not counts:
        raise ValueError("Counts must be a non-empty JSON object")
    shots = sum(int(value) for value in counts.values())
    if shots <= 0:
        raise ValueError("Counts must contain positive total shots")
    probabilities = np.zeros(2**width, dtype=float)
    for bitstring, count in counts.items():
        clean = str(bitstring).replace(" ", "")
        if len(clean) != width or any(bit not in "01" for bit in clean):
            raise ValueError("Count keys must match the declared width")
        probabilities[int(clean, 2)] += int(count) / shots
    return probabilities


def extrapolate_probability_distribution(
    factors: tuple[int, ...], factor_probabilities: tuple[np.ndarray, ...]
) -> ProbabilityExtrapolationDiagnostic:
    """Extrapolate basis probabilities without clipping negative or excess mass."""

    if len(factors) != len(factor_probabilities):
        raise ValueError("Every factor must have one probability vector")
    weights = linear_intercept_weights(factors)
    matrix = np.asarray(factor_probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Factor probabilities must be equal-length vectors")
    if np.any(~np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("Factor probabilities must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Every factor probability vector must sum to one")
    extrapolated = np.asarray(weights) @ matrix
    negative_mass = float(np.maximum(-extrapolated, 0.0).sum())
    above_one_mass = float(np.maximum(extrapolated - 1.0, 0.0).sum())
    normalization_error = abs(float(extrapolated.sum()) - 1.0)
    return ProbabilityExtrapolationDiagnostic(
        factors=factors,
        extrapolation_weights=weights,
        extrapolated_probabilities=tuple(float(value) for value in extrapolated),
        negative_mass=negative_mass,
        above_one_mass=above_one_mass,
        normalization_error=normalization_error,
        unphysical_mass_flag=(negative_mass + above_one_mass > UNPHYSICAL_MASS_THRESHOLD),
    )
