"""Analytic physical ranges and noise floors for supported observables."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.observables import validate_pauli

MAXIMALLY_MIXED_FLOOR_MODEL = "maximally_mixed_state_before_measurement"


@dataclass(frozen=True)
class ObservableFloor:
    """Physical and reference-relative limits for one observable."""

    observable: str
    physical_minimum: float
    physical_maximum: float
    expected_noise_floor: float
    reference_value: float
    maximum_attainable_improvement: float
    floor_model: str
    evidence_label: str


def pauli_observable_floor(
    pauli_q0_first: str,
    reference_value: float,
    *,
    evidence_label: str = "noiseless_simulated",
) -> ObservableFloor:
    """Return the maximally-mixed floor contract for a Pauli observable.

    The maximum attainable improvement is the absolute error that could be
    removed when moving from the declared floor back to the reference value.
    """

    validate_pauli(pauli_q0_first)
    if not math.isfinite(reference_value):
        raise ValueError("Reference value must be finite")
    identity = set(pauli_q0_first) == {"I"}
    physical_minimum, physical_maximum = (1.0, 1.0) if identity else (-1.0, 1.0)
    tolerance = 1e-12
    if not physical_minimum - tolerance <= reference_value <= physical_maximum + tolerance:
        raise ValueError("Reference value lies outside the observable physical range")
    floor = 1.0 if identity else 0.0
    return ObservableFloor(
        observable=pauli_q0_first,
        physical_minimum=physical_minimum,
        physical_maximum=physical_maximum,
        expected_noise_floor=floor,
        reference_value=float(reference_value),
        maximum_attainable_improvement=abs(float(reference_value) - floor),
        floor_model=MAXIMALLY_MIXED_FLOOR_MODEL,
        evidence_label=evidence_label,
    )
