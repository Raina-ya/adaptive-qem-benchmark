import math

import pytest

from src.observable_floors import pauli_observable_floor
from src.zne_validity import VALIDITY_CLASSIFICATIONS, assess_zne_validity

FLOOR = pauli_observable_floor("ZZ", 1.0, evidence_label="analytic")


def assess(
    estimates: tuple[float, float, float],
    uncertainties: tuple[float, float, float] = (0.01, 0.01, 0.01),
):
    return assess_zne_validity((1, 3, 5), estimates, uncertainties, FLOOR)


def test_signal_retention_outputs_all_factor_diagnostics() -> None:
    result = assess((0.9, 0.8, 0.7))
    assert result.signal_distances == pytest.approx((0.9, 0.8, 0.7))
    assert result.signal_retention_ratios == pytest.approx((1.0, 8 / 9, 7 / 9))
    assert result.factor_uncertainties == (0.01, 0.01, 0.01)
    assert result.monotonicity_status == "monotonic_decay"
    assert result.validity_classification == "signal_retained"


def test_decayed_signal_is_distinct_from_destroyed_signal() -> None:
    assert assess((0.8, 0.5, 0.3)).validity_classification == "signal_decayed"
    destroyed = assess((0.8, 0.4, 0.05), (0.01, 0.02, 0.03))
    assert destroyed.validity_classification == "signal_destroyed"


def test_unidentifiable_scale_one_signal_is_inconclusive() -> None:
    result = assess((0.01, 0.008, 0.006), (0.01, 0.01, 0.01))
    assert result.validity_classification == "inconclusive"


def test_nonmonotonic_signal_is_flagged_as_possible_artefact() -> None:
    result = assess((0.8, 0.9, 0.6))
    assert result.monotonicity_status == "non_monotonic"
    assert result.validity_classification == "artefact_suspected"


def test_floor_crossing_is_flagged_as_possible_artefact() -> None:
    result = assess((0.6, 0.2, -0.2))
    assert result.validity_classification == "artefact_suspected"


def test_unphysical_extrapolate_is_preserved_and_flagged() -> None:
    result = assess((1.1, 0.9, 0.7))
    assert result.extrapolated_value > 1.0
    assert result.physical_overshoot == pytest.approx(result.extrapolated_value - 1.0)
    assert result.validity_classification == "extrapolation_unstable"


def test_large_fit_residual_is_unstable() -> None:
    result = assess((0.9, 0.1, 0.8))
    assert result.normalized_fit_residual > 0.25
    assert result.validity_classification == "extrapolation_unstable"


def test_zero_baseline_retention_is_explicitly_undefined() -> None:
    result = assess((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert all(math.isnan(value) for value in result.signal_retention_ratios)
    assert result.validity_classification == "inconclusive"


@pytest.mark.parametrize(
    ("factors", "estimates", "uncertainties"),
    [
        ((1, 3), (0.9,), (0.01,)),
        ((3, 1), (0.9, 0.8), (0.01, 0.01)),
        ((1, 1), (0.9, 0.8), (0.01, 0.01)),
        ((1, 3), (0.9, 0.8), (0.01, -0.01)),
        ((1, 3), (0.9, float("nan")), (0.01, 0.01)),
    ],
)
def test_invalid_inputs_fail_explicitly(factors, estimates, uncertainties) -> None:
    with pytest.raises(ValueError):
        assess_zne_validity(factors, estimates, uncertainties, FLOOR)


def test_all_preregistered_classifications_are_exercised() -> None:
    observed = {
        assess((0.9, 0.8, 0.7)).validity_classification,
        assess((0.8, 0.5, 0.3)).validity_classification,
        assess((0.8, 0.4, 0.05), (0.01, 0.02, 0.03)).validity_classification,
        assess((1.1, 0.9, 0.7)).validity_classification,
        assess((0.8, 0.9, 0.6)).validity_classification,
        assess((0.01, 0.008, 0.006), (0.01, 0.01, 0.01)).validity_classification,
    }
    assert observed == VALIDITY_CLASSIFICATIONS
