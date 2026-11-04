import pytest

from src.observable_floors import (
    MAXIMALLY_MIXED_FLOOR_MODEL,
    pauli_observable_floor,
)


@pytest.mark.parametrize("observable", ["X", "XX", "ZZ", "XZX", "IIZ"])
def test_nonidentity_pauli_has_zero_maximally_mixed_floor(observable: str) -> None:
    contract = pauli_observable_floor(observable, 0.6)
    assert (contract.physical_minimum, contract.physical_maximum) == (-1.0, 1.0)
    assert contract.expected_noise_floor == 0.0
    assert contract.maximum_attainable_improvement == pytest.approx(0.6)
    assert contract.floor_model == MAXIMALLY_MIXED_FLOOR_MODEL


def test_identity_floor_is_not_assumed_to_be_zero() -> None:
    contract = pauli_observable_floor("III", 1.0, evidence_label="analytic")
    assert (contract.physical_minimum, contract.physical_maximum) == (1.0, 1.0)
    assert contract.expected_noise_floor == 1.0
    assert contract.maximum_attainable_improvement == 0.0
    assert contract.evidence_label == "analytic"


@pytest.mark.parametrize(
    ("observable", "reference"),
    [("Z", 1.01), ("Z", -1.01), ("II", 0.99)],
)
def test_reference_outside_physical_range_fails(observable: str, reference: float) -> None:
    with pytest.raises(ValueError, match="physical range"):
        pauli_observable_floor(observable, reference)


def test_invalid_or_nonfinite_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Pauli"):
        pauli_observable_floor("AB", 0.0)
    with pytest.raises(ValueError, match="finite"):
        pauli_observable_floor("Z", float("nan"))
