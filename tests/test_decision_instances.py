import pytest

from src.decision_instances import SUPPORTED_MARGINS, controlled_decision_instance
from src.observables import exact_expectation


@pytest.mark.parametrize("margin", SUPPORTED_MARGINS)
def test_controlled_instance_has_exact_margin(margin: float) -> None:
    instance = controlled_decision_instance(margin)
    exact_a = exact_expectation(instance.candidate_a.circuit, instance.candidate_a.observable)
    exact_b = exact_expectation(instance.candidate_b.circuit, instance.candidate_b.observable)
    assert exact_a == pytest.approx(instance.candidate_a.exact_expectation, abs=1e-12)
    assert exact_b == pytest.approx(instance.candidate_b.exact_expectation, abs=1e-12)
    assert exact_a - exact_b == pytest.approx(margin, abs=1e-12)


def test_unsupported_margin_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        controlled_decision_instance(0.03)
