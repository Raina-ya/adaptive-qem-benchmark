import pytest

from src.circuits import bell_circuit
from src.config import derive_seed
from src.mitigation.zne import PREREGISTERED_MODEL, linear_extrapolate, run_linear_zne
from src.noise_models import noise_spec


def test_linear_extrapolation_recovers_intercept() -> None:
    intercept, slope, rss = linear_extrapolate((1, 3, 5), (0.9, 0.7, 0.5))
    assert intercept == pytest.approx(1.0)
    assert slope == pytest.approx(-0.1)
    assert rss < 1e-28


def test_zne_charges_exact_total_budget_and_preserves_factor_data() -> None:
    result = run_linear_zne(
        bell_circuit(),
        "XX",
        total_shot_budget=4096,
        noise_spec=noise_spec("mixed", "medium"),
        seed=derive_seed("zne-test"),
    )
    assert sum(result.shots_by_factor) == 4096
    assert result.factors == (1, 3, 5)
    assert len(set(result.factor_seeds)) == 3
    assert len(result.factor_estimates) == 3
    assert "intercept" in PREREGISTERED_MODEL


def test_out_of_range_extrapolate_is_flagged_not_clipped() -> None:
    intercept, _, _ = linear_extrapolate((1, 3, 5), (1.1, 0.9, 0.7))
    assert intercept > 1.0
