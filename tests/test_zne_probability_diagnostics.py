import json

import numpy as np
import pytest

from src.circuits import bell_circuit
from src.config import derive_seed
from src.noise_models import noise_spec
from src.zne_probability_diagnostics import run_distribution_zne
from src.zne_validity import (
    extrapolate_probability_distribution,
    linear_intercept_weights,
    probability_vector_from_counts,
)


def test_distribution_runner_preserves_counts_and_budget() -> None:
    result = run_distribution_zne(
        bell_circuit(),
        "XX",
        total_shot_budget=1024,
        noise_spec=noise_spec("mixed", "low"),
        seed=derive_seed("zne-probability-test"),
    )
    assert sum(result.shots_by_factor) == 1024
    assert len(result.factor_counts_json) == 3
    assert len(result.factor_standard_errors) == 3
    assert len(set(result.factor_seeds)) == 3


def test_linear_weights_recover_intercept_and_sum_to_one() -> None:
    weights = np.asarray(linear_intercept_weights((1, 3, 5)))
    assert weights.sum() == pytest.approx(1.0)
    values = np.array([0.9, 0.7, 0.5])
    assert weights @ values == pytest.approx(1.0)


def test_counts_are_converted_without_bit_order_reinterpretation() -> None:
    vector = probability_vector_from_counts(json.dumps({"00": 1, "10": 3}), 2)
    assert vector == pytest.approx((0.25, 0.0, 0.75, 0.0))


def test_negative_and_above_one_mass_are_preserved() -> None:
    diagnostic = extrapolate_probability_distribution(
        (1, 3, 5),
        (
            np.array([0.95, 0.05]),
            np.array([0.50, 0.50]),
            np.array([0.20, 0.80]),
        ),
    )
    assert diagnostic.extrapolated_probabilities[0] > 1.0
    assert diagnostic.extrapolated_probabilities[1] < 0.0
    assert diagnostic.negative_mass > 0.0
    assert diagnostic.above_one_mass > 0.0
    assert diagnostic.normalization_error < 1e-12
    assert diagnostic.unphysical_mass_flag


def test_physical_distribution_has_zero_unphysical_mass() -> None:
    diagnostic = extrapolate_probability_distribution(
        (1, 3, 5),
        (
            np.array([0.7, 0.3]),
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
        ),
    )
    assert diagnostic.negative_mass == pytest.approx(0.0)
    assert diagnostic.above_one_mass == pytest.approx(0.0)
    assert not diagnostic.unphysical_mass_flag


@pytest.mark.parametrize(
    "probabilities",
    [
        (np.array([0.5, 0.5]),),
        (np.array([0.5, 0.5]), np.array([0.5, 0.4]), np.array([0.5, 0.5])),
        (np.array([0.5, 0.5]), np.array([0.5, -0.5]), np.array([0.5, 0.5])),
    ],
)
def test_invalid_probability_inputs_fail(probabilities) -> None:
    with pytest.raises(ValueError):
        extrapolate_probability_distribution((1, 3, 5), probabilities)
