import json

import pytest

from src.circuits import bell_circuit
from src.mitigation.combined import (
    CombinedBudget,
    allocate_combined_budget,
    run_combined_mitigation,
)
from src.mitigation.zne import linear_extrapolate
from src.noise_models import noise_spec


def test_combined_allocations_use_exact_total_budget() -> None:
    balanced = allocate_combined_budget(4096, "balanced")
    base_heavy = allocate_combined_budget(4096, "base_heavy")

    assert balanced.calibration_shots == 819
    assert balanced.factor_shots == (1093, 1092, 1092)
    assert base_heavy.factor_shots == (1639, 983, 655)
    assert balanced.calibration_shots + sum(balanced.factor_shots) == 4096
    assert base_heavy.calibration_shots + sum(base_heavy.factor_shots) == 4096


def test_inconsistent_combined_ledger_is_rejected() -> None:
    with pytest.raises(ValueError, match="must equal"):
        CombinedBudget(4096, 819, (1000, 1000, 1000), "balanced")


def test_unknown_allocation_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown allocation"):
        allocate_combined_budget(4096, "outcome_tuned")


def test_combined_preserves_intermediates_and_does_not_clip() -> None:
    result = run_combined_mitigation(
        bell_circuit(),
        "XX",
        total_shot_budget=4096,
        noise_spec=noise_spec("mixed", "medium"),
        seed=1729,
        allocation_policy="balanced",
    )

    assert result.status == "ok"
    assert len(result.factor_raw_counts_json) == 3
    assert len(result.factor_corrected_probabilities_json) == 3
    assert all(json.loads(payload) for payload in result.factor_raw_counts_json)
    expected, _, _ = linear_extrapolate(result.factors, result.factor_corrected_estimates)
    assert result.corrected_extrapolated_estimate == pytest.approx(expected)
    assert result.calibration_shots + sum(result.factor_shots) == result.total_shot_budget
    assert result.calibration_circuit_executions == 4
    assert result.folded_circuit_executions == 3


def test_combined_returns_explicit_conditioning_failure() -> None:
    result = run_combined_mitigation(
        bell_circuit(),
        "XX",
        total_shot_budget=4096,
        noise_spec=noise_spec("mixed", "medium"),
        seed=99,
        allocation_policy="base_heavy",
        condition_threshold=0.5,
    )

    assert result.status == "ill_conditioned"
    assert result.corrected_extrapolated_estimate is None
    assert result.validity_classification == "not_evaluated"


def test_combined_is_seed_reproducible() -> None:
    inputs = dict(
        circuit=bell_circuit(),
        pauli_q0_first="XX",
        total_shot_budget=4096,
        noise_spec=noise_spec("mixed", "low"),
        seed=20261104,
        allocation_policy="balanced",
    )

    assert run_combined_mitigation(**inputs) == run_combined_mitigation(**inputs)
