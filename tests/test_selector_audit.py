import json

import pandas as pd
import pytest

from src.metrics import CONDITION_COLUMNS
from src.selector import select_qem_method
from src.selector_audit import (
    AUDIT_PATH,
    CATEGORIES,
    CONFIG,
    POLICIES,
    SUMMARY_PATH,
    classify,
    counterfactual_without_gates,
    is_catastrophic,
    selector_inputs,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "selected_method": "raw",
        "oracle_label": "readout",
        "decision_compatible": False,
        "rmse_regret": 0.01,
        "reference": 1.0,
    }
    row.update(overrides)
    return row


def test_categories_match_the_requested_audit_set() -> None:
    assert len(CATEGORIES) == 8
    assert CONFIG["negligible_regret_max"] == 0.005
    assert POLICIES == ("selector", "always_raw", "always_readout", "always_zne")


def test_correct_selections_separate_useful_from_tied() -> None:
    useful = _row(selected_method="readout", decision_compatible=True, rmse_regret=0.0)
    assert classify(useful, None) == "correct_materially_useful"
    tied = _row(oracle_label="practical_tie", decision_compatible=True)
    assert classify(tied, None) == "correct_practically_tied"


@pytest.mark.parametrize(
    ("regret", "reference", "expected"),
    [
        (0.005, 1.0, "incorrect_negligible_regret"),
        (0.0051, 1.0, "incorrect_material_regret"),
        (0.05, 1.0, "catastrophic_regret"),
        (0.002, 0.001, "catastrophic_regret"),
    ],
)
def test_incorrect_selections_use_frozen_regret_bands(
    regret: float, reference: float, expected: str
) -> None:
    row = _row(rmse_regret=regret, reference=reference)
    assert classify(row, None) == expected


def test_abstention_scores_the_gate_free_counterfactual() -> None:
    row = _row(selected_method="abstain", rmse_regret=float("nan"))
    good = {"complete_repeats": 50.0, "rmse_regret": 0.001}
    assert classify(row, good) == "unnecessary_abstention"
    harmful = {"complete_repeats": 50.0, "rmse_regret": 0.02}
    assert classify(row, harmful) == "abstention_avoided_failure"
    failed = {"complete_repeats": 20.0, "rmse_regret": 0.0}
    assert classify(row, failed) == "abstention_avoided_failure"
    with pytest.raises(ValueError):
        classify(row, None)


def test_inconclusive_cells_are_not_scored_as_correct_or_incorrect() -> None:
    assert classify(_row(oracle_label="inconclusive"), None) == "inconclusive"
    assert classify(_row(selected_method="inconclusive"), None) == "inconclusive"


def test_catastrophic_rule_is_inclusive_on_both_limits() -> None:
    assert is_catastrophic(0.05, 1.0)
    assert is_catastrophic(0.01, -0.01)
    assert not is_catastrophic(0.049, 1.0)


def test_counterfactual_only_disables_abstention_gates() -> None:
    inputs = {
        "noise": "medium",
        "shots": 1024,
        "width": 6,
        "depth": 3,
        "gate_count": 16,
        "condition_number": 25.0,
        "fit_residual": 0.002,
    }
    assert select_qem_method(**inputs).method == "abstain"
    assert counterfactual_without_gates(inputs) == "raw"
    inputs_readout = {**inputs, "shots": 4096, "condition_number": 2.0}
    assert counterfactual_without_gates(inputs_readout) == "readout"


def test_committed_audit_is_complete_and_replays_selector_decisions() -> None:
    if not AUDIT_PATH.exists():
        pytest.skip("selector audit artifacts have not been generated")
    audit = pd.read_csv(AUDIT_PATH)
    assert len(audit) == 180 * len(POLICIES)
    assert set(audit.audit_category) <= set(CATEGORIES)
    assert audit.groupby("policy").size().eq(180).all()
    assert (audit.rmse_regret.dropna() >= -1e-12).all()
    selector = audit.loc[audit.policy == "selector"]
    assert selector.groupby("evaluation_set").size().to_dict() == {
        "holdout": 29,
        "noise_shift": 36,
        "rule_design": 115,
    }
    main_grid = pd.read_csv(AUDIT_PATH.parent / "benchmark_summary_v1.csv")
    for row in selector.loc[selector.evaluation_set != "noise_shift"].itertuples():
        condition = tuple(getattr(row, column) for column in CONDITION_COLUMNS)
        inputs = selector_inputs(main_grid, condition)
        assert select_qem_method(**inputs).method == row.selected_method
    summary = pd.read_csv(SUMMARY_PATH)
    totals = summary.groupby(["evaluation_set", "policy"]).category_cells.sum()
    cells = summary.groupby(["evaluation_set", "policy"]).cells.first()
    pd.testing.assert_series_equal(totals, cells, check_names=False)
    config = json.loads(
        (AUDIT_PATH.parents[2] / "experiments/configs/selector_audit_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["audit_categories"] == list(CATEGORIES)
