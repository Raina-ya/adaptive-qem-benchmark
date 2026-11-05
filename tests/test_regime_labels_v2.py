import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.regime_labels_v2 import (
    COMPARISON_PATH,
    CONFIG_PATH,
    EVALUATION_PATH,
    LABELS_PATH,
    build_labels_v2,
    evaluate_selections,
    label_condition,
)

ROOT = Path(__file__).resolve().parents[1]
CONDITION = {
    "family": "hardware_efficient",
    "circuit_id": "c",
    "width": 2,
    "depth": 1,
    "noise": "low",
    "shots": 4096,
}


def _group(
    raw: float,
    readout: float,
    zne: float,
    *,
    readout_ci: tuple[float, float] = (-0.2, -0.1),
    zne_ci: tuple[float, float] = (-0.2, -0.1),
    complete: int = 50,
) -> pd.DataFrame:
    rows = (
        ("raw", raw, 1.0, (0.0, 0.0), 50),
        ("readout", readout, 5.0, readout_ci, complete),
        ("zne", zne, 3.0, zne_ci, 50),
    )
    return pd.DataFrame(
        [
            {
                **CONDITION,
                "method": method,
                "rmse": rmse,
                "execution_overhead": overhead,
                "rmse_gain_ci_low": interval[0],
                "rmse_gain_ci_high": interval[1],
                "complete_repeats": repeats,
            }
            for method, rmse, overhead, interval, repeats in rows
        ]
    )


def test_config_records_thresholds_and_post_hoc_status() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["status"] == "post_hoc_reanalysis_of_inspected_v1_results"
    assert config["negligible_regret_max"] == 0.005
    assert config["catastrophic_regret_absolute"] == 0.05
    assert config["simulator_only"] is True


def test_decisive_raw_winner_is_labeled_raw() -> None:
    row = label_condition(_group(0.01, 0.20, 0.30))
    assert row["oracle_label"] == "raw"
    assert row["oracle_method"] == "raw"
    assert row["raw_win_basis"] == "accuracy"


def test_raw_win_from_cost_alone_is_labeled_and_distinguished() -> None:
    row = label_condition(_group(0.100, 0.100, 0.200, readout_ci=(-0.01, 0.01)))
    assert row["oracle_label"] == "raw"
    assert row["raw_win_basis"] == "cost_only"


def test_true_tie_stays_a_tie() -> None:
    row = label_condition(_group(0.100, 0.0955, 0.200, readout_ci=(0.001, 0.008)))
    assert row["oracle_label"] == "practical_tie"
    assert "raw" in json.loads(row["equivalent_methods_json"])


def test_mitigated_winner_rules_are_unchanged() -> None:
    clear = label_condition(_group(0.30, 0.10, 0.40, readout_ci=(0.1, 0.3)))
    assert clear["oracle_label"] == "readout"
    uncertain = label_condition(_group(0.30, 0.10, 0.40, readout_ci=(-0.1, 0.3)))
    assert uncertain["oracle_label"] == "inconclusive"


def test_incomplete_method_stays_inconclusive() -> None:
    row = label_condition(_group(0.1, 0.1, 0.1, complete=20))
    assert row["oracle_label"] == "inconclusive"
    assert pd.isna(row["oracle_method"])


def test_regret_is_nonnegative_and_abstention_is_not_dropped() -> None:
    summary = pd.concat(
        [
            _group(0.10, 0.05, 0.20),
            _group(0.10, 0.05, 0.20).assign(circuit_id="d"),
        ]
    )
    selections = pd.DataFrame(
        [
            {**CONDITION, "selected_method": "raw", "reference": 0.5},
            {**CONDITION, "circuit_id": "d", "selected_method": "abstain", "reference": 0.5},
        ]
    )
    frame = evaluate_selections(selections, summary)
    assert len(frame) == 2
    decided = frame.loc[frame.circuit_id == "c"].iloc[0]
    assert decided.rmse_regret == pytest.approx(0.05)
    assert decided.best_rmse_method == "readout"
    assert decided.overhead_difference == pytest.approx(-4.0)
    assert pd.isna(frame.loc[frame.circuit_id == "d"].iloc[0].rmse_regret)
    assert (frame.rmse_regret.dropna() >= 0.0).all()


def test_committed_v2_artifacts_preserve_v1_and_cover_every_cell() -> None:
    if not LABELS_PATH.exists():
        pytest.skip("v2 artifacts have not been generated")
    labels = pd.read_csv(LABELS_PATH)
    v1 = pd.read_csv(ROOT / "results/processed/regime_labels_v1.csv")
    assert len(labels) == len(v1) == 144
    merged = labels.merge(
        v1,
        on=["family", "circuit_id", "width", "depth", "noise", "shots"],
        suffixes=("", "_v1"),
    )
    assert len(merged) == 144
    changed = merged.loc[merged.oracle_label != merged.oracle_label_v1]
    assert set(changed.oracle_label) == {"raw"}
    assert set(changed.oracle_label_v1) == {"practical_tie"}
    assert (merged.split == merged.split_v1).all()
    evaluation = pd.read_csv(EVALUATION_PATH)
    assert set(evaluation.evaluation_set) == {"rule_design", "holdout", "noise_shift"}
    assert evaluation.groupby("evaluation_set").size().to_dict() == {
        "holdout": 29,
        "noise_shift": 36,
        "rule_design": 115,
    }
    assert (evaluation.rmse_regret.dropna() >= -1e-12).all()
    comparison = pd.read_csv(COMPARISON_PATH)
    assert np.isfinite(comparison.v2_value.to_numpy(dtype=float)).all()


def test_build_labels_keeps_one_row_per_condition() -> None:
    summary = pd.concat(
        [
            _group(0.01, 0.20, 0.30),
            _group(0.3, 0.1, 0.4, readout_ci=(0.1, 0.3)).assign(circuit_id="d"),
        ]
    )
    labels = build_labels_v2(summary)
    assert len(labels) == 2
    assert set(labels.oracle_label) == {"raw", "readout"}
