import pandas as pd
import pytest

from src.generate_combined_qem import (
    ALLOCATION_POLICIES,
    BUDGETS,
    CELLS,
    FIGURE_DIR,
    INTERACTIONS_PATH,
    METHODS,
    RAW_PATH,
    SUMMARY_PATH,
    VALIDITY_PATH,
    build_manifest,
    summarize_interactions,
)


def test_combined_manifest_is_complete_and_preregistered() -> None:
    manifest = build_manifest(repeats=2)

    assert len(manifest) == len(CELLS) * len(BUDGETS) * len(ALLOCATION_POLICIES) * len(METHODS) * 2
    assert set(manifest.method) == set(METHODS)
    assert set(manifest.total_shot_budget) == set(BUDGETS)
    assert set(manifest.allocation_policy) == set(ALLOCATION_POLICIES)
    assert manifest.groupby(
        ["cell_id", "total_shot_budget", "allocation_policy", "method"]
    ).size().eq(2).all()


def test_combined_manifest_contains_all_required_cell_roles() -> None:
    manifest = build_manifest(repeats=1)

    assert set(manifest.cell_id) == {
        "readout_dominated",
        "gate_noise_dominated",
        "practical_tie",
        "zne_invalid",
        "decision_near_tie",
        "decision_larger_margin",
    }
    margins = manifest.dropna(subset=["margin"]).groupby("cell_id").margin.first().to_dict()
    assert margins == {"decision_larger_margin": 0.20, "decision_near_tie": 0.02}


def test_interaction_uses_the_frozen_factorial_contrast() -> None:
    rows = []
    values = {"raw": 0.10, "readout": 0.06, "zne": 0.07, "combined": 0.02}
    for repeat in range(3):
        for method, error in values.items():
            rows.append(
                {
                    "cell_id": "sample",
                    "kind": "expectation",
                    "circuit_id": "sample",
                    "noise": "low",
                    "total_shot_budget": 4096,
                    "allocation_policy": "balanced",
                    "repeat": repeat,
                    "method": method,
                    "absolute_error": error,
                    "decision_error": float("nan"),
                    "record_kind": "primary",
                    "status": "complete",
                }
            )

    result = summarize_interactions(pd.DataFrame(rows))

    assert len(result) == 1
    assert result.iloc[0].interaction_mean == pytest.approx(-0.01)
    assert result.iloc[0].matched_repeats == 3


def test_committed_combined_artifacts_are_complete() -> None:
    raw = pd.read_csv(RAW_PATH)
    primary = raw.loc[raw.record_kind == "primary"]

    assert len(raw) == 5600
    assert len(primary) == 4800
    assert set(raw.status) == {"complete"}
    assert (raw.total_shots_used == raw.total_shot_budget).all()
    assert primary.groupby(
        ["cell_id", "total_shot_budget", "allocation_policy", "method"]
    ).size().eq(50).all()
    assert len(pd.read_csv(SUMMARY_PATH)) == 96
    assert len(pd.read_csv(INTERACTIONS_PATH)) == 24
    assert len(pd.read_csv(VALIDITY_PATH)) == 16
    for filename in (
        "combined_overhead_vs_gain.png",
        "combined_interaction_effect.png",
        "combined_decision_error.png",
    ):
        assert (FIGURE_DIR / filename).stat().st_size > 0
