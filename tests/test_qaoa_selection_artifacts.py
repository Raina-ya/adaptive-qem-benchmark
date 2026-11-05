import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/qaoa_selection_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/qaoa_selection_summary_v1.csv"
MANIFEST_PATH = ROOT / "experiments/manifests/qaoa_selection_v1_manifest.csv"
MANIFEST_HASH_PATH = ROOT / "experiments/manifests/qaoa_selection_v1_manifest.sha256"
FIGURES = (
    ROOT / "figures/qaoa_selection_error.png",
    ROOT / "figures/qaoa_simple_regret.png",
    ROOT / "figures/qaoa_shots_to_decision.png",
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, keep_default_na=False)


def test_frozen_manifest_hash_matches_committed_manifest() -> None:
    expected = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    recorded = MANIFEST_HASH_PATH.read_text(encoding="ascii").split()[0]

    assert recorded == expected


def test_raw_results_cover_every_manifest_row_exactly(raw: pd.DataFrame) -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    keys = [
        "graph_id",
        "candidate_id",
        "method",
        "total_selection_budget",
        "repeat",
        "seed",
    ]

    assert len(raw) == len(manifest) == 4500
    assert not raw.duplicated(keys).any()
    pd.testing.assert_frame_equal(
        raw[keys].sort_values(keys).reset_index(drop=True),
        manifest[keys].sort_values(keys).reset_index(drop=True),
    )
    assert raw.status.isin({"complete", "failed", "not_applicable"}).all()


def test_raw_budgets_counts_and_certification_records_are_preserved(
    raw: pd.DataFrame,
) -> None:
    group_keys = ["graph_id", "method", "total_selection_budget", "repeat"]
    for _, group in raw.groupby(group_keys, sort=False):
        charged = int(group.candidate_target_shots.sum())
        if group.method.iloc[0] == "readout":
            charged += int(group.calibration_shots.max())
        assert charged == int(group.selection_shots.iloc[0])
        assert len(group) == 5
        completed = group[group.status == "complete"]
        assert completed.raw_counts.ne("").all()
        assert sum(group.certification_checkpoint_data.ne("")) <= 1
        if group.certification_checkpoint_data.ne("").any():
            payload = json.loads(
                group.loc[group.certification_checkpoint_data.ne(""),
                "certification_checkpoint_data"].iloc[0]
            )
            assert len(payload) == 4
            assert [item["total_shots"] for item in payload] == sorted(
                item["total_shots"] for item in payload
            )


def test_processed_summary_has_all_preregistered_strata() -> None:
    summary = pd.read_csv(SUMMARY_PATH)

    assert len(summary) == 48
    assert set(summary.candidate_gap_regime) == {"overall", "clear", "near_tie"}
    assert set(summary.method) == {"raw", "readout", "zne"}
    assert set(summary.total_selection_budget) == {4096, 16384}
    assert set(summary.graph_family) == {"ring", "sparse", "medium_dense"}
    assert summary[summary.candidate_gap_regime == "overall"].shape[0] == 18
    assert summary[summary.candidate_gap_regime == "clear"].shape[0] == 18
    assert summary[summary.candidate_gap_regime == "near_tie"].shape[0] == 12


def test_figures_and_successful_run_log_exist() -> None:
    assert all(path.is_file() and path.stat().st_size > 0 for path in FIGURES)
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (ROOT / "results/runs").glob("*_generate_qaoa_selection.json")
    ]
    assert any(record["status"] == "ok" for record in records)
