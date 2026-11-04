import pandas as pd

from src.generate_readout_tensor_validation import (
    BUDGETS,
    DISTRIBUTIONS,
    FIGURE_PATH,
    METHODS,
    RAW_PATH,
    READOUT_MODELS,
    SUMMARY_PATH,
    WIDTHS,
    build_manifest,
    execute_manifest,
)


def test_tensor_validation_manifest_is_complete_and_deterministic() -> None:
    first = build_manifest(repeats=2)
    second = build_manifest(repeats=2)
    expected = (
        len(WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * 2
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected
    assert first.groupby(
        ["width", "distribution", "readout_model", "total_shot_budget", "repeat"]
    ).pair_seed.nunique().eq(1).all()
    assert set(first.status) == {"pending"}


def test_one_paired_cell_executes_with_exact_budget() -> None:
    manifest = build_manifest(repeats=1).iloc[:2].reset_index(drop=True)
    raw = execute_manifest(manifest)

    assert len(raw) == 2
    assert set(raw.status) == {"complete"}
    assert (raw.total_shots_used == raw.total_shot_budget).all()


def test_committed_tensor_validation_artifacts_are_complete() -> None:
    raw = pd.read_csv(RAW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    pair_keys = [
        "width",
        "distribution",
        "readout_model",
        "total_shot_budget",
        "repeat",
    ]
    raw_estimates = raw.pivot(index=pair_keys, columns="method", values="raw_estimate")

    assert len(raw) == 960
    assert len(summary) == 48
    assert set(raw.status) == {"complete"}
    assert (raw.total_shots_used == raw.total_shot_budget).all()
    assert (raw_estimates.full_matrix == raw_estimates.tensor_product).all()
    assert raw.raw_counts_json.str.len().gt(0).all()
    assert raw.calibration_counts_json.str.len().gt(0).all()
    assert FIGURE_PATH.stat().st_size > 0
