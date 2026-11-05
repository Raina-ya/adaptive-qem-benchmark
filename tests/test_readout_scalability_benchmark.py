import pandas as pd

from src.generate_readout_scalability_benchmark import (
    ACCURACY_FIGURE_PATH,
    BUDGETS,
    DISTRIBUTIONS,
    LARGE_WIDTH_REPEATS,
    MEMORY_FIGURE_PATH,
    METHODS,
    RAW_PATH,
    READOUT_MODELS,
    RUNTIME_FIGURE_PATH,
    SMALL_WIDTH_REPEATS,
    SUMMARY_PATH,
    UNCERTAINTY_FIGURE_PATH,
    WIDTHS,
    build_manifest,
    execute_manifest,
)


def test_scalability_manifest_is_complete_paired_and_deterministic() -> None:
    first = build_manifest(small_width_repeats=1, large_width_repeats=1)
    second = build_manifest(small_width_repeats=1, large_width_repeats=1)
    expected = (
        len(WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
    )
    pair_keys = ["width", "distribution", "readout_model", "total_shot_budget", "repeat"]

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected
    assert first.groupby(pair_keys).pair_seed.nunique().eq(1).all()
    assert set(first.method) == set(METHODS)


def test_one_small_width_pair_uses_identical_target_counts() -> None:
    manifest = build_manifest(small_width_repeats=1, large_width_repeats=0).iloc[:4]
    raw = execute_manifest(manifest)

    assert len(raw) == 4
    assert set(raw.status) == {"complete"}
    assert raw.raw_counts_json.nunique() == 1
    assert (raw.total_shots_used == raw.total_shot_budget).all()


def test_committed_scalability_artifacts_cover_the_frozen_grid() -> None:
    raw = pd.read_csv(RAW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    expected = (
        3
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * SMALL_WIDTH_REPEATS
        + 3
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * LARGE_WIDTH_REPEATS
    )

    assert len(raw) == expected
    assert len(summary) == len(WIDTHS) * 2 * 2 * 2 * 4
    assert set(raw.status) <= {"complete", "failed", "resource_limited"}
    complete = raw.loc[raw.status == "complete"]
    assert (complete.total_shots_used == complete.total_shot_budget).all()
    assert complete.raw_counts_json.str.len().gt(0).all()
    full_large = raw.loc[raw.method == "full_matrix"].query("width >= 10")
    assert (full_large.status == "resource_limited").all()
    assert (summary.uncertainty_amplification.dropna() >= 0).all()
    for path in (
        MEMORY_FIGURE_PATH,
        RUNTIME_FIGURE_PATH,
        ACCURACY_FIGURE_PATH,
        UNCERTAINTY_FIGURE_PATH,
    ):
        assert path.stat().st_size > 0
