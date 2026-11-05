import pandas as pd

from src.generate_readout_mismatch_benchmark import (
    ACCURACY_FIGURE_PATH,
    BIAS_FIGURE_PATH,
    BUDGETS,
    DECISION_FIGURE_PATH,
    DISTRIBUTIONS,
    METHODS,
    RAW_PATH,
    REPEATS,
    SUMMARY_PATH,
    TARGET_MODELS,
    WIDTHS,
    build_manifest,
    execute_manifest,
)


def test_mismatch_manifest_is_complete_paired_and_deterministic() -> None:
    first = build_manifest(repeats=2)
    second = build_manifest(repeats=2)
    expected = (
        len(WIDTHS)
        * len(DISTRIBUTIONS)
        * len(TARGET_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * 2
    )
    pair_keys = [
        "width",
        "distribution",
        "target_model",
        "total_shot_budget",
        "repeat",
    ]

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected
    assert first.groupby(pair_keys).pair_seed.nunique().eq(1).all()


def test_one_mismatch_pair_uses_identical_target_counts_and_budget() -> None:
    raw = execute_manifest(build_manifest(repeats=1).iloc[:4])

    assert len(raw) == 4
    assert set(raw.status) == {"complete"}
    assert raw.raw_counts_json.nunique() == 1
    assert (raw.total_shots_used == raw.total_shot_budget).all()


def test_committed_mismatch_artifacts_are_complete() -> None:
    raw = pd.read_csv(RAW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    expected = (
        len(WIDTHS)
        * len(DISTRIBUTIONS)
        * len(TARGET_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * REPEATS
    )

    assert len(raw) == expected
    assert len(summary) == len(WIDTHS) * 2 * 4 * 2 * 4
    assert set(raw.status) <= {"complete", "failed"}
    complete = raw.loc[raw.status == "complete"]
    assert (complete.total_shots_used == complete.total_shot_budget).all()
    assert complete.raw_counts_json.str.len().gt(0).all()
    assert complete.exact_decision_margin.eq(0.05).all()
    assert summary.decision_error_probability.gt(0.0).any()
    assert summary.decision_error_wilson_low.le(summary.decision_error_probability).all()
    assert summary.decision_error_probability.le(summary.decision_error_wilson_high).all()
    for path in (ACCURACY_FIGURE_PATH, DECISION_FIGURE_PATH, BIAS_FIGURE_PATH):
        assert path.stat().st_size > 0
