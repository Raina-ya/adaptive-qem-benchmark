import pandas as pd

from src.generate_readout_matrix_free_validation import (
    BUDGETS,
    DISTRIBUTIONS,
    FIGURE_PATH,
    PILOT_REPEATS,
    PILOT_WIDTHS,
    RAW_PATH,
    READOUT_MODELS,
    REFERENCE_REPEATS,
    REFERENCE_WIDTHS,
    SUMMARY_PATH,
    build_manifest,
    execute_manifest,
)


def test_matrix_free_manifest_is_complete_and_deterministic() -> None:
    first = build_manifest(reference_repeats=2, pilot_repeats=1)
    second = build_manifest(reference_repeats=2, pilot_repeats=1)
    expected_reference = (
        len(REFERENCE_WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * 2
    )
    expected_pilot = len(PILOT_WIDTHS) * len(READOUT_MODELS) * len(BUDGETS)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected_reference + expected_pilot
    assert set(first.status) == {"pending"}


def test_one_matrix_free_cell_executes_with_exact_budget() -> None:
    raw = execute_manifest(build_manifest(reference_repeats=1, pilot_repeats=0).iloc[:1])

    assert len(raw) == 1
    assert raw.status.iloc[0] == "complete"
    assert raw.total_shots_used.iloc[0] == raw.total_shot_budget.iloc[0]
    assert raw.materialized_matrix_entries.iloc[0] == 0


def test_committed_matrix_free_artifacts_are_complete() -> None:
    raw = pd.read_csv(RAW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    expected_reference = (
        len(REFERENCE_WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * REFERENCE_REPEATS
    )
    expected_pilot = (
        len(PILOT_WIDTHS) * len(READOUT_MODELS) * len(BUDGETS) * PILOT_REPEATS
    )

    assert len(raw) == expected_reference + expected_pilot
    assert len(summary) == 32
    assert set(raw.status) == {"complete"}
    assert raw.converged.all()
    assert (raw.total_shots_used == raw.total_shot_budget).all()
    assert (raw.materialized_matrix_entries == 0).all()
    assert raw.raw_counts_json.str.len().gt(0).all()
    assert raw.calibration_counts_json.str.len().gt(0).all()
    assert FIGURE_PATH.stat().st_size > 0
