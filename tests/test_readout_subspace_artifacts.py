import pandas as pd

from src.generate_readout_subspace_validation import (
    BUDGETS,
    DISTRIBUTIONS,
    FIGURE_PATH,
    METHODS,
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


def test_subspace_manifest_is_complete_and_deterministic() -> None:
    first = build_manifest(reference_repeats=2, pilot_repeats=1)
    second = build_manifest(reference_repeats=2, pilot_repeats=1)
    expected_reference = (
        len(REFERENCE_WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * 2
    )
    expected_pilot = len(PILOT_WIDTHS) * len(READOUT_MODELS) * len(BUDGETS)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected_reference + expected_pilot
    assert set(first.status) == {"pending"}
    assert set(first.loc[first.width.isin(PILOT_WIDTHS), "method"]) == {
        "observed_subspace"
    }


def test_one_subspace_pair_executes_with_identical_target_counts() -> None:
    manifest = build_manifest(reference_repeats=1, pilot_repeats=0).iloc[:2]
    raw = execute_manifest(manifest)

    assert len(raw) == 2
    assert set(raw.status) == {"complete"}
    assert (raw.total_shots_used == raw.total_shot_budget).all()
    assert raw.raw_counts_json.nunique() == 1


def test_committed_subspace_artifacts_are_complete() -> None:
    raw = pd.read_csv(RAW_PATH)
    summary = pd.read_csv(SUMMARY_PATH)
    expected_reference = (
        len(REFERENCE_WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(BUDGETS)
        * len(METHODS)
        * REFERENCE_REPEATS
    )
    expected_pilot = (
        len(PILOT_WIDTHS) * len(READOUT_MODELS) * len(BUDGETS) * PILOT_REPEATS
    )

    assert len(raw) == expected_reference + expected_pilot
    assert len(summary) == 56
    assert set(raw.status) == {"complete"}
    assert (raw.total_shots_used == raw.total_shot_budget).all()
    assert raw.raw_counts_json.str.len().gt(0).all()
    assert raw.calibration_counts_json.str.len().gt(0).all()
    assert (raw.discarded_probability_mass >= 0).all()
    assert FIGURE_PATH.stat().st_size > 0
