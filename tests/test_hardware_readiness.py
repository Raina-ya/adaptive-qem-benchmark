import pandas as pd
import pytest

from src.hardware_readiness import (
    CONFIG,
    MANIFEST_PATH,
    build_manifest,
    required_candidate_shots,
    required_repeats,
)


def test_manifest_freezes_all_roles_without_submitting_a_job() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    assert CONFIG["hardware_execution_authorized"] is False
    assert CONFIG["jobs_submitted"] == 0
    assert len(manifest) == 6
    assert set(manifest.submission_status) == {"not_submitted"}
    assert manifest[CONFIG["required_metadata"]].isna().all(axis=None)
    assert manifest.falsifiable_hypothesis.notna().all()
    assert manifest.falsifier.notna().all()


def test_committed_manifest_matches_regenerated_predictions() -> None:
    committed = pd.read_csv(MANIFEST_PATH)
    current = build_manifest()
    columns = ["reference_value", "predicted_raw", "predicted_readout", "predicted_zne"]
    for column in columns:
        assert committed[column].to_numpy() == pytest.approx(
            current[column].to_numpy(), rel=1e-9, abs=1e-11, nan_ok=True
        )
    assert list(committed.required_total_shots) == list(current.required_total_shots)


def test_near_tie_is_flagged_as_underpowered_at_the_reference_budget() -> None:
    manifest = pd.read_csv(MANIFEST_PATH).set_index("cell_id")
    near = manifest.loc["decision_near_tie"]
    assert near.required_shots_per_repeat > CONFIG["reference_budget"]
    assert near.max_decision_error_wilson_high > CONFIG["decision_error_tolerance"]
    clear = manifest.loc["decision_clear"]
    assert clear.required_shots_per_repeat < CONFIG["reference_budget"]
    assert clear.max_decision_error_wilson_high < CONFIG["decision_error_tolerance"]


def test_power_helpers_follow_the_normal_approximation() -> None:
    assert required_candidate_shots(0.1, 0.0) == 1570
    assert required_candidate_shots(0.05, 0.0) > 3 * required_candidate_shots(0.1, 0.0)
    assert required_repeats(1.0, 0.01) == 3
    assert required_repeats(0.1, 0.1) == 8
    with pytest.raises(ValueError):
        required_candidate_shots(0.0, 0.0)
    with pytest.raises(ValueError):
        required_repeats(0.0, 1.0)
