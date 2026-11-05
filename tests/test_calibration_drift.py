import json

import numpy as np
import pandas as pd
import pytest

from src.calibration_drift import (
    AGE_FIGURE,
    CONFIG_PATH,
    COST_FIGURE,
    RAW_PATH,
    _workflow_cost,
    assignment_matrix,
    drift_matrix,
    load_config,
    run_benchmark,
    should_recalibrate,
)


def _tiny_config() -> dict[str, object]:
    config = load_config().copy()
    config.update(
        {
            "workload_lengths": [5],
            "drift_models": ["slow_monotonic"],
            "policies": ["every_decision", "never"],
            "repeats": 2,
        }
    )
    return config


def test_assignment_and_all_drift_matrices_are_column_stochastic() -> None:
    config = load_config()
    np.testing.assert_allclose(assignment_matrix(0.03, 0.05).sum(axis=0), 1.0)
    for model in config["drift_models"]:
        for decision_index in (0, 10, 99):
            matrix = drift_matrix(str(model), decision_index, config)
            np.testing.assert_allclose(matrix.sum(axis=0), 1.0)
            assert np.all((matrix >= 0.0) & (matrix <= 1.0))


def test_drift_and_sampling_are_deterministic() -> None:
    config = _tiny_config()
    np.testing.assert_array_equal(
        drift_matrix("random_bounded", 7, config),
        drift_matrix("random_bounded", 7, config),
    )
    pd.testing.assert_frame_equal(run_benchmark(config), run_benchmark(config))


def test_directed_drift_models_move_observation_toward_decision_boundary() -> None:
    config = load_config()
    true = np.array([0.47, 0.53])
    initial = float((drift_matrix("slow_monotonic", 0, config) @ true)[1])
    for model in ("slow_monotonic", "abrupt_change", "asymmetric"):
        later = float((drift_matrix(model, 99, config) @ true)[1])
        assert later < initial


def test_recalibration_policies_do_not_need_true_drift() -> None:
    config = load_config()
    assert should_recalibrate("every_decision", 4, 3, config)
    assert should_recalibrate("fixed_interval", 10, 10, config)
    assert not should_recalibrate("fixed_interval", 9, 9, config)
    assert should_recalibrate("uncertainty_triggered", 4, 4, config)
    assert not should_recalibrate("never", 99, 99, config)
    with pytest.raises(ValueError, match="Unknown recalibration"):
        should_recalibrate("oracle", 1, 1, config)


def test_workflow_cost_charges_target_and_calibration_shots() -> None:
    config = load_config()
    every_events, every_cost = _workflow_cost("every_decision", 20, config)
    never_events, never_cost = _workflow_cost("never", 20, config)
    assert every_events == 20
    assert never_events == 1
    assert every_cost == 20 * (1024 + 1024)
    assert never_cost == 20 * 1024 + 1024


def test_tiny_run_preserves_every_decision_and_cost() -> None:
    frame = run_benchmark(_tiny_config())
    assert len(frame) == 5 * 2 * 2
    assert set(frame.status) == {"complete"}
    assert set(frame.result_label) == {"noisy_simulated"}
    keys = ["workload_length", "drift_model", "policy", "repeat", "decision_index"]
    assert not frame.duplicated(keys).any()
    charged = frame.groupby(["policy", "repeat"]).agg(
        target=("target_shots", "sum"),
        calibration=("calibration_shots_charged", "sum"),
        total=("workflow_total_shots", "first"),
    )
    assert ((charged.target + charged.calibration) == charged.total).all()
    first = frame[frame.decision_index == 0].pivot(
        index="repeat", columns="policy", values="observed_probability_one"
    )
    assert (first.every_decision == first.never).all()


def test_frozen_config_and_committed_artifacts() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["workload_lengths"] == [1, 5, 20, 100]
    assert config["scope_boundary"].startswith("synthetic")
    assert RAW_PATH.is_file()
    raw = pd.read_csv(RAW_PATH)
    expected = sum(config["workload_lengths"]) * 4 * 4 * config["repeats"]
    assert len(raw) == expected
    assert set(raw.status) <= {"complete", "ill_conditioned"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in (AGE_FIGURE, COST_FIGURE))


def test_invalid_matrix_and_drift_inputs_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="Readout error"):
        assignment_matrix(0.6, 0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        drift_matrix("slow_monotonic", -1, load_config())
    with pytest.raises(ValueError, match="Unknown drift"):
        drift_matrix("unknown", 0, load_config())
