import copy
import math

import numpy as np
import pandas as pd
import pytest

from src import generate_drift_policy as drift
from src import generate_readout_feasibility_v2 as feasibility
from src import generate_readout_mismatch_dose as mismatch
from src import readout_boundaries as rb
from src import readout_boundaries_analysis as analysis

CONFIG = feasibility.CONFIG


def full_matrix(channel):
    width = channel.width
    return np.column_stack([rb.apply_channel(rb.basis(width, i), channel) for i in range(2**width)])


def test_channel_application_matches_the_explicit_matrix() -> None:
    from src.readout_mismatch import TargetReadoutModel, target_assignment_matrix

    channel = rb.Channel.uniform(4, 0.02, 0.05, 0.03)
    reference = target_assignment_matrix(4, TargetReadoutModel("x", 0.02, 0.05, 0.03))
    assert full_matrix(channel) == pytest.approx(reference)


def test_mixture_is_the_weighted_channel() -> None:
    first, second = rb.Channel.uniform(3, 0.02, 0.02), rb.Channel.uniform(3, 0.05, 0.1)
    mixed = full_matrix(rb.Mixture(first, second, 0.3))
    assert mixed == pytest.approx(0.7 * full_matrix(first) + 0.3 * full_matrix(second))


@pytest.mark.parametrize("width", [4, 6])
def test_candidate_distributions_have_the_stated_parity(width) -> None:
    signs = rb.parity_signs(width)
    assert rb.product_distribution(width, 0.3) @ signs == pytest.approx(0.3)
    assert rb.sparse_distribution(width, 0.3) @ signs == pytest.approx(0.3)


def test_observable_can_see_pair_flips_at_every_configured_width() -> None:
    widths = {*CONFIG["feasibility"]["widths"], *CONFIG["mismatch"]["widths"]}
    assert all(width % 2 == 0 for width in widths)
    channel = rb.Channel.uniform(4, 0.0, 0.0, 0.1)
    observed = rb.apply_channel(rb.product_distribution(4, 0.3), channel)
    assert observed @ rb.parity_signs(4) == pytest.approx(0.3 * 0.8)


@pytest.mark.parametrize("pair_flip", [0.0, 0.04])
def test_exact_calibrations_recover_the_parity(pair_flip) -> None:
    channel = rb.Channel.uniform(4, 0.02, 0.05, pair_flip)
    ideal = rb.product_distribution(4, 0.3)
    counts = rb.apply_channel(ideal, channel) * 1e9
    matrix = full_matrix(channel)
    assert rb.correct_full(counts, matrix, 1.0)[0] == pytest.approx(0.3)
    rng = np.random.default_rng(0)
    blocks = rb.calibrate_pairs(rng, channel, 40_000_000)
    assert rb.correct_pairs(counts, blocks)[0] == pytest.approx(0.3, abs=2e-3)
    if pair_flip == 0.0:
        assert rb.correct_tensor(counts, np.full(4, 0.02), np.full(4, 0.05))[0] == pytest.approx(
            0.3
        )
        subspace = rb.correct_subspace(
            counts.round(), np.full(4, 0.02), np.full(4, 0.05), iterative=True
        )
        assert subspace[0] == pytest.approx(0.3, abs=1e-6)


def test_tensor_correction_misreads_pair_correlated_error() -> None:
    channel = rb.Channel.uniform(4, 0.02, 0.02, 0.05)
    counts = rb.apply_channel(rb.product_distribution(4, 0.3), channel) * 1e9
    p01, p10 = rb.calibrate_tensor(np.random.default_rng(1), channel, 40_000_000)
    assert abs(rb.correct_tensor(counts, p01, p10)[0] - 0.3) > 0.02


def test_missing_full_calibration_is_a_named_failure() -> None:
    counts = np.ones(16)
    assert rb.correct_full(counts, np.zeros((16, 16)), None)[1] == "no_calibration_shots"
    assert rb.correct_full(counts, np.eye(16), 1e9)[1] == "ill_conditioned"


def test_budgets_add_up_in_both_regimes() -> None:
    total = {"regime": "fixed_total", "setting": 4096}
    target, circuits, per = feasibility.shots_for(total, "full_matrix", 8, CONFIG)
    assert target + circuits * per <= 4096 and circuits == 256 and per == 3
    assert feasibility.shots_for(total, "none", 8, CONFIG) == (4096, 0, 0)
    quality = {"regime": "fixed_quality", "setting": 1024}
    target, circuits, per = feasibility.shots_for(quality, "full_matrix", 8, CONFIG)
    assert (target, circuits * per) == (16384, 256 * 1024)


def small_config():
    config = copy.deepcopy(CONFIG)
    config["feasibility"]["repeats"] = 3
    config["mismatch"].update(repeats=3, lambdas=[0.0, 0.5, 1.0])
    config["drift"]["repeats"] = 4
    return config


def test_feasibility_rows_are_complete_and_deterministic() -> None:
    config = small_config()
    regime = {"regime": "fixed_total", "setting": 4096}
    first = feasibility.run_cell(4, "independent", regime, config)
    assert first == feasibility.run_cell(4, "independent", regime, config)
    frame = pd.DataFrame(first)
    assert set(frame.method) == set(config["feasibility"]["methods"])
    summary = analysis.feasibility_summary(frame)
    assert summary.calibration_share.between(0.0, 1.0).all()
    none = summary.loc[summary.method == "none", "variance_amplification"]
    assert none.to_numpy() == pytest.approx(1.0)


def test_mismatch_reuses_one_stale_calibration_across_lambda() -> None:
    config = small_config()
    frame = pd.DataFrame(mismatch.run_cell(4, "asymmetric_drift", config))
    tensor = frame.loc[frame.method == "tensor_product"]
    assert tensor.groupby("repeat").condition_number.nunique().eq(1).all()
    summary = analysis.mismatch_summary(frame)
    assert summary.diagnostic_shots.eq(2 * CONFIG["mismatch"]["check_shots"]).all()
    table = analysis.thresholds(frame, summary)
    assert {"lambda_loses_to_raw", "warns_in_time_check_drift"} <= set(table.columns)


def test_check_statistic_grows_with_mismatch() -> None:
    first, second = rb.Channel.uniform(4, 0.02, 0.02), rb.Channel.uniform(4, 0.02, 0.15)
    rng = np.random.default_rng(3)
    p01, p10 = rb.calibrate_tensor(rng, first, 200_000)
    quiet = mismatch.check_statistic(rng, first, p01, p10, 4096)
    loud = mismatch.check_statistic(rng, second, p01, p10, 4096)
    assert quiet < 4.0 < loud


@pytest.mark.parametrize("shape", ["ramp", "abrupt", "sinusoid"])
def test_drift_paths_start_at_zero_and_stay_bounded(shape) -> None:
    path = drift.drift_path(shape, 0.05, 20, 100)
    assert path[0] == 0.0 and path.max() == pytest.approx(0.05, abs=1e-12)


def test_triggered_policy_charges_its_check_shots() -> None:
    settings = small_config()["drift"]
    scenario = {
        "amplitude": 0.0,
        "timescale": 10,
        "shape": "ramp",
        "run_length": 20,
        "margin": 0.05,
        "calibration_cost": 1024,
    }
    triggered = drift.simulate(scenario, "triggered", 100.0, settings)
    never = drift.simulate(scenario, "never", 0.0, settings)
    checks = 2 * settings["check_shots_per_state"] * 19
    assert (triggered.total_shots - never.total_shots).eq(checks).all()
    every = drift.simulate(scenario, "every_decision", 0.0, settings)
    assert every.recalibrations.eq(20).all()
    assert every.false_recalibrations.eq(19).all()


def test_never_policy_misses_a_large_abrupt_drift() -> None:
    settings = small_config()["drift"]
    scenario = {
        "amplitude": 0.08,
        "timescale": 5,
        "shape": "abrupt",
        "run_length": 60,
        "margin": 0.05,
        "calibration_cost": 1024,
    }
    never = drift.simulate(scenario, "never", 0.0, settings)
    assert never.missed_drift.all()
    assert never.violation_fraction.iloc[0] == pytest.approx(55 / 60)
    assert math.isclose(never.detection_delay.iloc[0], 55.0)


def test_config_hash_check_refuses_changes(tmp_path, monkeypatch) -> None:
    changed = tmp_path / "config.yaml"
    changed.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(feasibility, "CONFIG_PATH", changed)
    with pytest.raises(ValueError, match="differs from frozen"):
        feasibility.check_config_hash()


def test_committed_results_keep_every_row_and_failure() -> None:
    raw = pd.read_csv(feasibility.RAW_PATH, usecols=["width", "method", "regime", "status"])
    assert len(raw) == 230400
    full = raw.loc[
        (raw.method == "full_matrix") & (raw.width == 10) & (raw.regime == "fixed_total")
    ]
    assert full.status.ne("complete").all()
    assert len(pd.read_csv(mismatch.RAW_PATH, usecols=["repeat"])) == 346500
    assert len(pd.read_csv(drift.RAW_PATH, usecols=["repeat"])) == 100800


def test_threshold_table_names_a_first_crossing_or_none() -> None:
    table = pd.read_csv(analysis.THRESHOLDS_PATH)
    resolved = table.lambda_loses_to_raw_resolved.dropna()
    assert resolved.between(0.0, 1.0).all()
    pointwise = table.lambda_loses_to_raw
    both = table.dropna(subset=["lambda_loses_to_raw", "lambda_loses_to_raw_resolved"])
    assert (both.lambda_loses_to_raw <= both.lambda_loses_to_raw_resolved).all()
    assert pointwise.notna().sum() >= resolved.size
