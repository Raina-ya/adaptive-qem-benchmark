import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from src.sequential_decision import (
    CONFIG_PATH,
    RAW_PATH,
    SUMMARY_PATH,
    CellSpec,
    asymptotic_gap,
    batch_schedule,
    cell_seed,
    circuit_expectation,
    empirical_bernstein_halfwidth,
    first_certified_look,
    generate_rows,
    hoeffding_halfwidth,
    intercept_weights,
    load_config,
    mover_interval,
    observed_probabilities,
    policy_intervals,
    sampling_probability,
    score_policy,
    simulate_trajectory,
    summarize,
    wilson_bounds,
)

CONFIG = load_config()
ALPHA = 0.05


def _small_config(repeats: int = 2) -> dict[str, object]:
    return {
        **CONFIG,
        "repeats": repeats,
        "max_total_shots": [1024],
        "tasks": {
            "symmetric_gap": {**CONFIG["tasks"]["symmetric_gap"], "margins": [0.0, 0.2]},
            "readout_imbalanced": {**CONFIG["tasks"]["readout_imbalanced"], "margins": [0.2]},
        },
    }


def test_config_is_frozen_and_declares_no_hardware_or_tuning() -> None:
    assert CONFIG["version"] == "sequential_decision_v1"
    assert CONFIG["root_seed"] == 20261102
    assert CONFIG["confidence_level"] == 0.95
    assert CONFIG["batch_target_shots"] == 256
    assert CONFIG["simulator_only"] is True
    assert "no threshold is tuned" in CONFIG["threshold_selection"]
    assert CONFIG_PATH.name == "sequential_decision_v1.yaml"


def test_noiseless_circuit_matches_requested_expectation() -> None:
    for factor in (1, 3, 5):
        value = circuit_expectation(0.37, 3, factor, 0.0, 0.0)
        assert math.isclose(value, 0.37, abs_tol=1e-12)


def test_folding_reduces_signal_monotonically_under_noise() -> None:
    values = [circuit_expectation(0.6, 3, factor, 0.005, 0.03) for factor in (1, 3, 5)]
    assert values[0] > values[1] > values[2] > 0.0


def test_density_model_matches_aer_density_matrix() -> None:
    pytest.importorskip("qiskit_aer")
    from qiskit import transpile
    from qiskit.quantum_info import DensityMatrix, Pauli
    from qiskit_aer import AerSimulator

    from src.decision_instances import _candidate
    from src.mitigation.folding import fold_global
    from src.noise_models import build_noise_model, noise_spec

    spec = noise_spec("two_qubit", "high")
    spec = dataclasses.replace(spec, one_qubit_depolarizing=0.01)
    simulator = AerSimulator(method="density_matrix", noise_model=build_noise_model(spec))
    for depth in (1, 3):
        for factor in (1, 3, 5):
            circuit = fold_global(_candidate("a", 0.3, depth).circuit, factor)
            circuit.save_density_matrix()
            compiled = transpile(circuit, simulator, optimization_level=0)
            state = DensityMatrix(simulator.run(compiled).result().data()["density_matrix"])
            aer_value = float(state.expectation_value(Pauli("IZ")).real)
            assert math.isclose(
                aer_value,
                circuit_expectation(0.3, depth, factor, 0.01, 0.08),
                abs_tol=1e-9,
            )


def test_config_noise_matches_declared_project_strengths() -> None:
    pytest.importorskip("qiskit_aer")
    from src.noise_models import noise_spec

    for level, strength in CONFIG["noise_strengths"].items():
        spec = noise_spec("mixed", level)
        assert spec.readout_p01 == strength["readout_p01"]
        assert spec.one_qubit_depolarizing == strength["one_qubit"]
        assert spec.two_qubit_depolarizing == strength["two_qubit"]
        assert spec.readout_p10 == pytest.approx(
            CONFIG["readout_p10_fraction_of_p01"] * strength["readout_p01"]
        )


def test_intercept_weights_are_the_frozen_linear_fit() -> None:
    weights = intercept_weights((1, 3, 5))
    assert weights.sum() == pytest.approx(1.0)
    assert weights == pytest.approx([13.0 / 12.0, 1.0 / 3.0, -5.0 / 12.0])
    line = 0.8 - 0.05 * np.array([1.0, 3.0, 5.0])
    assert float(weights @ line) == pytest.approx(0.8)


def test_batch_schedule_spends_every_shot() -> None:
    assert batch_schedule(1024, 256) == (256, 256, 256, 256)
    assert batch_schedule(820, 256) == (256, 256, 256, 52)
    assert sum(batch_schedule(13108, 256)) == 13108
    with pytest.raises(ValueError):
        batch_schedule(0, 256)


@pytest.mark.parametrize("method", ["raw", "readout", "zne"])
@pytest.mark.parametrize("budget", [1024, 4096, 16384])
def test_fixed_total_budget_is_conserved(method: str, budget: int) -> None:
    spec = CellSpec("readout_imbalanced", 0.1, "medium", budget, method)
    trajectory = simulate_trajectory(spec, 11, CONFIG)
    assert int(trajectory.target_shots[-1]) + trajectory.calibration_shots == budget
    assert trajectory.trials[-1, : trajectory.trials.shape[1]].min() >= 1
    expected = int(budget * 0.2) if method == "readout" else 0
    assert trajectory.calibration_shots == expected


def test_calibration_is_charged_before_any_early_stop() -> None:
    spec = CellSpec("symmetric_gap", 0.2, "zero", 16384, "readout")
    outcome = score_policy(
        "wilson_stopping", simulate_trajectory(spec, 3, CONFIG), spec.margin, ALPHA
    )
    assert outcome["status"] == "certified"
    assert outcome["total_shots_used"] == outcome["target_shots_used"] + 3276
    assert outcome["calibration_shots"] == 3276


def test_trajectories_are_deterministic_and_seeds_are_unique() -> None:
    spec = CellSpec("gate_imbalanced", 0.05, "high", 4096, "zne")
    left = simulate_trajectory(spec, cell_seed(spec, 0, CONFIG), CONFIG)
    right = simulate_trajectory(spec, cell_seed(spec, 0, CONFIG), CONFIG)
    np.testing.assert_array_equal(left.estimate, right.estimate)
    assert cell_seed(spec, 0, CONFIG) != cell_seed(spec, 1, CONFIG)
    other = dataclasses.replace(spec, method="raw")
    assert cell_seed(spec, 0, CONFIG) != cell_seed(other, 0, CONFIG)


def test_wilson_bounds_match_project_implementation() -> None:
    from scipy.stats import norm

    from src.metrics import wilson_interval

    low, high = wilson_bounds(np.array(0.3), np.array(100.0), float(norm.ppf(0.975)))
    expected = wilson_interval(30, 100)
    assert float(low) == pytest.approx(expected[0])
    assert float(high) == pytest.approx(expected[1])


def test_correction_for_looks_widens_every_interval() -> None:
    estimate = np.full(16, 0.05)
    gradient = np.tile([2.0, -2.0], (16, 1))
    probability = np.tile([0.55, 0.5], (16, 1))
    trials = np.tile(np.arange(1.0, 17.0)[:, None] * 128.0, (1, 2))
    fixed_low, fixed_high = policy_intervals(
        "fixed_budget", estimate, gradient, probability, trials, ALPHA
    )
    corrected_low, corrected_high = policy_intervals(
        "wilson_stopping", estimate, gradient, probability, trials, ALPHA
    )
    assert np.all(corrected_low < fixed_low)
    assert np.all(corrected_high > fixed_high)


def test_bounds_shrink_with_more_shots() -> None:
    gradient = np.array([[2.0, -2.0]])
    probability = np.array([[0.6, 0.6]])
    for scale_low, scale_high in ((128.0, 1024.0), (1024.0, 8192.0)):
        small = np.full((1, 2), scale_low)
        large = np.full((1, 2), scale_high)
        assert hoeffding_halfwidth(gradient, small, ALPHA) > hoeffding_halfwidth(
            gradient, large, ALPHA
        )
        assert empirical_bernstein_halfwidth(
            gradient, probability, small, ALPHA
        ) > empirical_bernstein_halfwidth(gradient, probability, large, ALPHA)


def test_mover_reduces_to_wilson_for_one_component() -> None:
    low, high = mover_interval(
        np.array([0.3]), np.array([[1.0]]), np.array([[0.3]]), np.array([[100.0]]), ALPHA
    )
    from src.metrics import wilson_interval

    assert float(low[0]) == pytest.approx(wilson_interval(30, 100)[0])
    assert float(high[0]) == pytest.approx(wilson_interval(30, 100)[1])


def test_fixed_policy_reads_only_the_final_look() -> None:
    lower = np.array([0.1, -0.1, -0.1])
    upper = np.array([0.3, 0.1, 0.1])
    assert first_certified_look("fixed_budget", lower, upper) is None
    assert first_certified_look("wilson_stopping", lower, upper) == 0
    assert first_certified_look("wilson_stopping", -upper, -lower) == 0
    assert first_certified_look("wilson_stopping", lower[1:], upper[1:]) is None


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError):
        policy_intervals(
            "peek",
            np.zeros(1),
            np.array([[2.0, -2.0]]),
            np.array([[0.5, 0.5]]),
            np.full((1, 2), 10.0),
            ALPHA,
        )


def test_shared_readout_correction_preserves_raw_gap_sign() -> None:
    spec = CellSpec("symmetric_gap", 0.02, "high", 4096, "readout")
    for seed in range(20):
        trajectory = simulate_trajectory(spec, seed, CONFIG)
        raw_gap = 2.0 * (trajectory.probability[:, 0] - trajectory.probability[:, 1])
        np.testing.assert_array_equal(np.sign(trajectory.estimate), np.sign(raw_gap))


def test_asymptotic_gap_exposes_each_estimators_systematic_bias() -> None:
    symmetric = CellSpec("symmetric_gap", 0.1, "high", 4096, "raw")
    assert 0.0 < asymptotic_gap(symmetric, CONFIG) < 0.1
    gate = CellSpec("gate_imbalanced", 0.05, "high", 4096, "raw")
    assert asymptotic_gap(gate, CONFIG) < 0.0
    readout_gate = dataclasses.replace(gate, method="readout")
    assert asymptotic_gap(readout_gate, CONFIG) < 0.0
    imbalance = CellSpec("readout_imbalanced", 0.05, "high", 4096, "raw")
    assert asymptotic_gap(imbalance, CONFIG) < 0.0
    fixed = dataclasses.replace(imbalance, method="readout")
    assert asymptotic_gap(fixed, CONFIG) > 0.0


def test_observed_probabilities_stay_physical() -> None:
    for task, definition in CONFIG["tasks"].items():
        for noise in definition["noise"]:
            spec = CellSpec(task, 0.2, noise, 4096, "zne")
            for values in observed_probabilities(spec, CONFIG).values():
                assert np.all((values >= 0.0) & (values <= 1.0))


def test_exact_half_probability_is_stable_across_platform_rounding() -> None:
    for method in ("raw", "readout", "zne"):
        spec = CellSpec("symmetric_gap", 0.0, "zero", 1024, method)
        for values in observed_probabilities(spec, CONFIG).values():
            assert all(sampling_probability(value) == 0.5 for value in values)
    assert sampling_probability(0.5 + 2e-16) == 0.5
    assert sampling_probability(0.5 - 2e-16) == 0.5


def test_committed_rows_replay_from_their_seeds() -> None:
    if not RAW_PATH.exists():
        pytest.skip("sequential decision raw artifact has not been generated")
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    keys = ["task", "margin", "noise", "max_total_shots", "method"]
    sample = pd.concat(
        [
            raw.loc[(raw.margin == 0.0) & (raw.noise == "zero") & (raw.repeat == 0)],
            raw.loc[(raw.repeat == 1) & (raw.max_total_shots == 1024)],
        ]
    )
    for values, group in sample.groupby(keys, sort=False):
        spec = CellSpec(
            values[0], float(values[1]), values[2], int(values[3]), values[4]
        )
        for row in group.itertuples():
            assert row.seed == cell_seed(spec, int(row.repeat), CONFIG)
            outcome = score_policy(
                row.policy,
                simulate_trajectory(spec, int(row.seed), CONFIG),
                spec.margin,
                ALPHA,
            )
            assert outcome["status"] == row.status
            assert outcome["total_shots_used"] == row.total_shots_used
            assert outcome["gap_estimate"] == pytest.approx(row.gap_estimate, abs=1e-9)


def test_null_gap_scores_any_certification_as_wrong() -> None:
    spec = CellSpec("symmetric_gap", 0.0, "zero", 4096, "raw")
    outcomes = [
        score_policy("naive_wald_peeking", simulate_trajectory(spec, seed, CONFIG), 0.0, ALPHA)
        for seed in range(100)
    ]
    assert any(item["wrong_certified"] for item in outcomes)
    assert all(item["forced_error"] is None for item in outcomes)
    assert all(item["forced_regret"] == 0.0 for item in outcomes)


def test_corrected_looks_control_false_certification_where_naive_peeking_fails() -> None:
    spec = CellSpec("symmetric_gap", 0.0, "low", 16384, "raw")
    trajectories = [simulate_trajectory(spec, 5000 + seed, CONFIG) for seed in range(600)]

    def rate(policy: str) -> float:
        return float(
            np.mean(
                [score_policy(policy, item, 0.0, ALPHA)["wrong_certified"] for item in trajectories]
            )
        )

    assert rate("naive_wald_peeking") > 0.15
    for policy in (
        "fixed_budget",
        "wilson_stopping",
        "gap_confidence_stopping",
        "variance_aware_stopping",
    ):
        assert rate(policy) < 0.08


def test_generated_schema_keeps_every_policy_and_status() -> None:
    frame = generate_rows(_small_config(), repeats=2)
    assert set(frame.policy) == {*CONFIG["policies"], *CONFIG["negative_control_policies"]}
    assert set(frame.result_label) == {"noisy_simulated"}
    assert (frame.total_shots_used <= frame.max_total_shots).all()
    fixed = frame.loc[frame.policy == "fixed_budget"]
    assert (fixed.total_shots_used == fixed.max_total_shots).all()
    seeds = frame.groupby(["task", "margin", "noise", "method", "repeat"]).seed.nunique()
    assert (seeds == 1).all()


def test_summary_rates_recompute_from_rows() -> None:
    frame = generate_rows(_small_config(), repeats=4)
    summary = summarize(frame)
    assert int(summary.repeats.sum()) == len(frame)
    row = summary.iloc[0]
    assert row.unresolved_rate + row.certified_rate + row.failed_rate == pytest.approx(1.0)
    null = summary.loc[summary.margin == 0.0]
    assert null.forced_decision_error.isna().all()


def test_committed_artifacts_recompute_and_cover_the_frozen_grid() -> None:
    if not RAW_PATH.exists():
        pytest.skip("sequential decision raw artifact has not been generated")
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    tasks = CONFIG["tasks"]
    cells = sum(len(item["margins"]) * len(item["noise"]) for item in tasks.values())
    assert cells == 44
    policies = len(CONFIG["policies"]) + len(CONFIG["negative_control_policies"])
    expected = cells * 3 * 3 * policies * CONFIG["repeats"]
    assert len(raw) == expected
    assert raw.status.isin(["certified", "unresolved", "failed"]).all()
    assert (raw.total_shots_used <= raw.max_total_shots).all()
    assert raw.drop_duplicates(
        ["task", "margin", "noise", "max_total_shots", "method", "policy", "repeat"]
    ).shape[0] == len(raw)
    saved = pd.read_csv(SUMMARY_PATH)
    assert int(saved.repeats.sum()) == len(raw)
    wrong = raw.groupby(
        ["task", "margin", "noise", "max_total_shots", "method", "policy"]
    ).wrong_certified.mean()
    keyed = saved.set_index(
        ["task", "margin", "noise", "max_total_shots", "method", "policy"]
    ).wrong_certified_rate
    pd.testing.assert_series_equal(
        wrong.sort_index(), keyed.sort_index(), check_names=False, check_exact=False
    )
