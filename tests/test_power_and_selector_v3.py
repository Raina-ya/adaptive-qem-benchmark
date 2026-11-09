import ast
import copy
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import confirmatory_analysis_v1 as confirmatory
from src import (
    confirmatory_v1,
    decision_tasks_v3,
    power_analysis_v1,
    selector_v3,
    selector_v3_analysis,
    shots_to_reliability,
)
from src import generate_selector_v3 as generator
from src.density_sim import Gate, Noise, derive_seed, fold_global, ideal_expectation, rx
from src.sequential_decision import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_required_pairs_matches_the_mcnemar_formula() -> None:
    assert power_analysis_v1.required_pairs(0.2, 0.1, 0.05, 0.8) == 155
    # A rate below the minimum difference is impossible, so it is raised to it.
    assert power_analysis_v1.required_pairs(
        0.01, 0.1, 0.05, 0.8
    ) == power_analysis_v1.required_pairs(0.1, 0.1, 0.05, 0.8)


def test_interval_has_width_with_no_discordant_pairs() -> None:
    low, high = power_analysis_v1.paired_interval(0, 0, 100, 1.96)
    assert low < 0.0 < high


def test_classification_separates_close_from_underpowered() -> None:
    classify = power_analysis_v1.classify
    assert classify(0.02, 0.2, 0.03, 0.18, 0.1) == "resolved_difference"
    assert classify(-0.05, 0.05, -0.04, 0.04, 0.1) == "close_within_margin"
    assert classify(-0.2, 0.2, -0.15, 0.15, 0.1) == "underpowered"


def test_every_planned_comparison_has_an_effect_size_and_repeats() -> None:
    plan = pd.read_csv(power_analysis_v1.PLAN_PATH)
    assert plan.minimum_difference.gt(0.0).all()
    assert plan.pairs_required.gt(0).all()
    assert plan.pairs_planned.gt(0).all()
    assert plan.group.eq("confirmatory").all()
    assert set(plan.role) <= {"near_tie", "rank_uncertain", "core_counterexample"}
    assert plan.alpha_per_comparison.to_numpy() == pytest.approx(0.05 / len(plan), rel=1e-6)
    confirmatory_v1.check_manifest_hash()


def test_plan_keeps_core_counterexample_even_when_resolved() -> None:
    table = pd.read_csv(power_analysis_v1.COMPARISONS_PATH)
    plan = power_analysis_v1.build_plan(table)
    core = plan.loc[plan.role == "core_counterexample"]
    assert set(core.task) == {"medium_dense8"}
    assert not plan.cell.str.contains("0.1\\|").any()


def test_holm_stops_at_the_first_failure() -> None:
    assert confirmatory.holm_flags([0.001, 0.03, 0.04], 0.05) == [True, False, False]
    assert confirmatory.holm_flags([0.01, 0.02, 0.03], 0.05) == [True, True, True]


def test_null_labels_follow_the_frozen_rule() -> None:
    assert confirmatory.label(True, True) == "resolved_difference"
    assert confirmatory.label(False, True) == "close_within_margin"
    assert confirmatory.label(False, False) == "underpowered"


def test_confirmatory_seeds_differ_from_v1_engine_seeds() -> None:
    from src.sequential_decision import CellSpec, cell_seed

    source = load_config()
    spec = CellSpec("symmetric_gap", 0.05, "medium", 4096, "raw")
    rows = confirmatory_v1.sample_outcomes("confirmatory", spec, 3, source)
    assert all(row["seed"] != cell_seed(spec, row["repeat"], source) for row in rows)
    assert len({row["seed"] for row in rows}) == 3


def test_holdout_tasks_were_never_in_the_v1_engine() -> None:
    assert not set(confirmatory_v1.CONFIG["holdout_tasks"]) & set(load_config()["tasks"])


@pytest.mark.parametrize("method", ["raw", "readout", "zne", "combined"])
def test_every_method_spends_exactly_its_budget(method) -> None:
    factors = 3 if method in ("zne", "combined") else 1
    target, calibration = shots_to_reliability.method_shots(4099, method, factors, 0.2)
    assert target.sum() + calibration == 4099
    assert (calibration > 0) == (method in ("readout", "combined"))


@pytest.mark.parametrize("method", ["raw", "readout", "zne", "combined"])
def test_delta_variance_matches_the_sampled_spread(method) -> None:
    config = copy.deepcopy(shots_to_reliability.CONFIG)
    config.update(repeats=800, ladder=[8192, 16384])
    sample = shots_to_reliability.simulate_cell(
        "readout_imbalanced", 0.07, "medium", method, config, load_config()
    )
    sampled = sample["gap"][:, -1].std()
    predicted = float(np.sqrt(sample["variance"][:, -1]).mean())
    assert predicted == pytest.approx(sampled, rel=0.15)


def ladder_rows(highs, limit=0.1):
    return pd.DataFrame(
        [
            {
                "task": "t",
                "margin": 0.1,
                "noise": "n",
                "method": method,
                "budget": 1024 * 2**index,
                "decision_error": high,
                "decision_error_wilson_high": high,
                "reached_correct_stop": 0.5,
                "unresolved_at_budget": 0.5,
                "wrong_stop": 0.0,
                "total_executions": 2,
                "calibration_share": 0.0,
                "two_qubit_gate_shots": 100.0,
                "regret_at_stop": 0.0,
                "mean_shots_at_stop": 1024.0,
                "asymptotic_gap": limit,
                "result_label": "noisy_simulated",
            }
            for method in ("raw",)
            for index, high in enumerate(highs)
        ]
    )


def test_target_needs_every_larger_budget_to_stay_under_it() -> None:
    config = {"target_errors": [0.1]}
    table = shots_to_reliability.target_table(ladder_rows([0.3, 0.05, 0.2, 0.08, 0.04]), config)
    assert table.status.iloc[0] == "reached"
    assert table.shots_to_target.iloc[0] == 1024 * 2**3


def test_a_wrong_sign_limit_is_blocked_by_bias_not_budget() -> None:
    config = {"target_errors": [0.1]}
    blocked = shots_to_reliability.target_table(ladder_rows([0.6, 0.8], limit=-0.01), config)
    assert blocked.status.iloc[0] == "blocked_by_bias"
    short = shots_to_reliability.target_table(ladder_rows([0.6, 0.4], limit=0.01), config)
    assert short.status.iloc[0] == "not_reached_by_maximum_budget"


def test_failed_calibration_counts_as_a_decision_error() -> None:
    sample = {
        "gap": np.array([[0.5], [0.5]]),
        "variance": np.array([[1e-6], [1e-6]]),
        "failed": np.array([[True], [False]]),
        "asymptotic_gap": 0.5,
        "gate_shots": np.array([10.0]),
        "executions": 2,
        "calibrated": True,
    }
    config = {
        "ladder": [1024],
        "confidence_level": 0.95,
        "calibration_fraction": 0.2,
        "evidence_label": "noisy_simulated",
    }
    row = shots_to_reliability.summarize_cell({}, sample, 0.5, config)[0]
    assert row["decision_error"] == 0.5
    assert row["failed_calibration"] == 0.5
    assert row["unresolved_at_budget"] == 0.5


def test_task_manifest_uses_noiseless_values_only() -> None:
    tree = ast.parse((ROOT / "src/decision_tasks_v3.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"__future__", "pathlib", "src.density_sim"}
    generator.check_manifest_hash()


def test_task_manifest_has_real_gaps_and_every_family_in_every_split() -> None:
    frame = pd.read_csv(decision_tasks_v3.MANIFEST_PATH)
    assert frame.parameter_seed.is_unique
    for _, group in frame.groupby("task_id"):
        values = np.sort(group.ideal_expectation.to_numpy())[::-1]
        low, high = decision_tasks_v3.CONFIG["gap_classes"][group.gap_class.iloc[0]]
        assert low <= values[0] - values[1] < high
    tasks = frame.drop_duplicates("task_id")
    assert (pd.crosstab(tasks.family, tasks.split) > 0).all().all()


@pytest.mark.parametrize("family", decision_tasks_v3.CONFIG["families"])
def test_folding_preserves_each_family_value(family) -> None:
    gates = decision_tasks_v3.BUILDERS[family](4, 2, derive_seed("synthetic-family", family))
    pauli = decision_tasks_v3.observable(family, 4)
    assert ideal_expectation(fold_global(gates, 3), pauli) == pytest.approx(
        ideal_expectation(gates, pauli), abs=1e-12
    )


def test_rx_rotates_about_x() -> None:
    assert rx(math.pi)[0, 1] == pytest.approx(-1j)
    assert ideal_expectation([Gate("rx", (0,), 0.6)], "Z") == pytest.approx(math.cos(0.6))


def test_calibration_scale_only_moves_in_the_drift_condition() -> None:
    config = decision_tasks_v3.CONFIG
    assert generator.calibration_scale("mixed", 0.5, config) == 1.0
    assert generator.calibration_scale("readout_drift", 0.0, config) == 0.5
    assert generator.calibration_scale("readout_drift", 1.0, config) == 1.0


def synthetic_tasks(values=(0.9, 0.2, -0.3)) -> tuple[list, list, list]:
    """Build three candidates from plain rotations, with no manifest circuit involved."""

    dists, signs, paulis = [], [], []
    for value in values:
        angle = math.acos(value)
        gate = [Gate("ry", (0,), angle)]
        from src.density_sim import outcome_probabilities, parity_signs

        probabilities = outcome_probabilities(gate, "ZI", Noise("clean"))
        dists.append([probabilities, probabilities, probabilities])
        signs.append(parity_signs("ZI"))
        paulis.append("ZI")
    return dists, signs, paulis


def test_clear_gap_certifies_the_right_candidate_without_noise() -> None:
    dists, signs, paulis = synthetic_tasks()
    config = decision_tasks_v3.CONFIG
    rng = np.random.default_rng(3)
    outcome = generator.certify(rng, 16384, dists, signs, paulis, Noise("clean"), config)
    assert outcome["certified_winner"] == 0
    assert outcome["certified_look"] in config["certification"]["looks"]


def test_equal_candidates_are_rarely_certified() -> None:
    dists, signs, paulis = synthetic_tasks((0.4, 0.4, 0.4))
    config = decision_tasks_v3.CONFIG
    passed = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        outcome = generator.certify(rng, 4096, dists, signs, paulis, Noise("clean"), config)
        passed += outcome["l1_look"] > 0
    assert passed <= 6


def alarm_inputs():
    batches = 5
    corrected = np.tile([0.8, 0.2, 0.0], (batches, 1))
    corrected = corrected + np.linspace(-0.01, 0.01, batches)[:, None] * np.array([1.0, 0.5, -0.5])
    return {
        "corrected_f1": corrected,
        "plain_f1": corrected * 0.9,
        "corrected_f5": corrected * 0.5,
        "plain_var": np.full((batches, 3), 1e-4),
        "cal_rates": np.full((batches, 2, 2), 0.02),
    }


SETTINGS = decision_tasks_v3.CONFIG["certification"]


def fire(**changes) -> int:
    values = {**alarm_inputs(), **changes}
    return generator._alarms(5, 0, 1, *values.values(), 400, SETTINGS)


def test_quiet_data_fires_no_alarm() -> None:
    assert fire() == 0


def test_each_alarm_names_its_own_problem() -> None:
    base = alarm_inputs()
    assert fire(plain_f1=base["corrected_f1"][:, [1, 0, 2]]) & 1
    reversed_gap = base["corrected_f5"][:, [1, 0, 2]].copy()
    assert fire(corrected_f5=reversed_gap) & 2
    drifted = base["cal_rates"].copy()
    drifted[-1] = 0.2
    assert fire(cal_rates=drifted) & 4
    over = base["corrected_f1"].copy()
    over[:, 0] = 1.2 + np.linspace(-0.001, 0.001, 5)
    assert fire(corrected_f1=over) & 8
    noisy = base["corrected_f1"].copy()
    noisy[:, 2] = np.array([0.2, -0.2, 0.2, -0.2, 0.0])
    assert fire(corrected_f1=noisy) & 16
    flips = base["corrected_f1"].copy()
    flips[:, 1] = np.array([0.9, 0.9, 0.2, 0.2, 0.2])
    assert fire(corrected_f1=flips) & 32


def test_certificate_cannot_see_ideal_values() -> None:
    assert "ideal" not in " ".join(inspect.signature(generator.certify).parameters)
    assert "ideal" not in " ".join(inspect.signature(selector_v3.choose_branch).parameters)
    assert "ideal" not in (ROOT / "src/selector_v3.py").read_text(encoding="utf-8")


def selector_data(**changes):
    base = np.array([[0.5, 0.2, 0.1]])
    data = {
        "pilot_raw": base * 0.8,
        "pilot_raw_se": np.full((1, 3), 0.01),
        "pilot_readout": base,
        "pilot_fit": base,
        "pilot_fit_se": np.full((1, 3), 0.01),
        "branch_readout": base,
        "branch_readout_se": np.full((1, 3), 0.01),
        "branch_combined": base[:, [1, 0, 2]],
        "branch_combined_se": np.full((1, 3), 0.01),
        "branch_raw": base,
        "branch_raw_se": np.full((1, 3), 0.01),
        "drift_statistic": np.array([0.0]),
    }
    data.update(changes)
    return data


RULE = selector_v3.Rule(1.96, 3.0, True)


def test_selector_defaults_to_readout() -> None:
    branch, pick, abstain, reason = selector_v3.apply_rule(selector_data(), RULE)
    assert branch.tolist() == [0] and pick.tolist() == [0]
    assert not abstain[0] and reason[0] == 0


def test_selector_needs_strong_evidence_on_both_contenders_to_deviate() -> None:
    both = np.array([[0.8, 0.4, 0.1]])
    assert selector_v3.apply_rule(selector_data(pilot_fit=both), RULE)[0][0] == 1
    one = np.array([[0.8, 0.2, 0.1]])
    assert selector_v3.apply_rule(selector_data(pilot_fit=one), RULE)[0][0] == 0


def test_selector_uses_raw_only_when_correction_is_unresolved() -> None:
    same = np.array([[0.5, 0.2, 0.1]])
    assert selector_v3.apply_rule(selector_data(pilot_raw=same), RULE)[0][0] == 2
    plain = selector_v3.Rule(1.96, 3.0, False)
    assert selector_v3.apply_rule(selector_data(pilot_raw=same), plain)[0][0] == 0


def test_selector_abstains_with_a_named_reason() -> None:
    close = np.array([[0.5, 0.49, 0.1]])
    _, _, abstain, reason = selector_v3.apply_rule(selector_data(branch_readout=close), RULE)
    assert abstain[0] and reason[0] == 2
    _, _, abstain, reason = selector_v3.apply_rule(
        selector_data(drift_statistic=np.array([4.0])), RULE
    )
    assert abstain[0] and reason[0] == 1


def test_rule_choice_takes_no_holdout_argument() -> None:
    parameters = inspect.signature(selector_v3_analysis.choose_rule).parameters
    assert list(parameters) == ["frame_train", "data_train", "frame_val", "data_val", "config"]


def scored_frame(gate_count=4.0):
    return pd.DataFrame(
        {
            "task_id": ["a", "a", "b", "b"],
            "budget": [4096] * 4,
            "gate_count": [gate_count] * 4,
            "noise": ["n"] * 4,
            "ideal": [np.array([0.9, 0.1, 0.0])] * 4,
        }
    )


def test_every_policy_charges_its_folded_gate_shots() -> None:
    config = decision_tasks_v3.CONFIG
    frame = scored_frame()
    raw = selector_v3_analysis.gate_shots(frame, None, "always_raw", config)
    zne = selector_v3_analysis.gate_shots(frame, None, "always_zne", config)
    assert zne[0] == pytest.approx(3.0 * raw[0])
    readout = selector_v3_analysis.gate_shots(frame, None, "always_readout", config)
    assert readout[0] == pytest.approx(0.8 * raw[0])
    branch = np.array([0, 1, 2, 0])
    selector = selector_v3_analysis.gate_shots(frame, branch, "selector", config)
    assert selector[1] > selector[0]
    assert selector[0] == selector[2]


def test_abstentions_do_not_count_as_covered_errors() -> None:
    frame = scored_frame()
    outcome = selector_v3_analysis.score(
        frame, np.array([0, 1, 1, 0]), np.array([False, False, True, True])
    )
    row = selector_v3_analysis.summarize(frame, outcome, np.ones(4), 0.1)
    assert row["coverage"] == 0.5
    assert row["conditional_error"] == 0.5
    assert row["unconditional_error"] == 0.25
    assert row["reliability_coverage"] == 0.25
    assert row["forced_regret"] == pytest.approx(0.4)


def test_acceptance_fails_when_the_selector_does_no_better() -> None:
    frame = scored_frame()
    zero = {
        "regret": np.zeros(4),
        "error": np.zeros(4, dtype=bool),
        "abstain": np.zeros(4, dtype=bool),
    }
    policies = {
        "selector": (zero, np.full(4, 2.0), 0.1),
        "always_readout": (zero, np.full(4, 1.0), 0.2),
        "abstaining_readout": (zero, np.full(4, 1.0), 0.1),
    }
    table = selector_v3_analysis.acceptance_table(
        frame, policies, {**decision_tasks_v3.CONFIG, "bootstraps": 50}
    )
    assert not table.passed.any()


def test_a_wrong_certificate_counts_as_false_certification() -> None:
    frame = scored_frame().assign(
        certified_look=[5, 5, 0, 0],
        certified_winner=[0, 1, -1, -1],
        l1_look=[3, 3, 3, 0],
        l1_winner=[0, 1, 1, -1],
        naive_pass=[True, True, True, True],
        naive_winner=[0, 1, 1, 2],
    )
    rows = {
        row["rule"]: row for row in selector_v3_analysis.certificate_rows(frame, {"grouping": "a"})
    }
    assert rows["both_layers"]["false_certification_per_run"] == 0.25
    assert rows["layer_one_only"]["false_certification_given_certified"] == pytest.approx(2 / 3)
    assert rows["naive_independent_z"]["certified_rate"] == 1.0


def test_generated_row_has_every_column_the_analysis_parses() -> None:
    config = copy.deepcopy(decision_tasks_v3.CONFIG)
    config.update(repeats=2, budgets=[4096])
    tasks = pd.DataFrame(
        [
            {
                "task_id": "synthetic",
                "family": "tfim_trotter",
                "width": 2,
                "depth": 1,
                "parameter_seed": derive_seed("synthetic-row", index),
                "observable_q0_first": "ZZ",
                "ideal_expectation": 0.1 * index,
            }
            for index in range(3)
        ]
    )
    noise = Noise("mixed", **config["noise"]["mixed"])
    rows = generator.simulate_task(tasks, noise, config)
    frame = pd.DataFrame(rows)
    data = selector_v3_analysis.parse(frame, 3)
    assert all(values.shape == (2, 3) for name, values in data.items() if name != "drift_statistic")
    first = generator.simulate_task(tasks, noise, config)
    assert first == rows


def test_confirmatory_results_cover_the_whole_frozen_plan() -> None:
    table = pd.read_csv(confirmatory.RESULTS_PATH)
    plan = pd.read_csv(power_analysis_v1.PLAN_PATH)
    confirmed = table.loc[table.group == "confirmatory"]
    assert len(confirmed) == len(plan)
    assert confirmed.status.eq("run").all()
    assert set(table.classification) <= {
        "resolved_difference",
        "close_within_margin",
        "underpowered",
    }
    assert (confirmed.paired_repeats >= confirmed.pairs_planned).all()


def test_selector_acceptance_table_is_frozen_three_routes() -> None:
    table = pd.read_csv(selector_v3_analysis.ACCEPTANCE_PATH)
    assert len(table) == 3
    selection = pd.read_csv(selector_v3_analysis.RULE_PATH)
    assert selection.selected.sum() == 1
    assert not any("holdout" in column for column in selection.columns)
