import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import generate_signal_v2 as generator
from src import signal_manifest_v2 as manifest_module
from src import signal_strata_v1, signal_v2_analysis, zne_gate_audit_v1
from src import zne_gate_v2 as gate
from src import zne_gate_v2_diagnostics as diagnostics
from src.density_sim import Noise, derive_seed

ROOT = Path(__file__).resolve().parents[1]
SMALL = copy.deepcopy(manifest_module.CONFIG)
SMALL.update(budgets=[512], repeats=6, truth_draws=400)


def synthetic_row() -> pd.Series:
    """Use a seed outside the manifest's derivation, so no manifest circuit is simulated."""

    return pd.Series(
        {
            "circuit_id": "synthetic_w2",
            "width": 2,
            "depth": 2,
            "parameter_seed": derive_seed("synthetic-test", 1),
            "observable_q0_first": "XZ",
        }
    )


def test_band_edges_and_near_zero_has_no_relative_error() -> None:
    assert signal_strata_v1.signal_band(0.0599) == "near_zero"
    assert signal_strata_v1.signal_band(-0.06) == "weak"
    assert signal_strata_v1.signal_band(0.5) == "strong"
    assert signal_strata_v1.signal_band(-1.0) == "boundary_adjacent"
    assert np.isnan(signal_strata_v1.normalized_error(0.02, 0.01))
    assert signal_strata_v1.normalized_error(-0.1, 0.5) == pytest.approx(0.2)


def test_v1_strata_report_noise_free_cells_apart() -> None:
    summary = signal_strata_v1.summarize_bands(signal_strata_v1.build_cells())
    assert set(summary.noise_scope) == {"noisy", "noise_free"}
    strong = summary.loc[(summary.signal_band == "strong") & (summary.noise_scope == "noisy")]
    assert strong.independent_circuits.eq(0).all()


def test_audit_truth_uses_the_other_half() -> None:
    frame = zne_gate_audit_v1.load_inputs()
    cells = zne_gate_audit_v1.audit_cells(frame)
    assert len(cells) == 64
    even = frame.loc[(frame.repeat % 2 == 1) & (frame.circuit_id == cells.circuit_id.iloc[0])]
    row = cells.iloc[0]
    held = even.loc[(even.cell == row.cell) & (even.total_shot_budget == row.total_shot_budget)]
    assert row.gate_half == "even"
    assert row.held_gain_mean == pytest.approx(held.true_gain.mean())


def test_really_improved_needs_a_positive_interval() -> None:
    assert zne_gate_audit_v1.really_improved(np.array([0.1, 0.11, 0.09, 0.1])) == (True, True)
    assert zne_gate_audit_v1.really_improved(np.array([-0.1, 0.1, -0.1, 0.1]))[0] is False


def test_manifest_matches_its_hash_and_selection_rule() -> None:
    generator.check_manifest_hash()
    manifest = pd.read_csv(manifest_module.MANIFEST_PATH)
    assert manifest.circuit_id.is_unique and manifest.parameter_seed.is_unique
    assert (manifest.ideal_expectation.map(manifest_module.band_of) == manifest.signal_band).all()
    counts = manifest.signal_band.value_counts()
    assert (counts >= manifest_module.CONFIG["minimum_circuits_per_band"]).all()
    # A rebuild scans noiseless values only and gives the same rows.
    check = manifest.loc[manifest.width == 2]
    rebuilt = pd.DataFrame(
        [row for depth in (1, 3, 6) for row in manifest_module._select_cell(2, depth)]
    )
    assert rebuilt.circuit_id.tolist() == check.circuit_id.tolist()


def test_manifest_builder_reads_no_results() -> None:
    tree = ast.parse((ROOT / "src/signal_manifest_v2.py").read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"__future__", "pathlib", "src.density_sim"}


def test_splits_have_every_band_in_every_split() -> None:
    manifest = pd.read_csv(manifest_module.MANIFEST_PATH)
    table = pd.crosstab(manifest.signal_band, manifest.split)
    assert (table > 0).all().all()


def test_changed_manifest_is_refused(tmp_path, monkeypatch) -> None:
    changed = tmp_path / "manifest.csv"
    changed.write_text("circuit_id\nother\n", encoding="utf-8")
    monkeypatch.setattr(generator, "MANIFEST_PATH", changed)
    with pytest.raises(ValueError, match="differs from frozen"):
        generator.check_manifest_hash()


def test_fit_weights_recover_polynomials_at_zero() -> None:
    factors = (1, 3, 5, 7)
    assert np.polyval([1.0, -0.2], factors) @ generator.fit_weights(factors, 1) == pytest.approx(
        -0.2
    )
    values = np.polyval([0.5, 1.0, -0.2, 0.3], factors)
    assert values @ generator.fit_weights(factors, 3) == pytest.approx(0.3)


def test_exponential_fit_recovers_a_decay_and_flags_undefined_rows() -> None:
    factors = (1, 3, 5)
    values = np.array([[-0.8 * np.exp(-0.2 * f) for f in factors], [0.5, -0.1, 0.2]])
    estimates, undefined = generator.fit_estimates("exponential", factors, values)
    assert estimates[0] == pytest.approx(-0.8)
    assert undefined.tolist() == [False, True]
    assert estimates[1] == 0.5


def test_readout_weights_invert_the_assignment_matrix() -> None:
    from src.density_sim import assignment_matrix, parity_signs

    p01, p10 = np.array([0.1, 0.05]), np.array([0.2, 0.03])
    single = [assignment_matrix(1, p01[i], p10[i]) for i in range(2)]
    matrix = np.kron(single[1], single[0])
    expected = parity_signs("ZZ") @ np.linalg.inv(matrix)
    assert generator.readout_weights("ZZ", p01, p10) == pytest.approx(expected)
    identity = generator.readout_weights("ZI", p01, p10)
    assert identity == pytest.approx(
        parity_signs("ZI") @ np.linalg.inv(np.kron(np.eye(2), single[0]))
    )


def test_equal_budget_and_determinism() -> None:
    noise = Noise("mixed", depolarizing_2q=0.03, readout_p01=0.02, readout_p10=0.016)
    first = generator.simulate_condition(synthetic_row(), noise, SMALL)
    second = generator.simulate_condition(synthetic_row(), noise, SMALL)
    assert first[0] == second[0] and first[2] == second[2]
    for design in SMALL["designs"].values():
        assert sum(generator.split_evenly(512, len(design["factors"]))) == 512
    assert first[0][0]["result_label"] == "noisy_simulated"


def test_truth_matches_the_analytic_linear_mse() -> None:
    config = copy.deepcopy(SMALL)
    config["truth_draws"] = 40000
    noise = Noise("dep", depolarizing_2q=0.05)
    _, exact, truth = generator.simulate_condition(synthetic_row(), noise, config)
    means = [
        item["expectation"]
        for item in exact
        if item["folding"] == "global" and item["factor"] in (1, 3, 5)
    ]
    row = next(
        item for item in truth if item["design"] == "global_135" and item["model"] == "linear"
    )
    weights = generator.fit_weights((1, 3, 5), 1)
    shots = np.array(generator.split_evenly(512, 3))
    bias = np.dot(means, weights) - row["reference"]
    variance = np.sum(weights**2 * (1.0 - np.square(means)) / shots)
    assert row["fit_rmse"] == pytest.approx(np.sqrt(bias**2 + variance), rel=0.02)


def test_truth_keeps_the_unconstrained_estimate_beside_the_clipped_one() -> None:
    noise = Noise("none")
    _, _, truth = generator.simulate_condition(synthetic_row(), noise, SMALL)
    assert {"fit_rmse", "fit_rmse_clipped", "out_of_bounds_rate"} <= set(truth[0])


def constructed(values, garbage=None, corrected=None):
    values = np.asarray(values, dtype=float)
    errors = np.full(values.shape, 0.01)
    garbage = np.zeros(values.shape) if garbage is None else garbage
    corrected = values if corrected is None else corrected
    raw = np.full(len(values), float(values[:, 0].mean())) + np.linspace(-0.001, 0.001, len(values))
    return gate.runtime_checks(
        values, errors, garbage, corrected, raw, np.full(len(values), 0.005), 1.1
    )


def noisy_rows(means, count=40, spread=0.01):
    rng = np.random.default_rng(5)
    return np.asarray(means) + rng.normal(0.0, spread, size=(count, len(means)))


def test_physical_variants_differ_on_a_small_overshoot() -> None:
    checks = constructed(noisy_rows([0.97, 0.93, 0.89]))
    assert checks["physical_strict"] is False
    assert checks["physical_ci"] is True
    assert checks["physical_tolerance"] is True
    assert checks["physical_constrained"] is True


def test_a_large_overshoot_fails_every_non_constrained_bound() -> None:
    checks = constructed(noisy_rows([0.98, 0.60, 0.22]))
    assert not (checks["physical_strict"] or checks["physical_ci"] or checks["physical_tolerance"])


def test_trend_checks_catch_a_rise_and_curvature() -> None:
    rising = constructed(noisy_rows([0.3, 0.5, 0.7]))
    assert rising["trend_monotonic"] is False
    curved = constructed(noisy_rows([0.8, 0.3, 0.25], spread=0.002))
    assert curved["trend_quadratic"] is False
    assert curved["trend_leave_one_out"] is False
    straight = constructed(noisy_rows([0.8, 0.6, 0.4], spread=0.002))
    assert straight["trend_quadratic"] is True and straight["trend_leave_one_out"] is True


def test_corrected_monotonicity_reads_the_corrected_values() -> None:
    values = noisy_rows([0.8, 0.6, 0.4])
    checks = constructed(values, corrected=noisy_rows([0.3, 0.5, 0.7]))
    assert checks["trend_monotonic"] is True
    assert checks["trend_corrected_monotonic"] is False


def test_retention_rejects_an_unidentifiable_baseline() -> None:
    assert constructed(noisy_rows([0.001, 0.0, 0.0], spread=0.02))["retention"] is False
    assert constructed(noisy_rows([0.8, 0.6, 0.4]))["retention"] is True


def table(size: int, split: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        name: rng.random(size) > 0.3
        for name in (*gate.PHYSICAL, *gate.TREND, "garbage", "retention", "estimated_gain")
    }
    frame = pd.DataFrame(data)
    frame["physical_constrained"] = True
    frame["trend_none"] = True
    frame["improved"] = frame.physical_ci & frame.garbage
    frame["improved_clipped"] = frame.improved
    frame["split"] = split
    frame["circuit_id"] = [f"c{index % 8}" for index in range(size)]
    frame["noise"] = "n"
    frame["total_shot_budget"] = 512
    frame["signal_band"] = "weak"
    frame["v1_full"] = frame.physical_strict
    frame["v1_runtime"] = frame.physical_strict
    return frame


def test_selection_never_changes_when_holdout_changes() -> None:
    train, validation = table(200, "train", 1), table(120, "validation", 2)
    chosen, selection = gate.select_gate(train, validation)
    assert chosen.physical == "physical_ci" and chosen.garbage
    assert selection.selected.sum() == 1
    again, _ = gate.select_gate(train, validation)
    assert again == chosen


def test_every_rejection_names_a_mechanism() -> None:
    holdout = table(80, "holdout", 3)
    chosen = gate.Candidate("physical_ci", "trend_monotonic", True, False, True)
    reasons = gate.rejection_reasons(chosen, holdout)
    assert len(reasons) == int((~gate.decide(chosen, holdout)).sum())
    assert reasons.failure_mechanisms.str.len().gt(0).all()
    named = {item for text in reasons.failure_mechanisms for item in text.split(";")}
    assert named <= set(gate.MECHANISMS.values())
    assert all(
        name in gate.MECHANISMS for candidate in gate.candidates() for name in candidate.checks
    )


def test_holdout_table_reports_both_v1_forms_and_v2() -> None:
    holdout = table(80, "holdout", 4)
    chosen = gate.Candidate("physical_ci", "trend_none", True, False, False)
    frame = gate.holdout_table(chosen, holdout)
    assert len(frame) == 3
    assert {"false_accept", "false_reject", "fa_wilson_high", "fr_cluster_high"} <= set(frame)
    assert frame.conditions.eq(80).all()


def test_constrained_candidate_is_scored_on_the_clipped_estimate() -> None:
    frame = table(40, "train", 6)
    frame["improved"] = False
    frame["improved_clipped"] = True
    chosen = gate.Candidate("physical_constrained", "trend_none", False, False, False)
    assert gate._score(chosen, frame)["true_accept"] == 40


def test_end_to_end_on_synthetic_conditions_and_v1_uses_reference() -> None:
    rows, truth = [], []
    manifest_rows = []
    for index, split in enumerate(("train", "validation", "holdout")):
        row = synthetic_row().copy()
        row["circuit_id"] = f"synthetic_{split}"
        row["parameter_seed"] = derive_seed("synthetic-test", index)
        manifest_rows.append({**row.to_dict(), "split": split, "signal_band": "weak"})
        for name in ("none", "depolarizing"):
            noise = generator.noise_conditions(SMALL)[name]
            repeats, _, truths = generator.simulate_condition(row, noise, SMALL)
            rows.extend(repeats)
            truth.extend(truths)
    truth_frame = pd.DataFrame(truth).merge(
        pd.DataFrame(manifest_rows)[["circuit_id", "split"]], on="circuit_id"
    )
    conditions = gate.build_conditions(
        pd.DataFrame(rows), truth_frame, pd.DataFrame(manifest_rows), SMALL
    )
    assert len(conditions) == 6
    assert conditions.loc[conditions.noise == "none", "improved"].sum() == 0
    chosen, model = (
        gate.select_gate(
            conditions.loc[conditions.split == "train"],
            conditions.loc[conditions.split == "validation"],
        ),
        gate.select_model(truth_frame),
    )
    assert model[0][0] in SMALL["designs"]
    assert chosen[0] in gate.candidates()
    policies = gate.policy_table(
        chosen[0],
        model[0],
        conditions.loc[conditions.split == "holdout"],
        truth_frame.loc[truth_frame.split == "holdout"],
    )
    assert policies.policy.iloc[0] == "always_raw"
    assert policies.mean_gain_vs_raw.iloc[0] == 0.0


def test_analysis_blanks_normalized_error_near_zero() -> None:
    truth = pd.DataFrame(
        {
            "circuit_id": ["a", "b"],
            "noise": ["x", "x"],
            "total_shot_budget": [512, 512],
            "design": ["global_135"] * 2,
            "folding": ["global"] * 2,
            "model": ["linear"] * 2,
            "reference": [0.01, 0.5],
            "raw_mean": [0.0, 0.4],
            "raw_rmse": [0.02, 0.1],
            "fit_rmse": [0.03, 0.05],
        }
    )
    manifest = pd.DataFrame(
        {
            "circuit_id": ["a", "b"],
            "width": [2, 2],
            "depth": [1, 1],
            "signal_band": ["near_zero", "strong"],
            "split": ["train", "train"],
            "ideal_expectation": [0.01, 0.5],
        }
    )
    frame = signal_v2_analysis.prepare(truth, manifest)
    assert np.isnan(frame.fit_normalized_rmse.iloc[0])
    assert frame.fit_normalized_rmse.iloc[1] == pytest.approx(0.1)
    assert frame.outcome.tolist() == ["raw_better", "fit_better"]


def test_slope_check_needs_agreement_across_repeats() -> None:
    assert diagnostics.slope_stable(noisy_rows([0.8, 0.5, 0.2])) is True
    assert diagnostics.slope_stable(noisy_rows([0.5, 0.5, 0.5])) is False


def test_committed_holdout_scores_match_their_conditions() -> None:
    conditions = pd.read_csv(gate.CONDITIONS_PATH)
    holdout = conditions.loc[conditions.split == "holdout"]
    frozen = pd.read_csv(gate.SELECTION_PATH)
    assert frozen.selected.sum() == 1
    assert frozen.columns.tolist().count("validation_balanced_accuracy") == 1
    assert not any("holdout" in column for column in frozen.columns)
    table = pd.read_csv(gate.HOLDOUT_PATH)
    row = table.loc[table.gate == "v1_full_reads_reference"].iloc[0]
    assert row.conditions == len(holdout)
    assert row.false_reject == int((~holdout.v1_full & holdout.improved).sum())
    assert (holdout.loc[holdout.noise == "none", "improved"]).sum() == 0


def test_committed_rejections_are_complete() -> None:
    conditions = pd.read_csv(gate.CONDITIONS_PATH)
    holdout = conditions.loc[conditions.split == "holdout"]
    reasons = pd.read_csv(gate.REJECTIONS_PATH)
    assert len(reasons) == int((~holdout.trend_slope_sign).sum())
    assert reasons.failure_mechanisms.notna().all()


def test_raw_rows_keep_every_repeat() -> None:
    frame = pd.read_csv(generator.RAW_PATH, usecols=["status", "result_label"])
    assert len(frame) == 88200
    assert frame.status.eq("complete").all()
    assert frame.result_label.eq("noisy_simulated").all()
