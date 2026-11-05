import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qiskit.quantum_info import Statevector

from src.applications.qaoa_selection import (
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_POOL_PATH,
    CertificationCheckpoint,
    MaxCutGraph,
    benchmark_graphs,
    build_candidate_manifest,
    candidate_pool_frame,
    candidate_shot_allocation,
    certify_candidate_winner,
    certify_sampled_cuts,
    cut_value,
    evaluate_candidate_selection,
    exact_maxcut,
    expected_cut_from_counts,
    expected_cut_from_probabilities,
    generate_candidate_pool,
    maxcut_cost_operator,
    nested_shot_checkpoints,
    noiseless_probabilities,
    qaoa_circuit,
    select_benchmark_candidates,
    selection_budget_split,
    wilson_interval,
)


def test_ring_four_exact_optimum_preserves_complement_pair() -> None:
    graph = MaxCutGraph(
        "ring4", 4, ((0, 1), (0, 3), (1, 2), (2, 3)), "ring", 17
    )

    solution = exact_maxcut(graph)

    assert solution.optimum == 4
    assert solution.optimal_states == (5, 10)
    assert solution.optimal_bitstrings == ("0101", "1010")


@pytest.mark.parametrize(
    ("width", "edges", "match"),
    [
        (1, (), "at least two"),
        (3, ((0, 0),), "self-loops"),
        (3, ((0, 3),), "range"),
        (3, ((1, 0),), "canonical"),
        (3, ((0, 1), (0, 1)), "duplicate"),
        (3, ((0, 2), (0, 1)), "sorted"),
    ],
)
def test_graph_validation_rejects_invalid_edges(
    width: int, edges: tuple[tuple[int, int], ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MaxCutGraph("invalid", width, edges, "test", 1)


def test_cut_value_obeys_qiskit_bitstring_order() -> None:
    graph = MaxCutGraph("ordering", 3, ((0, 2),), "test", 1)

    assert cut_value(graph, "100") == 1  # q2=1, q0=0
    assert cut_value(graph, "001") == 1  # q2=0, q0=1
    assert cut_value(graph, "010") == 0
    assert cut_value(graph, 4) == cut_value(graph, "100")


def test_benchmark_graphs_are_deterministic_and_cover_required_widths() -> None:
    first = benchmark_graphs()
    second = benchmark_graphs()

    assert first == second
    assert {graph.width for graph in first} == {4, 6, 8}
    assert {graph.family for graph in first} == {"ring", "sparse", "medium_dense"}
    assert all(graph.edges == tuple(sorted(graph.edges)) for graph in first)


def test_complement_reduced_solver_matches_full_enumeration() -> None:
    for graph in benchmark_graphs():
        solution = exact_maxcut(graph)
        full_values = [cut_value(graph, state) for state in range(2**graph.width)]
        expected_states = tuple(
            state for state, value in enumerate(full_values) if value == max(full_values)
        )

        assert solution.optimum == max(full_values)
        assert solution.optimal_states == expected_states


def test_qaoa_circuit_regenerates_identically() -> None:
    graph = benchmark_graphs()[0]

    first = qaoa_circuit(graph, betas=(0.2,), gammas=(0.7,))
    second = qaoa_circuit(graph, betas=(0.2,), gammas=(0.7,))

    assert first == second
    assert first.num_qubits == graph.width
    assert first.name == "qaoa_ring4_p1"


def test_statevector_probability_expectation_matches_cost_operator() -> None:
    graph = benchmark_graphs()[1]
    circuit = qaoa_circuit(graph, betas=(0.31,), gammas=(0.82,))
    probabilities = noiseless_probabilities(circuit)

    distribution_value = expected_cut_from_probabilities(graph, probabilities)
    operator_value = Statevector.from_instruction(circuit).expectation_value(
        maxcut_cost_operator(graph)
    )

    assert probabilities.sum() == pytest.approx(1.0, abs=1e-12)
    assert distribution_value == pytest.approx(float(operator_value.real), abs=1e-12)


def test_expected_cut_from_counts_charges_one_complete_cut_per_shot() -> None:
    graph = benchmark_graphs()[0]
    counts = {"0101": 3, "0000": 1}

    assert expected_cut_from_counts(graph, counts) == pytest.approx(3.0)


def test_probability_validation_rejects_non_distribution() -> None:
    graph = benchmark_graphs()[0]
    invalid = np.zeros(2**graph.width)

    with pytest.raises(ValueError, match="sum to one"):
        expected_cut_from_probabilities(graph, invalid)


def test_qaoa_parameter_layers_must_match() -> None:
    graph = benchmark_graphs()[0]

    with pytest.raises(ValueError, match="same positive length"):
        qaoa_circuit(graph, betas=(0.1,), gammas=(0.2, 0.3))


def test_candidate_pool_and_ranks_regenerate_identically() -> None:
    graph = benchmark_graphs()[0]
    first = candidate_pool_frame((graph,))
    second = candidate_pool_frame((graph,))

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 13 * 24
    assert first.ideal_rank.tolist() == list(range(1, len(first) + 1))
    assert np.isfinite(first.exact_ideal_energy).all()
    assert first.ideal_gap_to_best.ge(0.0).all()
    assert first.loc[0, "ideal_gap_to_best"] == pytest.approx(0.0, abs=1e-12)


def test_candidate_generation_accepts_no_noisy_outcome_input() -> None:
    assert set(inspect.signature(generate_candidate_pool).parameters) == {"graph"}


def test_selected_candidates_are_unique_and_cover_decision_regimes() -> None:
    selected = {
        graph.graph_id: select_benchmark_candidates(graph) for graph in benchmark_graphs()
    }

    assert all(len(candidates) == 5 for candidates in selected.values())
    for candidates in selected.values():
        parameters = {(candidate.beta, candidate.gamma) for candidate in candidates}
        assert len(parameters) == len(candidates)
        assert candidates[0].candidate_role == "incumbent"
    regimes = {
        candidate.decision_regime
        for candidates in selected.values()
        for candidate in candidates
    }
    assert regimes >= {"near_tie", "clear", "control", "reference"}
    assert all(
        candidate.candidate_role != "near_tie_challenger"
        for candidate in selected["ring4"]
    )
    assert all(
        any(candidate.candidate_role == "near_tie_challenger" for candidate in selected[key])
        for key in ("sparse6", "medium_dense8")
    )


def test_manifest_is_complete_deterministic_and_has_unique_seeds() -> None:
    first = build_candidate_manifest(repeats=2)
    second = build_candidate_manifest(repeats=2)
    expected = 3 * 5 * 3 * 2 * 2

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected
    assert first.seed.is_unique
    assert set(first.method) == {"raw", "readout", "zne"}
    assert set(first.total_selection_budget) == {4096, 16384}
    assert set(first.status) == {"pending"}
    assert first.groupby("graph_id").candidate_id.nunique().eq(5).all()
    assert {
        "graph_id",
        "width",
        "graph_family",
        "edge_count",
        "qaoa_depth",
        "candidate_id",
        "candidate_role",
        "beta_parameters",
        "gamma_parameters",
        "exact_classical_maxcut",
        "exact_ideal_energy",
        "ideal_rank",
        "ideal_gap_to_best",
        "decision_regime",
        "method",
        "total_selection_budget",
        "repeat",
        "seed",
        "status",
    } <= set(first.columns)


def test_manifest_references_match_exact_graph_and_candidate_values() -> None:
    manifest = build_candidate_manifest(repeats=1)
    graphs = {graph.graph_id: graph for graph in benchmark_graphs()}
    for graph_id, rows in manifest.groupby("graph_id"):
        graph = graphs[graph_id]
        assert rows.exact_classical_maxcut.eq(exact_maxcut(graph).optimum).all()
        assert np.isfinite(rows.exact_ideal_energy).all()
        assert rows.ideal_gap_to_best.ge(0.0).all()
        assert rows.qaoa_depth.eq(1).all()


def test_selection_budget_is_frozen_at_half_selection_half_final() -> None:
    assert selection_budget_split(4096) == (2048, 2048)
    assert selection_budget_split(16384) == (8192, 8192)
    with pytest.raises(ValueError, match="positive even"):
        selection_budget_split(4095)


def test_candidate_shot_allocations_charge_every_selection_shot() -> None:
    raw = candidate_shot_allocation("raw", 4096, 5, width=4)
    readout = candidate_shot_allocation("readout", 4096, 5, width=4)
    zne = candidate_shot_allocation("zne", 4096, 5, width=4)

    assert sum(raw["candidate_target_shots"]) == 2048
    assert sum(readout["candidate_target_shots"]) + readout["calibration_shots"] == 2048
    assert sum(sum(shots) for shots in zne["factor_shots"]) == 2048
    assert all(sum(factors) == total for factors, total in zip(
        zne["factor_shots"], zne["candidate_target_shots"], strict=True
    ))


def test_sampled_cut_certification_uses_exact_classical_optimum() -> None:
    graph = benchmark_graphs()[0]
    result = certify_sampled_cuts(graph, {"0101": 3, "0000": 1}, alpha=0.95)

    assert result.exact_optimum == 4
    assert result.best_sampled_cut == 4
    assert result.reaches_exact_optimum
    assert result.at_least_alpha_fraction == pytest.approx(0.75)


def test_sampled_cut_certification_does_not_claim_suboptimal_sample_is_optimal() -> None:
    graph = benchmark_graphs()[0]
    result = certify_sampled_cuts(graph, {"0001": 5}, alpha=0.95)

    assert result.best_sampled_cut == 2
    assert not result.reaches_exact_optimum
    assert result.at_least_alpha_fraction == 0.0


def test_selection_metrics_exact_and_wrong_winners() -> None:
    exact = [3.0, 2.9, 1.0]
    correct = evaluate_candidate_selection(
        ["a", "b", "c"], exact, [2.8, 2.7, 1.1], top_k=2
    )
    wrong = evaluate_candidate_selection(
        ["a", "b", "c"], exact, [2.7, 2.8, 1.1], top_k=2
    )
    large_regret = evaluate_candidate_selection(
        ["a", "b", "c"], exact, [1.0, 1.1, 2.0], top_k=2
    )

    assert correct.set_valued_correct and correct.decision_error == 0
    assert wrong.selected_candidate_id == "b" and wrong.decision_error == 1
    assert wrong.top_k_recall == 1.0
    assert wrong.simple_regret == pytest.approx(0.1)
    assert large_regret.simple_regret > wrong.simple_regret


def test_degenerate_exact_winners_use_set_valued_correctness() -> None:
    result = evaluate_candidate_selection(
        ["a", "b", "c"], [3.0, 3.0, 1.0], [2.8, 2.9, 1.1], top_k=2
    )

    assert result.exact_winner_ids == ("a", "b")
    assert result.set_valued_correct
    assert result.unique_winner_correct is None


def test_noisy_tie_uses_visible_deterministic_candidate_id_rule() -> None:
    result = evaluate_candidate_selection(
        ["b", "a", "c"], [3.0, 2.0, 1.0], [2.5, 2.5, 1.0], top_k=2
    )

    assert result.noisy_winner_ids == ("a", "b")
    assert result.selected_candidate_id == "a"


def _checkpoint(total: int, a: float, b: float, error: float) -> CertificationCheckpoint:
    return CertificationCheckpoint(
        total,
        {"a": a, "b": b},
        {"a": error, "b": error},
        {"a": total // 2, "b": total // 2},
    )


def test_certification_preserves_unresolved_and_certifies_separation() -> None:
    unresolved = certify_candidate_winner(
        [_checkpoint(100, 1.0, 0.95, 0.2), _checkpoint(200, 1.0, 0.94, 0.12)]
    )
    certified = certify_candidate_winner(
        [_checkpoint(100, 1.0, 0.9, 0.2), _checkpoint(200, 1.5, 0.5, 0.05)]
    )

    assert unresolved.status == "unresolved"
    assert unresolved.shots_to_certified_decision is None
    assert unresolved.maximum_shots == 200
    assert certified.status == "certified"
    assert certified.winner_id == "a"
    assert certified.shots_to_certified_decision == 200


def test_certification_corrects_for_candidates_and_sequential_looks() -> None:
    result = certify_candidate_winner(
        [_checkpoint(100, 1.0, 0.9, 0.2), _checkpoint(200, 1.5, 0.5, 0.05)]
    )

    assert result.simultaneous_confidence == 0.95
    assert result.interval_confidence == pytest.approx(1.0 - 0.05 / 4)
    assert result.critical_value > 1.96


def test_nested_checkpoints_are_monotonic_and_reject_non_nested_input() -> None:
    assert nested_shot_checkpoints(2048) == (512, 1024, 1536, 2048)
    invalid = [
        _checkpoint(100, 1.0, 0.9, 0.2),
        CertificationCheckpoint(
            200,
            {"a": 1.2, "b": 0.8},
            {"a": 0.1, "b": 0.1},
            {"a": 50, "b": 150},
        ),
    ]
    with pytest.raises(ValueError, match="nested monotonically"):
        certify_candidate_winner(invalid)


def test_wilson_interval_contains_empirical_rate() -> None:
    low, high = wilson_interval(7, 10, confidence=0.95)

    assert low <= 0.7 <= high


def test_committed_candidate_artifacts_match_deterministic_generation() -> None:
    committed_pool = pd.read_csv(CANDIDATE_POOL_PATH)
    committed_manifest = pd.read_csv(CANDIDATE_MANIFEST_PATH)

    pd.testing.assert_frame_equal(committed_pool, candidate_pool_frame(), rtol=1e-12)
    pd.testing.assert_frame_equal(
        committed_manifest,
        build_candidate_manifest(),
        rtol=1e-12,
    )


def test_frozen_config_records_budget_and_certification_boundaries() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "experiments/configs/qaoa_selection_v1.yaml").read_text(encoding="utf-8")
    )

    assert config["qaoa_depth"] == 1
    assert not config["enable_qaoa_depth_2"]
    assert config["selection_budget_fraction"] == 0.5
    assert config["final_evaluation_budget_fraction"] == 0.5
    assert config["top_k"] == 2
    assert config["certification_confidence"] == 0.95
    assert config["certification_checkpoint_fractions"] == [0.25, 0.5, 0.75, 1.0]
    assert "Bonferroni" in config["certification_multiplicity"]
    assert config["success_alpha"] == 0.95
    assert config["near_tie_unavailable_graphs"] == ["ring4"]
    assert config["noise_model"] == "mixed_medium"
    assert "classical exact solver" in config["certification_boundary"]
