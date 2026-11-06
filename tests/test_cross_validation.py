import ast
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import cross_validation as cv
from src import independent_reference as ref
from src.applications import qaoa_selection as first_qaoa
from src.mitigation import readout as first_readout
from src.mitigation.zne import linear_extrapolate
from src.observables import expectation_from_counts, expectation_from_probabilities
from src.resource_accounting import split_shots

ROOT = Path(__file__).resolve().parents[1]


def _real_qiskit() -> bool:
    try:
        importlib.metadata.version("qiskit")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_qiskit = pytest.mark.skipif(not _real_qiskit(), reason="needs the real Qiskit package")
COUNTS = {"000": 5, "001": 11, "010": 2, "100": 7, "011": 3, "101": 1, "111": 13}


def test_reference_shares_no_code_with_the_first_path() -> None:
    tree = ast.parse(Path(ref.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "collections", "hashlib", "math", "numpy", "scipy"}


def test_committed_report_passes_every_check_without_a_declared_limitation() -> None:
    report = pd.read_csv(cv.OUTPUT_PATH, keep_default_na=False)
    assert report.loc[report.known_limitation == "", "passed"].astype(str).eq("True").all()
    limited = report.loc[report.known_limitation != ""]
    assert len(limited) == 1
    assert "iterative" in limited.check.iloc[0]
    current = cv.build_report()
    assert list(current.largest_difference) == list(report.largest_difference)


def test_exact_values_agree_at_the_required_tolerance() -> None:
    rows = cv.exact_checks()
    assert all(row["tolerance"] == 1e-12 and row["passed"] for row in rows)


def test_fixed_count_decisions_match_exactly() -> None:
    report = pd.DataFrame(cv.qaoa_checks())
    decisions = report.loc[
        report.check.str.contains("selected candidate|ranking|recall|accounting")
    ]
    assert len(decisions) == 12
    assert decisions.passed.all()
    assert (decisions.tolerance == 0.0).all()


def test_required_mutations_are_detected() -> None:
    table = cv.mutation_table()
    assert (table.convention_sensitive == table.detected).all()
    reversed_rows = table.loc[table.mutation == "reverse bit order"]
    assert reversed_rows.detected.sum() >= 4
    assert table.loc[table.mutation == "transpose assignment matrix", "detected"].all()


def test_symmetric_cells_cannot_detect_a_bit_reversal() -> None:
    """Record why a passing symmetric cell says nothing about bit ordering."""

    table = cv.mutation_table().set_index("target")
    for target in ("bell XX expectation", "ring4 energies from fixed counts"):
        assert not table.loc[target, "detected"]
    for target in ("sparse6 energies from fixed counts",):
        assert not table.loc[target, "detected"]


def test_reference_nnls_certifies_a_case_where_scipy_alone_is_not_optimal() -> None:
    frame = cv._qaoa_rows()
    calibration = frame.loc[
        (frame.graph_id == "sparse6")
        & (frame.method == "readout")
        & (frame.total_selection_budget == 4096)
        & (frame.repeat == 47)
    ]
    records = json.loads(calibration.calibration_data.iloc[0])
    row = calibration.loc[calibration.candidate_id == "p1_b04_g05"].iloc[0]
    matrix = ref.assignment_matrix(records, 6)
    observed = ref.probabilities_from_counts(json.loads(row.raw_counts), 6)
    solution = ref.solve_nnls(matrix, observed)
    assert max(ref._optimality_gaps(matrix, observed, solution)) <= ref.NNLS_TOLERANCE


def test_parity_matches_the_first_path_on_asymmetric_counts() -> None:
    for pauli in ("ZII", "IZI", "IIZ", "ZZI", "ZIZ", "XZI", "ZZZ"):
        assert ref.expectation_from_counts(COUNTS, pauli) == expectation_from_counts(COUNTS, pauli)
    assert ref.expectation_from_counts(COUNTS, "ZII", reverse_bits=True) != expectation_from_counts(
        COUNTS, "ZII"
    )


def test_probability_expectation_matches_the_first_path() -> None:
    probabilities = ref.probabilities_from_counts(COUNTS, 3)
    for pauli in ("ZII", "IZI", "ZZI"):
        assert ref.expectation_from_probabilities(probabilities, pauli) == pytest.approx(
            expectation_from_probabilities(probabilities, pauli), abs=1e-15
        )


def test_cut_values_graphs_and_optimum_match_the_first_path() -> None:
    graphs = ref.benchmark_graphs()
    for graph in first_qaoa.benchmark_graphs():
        width, edges = graphs[graph.graph_id]
        assert width == graph.width
        assert edges == graph.edges
        for index in range(2**width):
            assert ref.cut_of_index(index, edges) == first_qaoa.cut_value(graph, index)
        solution = first_qaoa.exact_maxcut(graph)
        optimum, states = ref.exact_maxcut(width, edges)
        assert optimum == solution.optimum
        assert len(states) > 1


def test_selection_metrics_match_the_first_path_including_ties() -> None:
    ids = ("a", "b", "c", "d", "e")
    exact = (3.0, 3.0, 2.5, 2.0, 1.0)
    for estimates in (
        (2.9, 2.8, 2.7, 1.0, 0.5),
        (2.0, 2.5, 3.0, 2.9, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (2.0, 2.0, 2.5, 2.5, 0.0),
    ):
        first = first_qaoa.evaluate_candidate_selection(ids, exact, estimates, top_k=2)
        second = ref.selection_metrics(ids, exact, estimates, top_k=2)
        assert second["selected"] == first.selected_candidate_id
        assert second["noisy_winners"] == first.noisy_winner_ids
        assert second["exact_winners"] == first.exact_winner_ids
        assert second["decision_error"] == first.decision_error
        assert second["simple_regret"] == first.simple_regret
        assert second["top_k_recall"] == first.top_k_recall
        assert second["energy_rmse"] == pytest.approx(first.energy_rmse, abs=1e-15)


def test_shot_splitting_and_extrapolation_match_the_first_path() -> None:
    for total, parts in ((2048, 5), (409, 64), (1639, 5), (7, 3)):
        assert ref.split_evenly(total, parts) == tuple(split_shots(total, parts))
    estimates = (0.9, 0.7, 0.45)
    assert ref.linear_intercept((1.0, 3.0, 5.0), estimates) == pytest.approx(
        linear_extrapolate((1, 3, 5), estimates)[0], abs=1e-14
    )
    allocation = first_qaoa.candidate_shot_allocation("readout", 4096, 5, width=6)
    assert allocation["calibration_shots"] == 2048 // 5
    assert tuple(allocation["candidate_target_shots"]) == ref.split_evenly(2048 - 409, 5)


def test_readout_correction_matches_the_first_path_on_fixed_counts() -> None:
    rows = cv._readout_rows()
    pauli = str(rows.observable_q0_first.iloc[0])
    for row in rows.head(10).itertuples():
        payload = json.loads(row.raw_payload_json)
        matrix = ref.assignment_matrix(json.loads(payload["calibration_counts_json"]), 2)
        counts = payload["raw_counts_json"]
        observed = first_readout.counts_probability_vector(counts, 2)
        assert np.array_equal(observed, ref.probabilities_from_counts(json.loads(counts), 2))
        first, condition, status = first_readout.constrained_readout_correction(matrix, observed)
        assert status == "ok"
        assert condition == pytest.approx(ref.infinity_condition_number(matrix), rel=1e-10)
        second = ref.correct_readout(matrix, observed)
        assert second == pytest.approx(first, abs=1e-12)
        assert ref.expectation_from_probabilities(second, pauli) == pytest.approx(
            expectation_from_probabilities(first, pauli), abs=1e-12
        )


@needs_qiskit
def test_exact_expectations_match_a_live_statevector() -> None:
    from src.circuits import bell_circuit, ghz_circuit, ghz_observables
    from src.hardware_efficient_circuits import hardware_efficient_circuit
    from src.observables import exact_expectation

    for pauli in ("XX", "YY", "ZZ", "XI", "ZI"):
        assert ref.pauli_expectation(ref.ghz_state(2), pauli) == pytest.approx(
            exact_expectation(bell_circuit(), pauli), abs=1e-12
        )
    for width in (2, 4, 6):
        state = ref.ghz_state_from_gates(width)
        for pauli in ghz_observables(width):
            assert ref.pauli_expectation(state, pauli) == pytest.approx(
                exact_expectation(ghz_circuit(width), pauli), abs=1e-12
            )
        for depth in (1, 3, 6):
            state = ref.hardware_efficient_state(width, depth)
            circuit = hardware_efficient_circuit(width, depth)
            for pauli in (ref.observable_for_hardware_efficient(width), "Z" + "I" * (width - 1)):
                assert ref.pauli_expectation(state, pauli) == pytest.approx(
                    exact_expectation(circuit, pauli), abs=1e-12
                )


@needs_qiskit
def test_qaoa_distribution_matches_a_live_statevector() -> None:
    graphs = {graph.graph_id: graph for graph in first_qaoa.benchmark_graphs()}
    frame = cv._qaoa_rows().drop_duplicates(["graph_id", "candidate_id"])
    for row in frame.itertuples():
        graph = graphs[row.graph_id]
        beta = json.loads(row.beta_parameters)[0]
        gamma = json.loads(row.gamma_parameters)[0]
        circuit = first_qaoa.qaoa_circuit(graph, betas=(beta,), gammas=(gamma,))
        first = first_qaoa.noiseless_probabilities(circuit)
        second = np.abs(ref.qaoa_state(graph.width, graph.edges, beta, gamma)) ** 2
        assert second == pytest.approx(first, abs=1e-12)
        assert ref.ideal_energy(graph.width, graph.edges, beta, gamma) == pytest.approx(
            first_qaoa.expected_cut_from_probabilities(graph, first), abs=1e-12
        )
