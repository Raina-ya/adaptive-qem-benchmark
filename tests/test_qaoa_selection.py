import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from src.applications.qaoa_selection import (
    MaxCutGraph,
    benchmark_graphs,
    cut_value,
    exact_maxcut,
    expected_cut_from_counts,
    expected_cut_from_probabilities,
    maxcut_cost_operator,
    noiseless_probabilities,
    qaoa_circuit,
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
