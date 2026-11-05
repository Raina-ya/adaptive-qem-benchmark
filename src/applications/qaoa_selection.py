"""Exact MaxCut references and deterministic QAOA circuit foundations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.config import derive_seed

Edge = tuple[int, int]


@dataclass(frozen=True)
class MaxCutGraph:
    """Immutable simple graph with canonical undirected edges."""

    graph_id: str
    width: int
    edges: tuple[Edge, ...]
    family: str
    seed: int

    def __post_init__(self) -> None:
        if self.width < 2:
            raise ValueError("MaxCut graphs require at least two vertices")
        if not self.graph_id or not self.family:
            raise ValueError("Graph id and family must be nonempty")
        if not isinstance(self.edges, tuple) or not self.edges:
            raise ValueError("Edges must be a nonempty tuple")
        for edge in self.edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise ValueError("Every edge must be a pair")
            left, right = edge
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError("Vertex indices must be integers")
            if left == right:
                raise ValueError("MaxCut graphs cannot contain self-loops")
            if not 0 <= left < self.width or not 0 <= right < self.width:
                raise ValueError("Vertex index is outside graph range")
            if left > right:
                raise ValueError("Edges must use canonical increasing endpoints")
        if len(set(self.edges)) != len(self.edges):
            raise ValueError("Graph contains a duplicate edge")
        if self.edges != tuple(sorted(self.edges)):
            raise ValueError("Edges must be sorted canonically")


@dataclass(frozen=True)
class ExactMaxCutSolution:
    """Exact classical optimum and every degenerate optimal basis state."""

    optimum: int
    optimal_states: tuple[int, ...]
    width: int

    @property
    def optimal_bitstrings(self) -> tuple[str, ...]:
        """Return Qiskit-style q[n-1]...q[0] strings."""

        return tuple(format(state, f"0{self.width}b") for state in self.optimal_states)


def _basis_state(graph: MaxCutGraph, assignment: int | str) -> int:
    if isinstance(assignment, str):
        if len(assignment) != graph.width or set(assignment) - {"0", "1"}:
            raise ValueError("Bitstrings must contain one binary digit per vertex")
        return int(assignment, 2)
    if not isinstance(assignment, int) or not 0 <= assignment < 2**graph.width:
        raise ValueError("Basis-state integer is outside graph range")
    return assignment


def cut_value(graph: MaxCutGraph, assignment: int | str) -> int:
    """Return the complete cut value of one computational-basis sample."""

    state = _basis_state(graph, assignment)
    return sum(((state >> left) & 1) != ((state >> right) & 1) for left, right in graph.edges)


def exact_maxcut(graph: MaxCutGraph) -> ExactMaxCutSolution:
    """Solve MaxCut exactly while enumerating one representative per complement pair."""

    best = -1
    representatives: list[int] = []
    for state in range(2 ** (graph.width - 1)):
        value = cut_value(graph, state)
        if value > best:
            best = value
            representatives = [state]
        elif value == best:
            representatives.append(state)
    complement_mask = 2**graph.width - 1
    optimal_states = tuple(
        sorted(
            {
                state
                for representative in representatives
                for state in (representative, representative ^ complement_mask)
            }
        )
    )
    return ExactMaxCutSolution(best, optimal_states, graph.width)


def _seeded_edges(
    width: int,
    *,
    family_code: int,
    density: float,
    mandatory: set[Edge],
) -> tuple[Edge, ...]:
    edges = set(mandatory)
    for left in range(width):
        for right in range(left + 1, width):
            score = derive_seed("maxcut-graph-edge", width, family_code, left, right) / 2**32
            if score < density:
                edges.add((left, right))
    return tuple(sorted(edges))


def benchmark_graphs() -> tuple[MaxCutGraph, ...]:
    """Return the frozen deterministic width-4/6/8 application instances."""

    ring4_edges = ((0, 1), (0, 3), (1, 2), (2, 3))
    sparse6_edges = _seeded_edges(
        6,
        family_code=1,
        density=0.12,
        mandatory={(vertex, vertex + 1) for vertex in range(5)},
    )
    dense8_edges = _seeded_edges(
        8,
        family_code=2,
        density=0.50,
        mandatory={*( (vertex, vertex + 1) for vertex in range(7) ), (0, 7)},
    )
    return (
        MaxCutGraph("ring4", 4, ring4_edges, "ring", derive_seed("maxcut-graph", 4, 0)),
        MaxCutGraph(
            "sparse6",
            6,
            sparse6_edges,
            "sparse",
            derive_seed("maxcut-graph", 6, 1),
        ),
        MaxCutGraph(
            "medium_dense8",
            8,
            dense8_edges,
            "medium_dense",
            derive_seed("maxcut-graph", 8, 2),
        ),
    )


def qaoa_circuit(
    graph: MaxCutGraph,
    *,
    betas: Sequence[float],
    gammas: Sequence[float],
) -> QuantumCircuit:
    """Build a deterministic MaxCut QAOA circuit for one or more layers."""

    beta_values = tuple(float(value) for value in betas)
    gamma_values = tuple(float(value) for value in gammas)
    if not beta_values or len(beta_values) != len(gamma_values):
        raise ValueError("Beta and gamma sequences must have the same positive length")
    if not all(math.isfinite(value) for value in (*beta_values, *gamma_values)):
        raise ValueError("QAOA parameters must be finite")

    circuit = QuantumCircuit(graph.width, name=f"qaoa_{graph.graph_id}_p{len(beta_values)}")
    circuit.h(range(graph.width))
    for beta, gamma in zip(beta_values, gamma_values, strict=True):
        for left, right in graph.edges:
            circuit.rzz(-gamma, left, right)
        for qubit in range(graph.width):
            circuit.rx(2.0 * beta, qubit)
    return circuit


def noiseless_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Return exact statevector probabilities indexed by Qiskit basis integer."""

    probabilities = Statevector.from_instruction(circuit).probabilities()
    return np.asarray(probabilities, dtype=float)


def expected_cut_from_probabilities(
    graph: MaxCutGraph, probabilities: np.ndarray
) -> float:
    """Evaluate ideal expected cut value from a complete basis distribution."""

    values = np.asarray(probabilities, dtype=float)
    if values.shape != (2**graph.width,):
        raise ValueError("Probability vector dimension does not match graph width")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Probabilities must be finite and nonnegative")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Probabilities must sum to one")
    cut_values = np.fromiter(
        (cut_value(graph, state) for state in range(values.size)),
        dtype=float,
        count=values.size,
    )
    return float(cut_values @ values)


def expected_cut_from_counts(graph: MaxCutGraph, counts: Mapping[str, int]) -> float:
    """Estimate expected cut value from complete bitstring samples."""

    if not counts:
        raise ValueError("Counts must contain at least one shot")
    total_shots = 0
    weighted_cut = 0
    for bitstring, count in counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError("Count values must be nonnegative integers")
        weighted_cut += cut_value(graph, bitstring) * count
        total_shots += count
    if total_shots <= 0:
        raise ValueError("Counts must contain at least one shot")
    return weighted_cut / total_shots


def maxcut_cost_operator(graph: MaxCutGraph) -> SparsePauliOp:
    """Return C=sum_(i,j)(I-Z_i Z_j)/2 in Qiskit's Pauli order."""

    terms: list[tuple[str, complex]] = [("I" * graph.width, len(graph.edges) / 2.0)]
    for left, right in graph.edges:
        label = ["I"] * graph.width
        label[graph.width - 1 - left] = "Z"
        label[graph.width - 1 - right] = "Z"
        terms.append(("".join(label), -0.5))
    return SparsePauliOp.from_list(terms).simplify()
