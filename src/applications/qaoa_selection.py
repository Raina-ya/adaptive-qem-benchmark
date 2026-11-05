"""Exact MaxCut references and deterministic QAOA circuit foundations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector

from src.config import derive_seed
from src.mitigation.folding import fold_global
from src.mitigation.readout import (
    constrained_readout_correction,
    counts_probability_vector,
    estimate_assignment_matrix,
)
from src.mitigation.zne import linear_extrapolate
from src.noise_models import NoiseSpec, aer_simulator
from src.resource_accounting import readout_mitigation_ledger, split_shots

Edge = tuple[int, int]

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_POOL_PATH = ROOT / "experiments/manifests/qaoa_candidate_pool_v1.csv"
CANDIDATE_MANIFEST_PATH = ROOT / "experiments/manifests/qaoa_selection_v1_manifest.csv"
BETA_GRID = tuple(math.pi * index / 24.0 for index in range(13))
GAMMA_GRID = tuple(math.pi * index / 12.0 for index in range(24))
METHODS = ("raw", "readout", "zne")
SELECTION_BUDGETS = (4096, 16384)
FULL_REPEATS = 50
NEAR_TIE_RELATIVE_GAP = 0.02
CLEAR_RELATIVE_GAP = 0.10
ENERGY_TOLERANCE = 1e-12
SUCCESS_ALPHA = 0.95
ZNE_FACTORS = (1, 3, 5)


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


@dataclass(frozen=True)
class QAOACandidate:
    """One deterministic p=1 parameter candidate and its ideal-only rank."""

    graph_id: str
    candidate_id: str
    beta: float
    gamma: float
    exact_ideal_energy: float
    ideal_rank: int
    ideal_gap_to_best: float
    candidate_role: str = "pool"
    decision_regime: str = "unassigned"


@dataclass(frozen=True)
class SampledCutCertification:
    """Classical verification of sampled cuts against the exact small-n optimum."""

    exact_optimum: int
    best_sampled_cut: int
    reaches_exact_optimum: bool
    alpha: float
    at_least_alpha_fraction: float


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


def expected_cut_from_corrected_probabilities(
    graph: MaxCutGraph, probabilities: np.ndarray
) -> float:
    """Evaluate MaxCut energy from a corrected computational-basis distribution."""

    return expected_cut_from_probabilities(graph, probabilities)


def candidate_shot_allocation(
    method: str, total_selection_budget: int, candidate_count: int, *, width: int
) -> dict[str, object]:
    """Allocate one graph-method-repeat budget without hidden executions."""

    selection_shots, final_shots = selection_budget_split(total_selection_budget)
    if candidate_count < 1:
        raise ValueError("Candidate count must be positive")
    if method == "raw":
        candidate_shots = split_shots(selection_shots, candidate_count)
        return {
            "candidate_target_shots": candidate_shots,
            "calibration_shots": 0,
            "factor_shots": tuple(() for _ in candidate_shots),
            "final_evaluation_shots": final_shots,
        }
    if method == "readout":
        ledger = readout_mitigation_ledger(selection_shots, width)
        candidate_shots = split_shots(ledger.target_shots, candidate_count)
        return {
            "candidate_target_shots": candidate_shots,
            "calibration_shots": ledger.calibration_shots,
            "factor_shots": tuple(() for _ in candidate_shots),
            "final_evaluation_shots": final_shots,
        }
    if method == "zne":
        candidate_totals = split_shots(selection_shots, candidate_count)
        factor_shots = tuple(split_shots(shots, len(ZNE_FACTORS)) for shots in candidate_totals)
        return {
            "candidate_target_shots": candidate_totals,
            "calibration_shots": 0,
            "factor_shots": factor_shots,
            "final_evaluation_shots": final_shots,
        }
    raise ValueError(f"Unsupported QAOA selection method: {method}")


def _sample_complete_bitstrings(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed: int,
    noise: NoiseSpec,
) -> dict[str, int]:
    """Sample complete cuts; one execution supplies every graph-edge contribution."""

    measured = circuit.copy()
    measured.measure_all()
    simulator = aer_simulator(noise)
    compiled = transpile(measured, simulator, optimization_level=0, seed_transpiler=seed)
    counts = simulator.run(compiled, shots=shots, seed_simulator=seed).result().get_counts()
    return {str(key).replace(" ", ""): int(value) for key, value in sorted(counts.items())}


def _counts_json(counts: Mapping[str, int]) -> str:
    return json.dumps(dict(sorted(counts.items())), sort_keys=True, separators=(",", ":"))


def _energy_from_vector(graph: MaxCutGraph, probabilities: np.ndarray) -> float:
    return expected_cut_from_corrected_probabilities(graph, probabilities)


def run_candidate_selection_group(
    graph: MaxCutGraph,
    candidates: Sequence[QAOACandidate],
    *,
    method: str,
    total_selection_budget: int,
    repeat: int,
    manifest_seeds: Sequence[int],
    noise: NoiseSpec,
) -> list[dict[str, object]]:
    """Run one fixed-budget graph-method-repeat selection group."""

    if len(candidates) != len(manifest_seeds):
        raise ValueError("Every candidate must have one frozen manifest seed")
    allocation = candidate_shot_allocation(
        method, total_selection_budget, len(candidates), width=graph.width
    )
    target_shots = allocation["candidate_target_shots"]
    factor_shots = allocation["factor_shots"]
    calibration_shots = int(allocation["calibration_shots"])
    final_shots = int(allocation["final_evaluation_shots"])
    calibration_counts = ""
    assignment_json = ""
    corrected_matrix: np.ndarray | None = None
    condition_number: float | None = None
    calibration_status = "ok"
    if method == "readout":
        corrected_matrix, _, calibration_counts = estimate_assignment_matrix(
            graph.width,
            calibration_shots,
            noise,
            seed=derive_seed(
                "qaoa-readout-calibration", graph.seed, total_selection_budget, repeat
            ),
        )
        assignment_json = json.dumps(corrected_matrix.tolist(), separators=(",", ":"))

    rows: list[dict[str, object]] = []
    for candidate_index, (candidate, manifest_seed) in enumerate(
        zip(candidates, manifest_seeds, strict=True)
    ):
        paired_seed = derive_seed(
            "qaoa-selection-paired-target",
            graph.seed,
            total_selection_budget,
            repeat,
            candidate_index,
        )
        candidate_executions = 1 if method in {"raw", "readout"} else len(ZNE_FACTORS)
        row: dict[str, object] = {
            "graph_id": graph.graph_id,
            "width": graph.width,
            "graph_family": graph.family,
            "edge_count": len(graph.edges),
            "candidate_id": candidate.candidate_id,
            "candidate_role": candidate.candidate_role,
            "beta_parameters": f"[{candidate.beta:.17g}]",
            "gamma_parameters": f"[{candidate.gamma:.17g}]",
            "exact_classical_maxcut": exact_maxcut(graph).optimum,
            "exact_ideal_energy": candidate.exact_ideal_energy,
            "ideal_rank": candidate.ideal_rank,
            "ideal_gap_to_best": candidate.ideal_gap_to_best,
            "decision_regime": candidate.decision_regime,
            "method": method,
            "noise": noise.name,
            "total_selection_budget": total_selection_budget,
            "selection_shots": total_selection_budget - final_shots,
            "final_evaluation_shots": final_shots,
            "candidate_target_shots": int(target_shots[candidate_index]),
            "calibration_shots": calibration_shots,
            "factor_level_shots": json.dumps(factor_shots[candidate_index]),
            "repeat": repeat,
            "seed": int(manifest_seed),
            "target_seed": paired_seed,
            "raw_counts": "",
            "factor_level_counts": "",
            "calibration_data": calibration_counts if candidate_index == 0 else "",
            "assignment_matrix": assignment_json if candidate_index == 0 else "",
            "raw_energy": math.nan,
            "corrected_or_extrapolated_estimate": math.nan,
            "condition_number": math.nan,
            "candidate_execution_count": candidate_executions,
            "selection_execution_count": (
                len(candidates) + (2**graph.width if method == "readout" else 0)
                if method != "zne"
                else len(candidates) * len(ZNE_FACTORS)
            ),
            "wall_seconds": math.nan,
            "result_label": "noisy_simulated",
            "status": "pending",
            "failure_reason": "",
        }
        try:
            circuit = qaoa_circuit(graph, betas=(candidate.beta,), gammas=(candidate.gamma,))
            if method in {"raw", "readout"}:
                counts = _sample_complete_bitstrings(
                    circuit,
                    shots=int(target_shots[candidate_index]),
                    seed=paired_seed,
                    noise=noise,
                )
                row["raw_counts"] = _counts_json(counts)
                row["raw_energy"] = expected_cut_from_counts(graph, counts)
                estimate = float(row["raw_energy"])
                if method == "readout":
                    observed = counts_probability_vector(str(row["raw_counts"]), graph.width)
                    corrected, condition_number, calibration_status = (
                        constrained_readout_correction(corrected_matrix, observed)
                    )
                    row["condition_number"] = condition_number
                    if corrected is None:
                        row["status"] = calibration_status
                        row["failure_reason"] = calibration_status
                        rows.append(row)
                        continue
                    estimate = _energy_from_vector(graph, corrected)
            else:
                estimates: list[float] = []
                counts_by_factor: dict[str, dict[str, int]] = {}
                for factor, shots in zip(
                    ZNE_FACTORS, factor_shots[candidate_index], strict=True
                ):
                    counts = _sample_complete_bitstrings(
                        fold_global(circuit, factor),
                        shots=shots,
                        seed=derive_seed("qaoa-zne-factor", paired_seed, factor),
                        noise=noise,
                    )
                    counts_by_factor[str(factor)] = counts
                    estimates.append(expected_cut_from_counts(graph, counts))
                row["factor_level_counts"] = json.dumps(
                    counts_by_factor, sort_keys=True, separators=(",", ":")
                )
                row["raw_counts"] = _counts_json(counts_by_factor["1"])
                row["raw_energy"] = estimates[0]
                row["factor_level_estimates"] = json.dumps(estimates, separators=(",", ":"))
                estimate, slope, residual = linear_extrapolate(ZNE_FACTORS, tuple(estimates))
                row["zne_slope"] = slope
                row["zne_residual_sum_squares"] = residual
            row["corrected_or_extrapolated_estimate"] = estimate
            row["status"] = (
                "nonphysical" if not 0.0 <= estimate <= len(graph.edges) else "complete"
            )
            if row["status"] == "nonphysical":
                row["failure_reason"] = "estimate_outside_physical_cut_range"
        except Exception as error:  # preserve every frozen candidate row
            row["status"] = "failed"
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def selection_budget_split(total_selection_budget: int) -> tuple[int, int]:
    """Apply the frozen 50/50 selection and final-evaluation allocation."""

    if total_selection_budget <= 0 or total_selection_budget % 2:
        raise ValueError("Total selection budget must be a positive even integer")
    selection_shots = total_selection_budget // 2
    return selection_shots, total_selection_budget - selection_shots


@cache
def generate_candidate_pool(graph: MaxCutGraph) -> tuple[QAOACandidate, ...]:
    """Evaluate the complete deterministic p=1 grid using noiseless statevectors."""

    evaluated: list[tuple[str, float, float, float]] = []
    for beta_index, beta in enumerate(BETA_GRID):
        for gamma_index, gamma in enumerate(GAMMA_GRID):
            circuit = qaoa_circuit(graph, betas=(beta,), gammas=(gamma,))
            energy = expected_cut_from_probabilities(graph, noiseless_probabilities(circuit))
            evaluated.append(
                (
                    f"p1_b{beta_index:02d}_g{gamma_index:02d}",
                    beta,
                    gamma,
                    energy,
                )
            )
    evaluated.sort(key=lambda item: (-item[3], item[0]))
    best = evaluated[0][3]
    candidates = []
    for rank, (candidate_id, beta, gamma, energy) in enumerate(evaluated, start=1):
        gap = best - energy
        if abs(gap) <= ENERGY_TOLERANCE:
            gap = 0.0
        candidates.append(
            QAOACandidate(
                graph.graph_id,
                candidate_id,
                beta,
                gamma,
                energy,
                rank,
                gap,
            )
        )
    return tuple(candidates)


def _first_unused(
    candidates: Sequence[QAOACandidate],
    used: set[str],
) -> QAOACandidate:
    for candidate in candidates:
        if candidate.candidate_id not in used:
            return candidate
    raise ValueError("Candidate pool cannot satisfy the frozen role assignment")


def select_benchmark_candidates(graph: MaxCutGraph) -> tuple[QAOACandidate, ...]:
    """Select five candidates by frozen ideal-only gap and quantile rules."""

    pool = generate_candidate_pool(graph)
    optimum = exact_maxcut(graph).optimum
    selected: list[QAOACandidate] = []
    used: set[str] = set()

    def add(candidate: QAOACandidate, role: str, regime: str) -> None:
        selected.append(replace(candidate, candidate_role=role, decision_regime=regime))
        used.add(candidate.candidate_id)

    add(pool[0], "incumbent", "reference")
    near = [
        candidate
        for candidate in pool[1:]
        if ENERGY_TOLERANCE < candidate.ideal_gap_to_best <= NEAR_TIE_RELATIVE_GAP * optimum
    ]
    if near:
        add(_first_unused(near, used), "near_tie_challenger", "near_tie")

    clear = [
        candidate
        for candidate in pool[1:]
        if candidate.ideal_gap_to_best >= CLEAR_RELATIVE_GAP * optimum
    ]
    add(_first_unused(clear, used), "clear_challenger", "clear")

    median_order = sorted(
        pool,
        key=lambda candidate: (
            abs(candidate.ideal_rank - (len(pool) + 1) / 2),
            candidate.ideal_rank,
        ),
    )
    add(_first_unused(median_order, used), "median_control", "control")
    add(_first_unused(tuple(reversed(pool)), used), "low_control", "control")

    if len(selected) < 5:
        non_tied = [
            candidate for candidate in pool if candidate.ideal_gap_to_best > ENERGY_TOLERANCE
        ]
        add(_first_unused(non_tied, used), "additional_control", "control")
    return tuple(selected)


def candidate_pool_frame(
    graphs: Sequence[MaxCutGraph] | None = None,
) -> pd.DataFrame:
    """Return every ideal-only grid candidate before benchmark subset selection."""

    rows = []
    for graph in graphs if graphs is not None else benchmark_graphs():
        classical = exact_maxcut(graph).optimum
        for candidate in generate_candidate_pool(graph):
            rows.append(
                {
                    "graph_id": graph.graph_id,
                    "width": graph.width,
                    "graph_family": graph.family,
                    "edge_count": len(graph.edges),
                    "qaoa_depth": 1,
                    "candidate_id": candidate.candidate_id,
                    "beta": candidate.beta,
                    "gamma": candidate.gamma,
                    "exact_classical_maxcut": classical,
                    "exact_ideal_energy": candidate.exact_ideal_energy,
                    "ideal_rank": candidate.ideal_rank,
                    "ideal_gap_to_best": candidate.ideal_gap_to_best,
                    "evidence_label": "noiseless_simulated",
                }
            )
    return pd.DataFrame(rows)


def build_candidate_manifest(repeats: int = FULL_REPEATS) -> pd.DataFrame:
    """Build the frozen full-run manifest without executing noisy circuits."""

    if repeats < 1:
        raise ValueError("Manifest repeats must be positive")
    rows = []
    for graph_index, graph in enumerate(benchmark_graphs()):
        classical = exact_maxcut(graph).optimum
        for candidate_index, candidate in enumerate(select_benchmark_candidates(graph)):
            for method_index, method in enumerate(METHODS):
                for budget_index, budget in enumerate(SELECTION_BUDGETS):
                    selection_shots, final_shots = selection_budget_split(budget)
                    for repeat_index in range(repeats):
                        rows.append(
                            {
                                "graph_id": graph.graph_id,
                                "width": graph.width,
                                "graph_family": graph.family,
                                "edge_count": len(graph.edges),
                                "qaoa_depth": 1,
                                "candidate_id": candidate.candidate_id,
                                "candidate_role": candidate.candidate_role,
                                "beta_parameters": f"[{candidate.beta:.17g}]",
                                "gamma_parameters": f"[{candidate.gamma:.17g}]",
                                "exact_classical_maxcut": classical,
                                "exact_ideal_energy": candidate.exact_ideal_energy,
                                "ideal_rank": candidate.ideal_rank,
                                "ideal_gap_to_best": candidate.ideal_gap_to_best,
                                "decision_regime": candidate.decision_regime,
                                "method": method,
                                "total_selection_budget": budget,
                                "selection_shots": selection_shots,
                                "final_evaluation_shots": final_shots,
                                "repeat": repeat_index,
                                "seed": derive_seed(
                                    "qaoa-selection-manifest",
                                    graph_index,
                                    candidate_index,
                                    method_index,
                                    budget_index,
                                    repeat_index,
                                ),
                                "status": "pending",
                            }
                        )
    return pd.DataFrame(rows)


def certify_sampled_cuts(
    graph: MaxCutGraph,
    counts: Mapping[str, int],
    *,
    alpha: float = SUCCESS_ALPHA,
) -> SampledCutCertification:
    """Classically verify sampled cut quality against the exact small-n solution."""

    if not 0.0 < alpha <= 1.0:
        raise ValueError("Certification alpha must lie in (0, 1]")
    if not counts:
        raise ValueError("Counts must contain at least one shot")
    optimum = exact_maxcut(graph).optimum
    best = -1
    qualifying = 0
    shots = 0
    for bitstring, count in counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError("Count values must be nonnegative integers")
        value = cut_value(graph, bitstring)
        best = max(best, value)
        if value >= alpha * optimum:
            qualifying += count
        shots += count
    if shots <= 0:
        raise ValueError("Counts must contain at least one shot")
    return SampledCutCertification(
        optimum,
        best,
        best == optimum,
        alpha,
        qualifying / shots,
    )


def write_candidate_contract() -> None:
    """Write the complete ideal pool and frozen noisy-run manifest."""

    CANDIDATE_POOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidate_pool_frame().to_csv(
        CANDIDATE_POOL_PATH, index=False, lineterminator="\n", float_format="%.17g"
    )
    build_candidate_manifest().to_csv(
        CANDIDATE_MANIFEST_PATH,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )


def maxcut_cost_operator(graph: MaxCutGraph) -> SparsePauliOp:
    """Return C=sum_(i,j)(I-Z_i Z_j)/2 in Qiskit's Pauli order."""

    terms: list[tuple[str, complex]] = [("I" * graph.width, len(graph.edges) / 2.0)]
    for left, right in graph.edges:
        label = ["I"] * graph.width
        label[graph.width - 1 - left] = "Z"
        label[graph.width - 1 - right] = "Z"
        terms.append(("".join(label), -0.5))
    return SparsePauliOp.from_list(terms).simplify()


if __name__ == "__main__":
    write_candidate_contract()
