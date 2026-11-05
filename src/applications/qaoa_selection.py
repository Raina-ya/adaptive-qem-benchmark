"""Exact MaxCut references and deterministic QAOA circuit foundations."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from statistics import NormalDist

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


@dataclass(frozen=True)
class CandidateSelectionMetrics:
    """Application-level metrics for one resolved candidate selection."""

    selected_candidate_id: str
    noisy_winner_ids: tuple[str, ...]
    exact_winner_ids: tuple[str, ...]
    set_valued_correct: bool
    unique_winner_correct: bool | None
    top_k_recall: float
    simple_regret: float
    energy_rmse: float
    decision_error: int


@dataclass(frozen=True)
class CertificationCheckpoint:
    """One cumulative checkpoint built from nested candidate samples."""

    total_shots: int
    estimates: Mapping[str, float]
    standard_errors: Mapping[str, float]
    candidate_shots: Mapping[str, int]


@dataclass(frozen=True)
class CertificationResult:
    """First certified winner, or an explicit unresolved result."""

    status: str
    winner_id: str | None
    shots_to_certified_decision: int | None
    maximum_shots: int
    simultaneous_confidence: float
    interval_confidence: float
    critical_value: float
    winner_tied: bool


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


def _validated_candidate_values(
    candidate_ids: Sequence[str], values: Sequence[float], *, name: str
) -> dict[str, float]:
    if not candidate_ids or len(candidate_ids) != len(values):
        raise ValueError(f"{name} must match a nonempty candidate list")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate identifiers must be unique")
    result = {
        candidate_id: float(value)
        for candidate_id, value in zip(candidate_ids, values, strict=True)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_candidate_selection(
    candidate_ids: Sequence[str],
    exact_energies: Sequence[float],
    estimates: Sequence[float],
    *,
    top_k: int,
    tolerance: float = ENERGY_TOLERANCE,
) -> CandidateSelectionMetrics:
    """Evaluate one noisy selection without exposing ideal values to the selector."""

    exact = _validated_candidate_values(candidate_ids, exact_energies, name="Exact energies")
    noisy = _validated_candidate_values(candidate_ids, estimates, name="Estimates")
    if not 1 <= top_k <= len(candidate_ids):
        raise ValueError("top_k must lie within the candidate count")
    noisy_best = max(noisy.values())
    noisy_winners = tuple(
        sorted(key for key, value in noisy.items() if noisy_best - value <= tolerance)
    )
    selected = noisy_winners[0]
    exact_best = max(exact.values())
    exact_winners = tuple(
        sorted(key for key, value in exact.items() if exact_best - value <= tolerance)
    )

    exact_order = sorted(candidate_ids, key=lambda key: (-exact[key], key))
    top_k_boundary = exact[exact_order[top_k - 1]]
    exact_top_k = {
        key for key, value in exact.items() if value >= top_k_boundary - tolerance
    }
    noisy_top_k = set(sorted(candidate_ids, key=lambda key: (-noisy[key], key))[:top_k])
    set_valued_correct = selected in exact_winners
    return CandidateSelectionMetrics(
        selected_candidate_id=selected,
        noisy_winner_ids=noisy_winners,
        exact_winner_ids=exact_winners,
        set_valued_correct=set_valued_correct,
        unique_winner_correct=(
            set_valued_correct if len(exact_winners) == 1 else None
        ),
        top_k_recall=len(noisy_top_k & exact_top_k) / len(exact_top_k),
        simple_regret=max(0.0, exact_best - exact[selected]),
        energy_rmse=math.sqrt(
            math.fsum((noisy[key] - exact[key]) ** 2 for key in candidate_ids)
            / len(candidate_ids)
        ),
        decision_error=int(not set_valued_correct),
    )


def wilson_interval(
    successes: int, total: int, *, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for one empirical probability."""

    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total and total > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence must lie in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def nested_shot_checkpoints(
    maximum_shots: int, fractions: Sequence[float] = (0.25, 0.5, 0.75, 1.0)
) -> tuple[int, ...]:
    """Create preregistered strictly increasing cumulative shot checkpoints."""

    if maximum_shots <= 0 or not fractions:
        raise ValueError("Maximum shots and checkpoint fractions must be positive")
    if tuple(fractions) != tuple(sorted(set(fractions))):
        raise ValueError("Checkpoint fractions must be unique and increasing")
    if fractions[-1] != 1.0 or any(not 0.0 < value <= 1.0 for value in fractions):
        raise ValueError("Checkpoint fractions must end at one and lie in (0, 1]")
    checkpoints = tuple(int(maximum_shots * value) for value in fractions)
    if checkpoints[0] <= 0 or any(
        right <= left for left, right in zip(checkpoints, checkpoints[1:], strict=False)
    ):
        raise ValueError("Shot budget is too small for distinct nested checkpoints")
    return checkpoints


def certify_candidate_winner(
    checkpoints: Sequence[CertificationCheckpoint],
    *,
    confidence: float = 0.95,
    tolerance: float = ENERGY_TOLERANCE,
) -> CertificationResult:
    """Certify separation using Bonferroni intervals across candidates and looks."""

    if not checkpoints or not 0.0 < confidence < 1.0:
        raise ValueError("Checkpoints are required and confidence must lie in (0, 1)")
    candidate_ids = tuple(sorted(checkpoints[0].estimates))
    if len(candidate_ids) < 2:
        raise ValueError("Certification requires at least two candidates")
    previous_total = 0
    previous_candidate_shots = {candidate_id: 0 for candidate_id in candidate_ids}
    for checkpoint in checkpoints:
        if tuple(sorted(checkpoint.estimates)) != candidate_ids:
            raise ValueError("Every checkpoint must contain the same candidates")
        if tuple(sorted(checkpoint.standard_errors)) != candidate_ids:
            raise ValueError("Every checkpoint must contain one standard error per candidate")
        if tuple(sorted(checkpoint.candidate_shots)) != candidate_ids:
            raise ValueError("Every checkpoint must contain cumulative candidate shots")
        if checkpoint.total_shots <= previous_total:
            raise ValueError("Checkpoint total shots must increase monotonically")
        for candidate_id in candidate_ids:
            estimate = float(checkpoint.estimates[candidate_id])
            standard_error = float(checkpoint.standard_errors[candidate_id])
            shots = int(checkpoint.candidate_shots[candidate_id])
            if not math.isfinite(estimate) or not math.isfinite(standard_error):
                raise ValueError("Checkpoint estimates and errors must be finite")
            if standard_error < 0.0 or shots <= previous_candidate_shots[candidate_id]:
                raise ValueError("Candidate shots must be positive and nested monotonically")
            previous_candidate_shots[candidate_id] = shots
        previous_total = checkpoint.total_shots

    family_size = len(candidate_ids) * len(checkpoints)
    alpha = 1.0 - confidence
    interval_confidence = 1.0 - alpha / family_size
    critical_value = NormalDist().inv_cdf(1.0 - alpha / (2.0 * family_size))
    last_tied = False
    for checkpoint in checkpoints:
        best_estimate = max(float(value) for value in checkpoint.estimates.values())
        tied = tuple(
            sorted(
                candidate_id
                for candidate_id, value in checkpoint.estimates.items()
                if best_estimate - float(value) <= tolerance
            )
        )
        last_tied = len(tied) > 1
        if last_tied:
            continue
        winner = tied[0]
        lower = float(checkpoint.estimates[winner]) - critical_value * float(
            checkpoint.standard_errors[winner]
        )
        challenger_uppers = [
            float(checkpoint.estimates[candidate_id])
            + critical_value * float(checkpoint.standard_errors[candidate_id])
            for candidate_id in candidate_ids
            if candidate_id != winner
        ]
        if lower > max(challenger_uppers):
            return CertificationResult(
                "certified",
                winner,
                checkpoint.total_shots,
                checkpoints[-1].total_shots,
                confidence,
                interval_confidence,
                critical_value,
                False,
            )
    return CertificationResult(
        "unresolved",
        None,
        None,
        checkpoints[-1].total_shots,
        confidence,
        interval_confidence,
        critical_value,
        last_tied,
    )


def selection_metrics_frame(frame: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    """Reduce candidate rows to one application-decision row per repeat."""

    required = {
        "graph_id",
        "graph_family",
        "width",
        "method",
        "total_selection_budget",
        "repeat",
        "candidate_id",
        "exact_ideal_energy",
        "corrected_or_extrapolated_estimate",
        "status",
        "candidate_target_shots",
        "calibration_shots",
        "factor_level_shots",
        "selection_shots",
        "selection_execution_count",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Selection results are missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    group_columns = ["graph_id", "method", "total_selection_budget", "repeat"]
    for keys, group in frame.groupby(group_columns, sort=True, dropna=False):
        graph_id, method, budget, repeat = keys
        overhead_shots = int(group.calibration_shots.max())
        if method == "zne":
            overhead_shots += sum(
                sum(json.loads(value)[1:]) for value in group.factor_level_shots
            )
        regime_groups = [("overall", group)]
        incumbent = group[group.candidate_role == "incumbent"]
        for regime, role in (
            ("clear", "clear_challenger"),
            ("near_tie", "near_tie_challenger"),
        ):
            challenger = group[group.candidate_role == role]
            if not incumbent.empty and not challenger.empty:
                regime_groups.append((regime, pd.concat([incumbent, challenger])))
        for regime, regime_group in regime_groups:
            estimates = pd.to_numeric(
                regime_group["corrected_or_extrapolated_estimate"], errors="coerce"
            )
            resolved = (
                regime_group.status.isin({"complete", "nonphysical"}).all()
                and estimates.notna().all()
            )
            certification_status = (
                str(group.certification_status.iloc[0])
                if regime == "overall" and "certification_status" in group
                else "not_evaluated"
            )
            certification_shots = (
                pd.to_numeric(group.shots_to_certified_decision, errors="coerce").iloc[0]
                if regime == "overall" and "shots_to_certified_decision" in group
                else math.nan
            )
            row: dict[str, object] = {
                "graph_id": graph_id,
                "graph_family": str(group.graph_family.iloc[0]),
                "width": int(group.width.iloc[0]),
                "candidate_gap_regime": regime,
                "method": method,
                "total_selection_budget": int(budget),
                "repeat": int(repeat),
                "resolved": bool(resolved),
                "unresolved": not resolved,
                "overhead_shots": overhead_shots,
                "shot_overhead_fraction": overhead_shots
                / int(group.selection_shots.iloc[0]),
                "selection_execution_count": int(
                    group.selection_execution_count.iloc[0]
                ),
                "execution_overhead_ratio": int(
                    group.selection_execution_count.iloc[0]
                )
                / len(group),
                "shots_to_certified_decision": certification_shots,
                "certification_status": certification_status,
            }
            if resolved:
                metrics = evaluate_candidate_selection(
                    regime_group.candidate_id.tolist(),
                    regime_group.exact_ideal_energy.tolist(),
                    estimates.tolist(),
                    top_k=min(top_k, len(regime_group)),
                )
                row.update(
                    {
                        "selected_candidate_id": metrics.selected_candidate_id,
                        "noisy_winner_ids": json.dumps(metrics.noisy_winner_ids),
                        "exact_winner_ids": json.dumps(metrics.exact_winner_ids),
                        "noisy_tie": len(metrics.noisy_winner_ids) > 1,
                        "exact_degeneracy": len(metrics.exact_winner_ids) > 1,
                        "best_candidate_selection_indicator": int(
                            metrics.set_valued_correct
                        ),
                        "unique_winner_indicator": metrics.unique_winner_correct,
                        "top_k_recall": metrics.top_k_recall,
                        "simple_regret": metrics.simple_regret,
                        "energy_rmse": metrics.energy_rmse,
                        "decision_error": metrics.decision_error,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_selection_results(
    frame: pd.DataFrame, *, top_k: int, confidence: float = 0.95
) -> pd.DataFrame:
    """Aggregate application metrics while retaining unresolved probability."""

    decisions = selection_metrics_frame(frame, top_k=top_k)
    rows: list[dict[str, object]] = []
    summary_groups = [
        "graph_id",
        "graph_family",
        "width",
        "candidate_gap_regime",
        "method",
        "total_selection_budget",
    ]
    for keys, group in decisions.groupby(summary_groups, sort=True):
        graph_id, graph_family, width, regime, method, budget = keys
        resolved = group[group.resolved]
        unresolved_count = int(group.unresolved.sum())
        unresolved_low, unresolved_high = wilson_interval(
            unresolved_count, len(group), confidence=confidence
        )
        row: dict[str, object] = {
            "graph_id": graph_id,
            "graph_family": graph_family,
            "width": int(width),
            "candidate_gap_regime": regime,
            "method": method,
            "total_selection_budget": int(budget),
            "repeats": len(group),
            "resolved_repeats": len(resolved),
            "unresolved_probability": unresolved_count / len(group),
            "unresolved_wilson_low": unresolved_low,
            "unresolved_wilson_high": unresolved_high,
            "mean_overhead_shots": group.overhead_shots.mean(),
            "mean_shot_overhead_fraction": group.shot_overhead_fraction.mean(),
            "mean_selection_execution_count": group.selection_execution_count.mean(),
            "total_selection_executions": int(group.selection_execution_count.sum()),
            "mean_execution_overhead_ratio": group.execution_overhead_ratio.mean(),
            "total_charged_shots": int(budget) * len(group),
        }
        if resolved.empty:
            row.update(
                {
                    "best_candidate_selection_probability": math.nan,
                    "selection_probability_wilson_low": math.nan,
                    "selection_probability_wilson_high": math.nan,
                    "mean_top_k_recall": math.nan,
                    "mean_simple_regret": math.nan,
                    "simple_regret_standard_error": math.nan,
                    "energy_rmse": math.nan,
                    "decision_error_probability": math.nan,
                    "decision_error_wilson_low": math.nan,
                    "decision_error_wilson_high": math.nan,
                    "mean_shots_to_certified_decision": math.nan,
                    "certified_probability": math.nan,
                }
            )
        else:
            errors = int(resolved.decision_error.sum())
            successes = len(resolved) - errors
            error_low, error_high = wilson_interval(
                errors, len(resolved), confidence=confidence
            )
            success_low, success_high = wilson_interval(
                successes, len(resolved), confidence=confidence
            )
            certified_shots = pd.to_numeric(
                resolved.loc[
                    resolved.certification_status == "certified",
                    "shots_to_certified_decision",
                ],
                errors="coerce",
            ).dropna()
            regret_standard_error = (
                float(resolved.simple_regret.std(ddof=1) / math.sqrt(len(resolved)))
                if len(resolved) > 1
                else math.nan
            )
            row.update(
                {
                    "best_candidate_selection_probability": resolved[
                        "best_candidate_selection_indicator"
                    ].mean(),
                    "selection_probability_wilson_low": success_low,
                    "selection_probability_wilson_high": success_high,
                    "mean_top_k_recall": resolved.top_k_recall.mean(),
                    "mean_simple_regret": resolved.simple_regret.mean(),
                    "simple_regret_standard_error": regret_standard_error,
                    "energy_rmse": math.sqrt(
                        math.fsum(float(value) ** 2 for value in resolved.energy_rmse)
                        / len(resolved)
                    ),
                    "decision_error_probability": errors / len(resolved),
                    "decision_error_wilson_low": error_low,
                    "decision_error_wilson_high": error_high,
                    "mean_shots_to_certified_decision": resolved[
                        "shots_to_certified_decision"
                    ].mean(),
                    "median_shots_to_certified_decision": (
                        certified_shots.median() if not certified_shots.empty else math.nan
                    ),
                    "q25_shots_to_certified_decision": (
                        certified_shots.quantile(0.25)
                        if not certified_shots.empty
                        else math.nan
                    ),
                    "q75_shots_to_certified_decision": (
                        certified_shots.quantile(0.75)
                        if not certified_shots.empty
                        else math.nan
                    ),
                    "certified_probability": resolved.certification_status.eq(
                        "certified"
                    ).mean(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


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


def _sample_complete_bitstring_memory(
    circuit: QuantumCircuit,
    *,
    shots: int,
    seed: int,
    noise: NoiseSpec,
) -> tuple[str, ...]:
    """Return an ordered sample record so checkpoints can use true prefixes."""

    measured = circuit.copy()
    measured.measure_all()
    simulator = aer_simulator(noise)
    compiled = transpile(measured, simulator, optimization_level=0, seed_transpiler=seed)
    memory = simulator.run(
        compiled, shots=shots, seed_simulator=seed, memory=True
    ).result().get_memory()
    return tuple(str(value).replace(" ", "") for value in memory)


def _memory_counts(memory: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(memory).items()))


def _cut_standard_error(graph: MaxCutGraph, counts: Mapping[str, int]) -> float:
    shots = sum(counts.values())
    if shots < 2:
        return math.inf
    mean = expected_cut_from_counts(graph, counts)
    squared_deviations = math.fsum(
        count * (cut_value(graph, bitstring) - mean) ** 2
        for bitstring, count in counts.items()
    )
    return math.sqrt(squared_deviations / (shots - 1) / shots)


def _zne_intercept_weights() -> tuple[float, ...]:
    design = np.column_stack((np.ones(len(ZNE_FACTORS)), np.asarray(ZNE_FACTORS, dtype=float)))
    return tuple(float(value) for value in np.linalg.pinv(design)[0])


def _attach_group_certification(
    rows: list[dict[str, object]],
    candidates: Sequence[QAOACandidate],
    estimates: Mapping[str, Sequence[float]],
    standard_errors: Mapping[str, Sequence[float]],
    candidate_shots: Mapping[str, Sequence[int]],
    *,
    calibration_shots: int,
) -> None:
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    maximum_shots = int(rows[0]["selection_shots"])
    if any(candidate_id not in estimates for candidate_id in candidate_ids):
        for row in rows:
            row["certification_status"] = "unresolved"
            row["maximum_certification_shots"] = maximum_shots
        return
    checkpoints: list[CertificationCheckpoint] = []
    records: list[dict[str, object]] = []
    for checkpoint_index in range(4):
        cumulative_candidate_shots = {
            candidate_id: int(candidate_shots[candidate_id][checkpoint_index])
            for candidate_id in candidate_ids
        }
        total_shots = calibration_shots + sum(cumulative_candidate_shots.values())
        checkpoint_estimates = {
            candidate_id: float(estimates[candidate_id][checkpoint_index])
            for candidate_id in candidate_ids
        }
        checkpoint_errors = {
            candidate_id: float(standard_errors[candidate_id][checkpoint_index])
            for candidate_id in candidate_ids
        }
        checkpoints.append(
            CertificationCheckpoint(
                total_shots,
                checkpoint_estimates,
                checkpoint_errors,
                cumulative_candidate_shots,
            )
        )
        records.append(
            {
                "total_shots": total_shots,
                "estimates": checkpoint_estimates,
                "standard_errors": checkpoint_errors,
                "candidate_shots": cumulative_candidate_shots,
            }
        )
    result = certify_candidate_winner(checkpoints)
    checkpoint_json = json.dumps(records, sort_keys=True, separators=(",", ":"))
    for row_index, row in enumerate(rows):
        row["certification_status"] = result.status
        row["shots_to_certified_decision"] = (
            result.shots_to_certified_decision
            if result.shots_to_certified_decision is not None
            else math.nan
        )
        row["maximum_certification_shots"] = result.maximum_shots
        row["certification_checkpoint_data"] = checkpoint_json if row_index == 0 else ""


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
    collect_checkpoints: bool = False,
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
    checkpoint_estimates: dict[str, list[float]] = {}
    checkpoint_standard_errors: dict[str, list[float]] = {}
    checkpoint_candidate_shots: dict[str, list[int]] = {}
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
            "correction_status": "not_applicable",
            "nonphysical_estimate": False,
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
            "certification_status": "not_evaluated",
            "shots_to_certified_decision": math.nan,
            "maximum_certification_shots": math.nan,
            "certification_checkpoint_data": "",
        }
        try:
            circuit = qaoa_circuit(graph, betas=(candidate.beta,), gammas=(candidate.gamma,))
            if method in {"raw", "readout"}:
                shots = int(target_shots[candidate_index])
                checkpoint_counts: list[dict[str, int]] = []
                if collect_checkpoints:
                    memory = _sample_complete_bitstring_memory(
                        circuit, shots=shots, seed=paired_seed, noise=noise
                    )
                    counts = _memory_counts(memory)
                    checkpoint_counts = [
                        _memory_counts(memory[:checkpoint])
                        for checkpoint in nested_shot_checkpoints(shots)
                    ]
                else:
                    counts = _sample_complete_bitstrings(
                        circuit, shots=shots, seed=paired_seed, noise=noise
                    )
                row["raw_counts"] = _counts_json(counts)
                row["raw_energy"] = expected_cut_from_counts(graph, counts)
                estimate = float(row["raw_energy"])
                candidate_checkpoint_estimates = [
                    expected_cut_from_counts(graph, prefix) for prefix in checkpoint_counts
                ]
                candidate_checkpoint_errors = [
                    _cut_standard_error(graph, prefix) for prefix in checkpoint_counts
                ]
                if method == "readout":
                    observed = counts_probability_vector(str(row["raw_counts"]), graph.width)
                    corrected, condition_number, calibration_status = (
                        constrained_readout_correction(
                            corrected_matrix,
                            observed,
                            precomputed_condition_number=condition_number,
                        )
                    )
                    row["condition_number"] = condition_number
                    row["correction_status"] = calibration_status
                    if corrected is None:
                        row["status"] = "failed"
                        row["failure_reason"] = calibration_status
                        rows.append(row)
                        continue
                    estimate = _energy_from_vector(graph, corrected)
                    corrected_checkpoint_estimates: list[float] = []
                    corrected_checkpoint_errors: list[float] = []
                    for prefix, raw_error in zip(
                        checkpoint_counts, candidate_checkpoint_errors, strict=True
                    ):
                        prefix_observed = counts_probability_vector(
                            _counts_json(prefix), graph.width
                        )
                        prefix_corrected, prefix_condition, prefix_status = (
                            constrained_readout_correction(
                                corrected_matrix,
                                prefix_observed,
                                precomputed_condition_number=condition_number,
                            )
                        )
                        if prefix_corrected is None:
                            raise RuntimeError(
                                f"checkpoint readout correction {prefix_status}"
                            )
                        corrected_checkpoint_estimates.append(
                            _energy_from_vector(graph, prefix_corrected)
                        )
                        corrected_checkpoint_errors.append(
                            raw_error * prefix_condition
                        )
                    candidate_checkpoint_estimates = corrected_checkpoint_estimates
                    candidate_checkpoint_errors = corrected_checkpoint_errors
                if collect_checkpoints:
                    checkpoint_estimates[candidate.candidate_id] = (
                        candidate_checkpoint_estimates
                    )
                    checkpoint_standard_errors[candidate.candidate_id] = (
                        candidate_checkpoint_errors
                    )
                    checkpoint_candidate_shots[candidate.candidate_id] = list(
                        nested_shot_checkpoints(shots)
                    )
            else:
                estimates: list[float] = []
                counts_by_factor: dict[str, dict[str, int]] = {}
                checkpoint_counts_by_factor: dict[int, list[dict[str, int]]] = {}
                for factor, shots in zip(
                    ZNE_FACTORS, factor_shots[candidate_index], strict=True
                ):
                    folded = fold_global(circuit, factor)
                    factor_seed = derive_seed("qaoa-zne-factor", paired_seed, factor)
                    if collect_checkpoints:
                        memory = _sample_complete_bitstring_memory(
                            folded, shots=shots, seed=factor_seed, noise=noise
                        )
                        counts = _memory_counts(memory)
                        checkpoint_counts_by_factor[factor] = [
                            _memory_counts(memory[:checkpoint])
                            for checkpoint in nested_shot_checkpoints(shots)
                        ]
                    else:
                        counts = _sample_complete_bitstrings(
                            folded, shots=shots, seed=factor_seed, noise=noise
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
                if collect_checkpoints:
                    weights = _zne_intercept_weights()
                    candidate_checkpoint_estimates = []
                    candidate_checkpoint_errors = []
                    candidate_checkpoint_totals = []
                    for checkpoint_index in range(4):
                        prefix_estimates = tuple(
                            expected_cut_from_counts(
                                graph, checkpoint_counts_by_factor[factor][checkpoint_index]
                            )
                            for factor in ZNE_FACTORS
                        )
                        prefix_errors = tuple(
                            _cut_standard_error(
                                graph, checkpoint_counts_by_factor[factor][checkpoint_index]
                            )
                            for factor in ZNE_FACTORS
                        )
                        extrapolated, _, _ = linear_extrapolate(
                            ZNE_FACTORS, prefix_estimates
                        )
                        candidate_checkpoint_estimates.append(extrapolated)
                        candidate_checkpoint_errors.append(
                            math.sqrt(
                                math.fsum(
                                    (weight * error) ** 2
                                    for weight, error in zip(
                                        weights, prefix_errors, strict=True
                                    )
                                )
                            )
                        )
                        candidate_checkpoint_totals.append(
                            sum(
                                sum(
                                    checkpoint_counts_by_factor[factor][checkpoint_index].values()
                                )
                                for factor in ZNE_FACTORS
                            )
                        )
                    checkpoint_estimates[candidate.candidate_id] = (
                        candidate_checkpoint_estimates
                    )
                    checkpoint_standard_errors[candidate.candidate_id] = (
                        candidate_checkpoint_errors
                    )
                    checkpoint_candidate_shots[candidate.candidate_id] = (
                        candidate_checkpoint_totals
                    )
            row["corrected_or_extrapolated_estimate"] = estimate
            row["nonphysical_estimate"] = not 0.0 <= estimate <= len(graph.edges)
            row["status"] = "complete"
            if row["nonphysical_estimate"]:
                row["failure_reason"] = "estimate_outside_physical_cut_range"
        except Exception as error:  # preserve every frozen candidate row
            row["status"] = "failed"
            row["failure_reason"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    if collect_checkpoints:
        _attach_group_certification(
            rows,
            candidates,
            checkpoint_estimates,
            checkpoint_standard_errors,
            checkpoint_candidate_shots,
            calibration_shots=calibration_shots,
        )
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
