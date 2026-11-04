"""Distribution-preserving ZNE sampling for signed-mass diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit

from src.config import derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.folding import fold_global
from src.mitigation.zne import linear_extrapolate
from src.noise_models import NoiseSpec
from src.resource_accounting import split_shots


@dataclass(frozen=True)
class DistributionZNEResult:
    """Factor estimates and raw counts retained for distribution extrapolation."""

    extrapolated_estimate: float
    fit_residual: float
    total_shot_budget: int
    factors: tuple[int, ...]
    shots_by_factor: tuple[int, ...]
    factor_estimates: tuple[float, ...]
    factor_standard_errors: tuple[float, ...]
    factor_counts_json: tuple[str, ...]
    factor_seeds: tuple[int, ...]


def run_distribution_zne(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    factors: tuple[int, ...] = (1, 3, 5),
) -> DistributionZNEResult:
    """Run fixed-budget ZNE while preserving every factor's sampled counts."""

    shots_by_factor = split_shots(total_shot_budget, len(factors))
    estimates: list[float] = []
    uncertainties: list[float] = []
    counts: list[str] = []
    seeds: list[int] = []
    for factor, shots in zip(factors, shots_by_factor, strict=True):
        child_seed = derive_seed("zne-distribution-factor", seed, factor)
        result = estimate_pauli(
            fold_global(circuit, factor),
            pauli_q0_first,
            shots=shots,
            seed=child_seed,
            noise_spec=noise_spec,
        )
        estimates.append(result.estimate)
        uncertainties.append(result.analytic_standard_error)
        counts.append(result.raw_counts_json)
        seeds.append(child_seed)
    extrapolated, _, residual = linear_extrapolate(factors, tuple(estimates))
    return DistributionZNEResult(
        extrapolated_estimate=extrapolated,
        fit_residual=residual,
        total_shot_budget=total_shot_budget,
        factors=factors,
        shots_by_factor=shots_by_factor,
        factor_estimates=tuple(estimates),
        factor_standard_errors=tuple(uncertainties),
        factor_counts_json=tuple(counts),
        factor_seeds=tuple(seeds),
    )
