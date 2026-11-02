"""Linear zero-noise extrapolation under a fixed total shot budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit

from src.config import derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.folding import fold_global
from src.noise_models import NoiseSpec
from src.resource_accounting import split_shots

PREREGISTERED_MODEL = "ordinary least squares: estimate = intercept + slope * noise_factor"


@dataclass(frozen=True)
class ZNEResult:
    extrapolated_estimate: float
    slope: float
    residual_sum_squares: float
    out_of_range: bool
    total_shot_budget: int
    factors: tuple[int, ...]
    shots_by_factor: tuple[int, ...]
    factor_estimates: tuple[float, ...]
    factor_seeds: tuple[int, ...]


def linear_extrapolate(
    factors: tuple[int, ...], estimates: tuple[float, ...]
) -> tuple[float, float, float]:
    """Fit the preregistered line and evaluate it at zero noise."""

    if len(factors) != len(estimates) or len(factors) < 2:
        raise ValueError("At least two matched factors and estimates are required")
    if len(set(factors)) != len(factors):
        raise ValueError("Noise factors must be distinct")
    design = np.column_stack((np.ones(len(factors)), np.asarray(factors, dtype=float)))
    coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(estimates), rcond=None)
    residuals = np.asarray(estimates) - design @ coefficients
    return float(coefficients[0]), float(coefficients[1]), float(residuals @ residuals)


def run_linear_zne(
    circuit: QuantumCircuit,
    pauli_q0_first: str,
    *,
    total_shot_budget: int,
    noise_spec: NoiseSpec,
    seed: int,
    factors: tuple[int, ...] = (1, 3, 5),
) -> ZNEResult:
    """Sample all folds with an equal-share integer split of one total budget."""

    shots_by_factor = split_shots(total_shot_budget, len(factors))
    estimates: list[float] = []
    seeds: list[int] = []
    for factor, shots in zip(factors, shots_by_factor, strict=True):
        child_seed = derive_seed("zne-factor", seed, factor)
        result = estimate_pauli(
            fold_global(circuit, factor),
            pauli_q0_first,
            shots=shots,
            seed=child_seed,
            noise_spec=noise_spec,
        )
        estimates.append(result.estimate)
        seeds.append(child_seed)
    extrapolated, slope, rss = linear_extrapolate(factors, tuple(estimates))
    return ZNEResult(
        extrapolated_estimate=extrapolated,
        slope=slope,
        residual_sum_squares=rss,
        out_of_range=not -1.0 <= extrapolated <= 1.0,
        total_shot_budget=total_shot_budget,
        factors=factors,
        shots_by_factor=shots_by_factor,
        factor_estimates=tuple(estimates),
        factor_seeds=tuple(seeds),
    )
