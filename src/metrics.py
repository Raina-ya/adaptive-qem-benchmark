"""Deterministic benchmark metrics and uncertainty intervals."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import derive_seed

CONDITION_COLUMNS = ["family", "circuit_id", "width", "depth", "noise", "shots"]


def bca_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic bias-corrected and accelerated bootstrap interval."""

    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 3:
        raise ValueError("BCa intervals require at least three scalar observations")
    observed = float(statistic(sample))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [statistic(sample[rng.integers(0, len(sample), len(sample))]) for _ in range(resamples)],
        dtype=float,
    )
    proportion = float(np.mean(draws < observed))
    proportion = min(max(proportion, 1.0 / (2 * resamples)), 1.0 - 1.0 / (2 * resamples))
    bias = norm.ppf(proportion)
    jackknife = np.array(
        [statistic(np.delete(sample, index)) for index in range(len(sample))], dtype=float
    )
    center = float(jackknife.mean())
    numerator = float(np.sum((center - jackknife) ** 3))
    denominator = 6.0 * float(np.sum((center - jackknife) ** 2)) ** 1.5
    acceleration = numerator / denominator if denominator else 0.0
    alpha = (1.0 - confidence) / 2.0

    def adjusted_probability(probability: float) -> float:
        score = bias + norm.ppf(probability)
        denominator_value = 1.0 - acceleration * score
        return float(norm.cdf(bias + score / denominator_value))

    lower_probability = adjusted_probability(alpha)
    upper_probability = adjusted_probability(1.0 - alpha)
    return (
        float(np.quantile(draws, lower_probability)),
        float(np.quantile(draws, upper_probability)),
    )


def paired_rmse_gain_interval(
    raw_errors: np.ndarray,
    method_errors: np.ndarray,
    *,
    seed: int,
    resamples: int = 1000,
) -> tuple[float, float, float]:
    """Estimate raw-minus-method RMSE gain with matched-repeat BCa resampling."""

    raw = np.asarray(raw_errors, dtype=float)
    method = np.asarray(method_errors, dtype=float)
    if raw.shape != method.shape:
        raise ValueError("Paired RMSE comparison requires matched arrays")
    pairs = np.arange(len(raw), dtype=float)

    def statistic(indices: np.ndarray) -> float:
        selected = indices.astype(int)
        return float(
            np.sqrt(np.mean(raw[selected] ** 2))
            - np.sqrt(np.mean(method[selected] ** 2))
        )

    estimate = statistic(pairs)
    lower, upper = bca_interval(pairs, statistic, seed=seed, resamples=resamples)
    return estimate, lower, upper


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial probability."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson interval requires 0 <= successes <= positive trials")
    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return center - radius, center + radius


def benchmark_summary(results: pd.DataFrame, *, resamples: int = 1000) -> pd.DataFrame:
    """Aggregate repeat rows while preserving paired mitigation comparisons."""

    frame = results.loc[results.status == "complete"].copy()
    reference_column = np.where(
        frame.family.eq("ghz"),
        frame.get("analytic_reference", 1.0),
        frame["noiseless_statevector_reference"],
    )
    frame["reference"] = reference_column.astype(float)
    frame["error_value"] = frame.estimate.astype(float) - frame.reference
    rows: list[dict[str, object]] = []
    for condition, condition_frame in frame.groupby(CONDITION_COLUMNS, sort=True):
        raw = condition_frame.loc[condition_frame.method == "raw", ["repeat", "error_value"]]
        for method, group in condition_frame.groupby("method", sort=True):
            errors = group.error_value.to_numpy(dtype=float)
            row = dict(zip(CONDITION_COLUMNS, condition, strict=True))
            row.update(
                method=method,
                complete_repeats=len(group),
                bias=float(errors.mean()),
                variance=float(errors.var(ddof=1)) if len(errors) > 1 else 0.0,
                rmse=float(np.sqrt(np.mean(errors**2))),
                execution_overhead=float(group.total_circuit_executions.mean()),
                folded_two_qubit_gate_count=float(group.folded_two_qubit_gate_count.mean()),
                condition_number_mean=float(group.condition_number.dropna().mean())
                if group.condition_number.notna().any()
                else np.nan,
                fit_residual_mean=float(group.fit_residual_sum_squares.dropna().mean())
                if group.fit_residual_sum_squares.notna().any()
                else np.nan,
                out_of_range_rate=float(
                    group.out_of_range.astype("boolean").fillna(False).mean()
                ),
                confidence_method="paired_bca_bootstrap",
                bootstrap_resamples=resamples,
            )
            matched = raw.merge(
                group[["repeat", "error_value"]], on="repeat", suffixes=("_raw", "_method")
            )
            if method == "raw":
                gain, lower, upper = 0.0, 0.0, 0.0
            elif len(matched) >= 3:
                gain, lower, upper = paired_rmse_gain_interval(
                    matched.error_value_raw.to_numpy(),
                    matched.error_value_method.to_numpy(),
                    seed=derive_seed(
                        "metric-bootstrap",
                        int(row["width"]),
                        int(row["depth"]),
                        int(row["shots"]),
                        len(rows),
                    ),
                    resamples=resamples,
                )
            else:
                gain, lower, upper = np.nan, np.nan, np.nan
            row.update(
                rmse_gain_vs_raw=gain,
                rmse_gain_ci_low=lower,
                rmse_gain_ci_high=upper,
            )
            rows.append(row)
    return pd.DataFrame(rows)
