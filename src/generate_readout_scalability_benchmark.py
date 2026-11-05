"""Run the unified width-scalability benchmark for four readout strategies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuits import ghz_circuit
from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import hardware_efficient_circuit
from src.mitigation.readout import (
    constrained_readout_correction,
    counts_probability_vector,
    run_readout_mitigation,
)
from src.mitigation.readout_matrix_free import (
    TensorSubspaceOperator,
    run_matrix_free_readout_mitigation,
    solve_matrix_free_nnls,
)
from src.mitigation.readout_subspace import (
    observed_subspace_correction,
    run_observed_subspace_mitigation,
    select_observed_support,
)
from src.mitigation.readout_tensor import (
    run_tensor_readout_mitigation,
    tensor_product_correction,
)
from src.noise_models import NoiseSpec
from src.observables import exact_expectation
from src.readout_scalability_metrics import planned_status, resource_estimate

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_scalability_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/readout_scalability_summary_v1.csv"
MEMORY_FIGURE_PATH = ROOT / "figures/readout_memory_vs_width.png"
RUNTIME_FIGURE_PATH = ROOT / "figures/readout_runtime_vs_support.png"
ACCURACY_FIGURE_PATH = ROOT / "figures/readout_accuracy_vs_width.png"
UNCERTAINTY_FIGURE_PATH = ROOT / "figures/readout_uncertainty_amplification.png"

WIDTHS = (2, 4, 6, 8, 10, 12)
DISTRIBUTIONS = ("sparse", "broad")
READOUT_MODELS = ("independent", "asymmetric")
BUDGETS = (4096, 16384)
METHODS = ("full_matrix", "tensor_product", "observed_subspace", "matrix_free_subspace")
SMALL_WIDTH_REPEATS = 50
LARGE_WIDTH_REPEATS = 10
MIN_COUNT = 2
DENSE_SUBSPACE_MAX_SUPPORT = 128


def _noise_spec(name: str) -> NoiseSpec:
    if name == "independent":
        return NoiseSpec(name, readout_p01=0.03, readout_p10=0.03)
    if name == "asymmetric":
        return NoiseSpec(name, readout_p01=0.015, readout_p10=0.06)
    raise ValueError(f"Unknown scalability readout model: {name}")


def _circuit_observable(width: int, distribution: str):
    if distribution == "sparse":
        return ghz_circuit(width), "Z" * width
    if distribution == "broad":
        return hardware_efficient_circuit(width, 3), "Z" * width
    raise ValueError(f"Unknown distribution family: {distribution}")


def build_manifest(
    small_width_repeats: int = SMALL_WIDTH_REPEATS,
    large_width_repeats: int = LARGE_WIDTH_REPEATS,
) -> pd.DataFrame:
    """Build one paired method row per frozen scalability repeat."""

    rows = []
    for width in WIDTHS:
        repeats = small_width_repeats if width <= 6 else large_width_repeats
        for distribution_index, distribution in enumerate(DISTRIBUTIONS):
            for model_index, readout_model in enumerate(READOUT_MODELS):
                for budget in BUDGETS:
                    for repeat in range(repeats):
                        pair_seed = derive_seed(
                            "readout-scalability-benchmark",
                            width,
                            distribution_index,
                            model_index,
                            budget,
                            repeat,
                        )
                        for method in METHODS:
                            status, reason = planned_status(method, resource_estimate(width))
                            rows.append(
                                {
                                    "width": width,
                                    "distribution": distribution,
                                    "readout_model": readout_model,
                                    "total_shot_budget": budget,
                                    "method": method,
                                    "repeat": repeat,
                                    "pair_seed": pair_seed,
                                    "planned_status": status,
                                    "planned_status_reason": reason,
                                }
                            )
    return pd.DataFrame(rows)


def _empty_metrics(width: int) -> dict[str, object]:
    return {
        "raw_estimate": math.nan,
        "corrected_estimate": math.nan,
        "sampling_error": math.nan,
        "absolute_error": math.nan,
        "decision_error": math.nan,
        "condition_estimate": math.nan,
        "assignment_dimension": 2**width,
        "observed_support_size": 0,
        "support_fraction": math.nan,
        "classical_storage_bytes": 0,
        "working_vector_bytes": 0,
        "retained_probability_mass": math.nan,
        "discarded_probability_mass": math.nan,
        "iterations": 0,
        "converged": False,
        "residual_norm": math.nan,
        "solve_wall_seconds": math.nan,
        "timing_kind": "not_measured",
        "total_shots_used": 0,
        "target_shots": 0,
        "calibration_shots": 0,
        "target_circuit_executions": 0,
        "calibration_circuit_executions": 0,
        "raw_counts_json": "",
        "calibration_counts_json": "",
        "support_states_json": "",
    }


def _timed_full_solve(result, width: int) -> float:
    matrix = np.asarray(json.loads(result.assignment_matrix_json), dtype=float)
    observed = counts_probability_vector(result.raw_counts_json, width)
    start = perf_counter()
    constrained_readout_correction(matrix, observed)
    return perf_counter() - start


def _timed_tensor_solve(result, width: int) -> float:
    matrices = tuple(
        np.asarray(matrix, dtype=float) for matrix in json.loads(result.assignment_matrices_json)
    )
    observed = counts_probability_vector(result.raw_counts_json, width)
    start = perf_counter()
    tensor_product_correction(matrices, observed)
    return perf_counter() - start


def _timed_subspace_solve(result, width: int) -> float:
    matrix = np.asarray(json.loads(result.assignment_matrix_json), dtype=float)
    support = tuple(json.loads(result.support_states_json))
    full_observed = counts_probability_vector(result.raw_counts_json, width)
    start = perf_counter()
    observed_subspace_correction(
        matrix,
        full_observed[list(support)],
        support,
        full_dimension=2**width,
    )
    return perf_counter() - start


def _timed_matrix_free_solve(result, width: int) -> float:
    matrices = tuple(
        np.asarray(matrix, dtype=float) for matrix in json.loads(result.assignment_matrices_json)
    )
    support = tuple(json.loads(result.support_states_json))
    full_observed = counts_probability_vector(result.raw_counts_json, width)
    operator = TensorSubspaceOperator(matrices, support)
    start = perf_counter()
    solve_matrix_free_nnls(operator, full_observed[list(support)])
    return perf_counter() - start


def _resource_limited_row(
    item: pd.Series,
    reference: float,
    reason: str,
    *,
    raw=None,
    support_size: int = 0,
    retained_mass: float = math.nan,
    discarded_mass: float = math.nan,
) -> dict[str, object]:
    width = int(item.width)
    metrics = _empty_metrics(width)
    if raw is not None:
        target_shots = sum(json.loads(raw.raw_counts_json).values())
        metrics.update(
            {
                "raw_estimate": raw.estimate,
                "observed_support_size": support_size,
                "support_fraction": support_size / 2**width,
                "retained_probability_mass": retained_mass,
                "discarded_probability_mass": discarded_mass,
                "total_shots_used": target_shots,
                "target_shots": target_shots,
                "target_circuit_executions": 1,
                "raw_counts_json": raw.raw_counts_json,
            }
        )
    return {
        **_base(item, reference),
        "status": "resource_limited",
        "method_status": "resource_limited",
        "status_reason": reason,
        "error": "",
        **metrics,
    }


def _base(item: pd.Series, reference: float) -> dict[str, object]:
    return {
        "width": int(item.width),
        "distribution": str(item.distribution),
        "readout_model": str(item.readout_model),
        "total_shot_budget": int(item.total_shot_budget),
        "method": str(item.method),
        "repeat": int(item["repeat"]),
        "pair_seed": int(item.pair_seed),
        "reference": reference,
        "result_label": "noisy_simulated",
    }


def _execute_result(item: pd.Series, reference: float) -> dict[str, object]:
    width = int(item.width)
    circuit, observable = _circuit_observable(width, str(item.distribution))
    spec = _noise_spec(str(item.readout_model))
    budget = int(item.total_shot_budget)
    seed = int(item.pair_seed)
    if item.method == "full_matrix" and item.planned_status == "resource_limited":
        return _resource_limited_row(item, reference, str(item.planned_status_reason))
    if item.method == "observed_subspace":
        calibration_shots = int(budget * READOUT_CALIBRATION_FRACTION)
        target_shots = budget - calibration_shots
        raw_preflight = estimate_pauli(
            circuit,
            observable,
            shots=target_shots,
            seed=derive_seed("readout-target", seed),
            noise_spec=spec,
        )
        support = select_observed_support(raw_preflight.raw_counts_json, width, min_count=MIN_COUNT)
        if len(support.states) > DENSE_SUBSPACE_MAX_SUPPORT:
            return _resource_limited_row(
                item,
                reference,
                "observed_support_exceeds_dense_limit",
                raw=raw_preflight,
                support_size=len(support.states),
                retained_mass=support.retained_probability_mass,
                discarded_mass=support.discarded_probability_mass,
            )

    if item.method == "full_matrix":
        result = run_readout_mitigation(
            circuit, observable, total_shot_budget=budget, noise_spec=spec, seed=seed
        )
        solve_time = _timed_full_solve(result, width)
        estimate = result.corrected_estimate
        support_size = 2**width
        storage = 8 * (2**width) ** 2
        working = 8 * 2**width
        retained, discarded = 1.0, 0.0
        iterations, converged, residual = 0, result.status == "ok", math.nan
        support_json = json.dumps(tuple(range(2**width)), separators=(",", ":"))
    elif item.method == "tensor_product":
        result = run_tensor_readout_mitigation(
            circuit, observable, total_shot_budget=budget, noise_spec=spec, seed=seed
        )
        solve_time = _timed_tensor_solve(result, width)
        estimate = result.corrected_estimate
        support_size = 2**width
        storage = 8 * (4 * width + 2**width)
        working = 8 * 2**width
        retained, discarded = 1.0, 0.0
        iterations, converged, residual = 0, result.status == "ok", math.nan
        support_json = json.dumps(tuple(range(2**width)), separators=(",", ":"))
    elif item.method == "observed_subspace":
        result = run_observed_subspace_mitigation(
            circuit,
            observable,
            total_shot_budget=budget,
            noise_spec=spec,
            seed=seed,
            min_count=MIN_COUNT,
        )
        solve_time = _timed_subspace_solve(result, width)
        estimate = result.corrected_estimate
        support_size = result.observed_support_size
        storage = result.reduced_matrix_bytes
        working = 8 * support_size
        retained = result.retained_probability_mass
        discarded = result.discarded_probability_mass
        iterations, converged, residual = 0, result.status == "ok", math.nan
        support_json = result.support_states_json
    else:
        result = run_matrix_free_readout_mitigation(
            circuit,
            observable,
            total_shot_budget=budget,
            noise_spec=spec,
            seed=seed,
            min_count=MIN_COUNT,
        )
        solve_time = _timed_matrix_free_solve(result, width)
        estimate = result.corrected_estimate
        support_size = result.observed_support_size
        storage = result.operator_storage_bytes + result.working_vector_bytes
        working = result.working_vector_bytes
        retained = result.retained_probability_mass
        discarded = result.discarded_probability_mass
        iterations = result.iterations
        converged = result.converged
        residual = result.residual_norm
        support_json = result.support_states_json

    complete = result.status == "ok" and estimate is not None
    calibration_json = result.calibration_counts_json
    return {
        **_base(item, reference),
        "status": "complete" if complete else "failed",
        "method_status": result.status,
        "status_reason": "",
        "error": "",
        "raw_estimate": result.raw_estimate,
        "corrected_estimate": estimate if estimate is not None else math.nan,
        "sampling_error": estimate - reference if estimate is not None else math.nan,
        "absolute_error": abs(estimate - reference) if estimate is not None else math.nan,
        "decision_error": (
            float(np.sign(estimate) != np.sign(reference))
            if complete and reference != 0.0
            else math.nan
        ),
        "condition_estimate": (
            result.condition_number if item.method == "full_matrix" else result.condition_estimate
        ),
        "assignment_dimension": 2**width,
        "observed_support_size": support_size,
        "support_fraction": support_size / 2**width,
        "classical_storage_bytes": storage,
        "working_vector_bytes": working,
        "retained_probability_mass": retained,
        "discarded_probability_mass": discarded,
        "iterations": iterations,
        "converged": converged,
        "residual_norm": residual,
        "solve_wall_seconds": solve_time,
        "timing_kind": "measured",
        "total_shots_used": result.target_shots + result.calibration_shots,
        "target_shots": result.target_shots,
        "calibration_shots": result.calibration_shots,
        "target_circuit_executions": result.target_circuit_executions,
        "calibration_circuit_executions": result.calibration_circuit_executions,
        "raw_counts_json": result.raw_counts_json,
        "calibration_counts_json": calibration_json,
        "support_states_json": support_json,
    }


def _result_row(item: pd.Series, reference: float) -> dict[str, object]:
    try:
        return _execute_result(item, reference)
    except Exception as error:  # every planned row remains visible
        return {
            **_base(item, reference),
            "status": "failed",
            "method_status": "exception",
            "status_reason": "",
            "error": f"{type(error).__name__}: {error}",
            **_empty_metrics(int(item.width)),
        }


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute the frozen grid and preserve complete, failed, and limited rows."""

    references: dict[tuple[int, str], float] = {}
    rows = []
    for _, item in manifest.iterrows():
        key = (int(item.width), str(item.distribution))
        if key not in references:
            circuit, observable = _circuit_observable(*key)
            references[key] = exact_expectation(circuit, observable)
        rows.append(_result_row(item, references[key]))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy, uncertainty, resources, support, and solve performance."""

    rows = []
    keys = ["width", "distribution", "readout_model", "total_shot_budget", "method"]
    for key, group in raw.groupby(keys, sort=True):
        complete = group.loc[group.status == "complete"]
        errors = complete.sampling_error.to_numpy(dtype=float)
        raw_errors = (complete.raw_estimate - complete.reference).to_numpy(dtype=float)
        corrected_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else math.nan
        raw_std = float(np.std(raw_errors, ddof=1)) if len(raw_errors) > 1 else math.nan
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "expected_repeats": len(group),
                "complete_repeats": len(complete),
                "failed_repeats": int((group.status == "failed").sum()),
                "resource_limited_repeats": int((group.status == "resource_limited").sum()),
                "bias": float(np.mean(errors)) if len(errors) else math.nan,
                "variance": float(np.var(errors, ddof=1)) if len(errors) > 1 else math.nan,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
                "uncertainty_amplification": (
                    corrected_std / raw_std
                    if math.isfinite(corrected_std) and math.isfinite(raw_std) and raw_std > 0
                    else math.nan
                ),
                "decision_error_probability": (
                    float(complete.decision_error.mean()) if len(complete) else math.nan
                ),
                "mean_support_size": (
                    float(complete.observed_support_size.mean()) if len(complete) else math.nan
                ),
                "mean_support_fraction": (
                    float(complete.support_fraction.mean()) if len(complete) else math.nan
                ),
                "mean_discarded_probability_mass": (
                    float(complete.discarded_probability_mass.mean())
                    if len(complete)
                    else math.nan
                ),
                "mean_classical_storage_bytes": (
                    float(complete.classical_storage_bytes.mean()) if len(complete) else math.nan
                ),
                "working_vector_bytes": (
                    int(complete.working_vector_bytes.max()) if len(complete) else 0
                ),
                "mean_calibration_circuit_executions": (
                    float(complete.calibration_circuit_executions.mean())
                    if len(complete)
                    else math.nan
                ),
                "calibration_shots": (
                    int(complete.calibration_shots.iloc[0]) if len(complete) else 0
                ),
                "mean_solve_wall_seconds": (
                    float(complete.solve_wall_seconds.mean()) if len(complete) else math.nan
                ),
                "mean_iterations": (
                    float(complete.iterations.mean()) if len(complete) else math.nan
                ),
                "convergence_probability": (
                    float(complete.converged.mean()) if len(complete) else math.nan
                ),
                "failure_probability": float((group.status == "failed").mean()),
                "timing_kind": "measured",
                "result_label": "noisy_simulated",
            }
        )
    return pd.DataFrame(rows)


def _plot_by_distribution(
    summary: pd.DataFrame,
    column: str,
    ylabel: str,
    path: Path,
    title: str,
    *,
    log_scale: bool = False,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True)
    for axis, distribution in zip(axes, DISTRIBUTIONS, strict=True):
        values = summary.loc[
            (summary.distribution == distribution)
            & (summary.readout_model == "independent")
            & (summary.total_shot_budget == 16384)
        ]
        for method in METHODS:
            cells = values.loc[values.method == method].sort_values("width")
            axis.plot(cells.width, cells[column], marker="o", label=method)
        axis.set_title(distribution)
        axis.set_xlabel("Width")
        axis.set_ylabel(ylabel)
        if log_scale:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_results(summary: pd.DataFrame) -> None:
    """Write the four preregistered scalability views."""

    _plot_by_distribution(
        summary,
        "mean_classical_storage_bytes",
        "Mean classical storage (bytes)",
        MEMORY_FIGURE_PATH,
        "Dense readout storage becomes impractical with width",
        log_scale=True,
    )
    runtime = summary.loc[summary.complete_repeats > 0]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in METHODS:
        cells = runtime.loc[runtime.method == method]
        axis.scatter(
            cells.mean_support_size,
            cells.mean_solve_wall_seconds,
            label=method,
            alpha=0.75,
        )
    axis.set_xlabel("Mean support size")
    axis.set_ylabel("Mean measured solve time (seconds)")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    figure.suptitle("Classical solve time grows with retained support")
    figure.tight_layout()
    RUNTIME_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(RUNTIME_FIGURE_PATH, dpi=180)
    plt.close(figure)
    _plot_by_distribution(
        summary,
        "rmse",
        "Expectation RMSE",
        ACCURACY_FIGURE_PATH,
        "Accuracy does not scale uniformly across readout strategies",
        log_scale=True,
    )
    _plot_by_distribution(
        summary,
        "uncertainty_amplification",
        "Std. dev. ratio (corrected / raw)",
        UNCERTAINTY_FIGURE_PATH,
        "Mitigation can amplify finite-shot uncertainty",
        log_scale=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Rebuild figures from the committed processed summary without simulation",
    )
    parsed = parser.parse_args(argv)
    if parsed.plots_only:
        plot_results(pd.read_csv(SUMMARY_PATH))
        return
    raw = execute_manifest(build_manifest())
    summary = summarize(raw)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.17g")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.17g")
    plot_results(summary)


if __name__ == "__main__":
    main()
