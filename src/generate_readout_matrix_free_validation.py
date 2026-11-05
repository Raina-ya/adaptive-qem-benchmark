"""Generate matrix-free observed-subspace validation artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuits import ghz_circuit
from src.config import derive_seed
from src.hardware_efficient_circuits import hardware_efficient_circuit
from src.mitigation.readout_matrix_free import run_matrix_free_readout_mitigation
from src.noise_models import NoiseSpec
from src.observables import exact_expectation

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_matrix_free_validation_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/readout_matrix_free_validation_summary_v1.csv"
FIGURE_PATH = ROOT / "figures/readout_matrix_free_convergence.png"

REFERENCE_WIDTHS = (2, 4, 6)
PILOT_WIDTHS = (8, 10)
DISTRIBUTIONS = ("sparse", "broad")
READOUT_MODELS = ("independent", "asymmetric")
BUDGETS = (4096, 16384)
REFERENCE_REPEATS = 50
PILOT_REPEATS = 10
MIN_COUNT = 2


def _noise_spec(name: str) -> NoiseSpec:
    if name == "independent":
        return NoiseSpec(name, readout_p01=0.03, readout_p10=0.03)
    if name == "asymmetric":
        return NoiseSpec(name, readout_p01=0.015, readout_p10=0.06)
    raise ValueError(f"Unknown validation readout model: {name}")


def _circuit_observable(width: int, distribution: str):
    if distribution == "sparse":
        return ghz_circuit(width), "Z" * width
    if distribution == "broad":
        return hardware_efficient_circuit(width, 3), "Z" * width
    raise ValueError(f"Unknown distribution family: {distribution}")


def build_manifest(
    reference_repeats: int = REFERENCE_REPEATS,
    pilot_repeats: int = PILOT_REPEATS,
) -> pd.DataFrame:
    """Build deterministic matrix-free validation and pilot rows."""

    rows = []
    for width in (*REFERENCE_WIDTHS, *PILOT_WIDTHS):
        distributions = DISTRIBUTIONS if width in REFERENCE_WIDTHS else ("sparse",)
        repeats = reference_repeats if width in REFERENCE_WIDTHS else pilot_repeats
        for distribution_index, distribution in enumerate(distributions):
            for model_index, readout_model in enumerate(READOUT_MODELS):
                for budget in BUDGETS:
                    for repeat in range(repeats):
                        rows.append(
                            {
                                "width": width,
                                "distribution": distribution,
                                "readout_model": readout_model,
                                "total_shot_budget": budget,
                                "repeat": repeat,
                                "seed": derive_seed(
                                    "readout-matrix-free-validation",
                                    width,
                                    distribution_index,
                                    model_index,
                                    budget,
                                    repeat,
                                ),
                                "validation_role": (
                                    "small_width_validation"
                                    if width in REFERENCE_WIDTHS
                                    else "sparse_scalability_pilot"
                                ),
                                "status": "pending",
                            }
                        )
    return pd.DataFrame(rows)


def _result_row(item: pd.Series, reference: float) -> dict[str, object]:
    base = {
        "width": int(item.width),
        "distribution": str(item.distribution),
        "readout_model": str(item.readout_model),
        "total_shot_budget": int(item.total_shot_budget),
        "method": "matrix_free_subspace",
        "repeat": int(item["repeat"]),
        "seed": int(item.seed),
        "validation_role": str(item.validation_role),
        "reference": reference,
        "result_label": "noisy_simulated",
    }
    circuit, observable = _circuit_observable(int(item.width), str(item.distribution))
    try:
        result = run_matrix_free_readout_mitigation(
            circuit,
            observable,
            total_shot_budget=int(item.total_shot_budget),
            noise_spec=_noise_spec(str(item.readout_model)),
            seed=int(item.seed),
            min_count=MIN_COUNT,
        )
        estimate = result.corrected_estimate
        complete = result.status == "ok" and estimate is not None
        return {
            **base,
            "status": "complete" if complete else "failed",
            "method_status": result.status,
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
            "condition_estimate": result.condition_estimate,
            "assignment_dimension": result.assignment_dimension,
            "observed_support_size": result.observed_support_size,
            "support_fraction": result.observed_support_size / result.assignment_dimension,
            "operator_storage_bytes": result.operator_storage_bytes,
            "working_vector_bytes": result.working_vector_bytes,
            "materialized_matrix_entries": result.materialized_matrix_entries,
            "retained_probability_mass": result.retained_probability_mass,
            "discarded_probability_mass": result.discarded_probability_mass,
            "iterations": result.iterations,
            "converged": result.converged,
            "residual_norm": result.residual_norm,
            "total_shots_used": result.target_shots + result.calibration_shots,
            "target_shots": result.target_shots,
            "calibration_shots": result.calibration_shots,
            "target_circuit_executions": result.target_circuit_executions,
            "calibration_circuit_executions": result.calibration_circuit_executions,
            "raw_counts_json": result.raw_counts_json,
            "calibration_counts_json": result.calibration_counts_json,
            "support_states_json": result.support_states_json,
            "assignment_matrices_json": result.assignment_matrices_json,
            "corrected_probabilities_json": result.corrected_probabilities_json or "",
        }
    except Exception as error:  # preserve every failed planned row
        return {
            **base,
            "status": "failed",
            "method_status": "exception",
            "error": f"{type(error).__name__}: {error}",
            "raw_estimate": math.nan,
            "corrected_estimate": math.nan,
            "sampling_error": math.nan,
            "absolute_error": math.nan,
            "decision_error": math.nan,
            "condition_estimate": math.nan,
            "assignment_dimension": 2 ** int(item.width),
            "observed_support_size": 0,
            "support_fraction": math.nan,
            "operator_storage_bytes": 0,
            "working_vector_bytes": 8 * 2 ** int(item.width),
            "materialized_matrix_entries": 0,
            "retained_probability_mass": math.nan,
            "discarded_probability_mass": math.nan,
            "iterations": 0,
            "converged": False,
            "residual_norm": math.nan,
            "total_shots_used": 0,
            "target_shots": 0,
            "calibration_shots": 0,
            "target_circuit_executions": 0,
            "calibration_circuit_executions": 0,
            "raw_counts_json": "",
            "calibration_counts_json": "",
            "support_states_json": "",
            "assignment_matrices_json": "",
            "corrected_probabilities_json": "",
        }


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute every row and preserve failures and convergence diagnostics."""

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
    """Aggregate accuracy, support, resource, and convergence metrics."""

    rows = []
    keys = ["width", "distribution", "readout_model", "total_shot_budget", "method"]
    for key, group in raw.groupby(keys, sort=True):
        complete = group.loc[group.status == "complete"]
        errors = complete.sampling_error.to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "expected_repeats": len(group),
                "complete_repeats": len(complete),
                "failed_repeats": int((group.status == "failed").sum()),
                "bias": float(np.mean(errors)) if len(errors) else math.nan,
                "variance": float(np.var(errors, ddof=1)) if len(errors) > 1 else math.nan,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
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
                "mean_iterations": (
                    float(complete.iterations.mean()) if len(complete) else math.nan
                ),
                "maximum_iterations": (
                    int(complete.iterations.max()) if len(complete) else 0
                ),
                "mean_residual_norm": (
                    float(complete.residual_norm.mean()) if len(complete) else math.nan
                ),
                "convergence_probability": float(group.converged.mean()),
                "mean_operator_storage_bytes": (
                    float(complete.operator_storage_bytes.mean()) if len(complete) else math.nan
                ),
                "working_vector_bytes": int(group.working_vector_bytes.iloc[0]),
                "materialized_matrix_entries": int(group.materialized_matrix_entries.max()),
                "mean_calibration_circuit_executions": (
                    float(complete.calibration_circuit_executions.mean())
                    if len(complete)
                    else math.nan
                ),
                "calibration_shots": (
                    int(complete.calibration_shots.iloc[0]) if len(complete) else 0
                ),
                "failure_probability": float((group.status == "failed").mean()),
                "result_label": "noisy_simulated",
            }
        )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame) -> None:
    """Plot convergence iterations against retained support size."""

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for axis, distribution in zip(axes, DISTRIBUTIONS, strict=True):
        values = summary.loc[summary.distribution == distribution]
        for model, marker in zip(READOUT_MODELS, ("o", "s"), strict=True):
            subset = values.loc[values.readout_model == model]
            for budget, color in zip(BUDGETS, ("tab:blue", "tab:orange"), strict=True):
                cells = subset.loc[subset.total_shot_budget == budget].sort_values("width")
                axis.plot(
                    cells.mean_support_size,
                    cells.mean_iterations,
                    marker=marker,
                    color=color,
                    label=f"{model}, {budget}",
                )
                for row in cells.itertuples():
                    axis.annotate(
                        str(row.width),
                        (row.mean_support_size, row.mean_iterations),
                        fontsize=7,
                    )
        axis.set_title(f"{distribution} output distribution")
        axis.set_xlabel("Mean observed support size")
        axis.set_ylabel("Mean solver iterations")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.suptitle("Matrix-free convergence cost grows with observed support")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    raw = execute_manifest(build_manifest())
    summary = summarize(raw)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.17g")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.17g")
    plot_summary(summary)


if __name__ == "__main__":
    main()
