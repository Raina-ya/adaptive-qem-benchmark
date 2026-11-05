"""Generate the observed-subspace validation and sparse-width pilot."""

from __future__ import annotations

import json
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
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.readout_subspace import run_observed_subspace_mitigation
from src.noise_models import NoiseSpec
from src.observables import exact_expectation

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_subspace_validation_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/readout_subspace_validation_summary_v1.csv"
FIGURE_PATH = ROOT / "figures/readout_support_bias_tradeoff.png"

REFERENCE_WIDTHS = (2, 4, 6)
PILOT_WIDTHS = (8, 10)
DISTRIBUTIONS = ("sparse", "broad")
READOUT_MODELS = ("independent", "asymmetric")
BUDGETS = (4096, 16384)
METHODS = ("full_matrix", "observed_subspace")
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
    """Build the frozen small-width comparison and sparse large-width pilot."""

    rows: list[dict[str, object]] = []
    for width in (*REFERENCE_WIDTHS, *PILOT_WIDTHS):
        distributions = DISTRIBUTIONS if width in REFERENCE_WIDTHS else ("sparse",)
        repeats = reference_repeats if width in REFERENCE_WIDTHS else pilot_repeats
        methods = METHODS if width in REFERENCE_WIDTHS else ("observed_subspace",)
        for distribution_index, distribution in enumerate(distributions):
            for model_index, readout_model in enumerate(READOUT_MODELS):
                for budget in BUDGETS:
                    for repeat in range(repeats):
                        pair_seed = derive_seed(
                            "readout-subspace-validation",
                            width,
                            distribution_index,
                            model_index,
                            budget,
                            repeat,
                        )
                        for method in methods:
                            rows.append(
                                {
                                    "width": width,
                                    "distribution": distribution,
                                    "readout_model": readout_model,
                                    "total_shot_budget": budget,
                                    "method": method,
                                    "repeat": repeat,
                                    "pair_seed": pair_seed,
                                    "validation_role": (
                                        "full_matrix_reference"
                                        if width in REFERENCE_WIDTHS
                                        else "sparse_scalability_pilot"
                                    ),
                                    "status": "pending",
                                }
                            )
    return pd.DataFrame(rows)


def _base_row(item: pd.Series, reference: float) -> dict[str, object]:
    return {
        "width": int(item.width),
        "distribution": str(item.distribution),
        "readout_model": str(item.readout_model),
        "total_shot_budget": int(item.total_shot_budget),
        "method": str(item.method),
        "repeat": int(item["repeat"]),
        "pair_seed": int(item.pair_seed),
        "validation_role": str(item.validation_role),
        "reference": reference,
        "result_label": "noisy_simulated",
    }


def _failure_row(item: pd.Series, reference: float, error: Exception) -> dict[str, object]:
    return {
        **_base_row(item, reference),
        "status": "failed",
        "method_status": "exception",
        "error": f"{type(error).__name__}: {error}",
        "raw_estimate": math.nan,
        "corrected_estimate": math.nan,
        "sampling_error": math.nan,
        "absolute_error": math.nan,
        "decision_error": math.nan,
        "paired_delta_vs_full": math.nan,
        "condition_estimate": math.nan,
        "assignment_dimension": 2 ** int(item.width),
        "observed_support_size": 0,
        "support_fraction": math.nan,
        "dense_matrix_bytes": 8 * (2 ** int(item.width)) ** 2,
        "reduced_matrix_bytes": 0,
        "retained_probability_mass": math.nan,
        "discarded_probability_mass": math.nan,
        "negative_probability_mass": math.nan,
        "total_shots_used": 0,
        "target_shots": 0,
        "calibration_shots": 0,
        "target_circuit_executions": 0,
        "calibration_circuit_executions": 0,
        "raw_counts_json": "",
        "calibration_counts_json": "",
        "support_states_json": "",
        "assignment_matrix_json": "",
        "corrected_probabilities_json": "",
    }


def _result_row(item: pd.Series, reference: float) -> dict[str, object]:
    circuit, observable = _circuit_observable(int(item.width), str(item.distribution))
    spec = _noise_spec(str(item.readout_model))
    dimension = 2 ** int(item.width)
    try:
        if item.method == "full_matrix":
            result = run_readout_mitigation(
                circuit,
                observable,
                total_shot_budget=int(item.total_shot_budget),
                noise_spec=spec,
                seed=int(item.pair_seed),
            )
            estimate = result.corrected_estimate
            condition = result.condition_number
            support_size = dimension
            reduced_bytes = 8 * dimension**2
            retained_mass = 1.0
            discarded_mass = 0.0
            support_json = json.dumps(tuple(range(dimension)), separators=(",", ":"))
            assignment_json = result.assignment_matrix_json
            negative_mass = (
                math.fsum(
                    max(-float(value), 0.0)
                    for value in json.loads(result.corrected_probabilities_json)
                )
                if result.corrected_probabilities_json is not None
                else math.nan
            )
        else:
            result = run_observed_subspace_mitigation(
                circuit,
                observable,
                total_shot_budget=int(item.total_shot_budget),
                noise_spec=spec,
                seed=int(item.pair_seed),
                min_count=MIN_COUNT,
            )
            estimate = result.corrected_estimate
            condition = result.condition_estimate
            support_size = result.observed_support_size
            reduced_bytes = result.reduced_matrix_bytes
            retained_mass = result.retained_probability_mass
            discarded_mass = result.discarded_probability_mass
            support_json = result.support_states_json
            assignment_json = result.assignment_matrix_json
            negative_mass = (
                result.negative_probability_mass
                if result.negative_probability_mass is not None
                else math.nan
            )
        complete = result.status == "ok" and estimate is not None
        decision_error = (
            float(np.sign(estimate) != np.sign(reference))
            if complete and reference != 0.0
            else math.nan
        )
        return {
            **_base_row(item, reference),
            "status": "complete" if complete else "failed",
            "method_status": result.status,
            "error": "",
            "raw_estimate": result.raw_estimate,
            "corrected_estimate": estimate if estimate is not None else math.nan,
            "sampling_error": estimate - reference if estimate is not None else math.nan,
            "absolute_error": abs(estimate - reference) if estimate is not None else math.nan,
            "decision_error": decision_error,
            "paired_delta_vs_full": math.nan,
            "condition_estimate": condition,
            "assignment_dimension": dimension,
            "observed_support_size": support_size,
            "support_fraction": support_size / dimension,
            "dense_matrix_bytes": 8 * dimension**2,
            "reduced_matrix_bytes": reduced_bytes,
            "retained_probability_mass": retained_mass,
            "discarded_probability_mass": discarded_mass,
            "negative_probability_mass": negative_mass,
            "total_shots_used": result.target_shots + result.calibration_shots,
            "target_shots": result.target_shots,
            "calibration_shots": result.calibration_shots,
            "target_circuit_executions": result.target_circuit_executions,
            "calibration_circuit_executions": result.calibration_circuit_executions,
            "raw_counts_json": result.raw_counts_json,
            "calibration_counts_json": result.calibration_counts_json,
            "support_states_json": support_json,
            "assignment_matrix_json": assignment_json,
            "corrected_probabilities_json": result.corrected_probabilities_json or "",
        }
    except Exception as error:  # preserve every planned row and failure reason
        return _failure_row(item, reference, error)


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute every planned row while preserving paired raw target records."""

    references: dict[tuple[int, str], float] = {}
    rows = []
    for _, item in manifest.iterrows():
        key = (int(item.width), str(item.distribution))
        if key not in references:
            circuit, observable = _circuit_observable(*key)
            references[key] = exact_expectation(circuit, observable)
        rows.append(_result_row(item, references[key]))
    raw = pd.DataFrame(rows)
    keys = [
        "width",
        "distribution",
        "readout_model",
        "total_shot_budget",
        "repeat",
    ]
    reference_rows = raw.loc[raw.width.isin(REFERENCE_WIDTHS)]
    estimates = reference_rows.pivot(
        index=keys, columns="method", values="corrected_estimate"
    )
    paired_delta = estimates.observed_subspace - estimates.full_matrix
    subspace_mask = raw.method == "observed_subspace"
    for index, row in raw.loc[subspace_mask & raw.width.isin(REFERENCE_WIDTHS)].iterrows():
        raw.loc[index, "paired_delta_vs_full"] = paired_delta.loc[
            tuple(row[key] for key in keys)
        ]
    raw.loc[raw.method == "full_matrix", "paired_delta_vs_full"] = 0.0
    return raw


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy, decision, support, and resource metrics separately."""

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
                "mean_paired_delta_vs_full": (
                    float(complete.paired_delta_vs_full.mean())
                    if len(complete) and complete.paired_delta_vs_full.notna().any()
                    else math.nan
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
                "mean_condition_estimate": (
                    float(complete.condition_estimate.mean()) if len(complete) else math.nan
                ),
                "dense_matrix_bytes": int(group.dense_matrix_bytes.iloc[0]),
                "mean_reduced_matrix_bytes": (
                    float(complete.reduced_matrix_bytes.mean()) if len(complete) else math.nan
                ),
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
    """Plot the support reduction and bias tradeoff without merging distributions."""

    subspace = summary.loc[summary.method == "observed_subspace"]
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for axis, distribution in zip(axes, DISTRIBUTIONS, strict=True):
        values = subspace.loc[subspace.distribution == distribution]
        for model, marker in zip(READOUT_MODELS, ("o", "s"), strict=True):
            subset = values.loc[values.readout_model == model]
            for budget, color in zip(BUDGETS, ("tab:blue", "tab:orange"), strict=True):
                cells = subset.loc[subset.total_shot_budget == budget].sort_values("width")
                axis.plot(
                    cells.mean_support_fraction,
                    cells.rmse,
                    marker=marker,
                    color=color,
                    label=f"{model}, {budget}",
                )
                for row in cells.itertuples():
                    axis.annotate(str(row.width), (row.mean_support_fraction, row.rmse), fontsize=7)
        axis.set_title(f"{distribution} output distribution")
        axis.set_xlabel("Mean retained support / full dimension")
        axis.set_ylabel("Expectation RMSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.suptitle("Observed-subspace savings can trade support size for estimator error")
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
