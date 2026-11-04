"""Generate the frozen full-matrix versus tensor-product validation study."""

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
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.readout_tensor import run_tensor_readout_mitigation
from src.noise_models import NoiseSpec
from src.observables import exact_expectation

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_tensor_validation_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/readout_tensor_validation_summary_v1.csv"
FIGURE_PATH = ROOT / "figures/readout_tensor_vs_full.png"

WIDTHS = (2, 4, 6)
DISTRIBUTIONS = ("sparse", "broad")
READOUT_MODELS = ("independent", "asymmetric")
BUDGETS = (4096, 16384)
METHODS = ("full_matrix", "tensor_product")
REPEATS = 20


def _noise_spec(name: str) -> NoiseSpec:
    if name == "independent":
        return NoiseSpec(name, readout_p01=0.03, readout_p10=0.03)
    if name == "asymmetric":
        return NoiseSpec(name, readout_p01=0.015, readout_p10=0.06)
    raise ValueError(f"Unknown validation readout model: {name}")


def _circuit_observable(width: int, distribution: str):
    if distribution == "sparse":
        return ghz_circuit(width), "X" * width
    if distribution == "broad":
        return hardware_efficient_circuit(width, 3), hardware_efficient_observable(width)
    raise ValueError(f"Unknown distribution family: {distribution}")


def build_manifest(repeats: int = REPEATS) -> pd.DataFrame:
    """Build paired full/tensor rows before any outcomes are examined."""

    rows = []
    for width in WIDTHS:
        for distribution_index, distribution in enumerate(DISTRIBUTIONS):
            for model_index, readout_model in enumerate(READOUT_MODELS):
                for budget in BUDGETS:
                    for repeat in range(repeats):
                        pair_seed = derive_seed(
                            "readout-tensor-validation",
                            width,
                            distribution_index,
                            model_index,
                            budget,
                            repeat,
                        )
                        for method in METHODS:
                            rows.append(
                                {
                                    "width": width,
                                    "distribution": distribution,
                                    "readout_model": readout_model,
                                    "total_shot_budget": budget,
                                    "method": method,
                                    "repeat": repeat,
                                    "pair_seed": pair_seed,
                                    "status": "pending",
                                }
                            )
    return pd.DataFrame(rows)


def _result_row(item: pd.Series, reference: float) -> dict[str, object]:
    circuit, observable = _circuit_observable(int(item.width), str(item.distribution))
    spec = _noise_spec(str(item.readout_model))
    common = {
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
            negative_mass = (
                math.fsum(
                    max(-float(value), 0.0)
                    for value in json.loads(result.corrected_probabilities_json)
                )
                if result.corrected_probabilities_json is not None
                else math.nan
            )
            assignment_json = result.assignment_matrix_json
        else:
            result = run_tensor_readout_mitigation(
                circuit,
                observable,
                total_shot_budget=int(item.total_shot_budget),
                noise_spec=spec,
                seed=int(item.pair_seed),
            )
            estimate = result.corrected_estimate
            condition = result.condition_estimate
            negative_mass = (
                result.negative_probability_mass
                if result.negative_probability_mass is not None
                else math.nan
            )
            assignment_json = result.assignment_matrices_json
        complete = result.status == "ok" and estimate is not None
        return {
            **common,
            "status": "complete" if complete else "failed",
            "method_status": result.status,
            "error": "",
            "raw_estimate": result.raw_estimate,
            "corrected_estimate": estimate if estimate is not None else math.nan,
            "sampling_error": estimate - reference if estimate is not None else math.nan,
            "absolute_error": abs(estimate - reference) if estimate is not None else math.nan,
            "paired_delta_vs_full": math.nan,
            "condition_estimate": condition,
            "negative_probability_mass": negative_mass,
            "total_shots_used": result.target_shots + result.calibration_shots,
            "target_shots": result.target_shots,
            "calibration_shots": result.calibration_shots,
            "target_circuit_executions": result.target_circuit_executions,
            "calibration_circuit_executions": result.calibration_circuit_executions,
            "raw_counts_json": result.raw_counts_json,
            "calibration_counts_json": result.calibration_counts_json,
            "assignment_model_json": assignment_json,
            "corrected_probabilities_json": result.corrected_probabilities_json or "",
        }
    except Exception as exc:  # explicit row-level failure preservation
        return {
            **common,
            "status": "failed",
            "method_status": "exception",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_estimate": math.nan,
            "corrected_estimate": math.nan,
            "sampling_error": math.nan,
            "absolute_error": math.nan,
            "paired_delta_vs_full": math.nan,
            "condition_estimate": math.nan,
            "negative_probability_mass": math.nan,
            "total_shots_used": 0,
            "target_shots": 0,
            "calibration_shots": 0,
            "target_circuit_executions": 0,
            "calibration_circuit_executions": 0,
            "raw_counts_json": "",
            "calibration_counts_json": "",
            "assignment_model_json": "",
            "corrected_probabilities_json": "",
        }


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute paired rows and retain every failure and intermediate record."""

    references = {}
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
    estimates = raw.pivot(index=keys, columns="method", values="corrected_estimate")
    paired_delta = estimates["tensor_product"] - estimates["full_matrix"]
    tensor_mask = raw.method == "tensor_product"
    raw.loc[tensor_mask, "paired_delta_vs_full"] = [
        paired_delta.loc[tuple(row)]
        for row in raw.loc[tensor_mask, keys].itertuples(index=False, name=None)
    ]
    raw.loc[raw.method == "full_matrix", "paired_delta_vs_full"] = 0.0
    return raw


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Separate sampling performance, paired method delta, and resource counts."""

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
                "mean_paired_delta_vs_full": (
                    float(complete.paired_delta_vs_full.mean()) if len(complete) else math.nan
                ),
                "mean_condition_estimate": (
                    float(complete.condition_estimate.mean()) if len(complete) else math.nan
                ),
                "mean_negative_probability_mass": (
                    float(complete.negative_probability_mass.mean())
                    if len(complete)
                    else math.nan
                ),
                "calibration_circuit_executions": (
                    int(complete.calibration_circuit_executions.iloc[0])
                    if len(complete)
                    else 0
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
    """Plot RMSE by width without merging sparse and broad distributions."""

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True, sharey=True)
    for axis, (distribution, readout_model) in zip(
        axes.flat,
        [(d, n) for d in DISTRIBUTIONS for n in READOUT_MODELS],
        strict=True,
    ):
        subset = summary.loc[
            (summary.distribution == distribution)
            & (summary.readout_model == readout_model)
        ]
        for method in METHODS:
            for budget, linestyle in zip(BUDGETS, ("--", "-"), strict=True):
                values = subset.loc[
                    (subset.method == method) & (subset.total_shot_budget == budget)
                ].sort_values("width")
                axis.plot(
                    values.width,
                    values.rmse,
                    marker="o",
                    linestyle=linestyle,
                    label=f"{method}, {budget}",
                )
        axis.set_title(f"{distribution}; {readout_model}")
        axis.set_xlabel("Width")
        axis.set_ylabel("Expectation RMSE")
        axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7)
    figure.suptitle("Tensor-product correction is validated against full assignment matrices")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    manifest = build_manifest()
    raw = execute_manifest(manifest)
    summary = summarize(raw)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.17g")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.17g")
    plot_summary(summary)


if __name__ == "__main__":
    main()
