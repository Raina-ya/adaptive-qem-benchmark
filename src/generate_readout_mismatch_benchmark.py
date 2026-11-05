"""Generate the correlated and calibration-mismatch readout benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import READOUT_CALIBRATION_FRACTION, derive_seed
from src.decision_metrics import decision_error_interval
from src.hardware_efficient_circuits import hardware_efficient_circuit
from src.mitigation.readout import (
    constrained_readout_correction,
    counts_probability_vector,
    estimate_assignment_matrix,
)
from src.mitigation.readout_matrix_free import (
    TensorSubspaceOperator,
    matrix_free_readout_ledger,
    solve_matrix_free_nnls,
    tensor_condition_estimate,
)
from src.mitigation.readout_subspace import (
    estimate_subspace_assignment_matrix,
    observed_subspace_correction,
    select_observed_support,
    subspace_readout_ledger,
)
from src.mitigation.readout_tensor import (
    estimate_tensor_assignment_matrices,
    tensor_product_correction,
    tensor_readout_ledger,
)
from src.observables import expectation_from_counts, expectation_from_probabilities
from src.readout_mismatch import (
    calibration_noise_spec,
    exact_model_mismatch_error,
    ideal_measurement_probabilities,
    sample_target_counts,
    target_readout_model,
)
from src.resource_accounting import readout_mitigation_ledger

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_mismatch_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/readout_mismatch_summary_v1.csv"
ACCURACY_FIGURE_PATH = ROOT / "figures/readout_mismatch_accuracy.png"
DECISION_FIGURE_PATH = ROOT / "figures/readout_mismatch_decision_error.png"
BIAS_FIGURE_PATH = ROOT / "figures/readout_mismatch_bias_vs_sampling.png"

WIDTHS = (2, 4, 6)
DISTRIBUTIONS = ("sparse", "broad")
TARGET_MODELS = (
    "independent",
    "asymmetric",
    "pair_correlated",
    "calibration_mismatch",
)
BUDGETS = (4096, 16384)
METHODS = ("full_matrix", "tensor_product", "observed_subspace", "matrix_free_subspace")
REPEATS = 50
MIN_COUNT = 2
DECISION_MARGIN = 0.05


def _circuit_observable(width: int, distribution: str) -> tuple[QuantumCircuit, str]:
    observable = "Z" + "I" * (width - 1)
    if distribution == "sparse":
        circuit = QuantumCircuit(width, name=f"sparse_readout_w{width}")
        circuit.ry(0.6, 0)
        if width > 2:
            circuit.ry(0.35, width - 1)
        return circuit, observable
    if distribution == "broad":
        return hardware_efficient_circuit(width, 3), observable
    raise ValueError(f"Unknown distribution family: {distribution}")


def build_manifest(repeats: int = REPEATS) -> pd.DataFrame:
    """Build paired target rows for every mitigation strategy."""

    rows = []
    for width in WIDTHS:
        for distribution_index, distribution in enumerate(DISTRIBUTIONS):
            for model_index, target_model in enumerate(TARGET_MODELS):
                for budget in BUDGETS:
                    for repeat in range(repeats):
                        pair_seed = derive_seed(
                            "readout-mismatch-benchmark",
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
                                    "target_model": target_model,
                                    "calibration_model": (
                                        "independent_symmetric"
                                        if target_model == "calibration_mismatch"
                                        else "independent_matched_marginals"
                                    ),
                                    "total_shot_budget": budget,
                                    "method": method,
                                    "repeat": repeat,
                                    "pair_seed": pair_seed,
                                    "status": "pending",
                                }
                            )
    return pd.DataFrame(rows)


def _base(item: pd.Series, reference: float, model_floor: float) -> dict[str, object]:
    direction = float(np.sign(reference)) if reference != 0.0 else 1.0
    decision_threshold = reference - direction * DECISION_MARGIN
    return {
        "width": int(item.width),
        "distribution": str(item.distribution),
        "target_model": str(item.target_model),
        "calibration_model": str(item.calibration_model),
        "total_shot_budget": int(item.total_shot_budget),
        "method": str(item.method),
        "repeat": int(item["repeat"]),
        "pair_seed": int(item.pair_seed),
        "reference": reference,
        "decision_threshold": decision_threshold,
        "exact_decision_margin": DECISION_MARGIN,
        "infinite_shot_model_error": model_floor,
        "result_label": "noisy_simulated",
    }


def _execute_method(
    item: pd.Series,
    raw_counts_json: str,
    reference: float,
    model_floor: float,
) -> dict[str, object]:
    width = int(item.width)
    budget = int(item.total_shot_budget)
    seed = int(item.pair_seed)
    calibration_spec = calibration_noise_spec(str(item.target_model))
    observed_full = counts_probability_vector(raw_counts_json, width)
    counts = json.loads(raw_counts_json)
    raw_estimate = expectation_from_counts(counts, "Z" + "I" * (width - 1))
    support = tuple(range(2**width))
    retained_mass, discarded_mass = 1.0, 0.0
    iterations, converged, residual_norm = 0, False, math.nan

    if item.method == "full_matrix":
        ledger = readout_mitigation_ledger(budget, width)
        assignment, _, calibration_json = estimate_assignment_matrix(
            width, ledger.calibration_shots, calibration_spec, seed=seed
        )
        corrected, condition, method_status = constrained_readout_correction(
            assignment, observed_full
        )
        storage = assignment.nbytes
    elif item.method == "tensor_product":
        ledger = tensor_readout_ledger(budget, width)
        matrices, _, calibration_json = estimate_tensor_assignment_matrices(
            width, ledger.calibration_shots, calibration_spec, seed=seed
        )
        corrected, condition, method_status = tensor_product_correction(
            matrices, observed_full
        )
        storage = sum(matrix.nbytes for matrix in matrices) + observed_full.nbytes
    elif item.method == "observed_subspace":
        selected = select_observed_support(raw_counts_json, width, min_count=MIN_COUNT)
        support = selected.states
        retained_mass = selected.retained_probability_mass
        discarded_mass = selected.discarded_probability_mass
        ledger = subspace_readout_ledger(budget, len(support))
        assignment, calibration_json = estimate_subspace_assignment_matrix(
            width, support, ledger.calibration_shots, calibration_spec, seed=seed
        )
        corrected, condition, method_status = observed_subspace_correction(
            assignment,
            observed_full[list(support)],
            support,
            full_dimension=2**width,
        )
        storage = assignment.nbytes
    else:
        selected = select_observed_support(raw_counts_json, width, min_count=MIN_COUNT)
        support = selected.states
        retained_mass = selected.retained_probability_mass
        discarded_mass = selected.discarded_probability_mass
        ledger = matrix_free_readout_ledger(budget, width)
        matrices, _, calibration_json = estimate_tensor_assignment_matrices(
            width, ledger.calibration_shots, calibration_spec, seed=seed
        )
        condition = tensor_condition_estimate(matrices)
        operator = TensorSubspaceOperator(matrices, support)
        solve = solve_matrix_free_nnls(operator, observed_full[list(support)])
        method_status = solve.status
        iterations = solve.iterations
        converged = solve.converged
        residual_norm = solve.residual_norm
        corrected = None
        if solve.solution is not None:
            corrected = np.zeros(2**width, dtype=float)
            corrected[list(support)] = solve.solution
        storage = sum(matrix.nbytes for matrix in matrices) + 8 * (2**width + len(support))

    estimate = (
        expectation_from_probabilities(corrected, "Z" + "I" * (width - 1))
        if corrected is not None
        else None
    )
    complete = method_status == "ok" and estimate is not None
    direction = float(np.sign(reference)) if reference != 0.0 else 1.0
    decision_threshold = reference - direction * DECISION_MARGIN
    return {
        **_base(item, reference, model_floor),
        "status": "complete" if complete else "failed",
        "method_status": method_status,
        "error": "",
        "raw_estimate": raw_estimate,
        "corrected_estimate": estimate if estimate is not None else math.nan,
        "sampling_error": estimate - reference if estimate is not None else math.nan,
        "absolute_error": abs(estimate - reference) if estimate is not None else math.nan,
        "decision_error": (
            float(
                np.sign(estimate - decision_threshold)
                != np.sign(reference - decision_threshold)
            )
            if complete
            else math.nan
        ),
        "condition_estimate": condition,
        "assignment_dimension": 2**width,
        "observed_support_size": len(support),
        "support_fraction": len(support) / 2**width,
        "classical_storage_bytes": storage,
        "retained_probability_mass": retained_mass,
        "discarded_probability_mass": discarded_mass,
        "iterations": iterations,
        "converged": converged,
        "residual_norm": residual_norm,
        "total_shots_used": ledger.target_shots + ledger.calibration_shots,
        "target_shots": ledger.target_shots,
        "calibration_shots": ledger.calibration_shots,
        "target_circuit_executions": ledger.target_circuit_executions,
        "calibration_circuit_executions": ledger.calibration_circuit_executions,
        "raw_counts_json": raw_counts_json,
        "calibration_counts_json": calibration_json,
        "support_states_json": json.dumps(support, separators=(",", ":")),
    }


def _failure_row(
    item: pd.Series, reference: float, model_floor: float, error: Exception
) -> dict[str, object]:
    return {
        **_base(item, reference, model_floor),
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
        "classical_storage_bytes": 0,
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
    }


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute paired target counts against method-specific calibration."""

    contexts: dict[tuple[int, str, str, int, int], tuple[str, float, float]] = {}
    rows = []
    for _, item in manifest.iterrows():
        key = (
            int(item.width),
            str(item.distribution),
            str(item.target_model),
            int(item.total_shot_budget),
            int(item["repeat"]),
        )
        if key not in contexts:
            circuit, observable = _circuit_observable(key[0], key[1])
            ideal = ideal_measurement_probabilities(circuit, observable)
            reference = expectation_from_probabilities(ideal, observable)
            calibration_spec = calibration_noise_spec(key[2])
            model_floor = exact_model_mismatch_error(
                ideal,
                observable,
                target_readout_model(key[2]),
                calibration_spec,
            )
            target_shots = key[3] - int(key[3] * READOUT_CALIBRATION_FRACTION)
            counts_json = sample_target_counts(
                ideal,
                target_readout_model(key[2]),
                shots=target_shots,
                seed=derive_seed("readout-target", int(item.pair_seed)),
            )
            contexts[key] = counts_json, reference, model_floor
        counts_json, reference, model_floor = contexts[key]
        try:
            rows.append(_execute_method(item, counts_json, reference, model_floor))
        except Exception as error:  # every planned method row remains visible
            rows.append(_failure_row(item, reference, model_floor, error))
    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Separate model bias, sampling spread, RMSE, decision risk, and overhead."""

    rows = []
    keys = [
        "width",
        "distribution",
        "target_model",
        "calibration_model",
        "total_shot_budget",
        "method",
    ]
    for key, group in raw.groupby(keys, sort=True):
        complete = group.loc[group.status == "complete"]
        errors = complete.sampling_error.to_numpy(dtype=float)
        decision = complete.decision_error.to_numpy(dtype=bool)
        if len(decision):
            probability, low, high = decision_error_interval(decision)
            low = min(low, probability)
            high = max(high, probability)
        else:
            probability = low = high = math.nan
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "expected_repeats": len(group),
                "complete_repeats": len(complete),
                "failed_repeats": int((group.status == "failed").sum()),
                "bias": float(np.mean(errors)) if len(errors) else math.nan,
                "sampling_std": (
                    float(np.std(errors, ddof=1)) if len(errors) > 1 else math.nan
                ),
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else math.nan,
                "mean_infinite_shot_model_error": (
                    float(complete.infinite_shot_model_error.mean())
                    if len(complete)
                    else math.nan
                ),
                "decision_error_probability": probability,
                "decision_error_wilson_low": low,
                "decision_error_wilson_high": high,
                "mean_support_size": (
                    float(complete.observed_support_size.mean()) if len(complete) else math.nan
                ),
                "mean_discarded_probability_mass": (
                    float(complete.discarded_probability_mass.mean())
                    if len(complete)
                    else math.nan
                ),
                "mean_classical_storage_bytes": (
                    float(complete.classical_storage_bytes.mean()) if len(complete) else math.nan
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


def plot_results(summary: pd.DataFrame) -> None:
    """Plot accuracy, decision error, and bias-versus-sampling separation."""

    selected = summary.loc[
        (summary.width == 6)
        & (summary.distribution == "broad")
        & (summary.total_shot_budget == 16384)
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(TARGET_MODELS))
    width = 0.18
    for index, method in enumerate(METHODS):
        cells = selected.loc[selected.method == method].set_index("target_model")
        axis.bar(
            x + (index - 1.5) * width,
            cells.loc[list(TARGET_MODELS), "rmse"],
            width,
            label=method,
        )
    axis.set_xticks(x, TARGET_MODELS, rotation=15)
    axis.set_ylabel("Expectation RMSE")
    axis.set_title("Calibration mismatch changes mitigation accuracy")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=7)
    figure.tight_layout()
    ACCURACY_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ACCURACY_FIGURE_PATH, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    for method in METHODS:
        cells = selected.loc[selected.method == method].set_index("target_model")
        values = cells.loc[list(TARGET_MODELS)]
        axis.errorbar(
            x,
            values.decision_error_probability,
            yerr=np.vstack(
                [
                    (values.decision_error_probability - values.decision_error_wilson_low).clip(
                        lower=0.0
                    ),
                    (values.decision_error_wilson_high - values.decision_error_probability).clip(
                        lower=0.0
                    ),
                ]
            ),
            marker="o",
            capsize=3,
            label=method,
        )
    axis.set_xticks(x, TARGET_MODELS, rotation=15)
    axis.set_ylabel("Decision-error probability (95% Wilson interval)")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Readout-model mismatch can change downstream decisions")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(DECISION_FIGURE_PATH, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    for axis, distribution in zip(axes, DISTRIBUTIONS, strict=True):
        cells = summary.loc[
            (summary.width == 6)
            & (summary.distribution == distribution)
            & (summary.total_shot_budget == 16384)
        ]
        for method in METHODS:
            values = cells.loc[cells.method == method]
            axis.scatter(values.bias.abs().clip(lower=1e-12), values.sampling_std, label=method)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(distribution)
        axis.set_xlabel("Absolute bias")
        axis.set_ylabel("Sampling standard deviation")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    figure.suptitle("Approximation bias and sampling uncertainty are distinct")
    figure.tight_layout()
    figure.savefig(BIAS_FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    raw = execute_manifest(build_manifest())
    summary = summarize(raw)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.17g")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.17g")
    plot_results(summary)


if __name__ == "__main__":
    main()
