"""Generate cost-aware regime, selector, and decision-error analyses."""

from __future__ import annotations

import json
import shutil
from functools import cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuits import ghz_circuit
from src.config import derive_seed
from src.generate_days08_12 import _write_canonical_csv
from src.metrics import CONDITION_COLUMNS, benchmark_summary, wilson_interval
from src.mitigation.folding import folded_metadata
from src.regime_map import build_regime_labels
from src.selector import select_qem_method

RAW = Path("results/raw")
PROCESSED = Path("results/processed")
FIGURES = Path("figures")


@cache
def _ghz_resource(width: int, method: str) -> tuple[int, int, int]:
    circuit = ghz_circuit(width)
    factors = (1, 3, 5) if method == "zne" else (1,)
    metadata = [
        folded_metadata(circuit, factor, seed=derive_seed("ghz-analysis-resource", width, factor))
        for factor in factors
    ]
    target_executions = len(factors)
    calibration_executions = 2**width if method == "readout" else 0
    return (
        target_executions + calibration_executions,
        sum(item["transpiled_two_qubit_gate_count"] for item in metadata),
        max(item["transpiled_depth"] for item in metadata),
    )


def _payload_value(payload: object, key: str) -> float | bool | None:
    if not isinstance(payload, str) or not payload:
        return None
    parsed = json.loads(payload)
    return parsed.get(key)


def _normalize_ghz(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["circuit_id"] = frame.width.map(lambda width: f"ghz_w{int(width)}")
    frame["noiseless_statevector_reference"] = frame.analytic_reference
    frame["condition_number"] = frame.raw_payload_json.map(
        lambda payload: _payload_value(payload, "condition_number")
    )
    frame["fit_residual_sum_squares"] = frame.raw_payload_json.map(
        lambda payload: _payload_value(payload, "residual_sum_squares")
    )
    frame["out_of_range"] = frame.raw_payload_json.map(
        lambda payload: _payload_value(payload, "out_of_range")
    )
    resources = [
        _ghz_resource(int(row.width), str(row.method)) for row in frame.itertuples()
    ]
    frame["total_circuit_executions"] = [item[0] for item in resources]
    frame["folded_two_qubit_gate_count"] = [item[1] for item in resources]
    frame["transpiled_depth"] = [item[2] for item in resources]
    return frame


def _selector_predictions(summary: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition, group in summary.groupby(CONDITION_COLUMNS, sort=True):
        condition_data = dict(zip(CONDITION_COLUMNS, condition, strict=True))
        readout = group.loc[group.method == "readout"]
        zne = group.loc[group.method == "zne"]
        raw = group.loc[group.method == "raw"]
        condition_number = (
            float(readout.condition_number_mean.iloc[0])
            if not readout.empty and pd.notna(readout.condition_number_mean.iloc[0])
            else None
        )
        fit_residual = (
            float(zne.fit_residual_mean.iloc[0])
            if not zne.empty and pd.notna(zne.fit_residual_mean.iloc[0])
            else None
        )
        gate_count = int(raw.folded_two_qubit_gate_count.iloc[0]) if not raw.empty else 0
        selection = select_qem_method(
            noise=str(condition_data["noise"]),
            shots=int(condition_data["shots"]),
            width=int(condition_data["width"]),
            depth=int(condition_data["depth"]),
            gate_count=gate_count,
            condition_number=condition_number,
            fit_residual=fit_residual,
        )
        rows.append(
            {
                **condition_data,
                "condition_number": condition_number,
                "fit_residual": fit_residual,
                "gate_count": gate_count,
                "selected_method": selection.method,
                "rule_trace": selection.trace,
            }
        )
    return pd.DataFrame(rows).merge(labels, on=CONDITION_COLUMNS, how="left")


def _evaluate_predictions(predictions: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    rmse_lookup = summary.set_index(CONDITION_COLUMNS + ["method"])["rmse"].to_dict()
    rows: list[dict[str, object]] = []
    for row in predictions.itertuples():
        condition = tuple(getattr(row, column) for column in CONDITION_COLUMNS)
        selected_rmse = rmse_lookup.get((*condition, row.selected_method))
        oracle_rmse = rmse_lookup.get((*condition, row.oracle_method))
        equivalent_methods = json.loads(row.equivalent_methods_json)
        strict_label_match = row.selected_method == row.oracle_label
        if row.oracle_label == "practical_tie":
            decision_compatible = row.selected_method in equivalent_methods
        elif row.oracle_label == "inconclusive":
            decision_compatible = row.selected_method in {"abstain", "inconclusive"}
        else:
            decision_compatible = strict_label_match
        rows.append(
            {
                **{column: getattr(row, column) for column in CONDITION_COLUMNS},
                "split": getattr(row, "split", "holdout"),
                "oracle_label": row.oracle_label,
                "oracle_method": row.oracle_method,
                "selected_method": row.selected_method,
                "rule_trace": row.rule_trace,
                "strict_label_match": strict_label_match,
                "decision_compatible": decision_compatible,
                "selected_rmse": selected_rmse,
                "oracle_rmse": oracle_rmse,
                "rmse_regret": (
                    float(selected_rmse - oracle_rmse)
                    if selected_rmse is not None and oracle_rmse is not None
                    else np.nan
                ),
                "inconclusive_or_abstain": row.selected_method
                in {"inconclusive", "abstain"},
            }
        )
    return pd.DataFrame(rows)


def _decision_error_summary(results: pd.DataFrame) -> pd.DataFrame:
    complete = results.loc[results.status == "complete"].copy()
    complete["decision_error"] = np.sign(complete.estimate) != np.sign(
        complete.noiseless_statevector_reference
    )
    rows: list[dict[str, object]] = []
    for condition, group in complete.groupby(CONDITION_COLUMNS + ["method"], sort=True):
        failures = int(group.decision_error.sum())
        lower, upper = wilson_interval(failures, len(group))
        rows.append(
            {
                **dict(zip(CONDITION_COLUMNS + ["method"], condition, strict=True)),
                "decision_threshold": 0.0,
                "decision_errors": failures,
                "trials": len(group),
                "decision_error_probability": failures / len(group),
                "wilson_ci_low": lower,
                "wilson_ci_high": upper,
            }
        )
    return pd.DataFrame(rows)


def _plot_rmse_heatmap(summary: pd.DataFrame, path: Path) -> None:
    table = summary.pivot_table(
        index="method", columns=["family", "noise"], values="rmse", aggfunc="mean"
    )
    figure, axis = plt.subplots(figsize=(8.5, 3.8))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xticks(
        range(len(table.columns)),
        [f"{family}\n{noise}" for family, noise in table.columns],
    )
    axis.set_title("RMSE depends on circuit family and noise regime")
    figure.colorbar(image, ax=axis, label="RMSE")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_bias_variance(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for method, group in summary.groupby("method", sort=True):
        axis.scatter(group.bias.abs(), group.variance, alpha=0.6, label=method)
    axis.set(xlabel="Absolute bias", ylabel="Variance")
    axis.set_title("Lower bias can coincide with higher variance")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_overhead_gain(summary: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    for method, group in summary.groupby("method", sort=True):
        axis.scatter(group.execution_overhead, group.rmse_gain_vs_raw, alpha=0.6, label=method)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Circuit executions", ylabel="Paired RMSE gain over raw")
    axis.set_title("Mitigation overhead does not guarantee accuracy gain")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_regime(labels: pd.DataFrame, path: Path) -> None:
    counts = labels.groupby(["noise", "oracle_label"]).size().unstack(fill_value=0)
    figure, axis = plt.subplots(figsize=(7.4, 4.4))
    counts.plot(kind="bar", stacked=True, ax=axis)
    axis.set(xlabel="Noise condition", ylabel="Operating cells")
    axis.set_title("Cost-aware regimes include ties and inconclusive cells")
    axis.legend(title="Oracle label", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_confusion(evaluation: pd.DataFrame, path: Path) -> None:
    labels = sorted(set(evaluation.oracle_label) | set(evaluation.selected_method))
    table = pd.crosstab(evaluation.oracle_label, evaluation.selected_method).reindex(
        index=labels, columns=labels, fill_value=0
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(table.to_numpy(), cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel="Selector", ylabel="Cost-aware oracle")
    axis.set_title("Frozen selector holdout outcomes")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, int(table.iloc[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Cells")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_regret(evaluation: pd.DataFrame, path: Path, title: str) -> None:
    usable = evaluation.dropna(subset=["rmse_regret"])
    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    if usable.empty:
        axis.text(0.5, 0.5, "No method-valued selector decisions", ha="center", va="center")
    else:
        grouped = usable.groupby("selected_method").rmse_regret.apply(list)
        axis.boxplot(grouped.tolist(), tick_labels=grouped.index)
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Selected method", ylabel="RMSE regret")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_decision_error(summary: pd.DataFrame, path: Path) -> None:
    aggregate = (
        summary.groupby(["noise", "method"], sort=True)
        .decision_error_probability.mean()
        .reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for method, group in aggregate.groupby("method", sort=True):
        axis.plot(group.noise, group.decision_error_probability, marker="o", label=method)
    axis.set(xlabel="Noise shift", ylabel="Decision-error probability", ylim=(0.0, 1.0))
    axis.set_title("Expectation accuracy and decision reliability are distinct")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ghz = _normalize_ghz(RAW / "day12_ghz_volumetric_v1_repeats-50.csv")
    hardware = pd.read_csv(RAW / "hardware_efficient_v1_repeats-50.csv")
    combined = pd.concat([ghz, hardware], ignore_index=True, sort=False)
    summary = benchmark_summary(combined, resamples=1000)
    _write_canonical_csv(summary, PROCESSED / "benchmark_summary_v1.csv")
    labels = build_regime_labels(summary)
    _write_canonical_csv(labels, PROCESSED / "regime_labels_v1.csv")
    predictions = _selector_predictions(summary, labels)
    design_predictions = predictions.loc[predictions.split == "rule_design"]
    _write_canonical_csv(
        design_predictions, PROCESSED / "selector_rule_design_predictions.csv"
    )
    holdout_predictions = predictions.loc[predictions.split == "holdout"]
    holdout_evaluation = _evaluate_predictions(holdout_predictions, summary)
    _write_canonical_csv(holdout_evaluation, PROCESSED / "selector_holdout_evaluation.csv")

    shifted = pd.read_csv(RAW / "noise_shift_holdout_v1_repeats-50.csv")
    shifted_summary = benchmark_summary(shifted, resamples=1000)
    shifted_labels = build_regime_labels(shifted_summary)
    shifted_predictions = _selector_predictions(shifted_summary, shifted_labels)
    shifted_evaluation = _evaluate_predictions(shifted_predictions, shifted_summary)
    _write_canonical_csv(shifted_evaluation, PROCESSED / "noise_shift_regret.csv")

    decision = _decision_error_summary(shifted)
    _write_canonical_csv(decision, PROCESSED / "decision_error_summary.csv")

    core = pd.DataFrame(
        [
            {"metric": "operating_cells", "value": len(labels)},
            {"metric": "holdout_cells", "value": len(holdout_evaluation)},
            {
                "metric": "holdout_selector_accuracy",
                "value": float(holdout_evaluation.decision_compatible.mean()),
            },
            {
                "metric": "holdout_strict_label_match",
                "value": float(holdout_evaluation.strict_label_match.mean()),
            },
            {
                "metric": "holdout_inconclusive_or_abstain_rate",
                "value": float(holdout_evaluation.inconclusive_or_abstain.mean()),
            },
            {
                "metric": "noise_shift_selector_accuracy",
                "value": float(shifted_evaluation.decision_compatible.mean()),
            },
            {
                "metric": "noise_shift_strict_label_match",
                "value": float(shifted_evaluation.strict_label_match.mean()),
            },
            {
                "metric": "noise_shift_inconclusive_or_abstain_rate",
                "value": float(shifted_evaluation.inconclusive_or_abstain.mean()),
            },
        ]
    )
    _write_canonical_csv(core, PROCESSED / "core_v1_summary.csv")

    _plot_rmse_heatmap(summary, FIGURES / "rmse_heatmap.png")
    _plot_bias_variance(summary, FIGURES / "bias_variance_tradeoff.png")
    _plot_overhead_gain(summary, FIGURES / "overhead_vs_rmse_gain.png")
    _plot_regime(labels, FIGURES / "operating_regime_map_v1.png")
    _plot_confusion(holdout_evaluation, FIGURES / "selector_confusion_matrix.png")
    _plot_regret(
        holdout_evaluation,
        FIGURES / "selector_rmse_regret.png",
        "In-distribution holdout regret",
    )
    _plot_regret(
        shifted_evaluation,
        FIGURES / "regret_under_noise_shift.png",
        "Noise-model mismatch regret",
    )
    _plot_decision_error(decision, FIGURES / "decision_error_probability.png")
    shutil.copyfile(
        FIGURES / "operating_regime_map_v1.png", FIGURES / "core_v1_regime_map.png"
    )


if __name__ == "__main__":
    main()
