"""Corrected oracle labels and regret for the frozen selector, keeping v1 unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import CONDITION_COLUMNS, benchmark_summary, wilson_interval
from src.regime_map import assign_holdout

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/selector_audit_v1.yaml"
LABELS_PATH = ROOT / "results/processed/regime_labels_v2.csv"
EVALUATION_PATH = ROOT / "results/processed/selector_evaluation_v2.csv"
COMPARISON_PATH = ROOT / "results/processed/regime_v1_v2_comparison.csv"
PROCESSED = ROOT / "results/processed"
SHIFT_RAW_PATH = ROOT / "results/raw/noise_shift_holdout_v1_repeats-50.csv"
METHODS = {"raw", "readout", "zne"}
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PRACTICAL = float(CONFIG["practical_rmse_threshold"])
PENALTY = float(CONFIG["overhead_penalty"])
GHZ_REFERENCE = 1.0


def label_condition(group: pd.DataFrame) -> dict[str, object]:
    """Label one cell, changing only how a decisive raw winner is named."""

    row: dict[str, object] = {column: group[column].iloc[0] for column in CONDITION_COLUMNS}
    if "split" in group:
        row["split"] = str(group.split.iloc[0])
    complete = group.loc[group.complete_repeats > 0].copy()
    if set(complete.method) != METHODS or (complete.complete_repeats != 50).any():
        row.update(
            oracle_label="inconclusive",
            oracle_method=None,
            equivalent_methods_json="[]",
            utility_gap=np.nan,
            raw_win_basis=None,
            label_reason="one or more methods lack 50 complete repeats",
        )
        return row
    maximum = max(float(complete.execution_overhead.max()), 1.0)
    complete["utility"] = complete.rmse + PENALTY * complete.execution_overhead / maximum
    ordered = complete.sort_values(["utility", "method"])
    winner, runner_up = ordered.iloc[0], ordered.iloc[1]
    raw = complete.loc[complete.method == "raw"].iloc[0]
    gap = float(runner_up.utility - winner.utility)
    equivalent = sorted(
        ordered.loc[ordered.utility <= float(winner.utility) + PRACTICAL, "method"].tolist()
    )
    basis = None
    if gap < PRACTICAL:
        label, reason = "practical_tie", "best two cost-aware utilities are practically equivalent"
    elif winner.method == "raw":
        label = "raw"
        reason = "raw has the lowest cost-aware utility by at least the practical threshold"
        basis = "accuracy" if float(runner_up.rmse_gain_ci_high) < 0.0 else "cost_only"
    elif float(winner.rmse_gain_ci_low) <= 0.0 <= float(winner.rmse_gain_ci_high):
        label, reason = "inconclusive", "paired RMSE-gain interval includes zero"
    elif float(raw.rmse - winner.rmse) < PRACTICAL:
        label, reason = "practical_tie", "accuracy gain over raw is below the practical threshold"
    else:
        label, reason = str(winner.method), "lowest cost-aware utility with conclusive gain"
    row.update(
        oracle_label=label,
        oracle_method=str(winner.method),
        equivalent_methods_json=json.dumps(equivalent, separators=(",", ":")),
        utility_gap=gap,
        raw_win_basis=basis,
        label_reason=reason,
    )
    return row


def build_labels_v2(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one corrected label per operating cell."""

    split = assign_holdout(summary)
    return pd.DataFrame(
        label_condition(group) for _, group in split.groupby(CONDITION_COLUMNS, sort=True)
    )


def evaluate_selections(selections: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Compute nonnegative RMSE regret and keep abstaining selections as rows."""

    rmse = summary.pivot_table(index=CONDITION_COLUMNS, columns="method", values="rmse")
    overhead = summary.pivot_table(
        index=CONDITION_COLUMNS, columns="method", values="execution_overhead"
    )
    repeats = summary.pivot_table(
        index=CONDITION_COLUMNS, columns="method", values="complete_repeats"
    )
    rows: list[dict[str, object]] = []
    for row in selections.to_dict("records"):
        key = tuple(row[column] for column in CONDITION_COLUMNS)
        selected = str(row["selected_method"])
        available = rmse.loc[key].dropna()
        best = str(available.idxmin())
        chosen = selected in available.index
        rows.append(
            {
                **row,
                "best_rmse_method": best,
                "best_rmse": float(available.min()),
                "selected_rmse": float(available[selected]) if chosen else np.nan,
                "rmse_regret": float(available[selected] - available.min()) if chosen else np.nan,
                "overhead_difference": (
                    float(overhead.loc[key, selected] - overhead.loc[key, best])
                    if chosen
                    else np.nan
                ),
                "selected_complete_repeats": (
                    float(repeats.loc[key, selected]) if chosen else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _compatible(row: pd.Series) -> bool:
    if row.oracle_label == "practical_tie":
        return row.selected_method in json.loads(row.equivalent_methods_json)
    if row.oracle_label == "inconclusive":
        return row.selected_method in {"abstain", "inconclusive"}
    return row.selected_method == row.oracle_label


def _with_labels(evaluation: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *CONDITION_COLUMNS,
        "oracle_label",
        "oracle_method",
        "equivalent_methods_json",
        "raw_win_basis",
    ]
    merged = evaluation.merge(labels[columns], on=CONDITION_COLUMNS, how="left")
    merged["decision_compatible"] = merged.apply(_compatible, axis=1)
    merged["strict_label_match"] = merged.selected_method == merged.oracle_label
    return merged


def _references(raw: pd.DataFrame) -> dict[str, float]:
    return (
        raw.groupby("circuit_id").noiseless_statevector_reference.first().astype(float).to_dict()
    )


def _comparison_rows(
    v1_labels: pd.DataFrame,
    labels: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add(metric: str, v1_value: float | None, v2_value: float) -> None:
        rows.append({"metric": metric, "v1_value": v1_value, "v2_value": v2_value})

    for label in ("practical_tie", "raw", "readout", "zne", "inconclusive"):
        add(
            f"cells_{label}",
            float((v1_labels.oracle_label == label).sum()),
            float((labels.oracle_label == label).sum()),
        )
    v1_sources = {
        "holdout": pd.read_csv(PROCESSED / "selector_holdout_evaluation.csv"),
        "noise_shift": pd.read_csv(PROCESSED / "noise_shift_regret.csv"),
    }
    for name, v1_frame in v1_sources.items():
        frame = evaluation.loc[evaluation.evaluation_set == name]
        count = int(frame.decision_compatible.sum())
        low, high = wilson_interval(count, len(frame))
        add(f"{name}_cells", float(len(v1_frame)), float(len(frame)))
        add(
            f"{name}_decision_compatibility",
            float(v1_frame.decision_compatible.mean()),
            count / len(frame),
        )
        add(f"{name}_compatibility_wilson_low", None, low)
        add(f"{name}_compatibility_wilson_high", None, high)
        add(
            f"{name}_strict_label_match",
            float(v1_frame.strict_label_match.mean()),
            float(frame.strict_label_match.mean()),
        )
        add(
            f"{name}_mean_regret",
            float(v1_frame.rmse_regret.mean()),
            float(frame.rmse_regret.mean()),
        )
        add(
            f"{name}_median_regret",
            float(v1_frame.rmse_regret.median()),
            float(frame.rmse_regret.median()),
        )
        add(
            f"{name}_negative_regret_cells",
            float((v1_frame.rmse_regret < 0.0).sum()),
            float((frame.rmse_regret < 0.0).sum()),
        )
        add(f"{name}_abstain_cells", None, float(frame.rmse_regret.isna().sum()))
    return pd.DataFrame(rows)


def main() -> None:
    summary = pd.read_csv(PROCESSED / "benchmark_summary_v1.csv")
    v1_labels = pd.read_csv(PROCESSED / "regime_labels_v1.csv")
    labels = build_labels_v2(summary)
    labels.to_csv(LABELS_PATH, index=False, lineterminator="\n", float_format="%.12g")

    shift_columns = [
        "width",
        "depth",
        "noise",
        "shots",
        "method",
        "repeat",
        "status",
        "family",
        "circuit_id",
        "estimate",
        "noiseless_statevector_reference",
        "total_circuit_executions",
        "folded_two_qubit_gate_count",
        "condition_number",
        "fit_residual_sum_squares",
        "out_of_range",
    ]
    shift_raw = pd.read_csv(SHIFT_RAW_PATH, usecols=shift_columns)
    shift_summary = benchmark_summary(shift_raw, resamples=1000)
    shift_labels = pd.DataFrame(
        label_condition(group)
        for _, group in shift_summary.groupby(CONDITION_COLUMNS, sort=True)
    )

    hardware_references = _references(
        pd.read_csv(
            ROOT / "results/raw/hardware_efficient_v1_repeats-50.csv",
            usecols=["circuit_id", "noiseless_statevector_reference"],
        )
    )
    selector_columns = [*CONDITION_COLUMNS, "selected_method", "rule_trace"]
    design = pd.read_csv(PROCESSED / "selector_rule_design_predictions.csv")[selector_columns]
    holdout = pd.read_csv(PROCESSED / "selector_holdout_evaluation.csv")[selector_columns]
    shifted = pd.read_csv(PROCESSED / "noise_shift_regret.csv")[selector_columns]

    def reference(frame: pd.DataFrame, table: dict[str, float]) -> pd.DataFrame:
        return frame.assign(reference=frame.circuit_id.map(table).fillna(GHZ_REFERENCE))

    parts = [
        _with_labels(
            evaluate_selections(reference(design, hardware_references), summary), labels
        ).assign(evaluation_set="rule_design"),
        _with_labels(
            evaluate_selections(reference(holdout, hardware_references), summary), labels
        ).assign(evaluation_set="holdout"),
        _with_labels(
            evaluate_selections(reference(shifted, _references(shift_raw)), shift_summary),
            shift_labels,
        ).assign(evaluation_set="noise_shift"),
    ]
    evaluation = pd.concat(parts, ignore_index=True)
    evaluation.to_csv(EVALUATION_PATH, index=False, lineterminator="\n", float_format="%.12g")
    _comparison_rows(v1_labels, labels, evaluation).to_csv(
        COMPARISON_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )


if __name__ == "__main__":
    main()
