"""Audit the frozen selector by decision consequences rather than label accuracy."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.metrics import CONDITION_COLUMNS, benchmark_summary, wilson_interval
from src.regime_labels_v2 import (
    GHZ_REFERENCE,
    _with_labels,
    evaluate_selections,
    label_condition,
    noise_shift_raw,
)
from src.selector import select_qem_method

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/selector_audit_v1.yaml"
PROCESSED = ROOT / "results/processed"
AUDIT_PATH = PROCESSED / "selector_audit_v1.csv"
SUMMARY_PATH = PROCESSED / "selector_audit_summary_v1.csv"
LABELS_PATH = PROCESSED / "regime_labels_v2.csv"
EVALUATION_PATH = PROCESSED / "selector_evaluation_v2.csv"
FIGURE_PATH = ROOT / "figures/selector_regret_categories.png"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
NEGLIGIBLE = float(CONFIG["negligible_regret_max"])
CATASTROPHIC = float(CONFIG["catastrophic_regret_absolute"])
RELATIVE = float(CONFIG["catastrophic_regret_relative_to_absolute_reference"])
CATEGORIES = tuple(CONFIG["audit_categories"])
POLICIES = tuple(CONFIG["comparison_policies"])
ABSTAIN_CONDITION_CAP = 8.0
ABSTAIN_RESIDUAL_CAP = 0.03
COMPLETE_REPEATS = 50
CATEGORY_LABELS = {
    "correct_materially_useful": "Correct, useful",
    "correct_practically_tied": "Correct, tied",
    "incorrect_negligible_regret": "Incorrect, negligible",
    "incorrect_material_regret": "Incorrect, material",
    "catastrophic_regret": "Catastrophic",
    "abstention_avoided_failure": "Abstained, avoided failure",
    "unnecessary_abstention": "Abstained, unnecessary",
    "inconclusive": "Inconclusive",
}
CATEGORY_COLORS = {
    "correct_materially_useful": "#2a78d6",
    "correct_practically_tied": "#1baf7a",
    "incorrect_negligible_regret": "#d2ad00",
    "incorrect_material_regret": "#eb6834",
    "catastrophic_regret": "#a8323a",
    "abstention_avoided_failure": "#6250d6",
    "unnecessary_abstention": "#e87ba4",
    "inconclusive": "#9a9a9a",
}
DARK_TEXT_CATEGORIES = {
    "correct_practically_tied",
    "incorrect_negligible_regret",
    "unnecessary_abstention",
    "inconclusive",
}
SET_TITLES = {
    "rule_design": "Rule design (115 cells)",
    "holdout": "Holdout (29 cells)",
    "noise_shift": "Noise shift (36 cells)",
}
POLICY_LABELS = {
    "selector": "Selector",
    "always_raw": "Always raw",
    "always_readout": "Always readout",
    "always_zne": "Always ZNE",
}
INK = "#2B2B2B"
MUTED = "#6B6B6B"


def is_catastrophic(regret: float, reference: float) -> bool:
    """Apply the frozen absolute-or-relative catastrophic rule."""

    return regret >= CATASTROPHIC or regret >= abs(reference) * RELATIVE


def classify(row: dict[str, object], counterfactual: dict[str, float] | None) -> str:
    """Return the single audit category for one selection."""

    selected = row["selected_method"]
    if selected == "abstain":
        if counterfactual is None:
            raise ValueError("Abstention needs a counterfactual method")
        failed = counterfactual["complete_repeats"] < COMPLETE_REPEATS
        harmful = counterfactual["rmse_regret"] > NEGLIGIBLE
        return "abstention_avoided_failure" if failed or harmful else "unnecessary_abstention"
    if selected == "inconclusive" or row["oracle_label"] == "inconclusive":
        return "inconclusive"
    if row["decision_compatible"]:
        if row["oracle_label"] == "practical_tie":
            return "correct_practically_tied"
        return "correct_materially_useful"
    regret = float(row["rmse_regret"])
    if is_catastrophic(regret, float(row["reference"])):
        return "catastrophic_regret"
    return "incorrect_negligible_regret" if regret <= NEGLIGIBLE else "incorrect_material_regret"


def selector_inputs(summary: pd.DataFrame, condition: tuple[object, ...]) -> dict[str, object]:
    """Return the same cell-level diagnostics that the frozen evaluation supplied."""

    group = summary.set_index(CONDITION_COLUMNS).loc[condition]
    values = dict(zip(CONDITION_COLUMNS, condition, strict=True))

    def diagnostic(method: str, column: str) -> float | None:
        rows = group.loc[group.method == method]
        if rows.empty or pd.isna(rows[column].iloc[0]):
            return None
        return float(rows[column].iloc[0])

    raw_rows = group.loc[group.method == "raw"]
    return {
        "noise": str(values["noise"]),
        "shots": int(values["shots"]),
        "width": int(values["width"]),
        "depth": int(values["depth"]),
        "gate_count": int(raw_rows.folded_two_qubit_gate_count.iloc[0]) if len(raw_rows) else 0,
        "condition_number": diagnostic("readout", "condition_number_mean"),
        "fit_residual": diagnostic("zne", "fit_residual_mean"),
    }


def counterfactual_without_gates(inputs: dict[str, object]) -> str:
    """Return the selector's choice with only its two abstention gates disabled."""

    condition = inputs["condition_number"]
    residual = inputs["fit_residual"]
    selection = select_qem_method(
        **{
            **inputs,
            "condition_number": (
                None if condition is None else min(float(condition), ABSTAIN_CONDITION_CAP)
            ),
            "fit_residual": (
                None if residual is None else min(float(residual), ABSTAIN_RESIDUAL_CAP)
            ),
        }
    )
    return selection.method


def _policy_rows(selector_rows: pd.DataFrame) -> pd.DataFrame:
    keep = [*CONDITION_COLUMNS, "reference", "evaluation_set"]
    frames = [selector_rows.assign(policy="selector")]
    for policy, method in (
        ("always_raw", "raw"),
        ("always_readout", "readout"),
        ("always_zne", "zne"),
    ):
        frames.append(
            selector_rows[keep].assign(
                policy=policy,
                selected_method=method,
                rule_trace=f"fixed policy: {method}",
            )
        )
    return pd.concat(frames, ignore_index=True)


def audit_set(
    rows: pd.DataFrame, summary: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    """Score selector and fixed policies on one evaluation set with identical rules."""

    identity = [*CONDITION_COLUMNS, "selected_method", "rule_trace", "reference"]
    output: list[pd.DataFrame] = []
    for policy, group in _policy_rows(rows).groupby("policy", sort=False):
        scored = _with_labels(evaluate_selections(group[identity], summary), labels)
        scored = scored.assign(policy=policy, evaluation_set=rows.evaluation_set.iloc[0])
        output.append(scored)
    audited = pd.concat(output, ignore_index=True)
    lookup = summary.set_index([*CONDITION_COLUMNS, "method"])
    best = summary.groupby(CONDITION_COLUMNS).rmse.min()
    categories: list[str] = []
    fallback_methods: list[str | None] = []
    fallback_regrets: list[float] = []
    for row in audited.to_dict("records"):
        counterfactual = None
        fallback, regret = None, np.nan
        if row["selected_method"] == "abstain":
            condition = tuple(row[column] for column in CONDITION_COLUMNS)
            fallback = counterfactual_without_gates(selector_inputs(summary, condition))
            method_key = (*condition, fallback)
            if method_key in lookup.index:
                regret = float(lookup.loc[method_key, "rmse"] - best.loc[condition])
                repeats = float(lookup.loc[method_key, "complete_repeats"])
            else:
                repeats = 0.0
            counterfactual = {"complete_repeats": repeats, "rmse_regret": regret}
        categories.append(classify(row, counterfactual))
        fallback_methods.append(fallback)
        fallback_regrets.append(regret)
    audited["audit_category"] = categories
    audited["counterfactual_method"] = fallback_methods
    audited["counterfactual_regret"] = fallback_regrets
    audited["catastrophic_absolute"] = audited.rmse_regret >= CATASTROPHIC
    audited["catastrophic_relative"] = audited.rmse_regret >= audited.reference.abs() * RELATIVE
    audited["low_signal_descriptive"] = (
        audited.reference.abs() < float(CONFIG["descriptive_low_signal_reference"])
    )
    return audited


def summarize_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Report every category count, including zero counts, with Wilson intervals."""

    rows: list[dict[str, object]] = []
    for (name, policy), group in audit.groupby(["evaluation_set", "policy"], sort=False):
        regret = group.rmse_regret.dropna()
        shared = {
            "evaluation_set": name,
            "policy": policy,
            "cells": len(group),
            "decided_cells": len(regret),
            "mean_regret": float(regret.mean()),
            "median_regret": float(regret.median()),
            "p90_regret": float(regret.quantile(0.9)),
            "max_regret": float(regret.max()),
            "catastrophic_absolute_cells": int(group.catastrophic_absolute.sum()),
            "catastrophic_relative_cells": int(group.catastrophic_relative.sum()),
        }
        for category in CATEGORIES:
            count = int((group.audit_category == category).sum())
            low, high = wilson_interval(count, len(group))
            rows.append(
                {
                    **shared,
                    "audit_category": category,
                    "category_cells": count,
                    "category_rate": count / len(group),
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_categories(summary: pd.DataFrame, path: Path = FIGURE_PATH) -> None:
    """Plot outcome shares per policy, with a panel for each evaluation set."""

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for axis, (name, title) in zip(axes, SET_TITLES.items(), strict=True):
        frame = summary.loc[summary.evaluation_set == name]
        for position, policy in enumerate(POLICIES):
            left = 0.0
            for category in CATEGORIES:
                row = frame.loc[
                    (frame.policy == policy) & (frame.audit_category == category)
                ].iloc[0]
                width = float(row.category_rate)
                if width == 0.0:
                    continue
                axis.barh(
                    position,
                    width,
                    left=left,
                    height=0.62,
                    color=CATEGORY_COLORS[category],
                    edgecolor="white",
                    linewidth=1.5,
                )
                if width >= 0.08:
                    axis.text(
                        left + width / 2.0,
                        position,
                        str(int(row.category_cells)),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color=INK if category in DARK_TEXT_CATEGORIES else "white",
                    )
                left += width
        axis.set_yticks(range(len(POLICIES)), [POLICY_LABELS[item] for item in POLICIES])
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        axis.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left", "bottom"):
            axis.spines[side].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[item], label=CATEGORY_LABELS[item])
        for item in CATEGORIES
    ]
    figure.legend(handles=handles, loc="lower center", ncols=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Selector and fixed policies by decision consequence (segment labels are cell counts)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    summary = pd.read_csv(PROCESSED / "benchmark_summary_v1.csv")
    labels = pd.read_csv(LABELS_PATH)
    shift_raw = noise_shift_raw()
    shift_summary = benchmark_summary(shift_raw, resamples=1000)
    shift_labels = pd.DataFrame(
        label_condition(group)
        for _, group in shift_summary.groupby(CONDITION_COLUMNS, sort=True)
    )
    evaluation = pd.read_csv(EVALUATION_PATH)
    keep = [*CONDITION_COLUMNS, "selected_method", "rule_trace", "reference", "evaluation_set"]
    parts = []
    for name, frame_summary, frame_labels in (
        ("rule_design", summary, labels),
        ("holdout", summary, labels),
        ("noise_shift", shift_summary, shift_labels),
    ):
        rows = evaluation.loc[evaluation.evaluation_set == name, keep].reset_index(drop=True)
        parts.append(audit_set(rows, frame_summary, frame_labels))
    audit = pd.concat(parts, ignore_index=True)
    audit["reference"] = audit.reference.fillna(GHZ_REFERENCE)
    audit.to_csv(AUDIT_PATH, index=False, lineterminator="\n", float_format="%.12g")
    result = summarize_audit(audit)
    result.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.12g")
    plot_categories(result)


if __name__ == "__main__":
    main()
