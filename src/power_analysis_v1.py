"""Size confirmatory comparisons from committed paired outcomes. This runs no circuits.

Every comparison is labelled one of three ways, so a null result says what it means:
- resolved_difference: the interval for the rate difference excludes zero.
- close_within_margin: the 90% interval lies inside the stated minimum difference.
- underpowered: neither, so the data cannot separate "close" from "not enough repeats".
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.qaoa_mechanism import decisions, load_rows

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
COMPARISONS_PATH = PROCESSED / "power_comparisons_v1.csv"
PLAN_PATH = ROOT / "experiments/manifests/confirmatory_v1_manifest.csv"
PLAN_HASH_PATH = ROOT / "experiments/manifests/confirmatory_v1_manifest.sha256"
ANALYSIS_ROLE = "exploratory_power_inputs"
DECISION_DELTA = 0.10
SELECTION_DELTA = 0.15
ALPHA = 0.05
POWER = 0.80
Z_95 = float(norm.ppf(0.975))
Z_90 = float(norm.ppf(0.95))
NEAR_TIE_MARGIN = 0.05
EXACT_ENGINE_CAP = 4000
AER_ENGINE_CAP = 400


def paired_interval(first_only: int, second_only: int, pairs: int, z: float) -> tuple[float, float]:
    """Return the Agresti-Min interval for second-rate minus first-rate of an outcome.

    Half a count is added to every cell, so zero discordant pairs still give a real width.
    """

    b, c, n = first_only + 0.5, second_only + 0.5, pairs + 2.0
    center = (c - b) / n
    spread = z * math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    return center - spread, center + spread


def required_pairs(discordant_rate: float, delta: float, alpha: float, power: float) -> int:
    """Return pairs for a two-sided McNemar test to detect delta at this discordance."""

    rate = max(discordant_rate, delta)
    z_alpha, z_power = float(norm.ppf(1.0 - alpha / 2.0)), float(norm.ppf(power))
    return math.ceil(
        (z_alpha * math.sqrt(rate) + z_power * math.sqrt(rate - delta**2)) ** 2 / delta**2
    )


def classify(low95: float, high95: float, low90: float, high90: float, delta: float) -> str:
    if low95 > 0.0 or high95 < 0.0:
        return "resolved_difference"
    if low90 >= -delta and high90 <= delta:
        return "close_within_margin"
    return "underpowered"


def compare(
    first: pd.Series, second: pd.Series, delta: float, identity: dict[str, object]
) -> dict[str, object]:
    """Compare two matched boolean bad-outcome series, indexed by repeat."""

    joined = pd.concat([first.rename("a"), second.rename("b")], axis=1).dropna()
    a, b = joined.a.astype(bool), joined.b.astype(bool)
    pairs = len(joined)
    first_only, second_only = int((a & ~b).sum()), int((~a & b).sum())
    low95, high95 = paired_interval(first_only, second_only, pairs, Z_95)
    low90, high90 = paired_interval(first_only, second_only, pairs, Z_90)
    rate = (first_only + second_only + 1.0) / (pairs + 2.0)
    return {
        **identity,
        "minimum_difference": delta,
        "expected_pairs": max(len(first), len(second)),
        "paired_repeats": pairs,
        "unpaired_excluded": max(len(first), len(second)) - pairs,
        "first_only_bad": first_only,
        "second_only_bad": second_only,
        "first_bad_rate": float(a.mean()) if pairs else math.nan,
        "second_bad_rate": float(b.mean()) if pairs else math.nan,
        "discordant_rate": rate,
        "difference_low95": low95,
        "difference_high95": high95,
        "difference_low90": low90,
        "difference_high90": high90,
        "classification": classify(low95, high95, low90, high90, delta),
        "pairs_for_80_power": required_pairs(rate, delta, ALPHA, POWER),
    }


def decision_margin_rows() -> list[dict[str, object]]:
    frame = pd.read_csv(
        ROOT / "results/raw/decision_margin_benchmark_v1.csv",
        usecols=["margin", "noise", "shots", "method", "repeat", "status", "decision_error"],
    )
    rows = []
    for (margin, noise, shots), group in frame.groupby(["margin", "noise", "shots"], sort=True):
        series = {
            method: part.loc[part.status == "complete"].set_index("repeat").decision_error
            for method, part in group.groupby("method")
        }
        for first, second in (("raw", "readout"), ("raw", "zne"), ("readout", "zne")):
            identity = {
                "source": "decision_margin_v1",
                "engine": "exact_channel_sampling",
                "cell": f"symmetric_gap|{margin}|{noise}|{shots}",
                "task": "symmetric_gap",
                "margin": margin,
                "noise": noise,
                "budget": int(shots),
                "first": first,
                "second": second,
                "outcome": "decision_error",
            }
            rows.append(compare(series[first], series[second], DECISION_DELTA, identity))
    return rows


def sequential_rows() -> list[dict[str, object]]:
    frame = pd.read_csv(
        ROOT / "results/raw/sequential_decision_v1.csv",
        usecols=[
            "task",
            "margin",
            "noise",
            "max_total_shots",
            "method",
            "policy",
            "repeat",
            "forced_error",
        ],
        low_memory=False,
    )
    frame = frame.loc[(frame.policy == "fixed_budget") & (frame.task != "symmetric_gap")]
    rows = []
    for (task, margin, noise, shots), group in frame.groupby(
        ["task", "margin", "noise", "max_total_shots"], sort=True
    ):
        series = {
            method: part.set_index("repeat").forced_error.map(
                {True: True, False: False, "True": True, "False": False}
            )
            for method, part in group.groupby("method")
        }
        for first, second in (("raw", "readout"), ("raw", "zne"), ("readout", "zne")):
            identity = {
                "source": "sequential_decision_v1",
                "engine": "exact_channel_sampling",
                "cell": f"{task}|{margin}|{noise}|{shots}",
                "task": task,
                "margin": margin,
                "noise": noise,
                "budget": int(shots),
                "first": first,
                "second": second,
                "outcome": "decision_error",
            }
            rows.append(compare(series[first], series[second], DECISION_DELTA, identity))
    return rows


def selection_rows() -> list[dict[str, object]]:
    chosen = decisions(load_rows())
    rows = []
    for (graph, budget), group in chosen.groupby(["graph_id", "total_selection_budget"], sort=True):
        series = {
            method: ~part.set_index("repeat").correct.astype(bool)
            for method, part in group.groupby("method")
        }
        for first, second in (("raw", "readout"), ("raw", "zne"), ("readout", "zne")):
            if first not in series or second not in series:
                continue
            identity = {
                "source": "qaoa_selection_v1",
                "engine": "aer",
                "cell": f"{graph}|{budget}",
                "task": graph,
                "margin": math.nan,
                "noise": "mixed_medium",
                "budget": int(budget),
                "first": first,
                "second": second,
                "outcome": "selection_error",
            }
            rows.append(compare(series[first], series[second], SELECTION_DELTA, identity))
    return rows


def role_of(row: pd.Series) -> str:
    """Name why a comparison would earn more repeats, or return none."""

    if row.task == "medium_dense8" and row.first == "raw" and row.second == "zne":
        return "core_counterexample"
    if row.outcome == "selection_error":
        rates = (row.first_bad_rate, row.second_bad_rate)
        return "rank_uncertain" if any(0.1 <= rate <= 0.9 for rate in rates) else "none"
    return "near_tie" if row.margin <= NEAR_TIE_MARGIN else "none"


def build_plan(table: pd.DataFrame) -> pd.DataFrame:
    """Keep only underpowered or core comparisons with a role, sized with Bonferroni."""

    frame = table.copy()
    frame["role"] = frame.apply(role_of, axis=1)
    keep = (frame.role != "none") & (
        (frame.classification == "underpowered") | (frame.role == "core_counterexample")
    )
    plan = frame.loc[keep].copy()
    count = len(plan)
    plan["family_size"] = count
    plan["alpha_per_comparison"] = ALPHA / count
    plan["pairs_required"] = [
        required_pairs(rate, delta, ALPHA / count, POWER)
        for rate, delta in zip(plan.discordant_rate, plan.minimum_difference, strict=True)
    ]
    cap = plan.engine.map({"exact_channel_sampling": EXACT_ENGINE_CAP, "aer": AER_ENGINE_CAP})
    plan["pairs_planned"] = np.minimum(plan.pairs_required, cap)
    plan["reaches_power_at_cap"] = plan.pairs_required <= cap
    plan["group"] = "confirmatory"
    columns = [
        "group",
        "role",
        "source",
        "engine",
        "cell",
        "task",
        "margin",
        "noise",
        "budget",
        "first",
        "second",
        "outcome",
        "minimum_difference",
        "discordant_rate",
        "family_size",
        "alpha_per_comparison",
        "pairs_required",
        "pairs_planned",
        "reaches_power_at_cap",
    ]
    return plan[columns].sort_values(["role", "cell", "first", "second"]).reset_index(drop=True)


def main() -> None:
    table = pd.DataFrame([*decision_margin_rows(), *sequential_rows(), *selection_rows()])
    table = table.assign(analysis_role=ANALYSIS_ROLE)
    table.to_csv(COMPARISONS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    build_plan(table).to_csv(PLAN_PATH, index=False, lineterminator="\n", float_format="%.12g")
    digest = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()
    PLAN_HASH_PATH.write_text(digest + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
