"""Apply the frozen Holm correction and null rule to the new repeats."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, norm

from src.confirmatory_qaoa_v1 import RAW_PATH as QAOA_PATH
from src.confirmatory_v1 import CONFIG, RAW_PATH
from src.power_analysis_v1 import PLAN_PATH, paired_interval

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results/processed/confirmatory_results_v1.csv"


def holm_flags(values: list[float], alpha: float) -> list[bool]:
    """Return whether each p-value survives Holm at family level alpha."""

    order = sorted(range(len(values)), key=values.__getitem__)
    survive = [False] * len(values)
    for rank, index in enumerate(order):
        if values[index] > alpha / (len(values) - rank):
            break
        survive[index] = True
    return survive


def compare_outcomes(first: pd.Series, second: pd.Series, delta: float, alpha: float) -> dict:
    """Compare matched bad-outcome series, and return a p-value and an equivalence check."""

    joined = pd.concat([first.rename("a"), second.rename("b")], axis=1).dropna()
    a, b = joined.a.astype(bool), joined.b.astype(bool)
    first_only, second_only = int((a & ~b).sum()), int((~a & b).sum())
    pairs = len(joined)
    discordant = first_only + second_only
    p_value = binomtest(first_only, discordant, 0.5).pvalue if discordant else 1.0
    low, high = paired_interval(first_only, second_only, pairs, float(norm.ppf(1.0 - alpha)))
    return {
        "paired_repeats": pairs,
        "first_bad_rate": float(a.mean()) if pairs else math.nan,
        "second_bad_rate": float(b.mean()) if pairs else math.nan,
        "first_only_bad": first_only,
        "second_only_bad": second_only,
        "p_value": float(p_value),
        "equivalence_low": low,
        "equivalence_high": high,
        "inside_margin": bool(low >= -delta and high <= delta),
    }


def label(survives: bool, inside: bool) -> str:
    if survives:
        return "resolved_difference"
    return "close_within_margin" if inside else "underpowered"


def _exact_series(frame: pd.DataFrame, row, method: str, group: str) -> pd.Series:
    part = frame.loc[
        (frame.group == group)
        & (frame.task == row.task)
        & (frame.margin == row.margin)
        & (frame.noise == row.noise)
        & (frame.budget == row.budget)
        & (frame.method == method)
    ]
    return part.set_index("repeat").decision_error


def selection_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Score QAOA selection per repeat. A failed candidate leaves that repeat unresolved."""

    rows = []
    keys = ["graph_id", "total_selection_budget", "method", "repeat"]
    for key, group in frame.groupby(keys, sort=True):
        complete = group.status.eq("complete").all()
        if complete:
            best = group.exact_ideal_energy.max()
            pick = group.sort_values(
                ["corrected_or_extrapolated_estimate", "candidate_id"], ascending=[False, True]
            ).iloc[0]
            error = best - pick.exact_ideal_energy > 1e-12
        else:
            error = float("nan")
        rows.append({**dict(zip(keys, key, strict=True)), "selection_error": error})
    return pd.DataFrame(rows)


def main() -> None:
    plan = pd.read_csv(PLAN_PATH)
    exact = pd.read_csv(RAW_PATH)
    qaoa = (
        selection_errors(pd.read_csv(QAOA_PATH, keep_default_na=False))
        if QAOA_PATH.exists()
        else pd.DataFrame()
    )
    rows = []
    family_alpha = float(plan.alpha_per_comparison.iloc[0])
    for row in plan.itertuples():
        identity = {
            "group": row.group,
            "role": row.role,
            "cell": row.cell,
            "first": row.first,
            "second": row.second,
            "minimum_difference": row.minimum_difference,
            "pairs_planned": row.pairs_planned,
            "reaches_power_at_cap": row.reaches_power_at_cap,
        }
        if row.engine == "aer":
            if qaoa.empty:
                rows.append({**identity, "status": "not_run"})
                continue
            part = qaoa.loc[
                (qaoa.graph_id == row.task) & (qaoa.total_selection_budget == row.budget)
            ]
            series = [
                part.loc[part.method == method].set_index("repeat").selection_error
                for method in (row.first, row.second)
            ]
        else:
            series = [_exact_series(exact, row, m, "confirmatory") for m in (row.first, row.second)]
        rows.append(
            {
                **identity,
                "status": "run",
                **compare_outcomes(*series, row.minimum_difference, family_alpha),
            }
        )
    delta = float(CONFIG["holdout_minimum_difference"])
    hold = exact.loc[exact.group == "holdout"]
    holdout_rows = []
    for (task, margin, noise, budget), group in hold.groupby(
        ["task", "margin", "noise", "budget"], sort=True
    ):
        for first, second in CONFIG["holdout_comparisons"]:
            series = [
                group.loc[group.method == method].set_index("repeat").decision_error
                for method in (first, second)
            ]
            holdout_rows.append(
                {
                    "group": "holdout",
                    "role": "holdout_confirmation",
                    "cell": f"{task}|{margin}|{noise}|{budget}",
                    "first": first,
                    "second": second,
                    "minimum_difference": delta,
                    "pairs_planned": int(CONFIG["holdout_repeats"]),
                    "reaches_power_at_cap": True,
                    "status": "run",
                    **compare_outcomes(*series, delta, 0.05),
                }
            )
    table = pd.DataFrame(rows)
    run = table.status == "run"
    survive = holm_flags(table.loc[run, "p_value"].tolist(), 0.05) if run.any() else []
    table["classification"] = "not_run"
    table.loc[run, "classification"] = [
        label(flag, inside)
        for flag, inside in zip(survive, table.loc[run, "inside_margin"], strict=True)
    ]
    holdout = pd.DataFrame(holdout_rows)
    survive = holm_flags(holdout.p_value.tolist(), 0.05)
    holdout["classification"] = [
        label(flag, inside) for flag, inside in zip(survive, holdout.inside_margin, strict=True)
    ]
    pd.concat([table, holdout], ignore_index=True).to_csv(
        RESULTS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )


if __name__ == "__main__":
    main()
