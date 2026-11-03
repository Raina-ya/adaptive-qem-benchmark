"""Cost-aware oracle labels and deterministic holdout assignment."""

from __future__ import annotations

import json
import math

import pandas as pd

from src.config import derive_seed
from src.metrics import CONDITION_COLUMNS

PRACTICAL_RMSE_THRESHOLD = 0.005
OVERHEAD_PENALTY = 0.01


def assign_holdout(summary: pd.DataFrame) -> pd.DataFrame:
    """Reserve at least one fifth of operating conditions before rule design."""

    conditions = summary[CONDITION_COLUMNS].drop_duplicates().copy()
    conditions["holdout_score"] = [
        derive_seed(
            "regime-holdout",
            int(row.width),
            int(row.depth),
            int(row.shots),
            sum(ord(character) for character in f"{row.family}:{row.noise}"),
        )
        for row in conditions.itertuples()
    ]
    holdout_count = max(1, math.ceil(0.20 * len(conditions)))
    holdout_indices = conditions.nsmallest(holdout_count, "holdout_score").index
    conditions["split"] = "rule_design"
    conditions.loc[holdout_indices, "split"] = "holdout"
    return summary.merge(
        conditions.drop(columns="holdout_score"),
        on=CONDITION_COLUMNS,
        how="left",
        validate="many_to_one",
    )


def build_regime_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """Label winners, practical ties, and statistically inconclusive cells."""

    split_summary = assign_holdout(summary)
    rows: list[dict[str, object]] = []
    for condition, group in split_summary.groupby(CONDITION_COLUMNS, sort=True):
        row = dict(zip(CONDITION_COLUMNS, condition, strict=True))
        row["split"] = str(group.split.iloc[0])
        complete = group.loc[group.complete_repeats > 0].copy()
        if set(complete.method) != {"raw", "readout", "zne"} or (
            complete.complete_repeats != 50
        ).any():
            row.update(
                oracle_label="inconclusive",
                oracle_method=None,
                equivalent_methods_json="[]",
                utility_gap=None,
                label_reason="one or more methods lack 50 complete repeats",
            )
            rows.append(row)
            continue
        maximum_overhead = max(float(complete.execution_overhead.max()), 1.0)
        complete["cost_aware_utility"] = complete.rmse + OVERHEAD_PENALTY * (
            complete.execution_overhead / maximum_overhead
        )
        ordered = complete.sort_values(["cost_aware_utility", "method"])
        winner = ordered.iloc[0]
        runner_up = ordered.iloc[1]
        raw = complete.loc[complete.method == "raw"].iloc[0]
        utility_gap = float(runner_up.cost_aware_utility - winner.cost_aware_utility)
        equivalent_methods = sorted(
            ordered.loc[
                ordered.cost_aware_utility
                <= float(winner.cost_aware_utility) + PRACTICAL_RMSE_THRESHOLD,
                "method",
            ].tolist()
        )
        if utility_gap < PRACTICAL_RMSE_THRESHOLD:
            label = "practical_tie"
            reason = "best two cost-aware utilities are practically equivalent"
        elif winner.method != "raw" and (
            float(winner.rmse_gain_ci_low) <= 0.0 <= float(winner.rmse_gain_ci_high)
        ):
            label = "inconclusive"
            reason = "paired RMSE-gain interval includes zero"
        elif float(raw.rmse - winner.rmse) < PRACTICAL_RMSE_THRESHOLD:
            label = "practical_tie"
            reason = "accuracy gain over raw is below the preregistered threshold"
        else:
            label = str(winner.method)
            reason = "lowest cost-aware utility with conclusive practical gain"
        row.update(
            oracle_label=label,
            oracle_method=str(winner.method),
            equivalent_methods_json=json.dumps(equivalent_methods, separators=(",", ":")),
            winner_rmse=float(winner.rmse),
            winner_execution_overhead=float(winner.execution_overhead),
            utility_gap=utility_gap,
            label_reason=reason,
            practical_rmse_threshold=PRACTICAL_RMSE_THRESHOLD,
            overhead_penalty=OVERHEAD_PENALTY,
        )
        rows.append(row)
    return pd.DataFrame(rows)
