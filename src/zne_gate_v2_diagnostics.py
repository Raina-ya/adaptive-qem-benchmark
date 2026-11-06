"""Stress the frozen v2 gate on holdout. Written after seeing the holdout, and changes no choice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import derive_seed
from src.generate_signal_v2 import RAW_PATH, split_evenly
from src.zne_gate_v2 import (
    CONDITIONS_PATH,
    SELECTION_PATH,
    _plus_values,
    confusion,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
GROUPS_PATH = PROCESSED / "zne_gate_v2_holdout_groups.csv"
REPEATS_PATH = PROCESSED / "zne_gate_v2_repeat_sensitivity.csv"
THRESHOLDS_PATH = PROCESSED / "zne_gate_v2_threshold_sensitivity.csv"
POLICY_PATH = PROCESSED / "zne_gate_v2_policy_contrasts.csv"
ANALYSIS_ROLE = "post_hoc_after_holdout"
REPEAT_COUNTS = (2, 5, 10, 20, 50)
THRESHOLDS = (0.0, 0.005, 0.01, 0.02)
BOOTSTRAPS = 2000


def selected_name() -> str:
    frame = pd.read_csv(SELECTION_PATH)
    return str(frame.loc[frame.selected, "name"].iloc[0])


def slope_stable(values: np.ndarray) -> bool:
    centered = np.array([1.0, 3.0, 5.0]) - 3.0
    slopes = values @ centered
    return bool(np.mean(np.sign(slopes) == np.sign(slopes.mean())) >= 0.9)


def holdout_groups(holdout: pd.DataFrame) -> pd.DataFrame:
    """Break the frozen gate and v1 down by noise, band, width, and depth."""

    rows = []
    for column in ("noise", "signal_band", "width", "depth", "total_shot_budget"):
        for value, group in holdout.groupby(column, sort=True):
            for gate, accepted, truth in (
                ("v1_full", group.v1_full, group.improved),
                ("v1_runtime", group.v1_runtime, group.improved),
                ("v2_selected", group.trend_slope_sign, group.improved_clipped),
            ):
                rows.append(
                    {
                        "grouping": column,
                        "value": value,
                        "gate": gate,
                        "independent_circuits": int(group.circuit_id.nunique()),
                        "conditions": len(group),
                        **confusion(accepted.astype(bool), truth.astype(bool)),
                    }
                )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def repeat_sensitivity(holdout: pd.DataFrame, repeats: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen check to fewer repeats, as a cheaper real run would have to."""

    keys = ["circuit_id", "noise", "total_shot_budget"]
    wanted = holdout[keys]
    frame = repeats.merge(wanted, on=keys)
    decisions = {count: [] for count in REPEAT_COUNTS}
    index = []
    for key, group in frame.groupby(keys, sort=True):
        group = group.sort_values("repeat")
        values, _ = _plus_values(group.global_135_plus, split_evenly(int(key[2]), 3))
        for count in REPEAT_COUNTS:
            decisions[count].append(slope_stable(values[:count]))
        index.append(key)
    labels = pd.DataFrame(index, columns=keys).merge(holdout[[*keys, "improved_clipped"]], on=keys)
    rows = [
        {
            "repeats_read": count,
            **confusion(pd.Series(decisions[count]), labels.improved_clipped.astype(bool)),
        }
        for count in REPEAT_COUNTS
    ]
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def threshold_sensitivity(holdout: pd.DataFrame) -> pd.DataFrame:
    """Re-score the same decisions under other definitions of a real improvement."""

    rows = []
    for threshold in THRESHOLDS:
        truth = holdout.raw_rmse - holdout.fit_rmse > threshold
        clipped = holdout.raw_rmse - holdout.fit_rmse_clipped > threshold
        for gate, accepted, labels in (
            ("v1_full", holdout.v1_full, truth),
            ("v1_runtime", holdout.v1_runtime, truth),
            ("v2_selected", holdout.trend_slope_sign, clipped),
            ("estimated_gain_only", holdout.estimated_gain, truth),
        ):
            rows.append(
                {
                    "practical_threshold": threshold,
                    "gate": gate,
                    **confusion(accepted.astype(bool), labels.astype(bool)),
                }
            )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def policy_contrasts(holdout: pd.DataFrame) -> pd.DataFrame:
    """Compare policies pairwise, resampling circuits, and separate the effect of clipping."""

    gate = holdout.trend_slope_sign.to_numpy(bool)
    unclipped = np.where(gate, holdout.fit_rmse, holdout.raw_rmse)
    clipped = np.where(gate, holdout.fit_rmse_clipped, holdout.raw_rmse)
    linear = holdout.fit_rmse.to_numpy()
    v1 = np.where(holdout.v1_runtime.to_numpy(bool), holdout.fit_rmse, holdout.raw_rmse)
    gain_only = np.where(holdout.estimated_gain.to_numpy(bool), holdout.fit_rmse, holdout.raw_rmse)
    contrasts = {
        "v2_clipped_minus_always_linear": linear - clipped,
        "v2_clipped_minus_v2_unclipped": unclipped - clipped,
        "v2_unclipped_minus_always_linear": linear - unclipped,
        "v2_clipped_minus_v1_runtime": v1 - clipped,
        "v2_unclipped_minus_estimated_gain_only": gain_only - unclipped,
    }
    circuits = holdout.circuit_id.to_numpy()
    unique = np.unique(circuits)
    rng = np.random.default_rng(derive_seed("zne-gate-v2-diagnostics"))
    picks = rng.integers(0, len(unique), size=(BOOTSTRAPS, len(unique)))
    rows = []
    for name, values in contrasts.items():
        per = pd.DataFrame({"circuit": circuits, "value": values}).groupby("circuit").value
        totals, counts = per.sum().to_numpy(), per.size().to_numpy()
        boots = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
        rows.append(
            {
                "contrast_rmse_reduction": name,
                "estimate": float(values.mean()),
                "cluster_low": float(np.quantile(boots, 0.025)),
                "cluster_high": float(np.quantile(boots, 0.975)),
            }
        )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def main() -> None:
    if "trend_slope_sign" not in selected_name():
        raise ValueError("These diagnostics were written for the slope-sign selection")
    conditions = pd.read_csv(CONDITIONS_PATH)
    holdout = conditions.loc[conditions.split == "holdout"].reset_index(drop=True)
    tables = {
        GROUPS_PATH: holdout_groups(holdout),
        REPEATS_PATH: repeat_sensitivity(holdout, pd.read_csv(RAW_PATH)),
        THRESHOLDS_PATH: threshold_sensitivity(holdout),
        POLICY_PATH: policy_contrasts(holdout),
    }
    for path, table in tables.items():
        table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


if __name__ == "__main__":
    main()
