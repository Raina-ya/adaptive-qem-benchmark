"""Summarize the three Part 9 studies with the rules frozen in readout_boundaries_v1.yaml."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import derive_seed
from src.generate_drift_policy import RAW_PATH as DRIFT_RAW
from src.generate_readout_feasibility_v2 import CONFIG
from src.generate_readout_feasibility_v2 import RAW_PATH as FEASIBILITY_RAW
from src.generate_readout_mismatch_dose import RAW_PATH as MISMATCH_RAW

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
FEASIBILITY_PATH = PROCESSED / "readout_feasibility_v2_summary.csv"
MISMATCH_PATH = PROCESSED / "readout_mismatch_dose_summary_v1.csv"
THRESHOLDS_PATH = PROCESSED / "readout_mismatch_thresholds_v1.csv"
DRIFT_PATH = PROCESSED / "drift_policy_summary_v1.csv"
Z_95 = 1.959963984540054
BOOTSTRAPS = 1000


def ideal_values(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = {name: values for name, values in CONFIG["candidates"].items()}
    frame = frame.copy()
    frame["ideal_high"] = frame.candidate_pair.map(lambda name: pairs[name][0])
    frame["ideal_low"] = frame.candidate_pair.map(lambda name: pairs[name][1])
    # A failed correction is a wrong decision, and its estimate is missing.
    frame["decision_error"] = ~(frame.estimate_high > frame.estimate_low)
    frame["error_high"] = frame.estimate_high - frame.ideal_high
    return frame


def feasibility_summary(raw: pd.DataFrame) -> pd.DataFrame:
    frame = ideal_values(raw)
    keys = ["width", "model", "regime", "regime_setting", "distribution", "candidate_pair"]
    raw_sd = frame.loc[frame.method == "none"].groupby(keys).estimate_high.std().rename("raw_sd")
    rows = []
    for key, group in frame.groupby([*keys, "method"], sort=True):
        complete = group.status == "complete"
        errors = group.error_high[complete]
        target = float(group.target_shots.iloc[0])
        calibration = float(group.calibration_shots.iloc[0])
        rows.append(
            {
                **dict(zip([*keys, "method"], key, strict=True)),
                "repeats": len(group),
                "failure_probability": float((~complete).mean()),
                "failure_reasons": ";".join(sorted(group.status[~complete].unique())),
                "bias": float(errors.mean()) if len(errors) else np.nan,
                "sd": float(errors.std(ddof=1)) if len(errors) > 1 else np.nan,
                "rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
                "decision_error": float(group.decision_error.mean()),
                "median_condition": float(group.condition_number.median()),
                "zero_diagonal_fraction": float(group.zero_diagonal_fraction.mean()),
                "calibration_circuits": int(group.calibration_circuits.iloc[0]),
                "shots_per_calibration_circuit": int(group.shots_per_calibration_circuit.iloc[0]),
                "calibration_shots": calibration,
                "calibration_share": calibration / (calibration + target),
                "median_support": float(group.support_high.median()),
                "median_iterations": float(group.iterations_high.median()),
            }
        )
    table = pd.DataFrame(rows)
    table = table.merge(raw_sd.reset_index(), on=keys, how="left")
    table["variance_amplification"] = table.sd / table.raw_sd
    return table.drop(columns="raw_sd")


def mismatch_summary(raw: pd.DataFrame) -> pd.DataFrame:
    frame = ideal_values(raw)
    frame["covered"] = (frame.error_high.abs() <= Z_95 * frame.standard_error_high).where(
        frame.standard_error_high.notna()
    )
    keys = ["width", "scenario", "distribution", "candidate_pair", "method", "lambda"]
    grouped = frame.groupby(keys, sort=True)
    table = grouped.agg(
        repeats=("repeat", "size"),
        failure_probability=("status", lambda s: float((s != "complete").mean())),
        bias=("error_high", "mean"),
        rmse=("error_high", lambda e: float(np.sqrt(np.nanmean(e**2)))),
        decision_error=("decision_error", "mean"),
        coverage=("covered", "mean"),
        median_condition=("condition_number", "median"),
        median_negative_mass=("negative_mass_high", "median"),
        check_alarm_share=("check_z", lambda z: float((z > 3.0).mean())),
    )
    table = table.reset_index()
    table["diagnostic_shots"] = 2 * int(CONFIG["mismatch"]["check_shots"])
    table["result_label"] = CONFIG["evidence_label"]
    return table


def _bootstrap_loses(raw_errors: np.ndarray, method_errors: np.ndarray, seed: int) -> float:
    """Return the lower 95% bound of method RMSE minus raw RMSE, resampling paired repeats."""

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(raw_errors), size=(BOOTSTRAPS, len(raw_errors)))
    difference = np.sqrt(np.nanmean(method_errors[picks] ** 2, axis=1)) - np.sqrt(
        np.nanmean(raw_errors[picks] ** 2, axis=1)
    )
    return float(np.quantile(difference, 0.025))


def thresholds(raw: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen first-crossing rules, and ask whether each diagnostic warned in time."""

    frame = ideal_values(raw)
    keys = ["width", "scenario", "distribution", "candidate_pair"]
    rows = []
    for key, part in summary.groupby(keys, sort=True):
        raw_curve = part.loc[part.method == "none"].set_index("lambda")
        for method in sorted(set(part.method) - {"none"}):
            curve = part.loc[part.method == method].set_index("lambda").sort_index()
            lambdas = curve.index.to_numpy()
            loses = lambdas[(curve.rmse > raw_curve.rmse.loc[lambdas]).to_numpy()]
            resolved = []
            for weight in lambdas:
                selected = frame.loc[
                    (frame.width == key[0])
                    & (frame.scenario == key[1])
                    & (frame.distribution == key[2])
                    & (frame.candidate_pair == key[3])
                    & (frame["lambda"] == weight)
                ]
                raw_errors = (
                    selected.loc[selected.method == "none"].sort_values("repeat").error_high
                )
                method_errors = (
                    selected.loc[selected.method == method].sort_values("repeat").error_high
                )
                seed = derive_seed("mismatch-threshold", *key, method, weight)
                if _bootstrap_loses(raw_errors.to_numpy(), method_errors.to_numpy(), seed) > 0.0:
                    resolved.append(weight)
            base_error = float(curve.decision_error.iloc[0])
            steep = lambdas[(curve.decision_error >= base_error + 0.10).to_numpy()]
            base_condition = float(curve.median_condition.iloc[0])
            warnings = {
                "condition_number": lambdas[
                    (curve.median_condition > 2.0 * base_condition).to_numpy()
                ],
                "negative_mass": lambdas[(curve.median_negative_mass > 0.05).to_numpy()],
                "check_drift": lambdas[(curve.check_alarm_share >= 0.5).to_numpy()],
                "coverage": lambdas[(curve.coverage < 0.90).to_numpy()],
            }
            row = {
                **dict(zip(keys, key, strict=True)),
                "method": method,
                "lambda_loses_to_raw": _first(loses),
                "lambda_loses_to_raw_resolved": _first(resolved),
                "lambda_decision_error_steep": _first(steep),
                "decision_error_at_zero": base_error,
            }
            for name, values in warnings.items():
                first = _first(values)
                row[f"warning_{name}"] = first
                target = row["lambda_loses_to_raw_resolved"]
                row[f"warns_in_time_{name}"] = bool(
                    not np.isnan(first) and (np.isnan(target) or first <= target)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _first(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.min()) if values.size else np.nan


def drift_summary(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "amplitude",
        "timescale",
        "shape",
        "run_length",
        "margin",
        "calibration_cost",
        "policy",
        "policy_parameter",
    ]
    metrics = [
        "total_shots",
        "shots_per_decision",
        "decision_error",
        "mean_stale_bias",
        "recalibrations",
        "false_recalibrations",
        "detection_delay",
        "missed_drift",
        "violation_fraction",
    ]
    table = raw.groupby(keys, sort=True)[metrics].mean().reset_index()
    table["decision_error_se"] = raw.groupby(keys, sort=True).decision_error.std(
        ddof=1
    ).to_numpy() / np.sqrt(raw.groupby(keys, sort=True).size().to_numpy())
    return table


def main() -> None:
    feasibility = pd.read_csv(FEASIBILITY_RAW)
    feasibility_summary(feasibility).to_csv(
        FEASIBILITY_PATH, index=False, lineterminator="\n", float_format="%.10g"
    )
    mismatch = pd.read_csv(MISMATCH_RAW)
    summary = mismatch_summary(mismatch)
    summary.to_csv(MISMATCH_PATH, index=False, lineterminator="\n", float_format="%.10g")
    thresholds(mismatch, summary).to_csv(
        THRESHOLDS_PATH, index=False, lineterminator="\n", float_format="%.10g"
    )
    drift_summary(pd.read_csv(DRIFT_RAW)).to_csv(
        DRIFT_PATH, index=False, lineterminator="\n", float_format="%.10g"
    )


if __name__ == "__main__":
    main()
