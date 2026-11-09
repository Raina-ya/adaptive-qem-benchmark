"""Measure the total shots each method needs to reach a decision-error target.

Every shot is charged: candidates, folded circuits, and readout calibration.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.density_sim import derive_seed
from src.sequential_decision import (
    CellSpec,
    _candidate_noise,
    intercept_weights,
    observed_probabilities,
)
from src.sequential_decision import load_config as load_sequential_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/shots_reliability_v1.yaml"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
LADDER_PATH = ROOT / "results/processed/shots_reliability_ladder_v1.csv"
TARGETS_PATH = ROOT / "results/processed/shots_reliability_targets_v1.csv"
Z_95 = float(norm.ppf(0.975))
PROBABILITY_DECIMALS = 12


def split_evenly(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (index < extra) for index in range(parts)]


def method_shots(budget: int, method: str, factors: int, calibration_fraction: float):
    """Return target shots per candidate per factor, and calibration shots per state."""

    uses_calibration = method in ("readout", "combined")
    calibration = int(budget * calibration_fraction) if uses_calibration else 0
    candidates = split_evenly(budget - calibration, 2)
    target = np.array([split_evenly(shots, factors) for shots in candidates], dtype=np.int64)
    return target, calibration


def calibration_state_count(task: dict[str, object]) -> int:
    return 2 if task["calibration"] == "shared" else 4


def asymptotic_gap(probabilities: dict, readout: dict, method: str, factors) -> float:
    """Return the value this estimator's gap converges to with unlimited shots."""

    gaps = []
    weights = intercept_weights(factors) if len(factors) > 1 else np.array([1.0])
    for label in ("a", "b"):
        observed = float(weights @ (2.0 * probabilities[label] - 1.0))
        if method in ("readout", "combined"):
            p01, p10 = readout[label]["p01"], readout[label]["p10"]
            observed = (observed - (p10 - p01)) / (1.0 - p01 - p10)
        gaps.append(observed)
    return gaps[0] - gaps[1]


def simulate_cell(
    task_name: str, margin: float, noise: str, method: str, config: dict, source: dict
) -> dict[str, np.ndarray]:
    """Sample cumulative counts up the ladder, for all repeats at once."""

    ladder = config["ladder"]
    repeats = int(config["repeats"])
    task = source["tasks"][task_name]
    uses_folds = method in ("zne", "combined")
    factors = tuple(config["zne_factors"]) if uses_folds else (1,)
    spec = CellSpec(task_name, margin, noise, ladder[0], "zne" if uses_folds else "raw")
    probabilities = observed_probabilities(spec, source)
    readout = _candidate_noise(spec, source)
    weights = intercept_weights(factors) if uses_folds else np.array([1.0])
    rng = np.random.default_rng(
        derive_seed("shots-reliability-v1", task_name, margin, noise, method)
    )
    labels = ("a", "b")
    shared = task["calibration"] == "shared"
    previous_target = np.zeros((2, len(factors)), dtype=np.int64)
    previous_calibration = 0
    plus = np.zeros((repeats, 2, len(factors)))
    cal_errors = np.zeros((repeats, 2, 2))
    cal_trials = np.zeros((2, 2))
    gaps, variances, failed = [], [], []
    for budget in ladder:
        target, calibration = method_shots(
            budget, method, len(factors), float(config["calibration_fraction"])
        )
        for candidate, label in enumerate(labels):
            for index in range(len(factors)):
                probability = round(float(probabilities[label][index]), PROBABILITY_DECIMALS)
                extra = int(target[candidate, index] - previous_target[candidate, index])
                plus[:, candidate, index] += rng.binomial(extra, probability, size=repeats)
        previous_target = target
        if calibration:
            pairs = 1 if shared else 2
            state = split_evenly(calibration - previous_calibration, 2 * pairs)
            cursor = 0
            for pair in range(pairs):
                source_noise = readout[labels[pair]]
                for kind, key in enumerate(("p01", "p10")):
                    cal_errors[:, pair, kind] += rng.binomial(
                        state[cursor], source_noise[key], size=repeats
                    )
                    cal_trials[pair, kind] += state[cursor]
                    cursor += 1
            previous_calibration = calibration
        observed = 2.0 * plus / target[None, :, :] - 1.0
        variance = (1.0 - observed**2) / target[None, :, :]
        # Plug-in variance stays above one count's resolution.
        variance = np.maximum(variance, 1.0 / target[None, :, :] ** 2)
        point = observed @ weights
        spread = variance @ weights**2
        if calibration:
            pairs = 1 if shared else 2
            rates = cal_errors / np.where(cal_trials > 0, cal_trials, 1.0)
            if shared:
                rates[:, 1] = rates[:, 0]
            a_hat, b_hat = rates[:, :, 0], rates[:, :, 1]
            scale = 1.0 - a_hat - b_hat
            bad = np.any(scale <= 0.0, axis=1)
            safe = np.where(scale > 0.0, scale, 1.0)
            corrected = (point - (b_hat - a_hat)) / safe
            gap = corrected[:, 0] - corrected[:, 1]
            trials = np.where(cal_trials > 0, cal_trials, 1.0)
            if shared:
                var_a = a_hat[:, 0] * (1.0 - a_hat[:, 0]) / trials[0, 0]
                var_b = b_hat[:, 0] * (1.0 - b_hat[:, 0]) / trials[0, 1]
                # With one calibration, its offset cancels and only the common scale moves the gap.
                calibration_term = (gap / safe[:, 0]) ** 2 * (var_a + var_b)
            else:
                var_a = a_hat * (1.0 - a_hat) / trials[:, 0][None, :]
                var_b = b_hat * (1.0 - b_hat) / trials[:, 1][None, :]
                calibration_term = np.sum(
                    ((1.0 + corrected) / safe) ** 2 * var_a
                    + ((1.0 - corrected) / safe) ** 2 * var_b,
                    axis=1,
                )
            total = np.sum(spread / safe**2, axis=1) + calibration_term
        else:
            gap = point[:, 0] - point[:, 1]
            total = spread.sum(axis=1)
            bad = np.zeros(repeats, dtype=bool)
        gaps.append(gap.copy())
        variances.append(total.copy())
        failed.append(bad.copy())
    limit = asymptotic_gap(probabilities, readout, method, factors)
    calibrated = method in ("readout", "combined")
    depths = np.array([int(task["depth_a"]), int(task["depth_b"])], dtype=float)
    factor_array = np.array(factors, dtype=float)
    gate_shots = []
    for budget in ladder:
        target, _ = method_shots(
            budget, method, len(factors), float(config["calibration_fraction"])
        )
        gate_shots.append(float((target * factor_array * depths[:, None]).sum()))
    return {
        "gap": np.array(gaps).T,
        "variance": np.array(variances).T,
        "failed": np.array(failed).T,
        "asymptotic_gap": limit,
        "gate_shots": np.array(gate_shots),
        "executions": 2 * len(factors) + (calibration_state_count(task) if calibrated else 0),
        "calibrated": calibrated,
    }


def wilson_upper(errors: int, trials: int) -> float:
    rate = errors / trials
    denominator = 1.0 + Z_95**2 / trials
    center = rate + Z_95**2 / (2.0 * trials)
    spread = Z_95 * math.sqrt(rate * (1.0 - rate) / trials + Z_95**2 / (4.0 * trials**2))
    return (center + spread) / denominator


def predicted_error(limit: float, budget_variance: float) -> float:
    """Return the normal-approximation error, for comparison with sampling only."""

    if budget_variance <= 0.0:
        return 0.0 if limit > 0.0 else 1.0
    return float(norm.cdf(-limit / math.sqrt(budget_variance)))


def summarize_cell(identity: dict, sample: dict, margin: float, config: dict) -> list[dict]:
    """Score fixed-budget and stopped decisions at each ladder budget. The margin is positive."""

    ladder = config["ladder"]
    alpha = 1.0 - float(config["confidence_level"])
    z = float(norm.ppf(1.0 - alpha / (2.0 * len(ladder))))
    half = z * np.sqrt(sample["variance"])
    gap = sample["gap"]
    separated = (np.abs(gap) > half) & ~sample["failed"]
    stop = np.where(separated.any(axis=1), separated.argmax(axis=1), len(ladder))
    rows = []
    for look, budget in enumerate(ladder):
        error = (gap[:, look] <= 0.0) | sample["failed"][:, look]
        stopped = stop <= look
        index = np.arange(len(gap))
        at_stop = np.minimum(stop, look)
        wrong_stop = stopped & (gap[index, at_stop] <= 0.0)
        forced_regret = np.where((gap[index, at_stop] <= 0.0), margin, 0.0)
        reached = stopped & ~wrong_stop
        spent = np.array(ladder, dtype=float)[at_stop]
        calibration = (
            int(budget * float(config["calibration_fraction"])) if sample["calibrated"] else 0
        )
        rows.append(
            {
                **identity,
                "budget": budget,
                "repeats": len(gap),
                "decision_error": float(error.mean()),
                "decision_error_wilson_high": wilson_upper(int(error.sum()), len(gap)),
                "reached_correct_stop": float(reached.mean()),
                "unresolved_at_budget": float((~stopped).mean()),
                "wrong_stop": float(wrong_stop.mean()),
                "failed_calibration": float(sample["failed"][:, look].mean()),
                "mean_shots_at_stop": float(spent.mean()),
                "regret_at_stop": float(forced_regret.mean()),
                "total_executions": sample["executions"],
                "calibration_share": calibration / budget,
                "two_qubit_gate_shots": sample["gate_shots"][look],
                "asymptotic_gap": sample["asymptotic_gap"],
                "result_label": config["evidence_label"],
            }
        )
    return rows


def target_table(ladder_table: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Find the smallest budget that stays under each target, and say why if there is none."""

    keys = ["task", "margin", "noise", "method"]
    rows = []
    raw_gate = ladder_table.loc[ladder_table.method == "raw"].set_index(
        ["task", "margin", "noise", "budget"]
    )
    for key, group in ladder_table.groupby(keys, sort=True):
        ordered = group.sort_values("budget")
        highs = ordered.decision_error_wilson_high.to_numpy()
        budgets = ordered.budget.to_numpy()
        for target in config["target_errors"]:
            passing = highs <= target
            tail = np.logical_and.accumulate(passing[::-1])[::-1]
            if tail.any():
                position = int(tail.argmax())
                status = "reached"
                shots = int(budgets[position])
            else:
                position = len(budgets) - 1
                shots = None
                status = (
                    "blocked_by_bias"
                    if ordered.asymptotic_gap.iloc[0] <= 0.0
                    else "not_reached_by_maximum_budget"
                )
            row = ordered.iloc[position]
            raw_row = raw_gate.loc[(key[0], key[1], key[2], int(row.budget))]
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "target_error": target,
                    "status": status,
                    "shots_to_target": shots,
                    "decision_error_at_budget": row.decision_error,
                    "probability_reached_correct_stop": row.reached_correct_stop,
                    "unresolved_at_budget": row.unresolved_at_budget,
                    "wrong_stop": row.wrong_stop,
                    "total_executions": row.total_executions,
                    "calibration_share": row.calibration_share,
                    "overhead_ratio": row.two_qubit_gate_shots
                    / max(float(raw_row.two_qubit_gate_shots), 1.0),
                    "regret_at_stop": row.regret_at_stop,
                    "mean_shots_at_stop": row.mean_shots_at_stop,
                    "result_label": row.result_label,
                }
            )
    return pd.DataFrame(rows)


def build_ladder(config: dict, source: dict) -> pd.DataFrame:
    rows = []
    for task_name, task in source["tasks"].items():
        for margin in task["margins"]:
            if float(margin) <= 0.0:
                continue
            for noise in task["noise"]:
                for method in config["methods"]:
                    identity = {
                        "task": task_name,
                        "margin": float(margin),
                        "noise": noise,
                        "method": method,
                    }
                    sample = simulate_cell(task_name, float(margin), noise, method, config, source)
                    rows.extend(summarize_cell(identity, sample, float(margin), config))
    return pd.DataFrame(rows)


def main() -> None:
    source = load_sequential_config()
    ladder = build_ladder(CONFIG, source)
    ladder.to_csv(LADDER_PATH, index=False, lineterminator="\n", float_format="%.12g")
    target_table(ladder, CONFIG).to_csv(
        TARGETS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )


if __name__ == "__main__":
    main()
