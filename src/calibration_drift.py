"""Deterministic calibration-amortization and synthetic drift benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import derive_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/calibration_lifetime_v1.yaml"
RAW_PATH = ROOT / "results/raw/calibration_lifetime_v1.csv"
AGE_FIGURE = ROOT / "figures/calibration_age_vs_decision_error.png"
COST_FIGURE = ROOT / "figures/amortized_cost_vs_reliability.png"


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    """Load the JSON-compatible frozen YAML without an extra dependency."""

    return json.loads(path.read_text(encoding="utf-8"))


def assignment_matrix(p01: float, p10: float) -> np.ndarray:
    """Return A[observed, prepared] for one-qubit assignment error."""

    if not 0.0 <= p01 < 0.5 or not 0.0 <= p10 < 0.5:
        raise ValueError("Readout error probabilities must be in [0, 0.5)")
    return np.array([[1.0 - p01, p10], [p01, 1.0 - p10]], dtype=float)


def drift_matrix(model: str, decision_index: int, config: dict[str, object]) -> np.ndarray:
    """Return a deterministic, bounded synthetic assignment matrix at time t."""

    if decision_index < 0:
        raise ValueError("decision_index must be nonnegative")
    p01 = float(config["initial_readout_p01"])
    p10 = float(config["initial_readout_p10"])
    if model == "slow_monotonic":
        p01 += min(0.00015 * decision_index, 0.015)
        p10 += min(0.0004 * decision_index, 0.04)
    elif model == "random_bounded":
        rng = np.random.default_rng(derive_seed("calibration-random-drift", decision_index))
        p01 += float(rng.uniform(-0.015, 0.015))
        p10 += float(rng.uniform(-0.015, 0.015))
    elif model == "abrupt_change":
        if decision_index >= 10:
            p01 += 0.005
            p10 += 0.05
    elif model == "asymmetric":
        p01 -= min(0.00015 * decision_index, 0.015)
        p10 += min(0.0006 * decision_index, 0.06)
    else:
        raise ValueError(f"Unknown drift model: {model}")
    return assignment_matrix(p01, p10)


def should_recalibrate(
    policy: str,
    decision_index: int,
    calibration_age: int | None,
    config: dict[str, object],
) -> bool:
    """Apply a preregistered policy without access to the true drift state."""

    if calibration_age is None or decision_index == 0:
        return True
    if policy == "every_decision":
        return True
    if policy == "fixed_interval":
        return calibration_age >= int(config["fixed_calibration_interval"])
    if policy == "uncertainty_triggered":
        bound = float(config["trigger_base_uncertainty"]) + calibration_age * float(
            config["trigger_growth_per_decision"]
        )
        return bound >= float(config["trigger_threshold"])
    if policy == "never":
        return False
    raise ValueError(f"Unknown recalibration policy: {policy}")


def _estimate_calibration(matrix: np.ndarray, shots: int, seed: int) -> np.ndarray:
    shots_by_state = (shots // 2, shots - shots // 2)
    estimated = np.zeros((2, 2), dtype=float)
    for prepared, state_shots in enumerate(shots_by_state):
        rng = np.random.default_rng(derive_seed("calibration-drift-cal", seed, prepared))
        sampled = rng.multinomial(state_shots, matrix[:, prepared])
        estimated[:, prepared] = sampled / state_shots
    return estimated


def _correct_probability(observed_one: float, calibration: np.ndarray) -> tuple[float, str]:
    denominator = float(calibration[1, 1] - calibration[1, 0])
    if abs(denominator) <= 0.1:
        return math.nan, "ill_conditioned"
    return (observed_one - float(calibration[1, 0])) / denominator, "complete"


def _workflow_cost(policy: str, length: int, config: dict[str, object]) -> tuple[int, int]:
    events = 0
    age: int | None = None
    for decision_index in range(length):
        if should_recalibrate(policy, decision_index, age, config):
            events += 1
            age = 0
        else:
            assert age is not None
            age += 1
    total = length * int(config["target_shots_per_decision"]) + events * int(
        config["calibration_shots_per_event"]
    )
    return events, total


def run_benchmark(config: dict[str, object] | None = None) -> pd.DataFrame:
    """Execute the frozen repeat-level synthetic drift benchmark."""

    config = load_config() if config is None else config
    rows: list[dict[str, object]] = []
    p1 = float(config["true_probability_one"])
    threshold = float(config["decision_threshold"])
    target_shots = int(config["target_shots_per_decision"])
    calibration_shots = int(config["calibration_shots_per_event"])
    for length in config["workload_lengths"]:
        length = int(length)
        for drift_index, drift in enumerate(config["drift_models"]):
            for policy in config["policies"]:
                events, workflow_total = _workflow_cost(str(policy), length, config)
                for repeat in range(int(config["repeats"])):
                    calibration: np.ndarray | None = None
                    calibration_age: int | None = None
                    calibration_time = -1
                    for decision_index in range(length):
                        recalibrated = should_recalibrate(
                            str(policy), decision_index, calibration_age, config
                        )
                        actual = drift_matrix(str(drift), decision_index, config)
                        if recalibrated:
                            calibration_time = decision_index
                            calibration_age = 0
                            calibration = _estimate_calibration(
                                actual,
                                calibration_shots,
                                derive_seed(
                                    "calibration-lifetime",
                                    length,
                                    drift_index,
                                    repeat,
                                    decision_index,
                                ),
                            )
                        else:
                            assert calibration_age is not None
                            calibration_age += 1
                        assert calibration is not None and calibration_age is not None
                        observed_probability = float((actual @ np.array([1.0 - p1, p1]))[1])
                        seed = derive_seed(
                            "calibration-lifetime-target",
                            length,
                            drift_index,
                            repeat,
                            decision_index,
                        )
                        observed_ones = int(
                            np.random.default_rng(seed).binomial(target_shots, observed_probability)
                        )
                        raw_probability = observed_ones / target_shots
                        corrected, status = _correct_probability(raw_probability, calibration)
                        decision_error = (
                            math.nan
                            if status != "complete"
                            else int((corrected > threshold) != (p1 > threshold))
                        )
                        rows.append(
                            {
                                "workload_length": length,
                                "drift_model": drift,
                                "policy": policy,
                                "repeat": repeat,
                                "decision_index": decision_index,
                                "seed": seed,
                                "calibration_time": calibration_time,
                                "calibration_age": calibration_age,
                                "recalibrated": recalibrated,
                                "actual_p01": actual[1, 0],
                                "actual_p10": actual[0, 1],
                                "calibrated_p01": calibration[1, 0],
                                "calibrated_p10": calibration[0, 1],
                                "true_probability_one": p1,
                                "observed_probability_one": raw_probability,
                                "corrected_probability_one": corrected,
                                "decision_threshold": threshold,
                                "decision_error": decision_error,
                                "target_shots": target_shots,
                                "calibration_shots_charged": calibration_shots
                                if recalibrated
                                else 0,
                                "calibration_events_in_workflow": events,
                                "workflow_total_shots": workflow_total,
                                "amortized_shots_per_decision": workflow_total / length,
                                "status": status,
                                "result_label": "noisy_simulated",
                            }
                        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate reliability and costs without dropping failed decisions."""

    keys = ["workload_length", "drift_model", "policy"]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        complete = group[group.status == "complete"]
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "decisions": len(group),
                "complete_decisions": len(complete),
                "decision_error_probability": complete.decision_error.mean(),
                "mean_calibration_age": complete.calibration_age.mean(),
                "amortized_shots_per_decision": group.amortized_shots_per_decision.iloc[0],
                "workflow_total_shots": group.workflow_total_shots.iloc[0],
            }
        )
    return pd.DataFrame(rows)


def _plot_age(frame: pd.DataFrame) -> None:
    grouped = (
        frame[frame.status == "complete"]
        .groupby(["drift_model", "calibration_age"], as_index=False)
        .decision_error.mean()
    )
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    for drift, group in grouped.groupby("drift_model"):
        axis.plot(group.calibration_age, group.decision_error, marker="o", label=drift)
    axis.set(xlabel="Calibration age (decisions)", ylabel="Decision-error probability")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(AGE_FIGURE, dpi=180)
    plt.close(figure)


def _plot_cost(summary: pd.DataFrame) -> None:
    grouped = (
        summary.groupby(["workload_length", "policy"], as_index=False)
        .agg(
            amortized_shots_per_decision=("amortized_shots_per_decision", "mean"),
            decision_error_probability=("decision_error_probability", "mean"),
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), sharex=True, sharey=True)
    policies = sorted(grouped.policy.unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(policies)]
    colors = dict(zip(policies, palette, strict=True))
    workloads = grouped.groupby("workload_length")
    for axis, (length, workload) in zip(axes.flat, workloads, strict=True):
        for row in workload.itertuples():
            axis.scatter(
                row.amortized_shots_per_decision,
                row.decision_error_probability,
                color=colors[row.policy],
                label=row.policy,
                s=42,
            )
        axis.set_title(f"{length} decision{'s' if length != 1 else ''}")
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    figure.supxlabel("Amortized shots per decision")
    figure.supylabel("Mean decision-error probability across drift models")
    figure.subplots_adjust(top=0.86, bottom=0.10, left=0.11, right=0.98, hspace=0.32)
    figure.savefig(COST_FIGURE, dpi=180)
    plt.close(figure)


def main() -> None:
    """Generate the frozen raw artifact and both analysis figures."""

    frame = run_benchmark()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGE_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.12g")
    summary = summarize(frame)
    _plot_age(frame)
    _plot_cost(summary)


if __name__ == "__main__":
    main()
