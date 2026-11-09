"""Recalibration policies under synthetic readout drift: a simulator sensitivity analysis.

One qubit, one threshold decision per step: is the Z expectation above the center? The true value
sits one margin above or below it. Policies see only calibration counts, check counts, and age.
"""

from __future__ import annotations

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import derive_seed
from src.generate_readout_feasibility_v2 import CONFIG, check_config_hash

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/drift_policy_v1.csv"


def drift_path(shape: str, amplitude: float, timescale: int, length: int) -> np.ndarray:
    """Return the added p10 at each decision, starting at zero."""

    steps = np.arange(length, dtype=float)
    if shape == "ramp":
        return amplitude * np.minimum(steps / timescale, 1.0)
    if shape == "abrupt":
        return np.where(steps >= timescale, amplitude, 0.0)
    if shape == "sinusoid":
        return amplitude * (1.0 - np.cos(2.0 * np.pi * steps / timescale)) / 2.0
    raise ValueError(f"Unknown drift shape: {shape}")


def policies(settings: dict) -> list[tuple[str, float]]:
    output = [("never", 0.0), ("every_decision", 0.0)]
    output += [("fixed_interval", float(k)) for k in settings["fixed_intervals"]]
    output += [("triggered", float(z)) for z in settings["trigger_thresholds"]]
    return output


def simulate(scenario: dict, policy: str, parameter: float, settings: dict) -> pd.DataFrame:
    """Run every repeat of one scenario and policy at once, vectorized over repeats."""

    repeats = int(settings["repeats"])
    length = int(scenario["run_length"])
    drift = drift_path(scenario["shape"], scenario["amplitude"], scenario["timescale"], length)
    p10 = settings["base_p10"] + drift
    p01 = settings["base_p01"] + settings["p01_fraction_of_p10_drift"] * drift
    rng = np.random.default_rng(derive_seed("drift-policy", *scenario.values(), policy, parameter))
    calibration_cost = int(scenario["calibration_cost"])
    check_shots = int(settings["check_shots_per_state"])
    target = int(settings["target_shots_per_decision"])
    center, margin, tolerance = (
        settings["decision_center"],
        scenario["margin"],
        settings["tolerance"],
    )
    side = rng.choice([-1.0, 1.0], size=(repeats, length))
    truth = center + side * margin

    cal01 = np.zeros(repeats)
    cal10 = np.zeros(repeats)
    true01_at_cal = np.zeros(repeats)
    true10_at_cal = np.zeros(repeats)
    age = np.zeros(repeats)
    recalibrations = np.zeros(repeats)
    false_recal = np.zeros(repeats)
    errors = np.zeros(repeats)
    stale = np.zeros(repeats)
    violation = np.zeros((repeats, length), dtype=bool)
    recal_steps = np.zeros((repeats, length), dtype=bool)
    shots = np.zeros(repeats)
    half = calibration_cost // 2
    for step in range(length):
        if step == 0:
            due = np.ones(repeats, dtype=bool)
        elif policy == "never":
            due = np.zeros(repeats, dtype=bool)
        elif policy == "every_decision":
            due = np.ones(repeats, dtype=bool)
        elif policy == "fixed_interval":
            due = age >= parameter
        else:
            # The check measures both basis states now and is charged to the budget.
            seen01 = rng.binomial(check_shots, p01[step], size=repeats) / check_shots
            seen10 = rng.binomial(check_shots, p10[step], size=repeats) / check_shots
            shots += 2 * check_shots
            z = np.maximum(
                _z(seen01, cal01, check_shots, half), _z(seen10, cal10, check_shots, half)
            )
            due = z > parameter
        if step > 0:
            still_good = (np.abs(p01[step] - true01_at_cal) < tolerance) & (
                np.abs(p10[step] - true10_at_cal) < tolerance
            )
            false_recal += due & still_good
        new01 = rng.binomial(half, p01[step], size=repeats) / half
        new10 = rng.binomial(calibration_cost - half, p10[step], size=repeats) / (
            calibration_cost - half
        )
        cal01 = np.where(due, new01, cal01)
        cal10 = np.where(due, new10, cal10)
        true01_at_cal = np.where(due, p01[step], true01_at_cal)
        true10_at_cal = np.where(due, p10[step], true10_at_cal)
        age = np.where(due, 0.0, age + 1.0)
        recalibrations += due
        recal_steps[:, step] = due
        shots += due * calibration_cost + target

        observed_mean = (1.0 - p01[step] - p10[step]) * truth[:, step] + (p10[step] - p01[step])
        plus = rng.binomial(target, (1.0 + observed_mean) / 2.0)
        raw = 2.0 * plus / target - 1.0
        scale = np.where(1.0 - cal01 - cal10 > 0.05, 1.0 - cal01 - cal10, np.nan)
        corrected = (raw - (cal10 - cal01)) / scale
        errors += np.where(np.isnan(corrected), True, np.sign(corrected - center) != side[:, step])
        expected = (observed_mean - (cal10 - cal01)) / scale
        stale += np.abs(expected - truth[:, step])
        violation[:, step] = (np.abs(p01[step] - true01_at_cal) >= tolerance) | (
            np.abs(p10[step] - true10_at_cal) >= tolerance
        )
    delay, missed = _delays(violation, recal_steps, int(settings["missed_drift_window"]))
    return pd.DataFrame(
        {
            **{key: [value] * repeats for key, value in scenario.items()},
            "policy": policy,
            "policy_parameter": parameter,
            "repeat": np.arange(repeats),
            "total_shots": shots,
            "shots_per_decision": shots / length,
            "decision_error": errors / length,
            "mean_stale_bias": stale / length,
            "recalibrations": recalibrations,
            "false_recalibrations": false_recal,
            "detection_delay": delay,
            "missed_drift": missed,
            "violation_fraction": violation.mean(axis=1),
            "result_label": "noisy_simulated",
        }
    )


def _z(seen, old, check_shots, calibration_shots):
    pooled = (seen * check_shots + old * calibration_shots) / (check_shots + calibration_shots)
    spread = np.sqrt(
        np.maximum(pooled * (1.0 - pooled), 1e-9) * (1.0 / check_shots + 1.0 / calibration_shots)
    )
    return np.abs(seen - old) / spread


def _delays(violation: np.ndarray, recal: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return mean steps from each violation onset to the next recalibration, and missed drift.

    A violation onset is a step in violation after a step out of it. A violation lasting the full
    window without any recalibration counts as missed drift.
    """

    repeats, length = violation.shape
    delays = np.full(repeats, np.nan)
    missed = np.zeros(repeats, dtype=bool)
    for row in range(repeats):
        onsets = np.flatnonzero(violation[row] & ~np.r_[False, violation[row, :-1]])
        values = []
        for onset in onsets:
            later = np.flatnonzero(recal[row, onset:])
            values.append(later[0] if later.size else length - onset)
            run = violation[row, onset : onset + window]
            if run.size == window and run.all() and not recal[row, onset : onset + window].any():
                missed[row] = True
        if values:
            delays[row] = float(np.mean(values))
    return delays, missed


def scenarios(settings: dict) -> list[dict]:
    keys = ("amplitude", "timescale", "shape", "run_length", "margin", "calibration_cost")
    grids = (
        settings["amplitudes"],
        settings["timescales"],
        settings["shapes"],
        settings["run_lengths"],
        settings["margins"],
        settings["calibration_costs"],
    )
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*grids)]


def _task(item):
    scenario, settings = item
    return pd.concat([simulate(scenario, p, k, settings) for p, k in policies(settings)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_config_hash()
    settings = CONFIG["drift"]
    items = [(scenario, settings) for scenario in scenarios(settings)]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_task, items))
    else:
        results = [_task(item) for item in items]
    pd.concat(results, ignore_index=True).to_csv(
        RAW_PATH, index=False, lineterminator="\n", float_format="%.7g"
    )


if __name__ == "__main__":
    main()
