"""Run the frozen v3 decision tasks: baselines, selector evidence, and certification outcomes."""

from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from src.decision_tasks_v3 import BUILDERS, CONFIG, HASH_PATH, MANIFEST_PATH
from src.density_sim import Noise, derive_seed, fold_global, outcome_probabilities, parity_signs
from src.generate_signal_v2 import fit_weights, readout_weights, split_evenly

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/selector_v3_repeats.csv"
ALARMS = (
    "raw_mitigated_winner_differs",
    "fold_gap_reversal",
    "calibration_age_drift",
    "outside_physical_bound",
    "variance_amplification",
    "batch_sign_instability",
)
PROBABILITY_DECIMALS = 12
LINEAR_WEIGHTS = fit_weights((1, 3, 5), 1)


def check_manifest_hash() -> None:
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    frozen = HASH_PATH.read_text(encoding="utf-8").strip()
    if digest != frozen:
        raise ValueError(f"Manifest hash {digest} differs from frozen {frozen}")


def calibration_scale(noise_name: str, position: float, config: dict) -> float:
    """Return how far the readout at calibration time sits from the readout at measurement."""

    drift = config["drift"]
    if noise_name != drift["condition"]:
        return 1.0
    first, last = drift["calibration_scale_first"], drift["calibration_scale_last"]
    return first + (last - first) * position


def sample_calibration(rng, shots: int, width: int, noise: Noise, scale: float):
    """Prepare all-zero and all-one states, and return per-qubit error counts and trials."""

    zero, one = split_evenly(shots, 2)
    errors_01 = rng.binomial(zero, min(noise.readout_p01 * scale, 1.0), size=width)
    errors_10 = rng.binomial(one, min(noise.readout_p10 * scale, 1.0), size=width)
    return errors_01, errors_10, zero, one


def corrected(
    rng, shots: int, distribution: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    counts = rng.multinomial(shots, distribution)
    mean = float(counts @ weights) / shots
    variance = float(counts @ weights**2) / shots - mean**2
    return mean, float(np.sqrt(max(variance, 0.0) / shots))


def plain(rng, shots: int, expectation: float) -> tuple[float, float]:
    estimate = 2.0 * rng.binomial(shots, (1.0 + expectation) / 2.0) / shots - 1.0
    return estimate, float(np.sqrt(max(1.0 - estimate**2, 1.0 / shots) / shots))


def _weights(pauli: str, errors_01, errors_10, zero: int, one: int) -> np.ndarray:
    return readout_weights(pauli, errors_01 / zero, errors_10 / one)


def _fold_sets(rng, shots: int, dists, means, pauli_weights):
    """Sample factors 1, 3, 5 once, returning plain and corrected values per candidate."""

    candidates = split_evenly(shots, len(dists))
    plain_values = np.zeros((len(dists), 3))
    plain_errors = np.zeros((len(dists), 3))
    corrected_values = np.zeros((len(dists), 3))
    corrected_errors = np.zeros((len(dists), 3))
    for k, (dist, mean, weight) in enumerate(zip(dists, means, pauli_weights, strict=True)):
        for index, factor_shots in enumerate(split_evenly(candidates[k], 3)):
            counts = rng.multinomial(factor_shots, dist[index])
            plain_value = float(counts @ mean[index]) / factor_shots
            plain_values[k, index] = plain_value
            plain_errors[k, index] = np.sqrt(
                max(1.0 - plain_value**2, 1.0 / factor_shots) / factor_shots
            )
            value = float(counts @ weight) / factor_shots
            variance = float(counts @ weight**2) / factor_shots - value**2
            corrected_values[k, index] = value
            corrected_errors[k, index] = np.sqrt(max(variance, 0.0) / factor_shots)
    return plain_values, plain_errors, corrected_values, corrected_errors


def certify(rng, budget: int, dists, signs, paulis, noise: Noise, config: dict) -> dict:
    """Apply the two certification layers at each look, and report the first passing look."""

    settings = config["certification"]
    batches = int(settings["batches"])
    width = len(paulis[0])
    count = len(dists)
    corrected_f1 = np.zeros((batches, count))
    plain_f1 = np.zeros((batches, count))
    corrected_f5 = np.zeros((batches, count))
    corrected_f3 = np.zeros((batches, count))
    cal_rates = np.zeros((batches, width, 2))
    plain_var = np.zeros((batches, count))
    batch_budget = split_evenly(budget, batches)
    trials_per_state = int(batch_budget[-1] * settings["calibration_fraction"]) // 2
    for batch, shots in enumerate(batch_budget):
        position = batch / max(batches - 1, 1)
        scale = calibration_scale(noise.name, position, config)
        cal_shots = int(shots * settings["calibration_fraction"])
        errors_01, errors_10, zero, one = sample_calibration(rng, cal_shots, width, noise, scale)
        cal_rates[batch, :, 0], cal_rates[batch, :, 1] = errors_01 / zero, errors_10 / one
        base = split_evenly(int(shots * settings["base_fraction"]), count)
        fold3 = split_evenly(int(shots * settings["fold3_fraction"]), count)
        fold5 = split_evenly(int(shots * settings["fold5_fraction"]), count)
        for k in range(count):
            weight = _weights(paulis[k], errors_01, errors_10, zero, one)
            counts = rng.multinomial(base[k], dists[k][0])
            corrected_f1[batch, k] = float(counts @ weight) / base[k]
            plain_value = float(counts @ signs[k]) / base[k]
            plain_f1[batch, k] = plain_value
            plain_var[batch, k] = max(1.0 - plain_value**2, 1.0 / base[k]) / base[k]
            corrected_f3[batch, k], _ = corrected(rng, fold3[k], dists[k][1], weight)
            corrected_f5[batch, k], _ = corrected(rng, fold5[k], dists[k][2], weight)
    looks = settings["looks"]
    alpha = float(settings["alpha"])
    per_test = alpha / max(count - 1, 1) / len(looks)
    outcome = {
        "l1_look": 0,
        "l1_winner": -1,
        "certified_look": 0,
        "certified_winner": -1,
        "first_alarm_bits": 0,
    }
    for look in looks:
        means = corrected_f1[:look].mean(axis=0)
        winner = int(np.argmax(means))
        others = [k for k in range(count) if k != winner]
        differences = corrected_f1[:look, winner][:, None] - corrected_f1[:look][:, others]
        spread = differences.std(axis=0, ddof=1) / np.sqrt(look)
        ratios = differences.mean(axis=0) / np.maximum(spread, 1e-12)
        critical = float(student_t.ppf(1.0 - per_test, look - 1))
        if not np.all(ratios >= critical):
            continue
        if outcome["l1_look"] == 0:
            outcome["l1_look"], outcome["l1_winner"] = look, winner
        runner = others[int(np.argmin(differences.mean(axis=0)))]
        bits = _alarms(
            look,
            winner,
            runner,
            corrected_f1,
            plain_f1,
            corrected_f5,
            plain_var,
            cal_rates,
            trials_per_state,
            settings,
        )
        if outcome["l1_look"] == look:
            outcome["first_alarm_bits"] = bits
        if bits == 0:
            outcome["certified_look"], outcome["certified_winner"] = look, winner
            break
    # The naive control pools batches and treats all estimates as independent.
    pooled = corrected_f1.mean(axis=0)
    winner = int(np.argmax(pooled))
    naive_error = np.sqrt(plain_var.mean(axis=0) / batches)
    gaps = pooled[winner] - np.delete(pooled, winner)
    naive = np.hypot(naive_error[winner], np.delete(naive_error, winner))
    outcome["naive_pass"] = bool(np.all(gaps >= 1.96 * naive))
    outcome["naive_winner"] = winner
    return outcome


def _alarms(
    look,
    winner,
    runner,
    corrected_f1,
    plain_f1,
    corrected_f5,
    plain_var,
    cal_rates,
    trials_per_state,
    settings,
) -> int:
    """Return a bitmask over ALARMS, using only data up to this look."""

    bits = 0
    if int(np.argmax(plain_f1[:look].mean(axis=0))) != winner:
        bits |= 1
    fold_gap = corrected_f5[:look, winner] - corrected_f5[:look, runner]
    fold_error = fold_gap.std(ddof=1) / np.sqrt(look)
    if fold_gap.mean() < -float(settings["fold_reversal_z"]) * fold_error:
        bits |= 2
    first, last = cal_rates[0], cal_rates[look - 1]
    pooled = (first + last) / 2.0
    spread = np.sqrt(pooled * (1.0 - pooled) * 2.0 / trials_per_state)
    if np.max(np.abs(first - last) / np.maximum(spread, 1e-9)) > float(settings["drift_z"]):
        bits |= 4
    means = corrected_f1[:look].mean(axis=0)
    errors = corrected_f1[:look].std(axis=0, ddof=1) / np.sqrt(look)
    if np.any(np.abs(means) > 1.0 + float(settings["bound_z"]) * errors):
        bits |= 8
    plain_error = np.sqrt(plain_var[:look].mean(axis=0) / look)
    if np.any(errors > float(settings["variance_ratio_limit"]) * plain_error):
        bits |= 16
    signs = np.sign(corrected_f1[:look, winner] - corrected_f1[:look, runner])
    if int((signs > 0).sum()) < min(int(settings["minimum_batch_sign_agreement"]), look):
        bits |= 32
    return bits


def exact_distributions(tasks: pd.DataFrame, noise: Noise, config: dict):
    factors = config["factors"]
    dists, signs, paulis, ideals, gate_counts = [], [], [], [], []
    for row in tasks.itertuples():
        gates = BUILDERS[row.family](int(row.width), int(row.depth), int(row.parameter_seed))
        paulis.append(row.observable_q0_first)
        signs.append(parity_signs(row.observable_q0_first))
        ideals.append(float(row.ideal_expectation))
        gate_counts.append(sum(1 for gate in gates if gate.name == "cx"))
        candidate = []
        for factor in factors:
            probability = outcome_probabilities(
                fold_global(gates, factor), row.observable_q0_first, noise
            )
            values = np.round(probability, PROBABILITY_DECIMALS)
            candidate.append(values / values.sum())
        dists.append(candidate)
    return dists, signs, paulis, ideals, gate_counts


def _join(values) -> str:
    return ";".join(f"{float(value):.6f}" for value in np.ravel(values))


def simulate_task(tasks: pd.DataFrame, noise: Noise, config: dict) -> list[dict]:
    """Return every repeat of one task and noise condition, at every budget."""

    dists, signs, paulis, ideals, gate_counts = exact_distributions(tasks, noise, config)
    count = len(dists)
    width = len(paulis[0])
    task_id = tasks.task_id.iloc[0]
    rows = []
    means = [[sign for _ in range(3)] for sign in signs]
    expectations = [[float(d @ s) for d in dist] for dist, s in zip(dists, signs, strict=True)]
    for budget in config["budgets"]:
        for repeat in range(int(config["repeats"])):
            seed = derive_seed("selector-v3-sample", task_id, noise.name, budget, repeat)
            rng = np.random.default_rng(seed)
            per_candidate = split_evenly(budget, count)
            base_raw = [plain(rng, per_candidate[k], expectations[k][0])[0] for k in range(count)]
            base_zne = []
            for k in range(count):
                shots = split_evenly(per_candidate[k], 3)
                base_zne.append(
                    sum(
                        LINEAR_WEIGHTS[i] * plain(rng, shots[i], expectations[k][i])[0]
                        for i in range(3)
                    )
                )
            calibration = int(budget * float(config["baseline_calibration_fraction"]))
            scale = calibration_scale(noise.name, 0.0, config)
            errors = sample_calibration(rng, calibration, width, noise, scale)
            weights = [_weights(paulis[k], *errors) for k in range(count)]
            target = split_evenly(budget - calibration, count)
            base_readout = [
                corrected(rng, target[k], dists[k][0], weights[k])[0] for k in range(count)
            ]
            errors = sample_calibration(rng, calibration, width, noise, scale)
            weights = [_weights(paulis[k], *errors) for k in range(count)]
            _, _, fold_values, _ = _fold_sets(rng, budget - calibration, dists, means, weights)
            base_combined = fold_values @ LINEAR_WEIGHTS

            # The pilot spends a fixed part of the budget, and the rest goes to one branch.
            pilot_cal = int(budget * float(config["pilot_calibration_fraction"]))
            pre, post = split_evenly(pilot_cal, 2)
            first = sample_calibration(rng, pre, width, noise, scale)
            second = sample_calibration(
                rng, post, width, noise, calibration_scale(noise.name, 1.0, config)
            )
            pooled = (
                first[0] + second[0],
                first[1] + second[1],
                first[2] + second[2],
                first[3] + second[3],
            )
            drift = _drift_statistic(first, second)
            weights = [_weights(paulis[k], *pooled) for k in range(count)]
            pilot_folds = int(budget * float(config["pilot_fold_fraction"]))
            p_plain, p_plain_se, p_corr, p_corr_se = _fold_sets(
                rng, pilot_folds, dists, means, weights
            )
            pilot_fit = p_corr @ LINEAR_WEIGHTS
            pilot_fit_se = np.sqrt(p_corr_se**2 @ LINEAR_WEIGHTS**2)

            remaining = budget - pilot_cal - pilot_folds
            split = split_evenly(remaining, count)
            branch_raw = [plain(rng, split[k], expectations[k][0]) for k in range(count)]
            branch_readout = [
                corrected(rng, split[k], dists[k][0], weights[k]) for k in range(count)
            ]
            b_plain, b_plain_se, b_corr, b_corr_se = _fold_sets(
                rng, remaining, dists, means, weights
            )
            outcome = certify(rng, budget, dists, signs, paulis, noise, config)
            rows.append(
                {
                    "task_id": task_id,
                    "noise": noise.name,
                    "budget": budget,
                    "repeat": repeat,
                    "seed": seed,
                    "base_raw": _join(base_raw),
                    "base_readout": _join(base_readout),
                    "base_zne": _join(base_zne),
                    "base_combined": _join(base_combined),
                    "drift_statistic": round(drift, 4),
                    "pilot_raw": _join(p_plain[:, 0]),
                    "pilot_raw_se": _join(p_plain_se[:, 0]),
                    "pilot_readout": _join(p_corr[:, 0]),
                    "pilot_fit": _join(pilot_fit),
                    "pilot_fit_se": _join(pilot_fit_se),
                    "branch_raw": _join([item[0] for item in branch_raw]),
                    "branch_raw_se": _join([item[1] for item in branch_raw]),
                    "branch_readout": _join([item[0] for item in branch_readout]),
                    "branch_readout_se": _join([item[1] for item in branch_readout]),
                    "branch_zne": _join(b_plain @ LINEAR_WEIGHTS),
                    "branch_zne_se": _join(np.sqrt(b_plain_se**2 @ LINEAR_WEIGHTS**2)),
                    "branch_combined": _join(b_corr @ LINEAR_WEIGHTS),
                    "branch_combined_se": _join(np.sqrt(b_corr_se**2 @ LINEAR_WEIGHTS**2)),
                    **outcome,
                    "result_label": config["evidence_label"],
                    "status": "complete",
                }
            )
    return rows


def _drift_statistic(first, second) -> float:
    """Return the largest per-qubit z between the pilot's two calibrations."""

    values = []
    for index in (0, 1):
        trials_a, trials_b = first[2 + index], second[2 + index]
        rate_a, rate_b = first[index] / trials_a, second[index] / trials_b
        pooled = (first[index] + second[index]) / (trials_a + trials_b)
        spread = np.sqrt(pooled * (1.0 - pooled) * (1.0 / trials_a + 1.0 / trials_b))
        values.append(np.abs(rate_a - rate_b) / np.maximum(spread, 1e-9))
    return float(np.max(values))


def _task(item) -> list[dict]:
    record, noise_name, config = item
    frame = pd.DataFrame(record)
    noise = Noise(noise_name, **config["noise"][noise_name])
    return simulate_task(frame, noise, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_manifest_hash()
    manifest = pd.read_csv(MANIFEST_PATH)
    items = [
        (group.to_dict("records"), noise, CONFIG)
        for _, group in manifest.groupby("task_id", sort=True)
        for noise in CONFIG["noise"]
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_task, items, chunksize=4))
    else:
        results = [_task(item) for item in items]
    frame = pd.DataFrame([row for result in results for row in result])
    frame.to_csv(RAW_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
