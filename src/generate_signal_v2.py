"""Run the frozen v2 benchmark by sampling shots from exact density-matrix probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import (
    Noise,
    derive_seed,
    fold_global,
    fold_local,
    garbage_fold,
    hardware_efficient_gates,
    ideal_expectation,
    outcome_probabilities,
    parity_signs,
)
from src.signal_manifest_v2 import CONFIG, HASH_PATH, MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/signal_v2_repeats.csv"
EXACT_PATH = ROOT / "results/processed/signal_v2_exact.csv"
TRUTH_PATH = ROOT / "results/processed/signal_v2_truth.csv"
RESULT_LABEL = "noisy_simulated"
GARBAGE_FACTORS = (1, 3, 5)
# Equal rounding keeps sampled counts the same across platforms' floating-point paths.
PROBABILITY_DECIMALS = 12


def check_manifest_hash() -> None:
    """Refuse to run if the manifest differs from the frozen one."""

    digest = hashlib.sha256(MANIFEST_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    frozen = HASH_PATH.read_text(encoding="utf-8").strip()
    if digest != frozen:
        raise ValueError(f"Manifest hash {digest} differs from frozen {frozen}")


def noise_conditions(config: dict | None = None) -> dict[str, Noise]:
    source = (config or CONFIG)["noise"]
    return {name: Noise(name, **values) for name, values in source.items()}


def split_evenly(total: int, parts: int) -> list[int]:
    base, extra = divmod(total, parts)
    return [base + (index < extra) for index in range(parts)]


def fit_weights(factors: tuple[int, ...], degree: int) -> np.ndarray:
    """Return least-squares weights that map factor estimates to the value at factor zero."""

    vandermonde = np.vander(np.asarray(factors, dtype=float), degree + 1, increasing=True)
    return np.linalg.pinv(vandermonde)[0]


def model_degree(model: str, factors: tuple[int, ...]) -> int:
    if model == "linear":
        return 1
    if model == "quadratic":
        return 2
    if model == "richardson":
        return len(factors) - 1
    raise ValueError(f"No polynomial degree for {model}")


def fit_estimates(
    model: str, factors: tuple[int, ...], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extrapolate each row to zero, and flag rows whose fit is undefined.

    The exponential model is a line through log |value|. It needs one sign at every factor, so a
    row without that falls back to its factor-one estimate and stays flagged.
    """

    if model != "exponential":
        weights = fit_weights(factors, model_degree(model, factors))
        return values @ weights, np.zeros(len(values), dtype=bool)
    signs = np.where(values[:, 0] < 0.0, -1.0, 1.0)
    aligned = values * signs[:, None]
    undefined = np.any(aligned <= 0.0, axis=1)
    safe = np.where(aligned > 0.0, aligned, 1.0)
    logs = np.log(safe) @ fit_weights(factors, 1)
    return np.where(undefined, values[:, 0], signs * np.exp(logs)), undefined


def readout_weights(pauli: str, p01: np.ndarray, p10: np.ndarray) -> np.ndarray:
    """Return per-outcome values of the corrected observable, using the tensor-product inverse."""

    weights = np.array([1.0])
    for qubit, symbol in enumerate(pauli):
        if symbol == "I":
            factor = np.array([1.0, 1.0])
        else:
            scale = 1.0 - p01[qubit] - p10[qubit]
            factor = np.array([1.0 - p10[qubit] + p01[qubit], -(1.0 - p01[qubit] + p10[qubit])])
            factor = factor / scale
        weights = np.kron(factor, weights)
    return weights


def _corrected(
    rng: np.random.Generator,
    probabilities: list[np.ndarray],
    shots: list[int],
    calibration_shots: int,
    pauli: str,
    noise: Noise,
) -> tuple[list[float], list[float]]:
    """Sample one calibration, then correct every factor with it."""

    width = len(pauli)
    zero, one = split_evenly(calibration_shots, 2)
    p01 = rng.binomial(zero, noise.readout_p01, size=width) / zero
    p10 = rng.binomial(one, noise.readout_p10, size=width) / one
    weights = readout_weights(pauli, p01, p10)
    estimates, errors = [], []
    for probability, count in zip(probabilities, shots, strict=True):
        counts = rng.multinomial(count, probability)
        mean = float(counts @ weights) / count
        variance = float(counts @ weights**2) / count - mean**2
        estimates.append(mean)
        errors.append(float(np.sqrt(max(variance, 0.0) / count)))
    return estimates, errors


def _rounded(probability: np.ndarray) -> np.ndarray:
    values = np.round(probability, PROBABILITY_DECIMALS)
    return values / values.sum()


def _exact(row: pd.Series, noise: Noise, config: dict) -> tuple[dict, dict, float]:
    """Compute each fold's outcome distribution once."""

    pauli = row.observable_q0_first
    gates = hardware_efficient_gates(int(row.width), int(row.depth), int(row.parameter_seed))
    reference = ideal_expectation(gates, pauli)
    folded = {"global": fold_global, "local": fold_local}
    distributions: dict[tuple[str, int], np.ndarray] = {}
    for design in config["designs"].values():
        for factor in design["factors"]:
            key = (design["folding"], factor)
            if key not in distributions:
                distributions[key] = _rounded(
                    outcome_probabilities(folded[design["folding"]](gates, factor), pauli, noise)
                )
    for factor in GARBAGE_FACTORS:
        seed = derive_seed("signal-v2-garbage", row.circuit_id, factor)
        distributions[("garbage", factor)] = _rounded(
            outcome_probabilities(garbage_fold(gates, factor, seed), pauli, noise)
        )
    signs = parity_signs(pauli)
    expectations = {key: float(value @ signs) for key, value in distributions.items()}
    return distributions, expectations, reference


def simulate_condition(row: pd.Series, noise: Noise, config: dict) -> tuple[list, list, list]:
    """Return repeat rows, exact-mean rows, and truth rows for one circuit and noise condition."""

    pauli = row.observable_q0_first
    distributions, expectations, reference = _exact(row, noise, config)
    identity = {"circuit_id": row.circuit_id, "noise": noise.name}
    exact_rows = [
        {**identity, "folding": key[0], "factor": key[1], "expectation": value}
        for key, value in expectations.items()
    ]
    designs = {
        name: (
            design["folding"],
            tuple(design["factors"]),
            [expectations[(design["folding"], factor)] for factor in design["factors"]],
        )
        for name, design in config["designs"].items()
    }
    garbage_means = [expectations[("garbage", factor)] for factor in GARBAGE_FACTORS]
    repeat_rows, truth_rows = [], []
    for budget in config["budgets"]:
        truth_rows.extend(_truth(row, noise, budget, reference, designs, config))
        calibration = int(round(config["readout_fraction"] * budget))
        for repeat in range(config["repeats"]):
            seed = derive_seed("signal-v2-sample", row.circuit_id, noise.name, budget, repeat)
            rng = np.random.default_rng(seed)
            output = {
                **identity,
                "total_shot_budget": budget,
                "repeat": repeat,
                "seed": seed,
                "raw_plus": int(rng.binomial(budget, (1.0 + designs["global_135"][2][0]) / 2.0)),
            }
            for name, (_, factors, means) in designs.items():
                shots = split_evenly(budget, len(factors))
                plus = rng.binomial(shots, (1.0 + np.asarray(means)) / 2.0)
                output[f"{name}_plus"] = ";".join(str(int(value)) for value in plus)
            shots = split_evenly(budget, len(GARBAGE_FACTORS))
            plus = rng.binomial(shots, (1.0 + np.asarray(garbage_means)) / 2.0)
            output["garbage_135_plus"] = ";".join(str(int(value)) for value in plus)
            # Correction spends part of the same budget on calibration, so its cost is counted.
            remaining = budget - calibration
            base = distributions[("global", 1)]
            estimate, error = _corrected(rng, [base], [remaining], calibration, pauli, noise)
            output["readout_estimate"] = round(estimate[0], 10)
            output["readout_se"] = round(error[0], 10)
            shots = split_evenly(remaining, 3)
            folds = [distributions[("global", factor)] for factor in (1, 3, 5)]
            estimates, errors = _corrected(rng, folds, shots, calibration, pauli, noise)
            output["corrected_135"] = ";".join(f"{value:.10f}" for value in estimates)
            output["corrected_135_se"] = ";".join(f"{value:.10f}" for value in errors)
            output["result_label"] = RESULT_LABEL
            output["status"] = "complete"
            repeat_rows.append(output)
    return repeat_rows, exact_rows, truth_rows


def _truth(
    row: pd.Series, noise: Noise, budget: int, reference: float, designs: dict, config: dict
) -> list[dict[str, object]]:
    """Measure each fit's true error from independent draws, never from the repeat rows."""

    base = designs["global_135"][2][0]
    raw_rmse = float(np.sqrt((base - reference) ** 2 + (1.0 - base**2) / budget))
    draws = int(config["truth_draws"])
    rows = []
    for name, (folding, factors, means) in designs.items():
        rng = np.random.default_rng(
            derive_seed("signal-v2-truth", row.circuit_id, noise.name, budget, name)
        )
        shots = np.asarray(split_evenly(budget, len(factors)))
        probabilities = (1.0 + np.asarray(means)) / 2.0
        values = 2.0 * rng.binomial(shots, probabilities, size=(draws, len(factors))) / shots - 1.0
        for model in config["designs"][name]["models"]:
            estimates, undefined = fit_estimates(model, factors, values)
            clipped = np.clip(estimates, -1.0, 1.0)
            rows.append(
                {
                    "circuit_id": row.circuit_id,
                    "noise": noise.name,
                    "total_shot_budget": budget,
                    "design": name,
                    "folding": folding,
                    "model": model,
                    "reference": reference,
                    "raw_mean": base,
                    "raw_rmse": raw_rmse,
                    "fit_bias": float(estimates.mean() - reference),
                    "fit_rmse": float(np.sqrt(np.mean((estimates - reference) ** 2))),
                    "fit_rmse_clipped": float(np.sqrt(np.mean((clipped - reference) ** 2))),
                    "out_of_bounds_rate": float(np.mean(np.abs(estimates) > 1.0)),
                    "undefined_rate": float(undefined.mean()),
                    "draws": draws,
                }
            )
    return rows


def _task(item: tuple[dict, str, dict]) -> tuple[list, list, list]:
    record, noise_name, config = item
    noise = noise_conditions(config)[noise_name]
    return simulate_condition(pd.Series(record), noise, config)


def run(
    manifest: pd.DataFrame, config: dict, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tasks = [
        (record, noise, config)
        for record in manifest.to_dict("records")
        for noise in config["noise"]
    ]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_task, tasks, chunksize=4))
    else:
        results = [_task(task) for task in tasks]
    frames = (
        pd.DataFrame([row for result in results for row in result[index]]) for index in range(3)
    )
    return tuple(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_manifest_hash()
    manifest = pd.read_csv(MANIFEST_PATH)
    repeats, exact, truth = run(manifest, CONFIG, args.workers)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    repeats.to_csv(RAW_PATH, index=False, lineterminator="\n")
    exact.to_csv(EXACT_PATH, index=False, lineterminator="\n", float_format="%.12g")
    truth.to_csv(TRUTH_PATH, index=False, lineterminator="\n", float_format="%.12g")
    print(json.dumps({"repeat_rows": len(repeats), "truth_rows": len(truth)}))


if __name__ == "__main__":
    main()
