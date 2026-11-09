"""Width feasibility of readout correction, at fixed total budget and fixed calibration quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import readout_boundaries as rb
from src.density_sim import derive_seed

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/readout_boundaries_v1.yaml"
HASH_PATH = ROOT / "experiments/configs/readout_boundaries_v1.sha256"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RAW_PATH = ROOT / "results/raw/readout_feasibility_v2.csv"


def config_digest() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_config_hash() -> None:
    frozen = HASH_PATH.read_text(encoding="utf-8").strip()
    if config_digest() != frozen:
        raise ValueError(f"Config hash differs from frozen {frozen}")


def channel_for(width: int, settings: dict) -> rb.Channel:
    return rb.Channel.uniform(width, settings["p01"], settings["p10"], settings["pair_flip"])


def candidate_distributions(
    width: int, config: dict
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    builders = {"product": rb.product_distribution, "sparse": rb.sparse_distribution}
    output = []
    for family in config["distributions"]:
        for name, (high, low) in config["candidates"].items():
            output.append(
                (family, name, builders[family](width, high), builders[family](width, low))
            )
    return output


def regimes(config: dict) -> list[dict]:
    settings = config["feasibility"]
    rows = []
    for budget in settings["fixed_total_budgets"]:
        rows.append({"regime": "fixed_total", "setting": budget})
    for shots in settings["fixed_quality_shots_per_calibration_circuit"]:
        rows.append({"regime": "fixed_quality", "setting": shots})
    return rows


def shots_for(regime: dict, method: str, width: int, config: dict) -> tuple[int, int, int]:
    """Return target shots, calibration circuits, and shots per calibration circuit."""

    settings = config["feasibility"]
    circuits = {
        "none": 0,
        "full_matrix": 2**width,
        "tensor_product": 2,
        "pairwise_local": 4,
        "observed_subspace": 2,
        "matrix_free": 2,
    }[method]
    if regime["regime"] == "fixed_total":
        budget = int(regime["setting"])
        if method == "none":
            return budget, 0, 0
        calibration = int(budget * settings["fixed_total_calibration_fraction"])
        return budget - calibration, circuits, calibration // circuits
    target = int(settings["fixed_quality_target_shots"])
    return target, circuits, int(regime["setting"]) if circuits else 0


def run_cell(width: int, model: str, regime: dict, config: dict) -> list[dict]:
    settings = config["feasibility"]
    channel = channel_for(width, settings["models"][model])
    signs = rb.parity_signs(width)
    full_columns = None
    if width <= settings["full_matrix_max_width"]:
        full_columns = np.column_stack(
            [rb.apply_channel(rb.basis(width, index), channel) for index in range(2**width)]
        )
    candidates = candidate_distributions(width, config)
    observed = [
        (family, name, rb.apply_channel(high, channel), rb.apply_channel(low, channel))
        for family, name, high, low in candidates
    ]
    rows = []
    for repeat in range(int(settings["repeats"])):
        for method in settings["methods"]:
            if method == "full_matrix" and full_columns is None:
                continue
            seed = derive_seed(
                "readout-feasibility-v2",
                width,
                model,
                regime["regime"],
                regime["setting"],
                method,
                repeat,
            )
            rng = np.random.default_rng(seed)
            target, circuits, per_circuit = shots_for(regime, method, width, config)
            calibration, extra = _calibrate(rng, method, channel, per_circuit, full_columns)
            condition = _condition(method, calibration, width, repeat, settings)
            for family, name, high, low in observed:
                halves = (target - target // 2, target // 2)
                estimates, statuses, details = [], [], []
                for distribution, shots in zip((high, low), halves, strict=True):
                    counts = rng.multinomial(shots, distribution).astype(float)
                    estimate, status, detail = _correct(
                        method, counts, calibration, condition, signs
                    )
                    estimates.append(estimate)
                    statuses.append(status)
                    details.append(detail)
                rows.append(
                    {
                        "width": width,
                        "model": model,
                        "regime": regime["regime"],
                        "regime_setting": regime["setting"],
                        "distribution": family,
                        "candidate_pair": name,
                        "method": method,
                        "repeat": repeat,
                        "seed": seed,
                        "target_shots": target,
                        "calibration_circuits": circuits,
                        "shots_per_calibration_circuit": per_circuit,
                        "calibration_shots": circuits * per_circuit,
                        "estimate_high": round(estimates[0], 10),
                        "estimate_low": round(estimates[1], 10),
                        "status": statuses[0] if statuses[0] != "complete" else statuses[1],
                        "condition_number": condition,
                        "negative_mass_high": details[0].get("negative_mass", math.nan),
                        "support_high": details[0].get("support", math.nan),
                        "iterations_high": details[0].get("iterations", math.nan),
                        "zero_diagonal_fraction": extra.get("zero_diagonal_fraction", math.nan),
                        "result_label": config["evidence_label"],
                    }
                )
    return rows


def _calibrate(rng, method, channel, per_circuit, full_columns):
    if method == "none":
        return None, {}
    if method == "full_matrix":
        if per_circuit <= 0:
            return np.zeros_like(full_columns), {"zero_diagonal_fraction": 1.0}
        estimate = np.column_stack(
            [rng.multinomial(per_circuit, column) / per_circuit for column in full_columns.T]
        )
        return estimate, {"zero_diagonal_fraction": float(np.mean(np.diag(estimate) == 0.0))}
    if method == "pairwise_local":
        return rb.calibrate_pairs(rng, channel, 4 * per_circuit), {}
    return rb.calibrate_tensor(rng, channel, 2 * per_circuit), {}


def _condition(method, calibration, width, repeat, settings) -> float:
    if method == "none" or calibration is None:
        return math.nan
    if method == "full_matrix":
        if not np.any(calibration):
            return math.inf
        if width >= 10 and repeat >= int(settings["condition_repeats_at_width_10"]):
            return math.nan
        return float(np.linalg.cond(calibration))
    if method == "pairwise_local":
        return float(np.prod([np.linalg.cond(block) for block in calibration]))
    p01, p10 = calibration
    return float(np.prod([np.linalg.cond(rb.local(a, b)) for a, b in zip(p01, p10, strict=True)]))


def _correct(method, counts, calibration, condition, signs):
    if method == "none":
        return rb.raw_parity(counts, signs), "complete", {}
    if method == "full_matrix":
        known = None if math.isnan(condition) else condition
        return rb.correct_full(counts, calibration, known)
    if method == "pairwise_local":
        return rb.correct_pairs(counts, calibration)
    p01, p10 = calibration
    if method == "tensor_product":
        return rb.correct_tensor(counts, p01, p10)
    return rb.correct_subspace(counts, p01, p10, iterative=method == "matrix_free")


def _task(item):
    width, model, regime, config = item
    return run_cell(width, model, regime, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_config_hash()
    settings = CONFIG["feasibility"]
    items = [
        (width, model, regime, CONFIG)
        for width in sorted(settings["widths"], reverse=True)
        for model in settings["models"]
        for regime in regimes(CONFIG)
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_task, items))
    else:
        results = [_task(item) for item in items]
    frame = pd.DataFrame([row for result in results for row in result])
    frame = frame.sort_values(
        ["width", "model", "regime", "regime_setting", "method", "repeat"], kind="mergesort"
    )
    frame.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.8g")


if __name__ == "__main__":
    main()
