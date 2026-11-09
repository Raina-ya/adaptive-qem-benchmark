"""Mismatch dose-response: calibrate on A_cal, measure under (1-lambda) A_cal + lambda A_mis."""

from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src import readout_boundaries as rb
from src.density_sim import derive_seed
from src.generate_readout_feasibility_v2 import CONFIG, candidate_distributions, check_config_hash

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/readout_mismatch_dose_v1.csv"


def scenario_channels(width: int, scenario: dict) -> tuple[rb.Channel, rb.Channel]:
    return rb.Channel.uniform(width, *scenario["cal"]), rb.Channel.uniform(
        width, *scenario["mismatch"]
    )


def check_statistic(rng, channel, cal_p01, cal_p10, shots: int) -> float:
    """Measure all zeros and all ones under the current channel, and compare with calibration.

    Return the largest per-qubit two-sample z. Its shots are diagnostic and are reported.
    """

    current_p01, current_p10 = rb.calibrate_tensor(rng, channel, 2 * shots)
    values = []
    for current, old in ((current_p01, cal_p01), (current_p10, cal_p10)):
        pooled = (current + old) / 2.0
        spread = np.sqrt(np.maximum(pooled * (1.0 - pooled), 1e-12) * 2.0 / shots)
        values.append(np.abs(current - old) / spread)
    return float(np.max(values))


def run_cell(width: int, name: str, config: dict) -> list[dict]:
    settings = config["mismatch"]
    calibrated, mismatched = scenario_channels(width, settings["scenarios"][name])
    signs = rb.parity_signs(width)
    candidates = [
        item
        for item in candidate_distributions(width, config)
        if item[1] in settings["candidate_pairs"]
    ]
    full_columns = None
    if width <= settings["full_matrix_max_width"]:
        full_columns = np.column_stack(
            [rb.apply_channel(rb.basis(width, index), calibrated) for index in range(2**width)]
        )
    rows = []
    for repeat in range(int(settings["repeats"])):
        rng = np.random.default_rng(derive_seed("readout-mismatch-dose", width, name, repeat))
        # One calibration per repeat is reused across every lambda, as a stale calibration would be.
        p01, p10 = rb.calibrate_tensor(rng, calibrated, int(settings["tensor_shots"]))
        blocks = rb.calibrate_pairs(rng, calibrated, int(settings["pair_shots"]))
        full = None
        if full_columns is not None:
            shots = int(settings["full_shots_per_state"])
            full = np.column_stack([rng.multinomial(shots, c) / shots for c in full_columns.T])
        conditions = {
            "tensor_product": float(
                np.prod([np.linalg.cond(rb.local(a, b)) for a, b in zip(p01, p10, strict=True)])
            ),
            "pairwise_local": float(np.prod([np.linalg.cond(block) for block in blocks])),
            "full_matrix": float(np.linalg.cond(full)) if full is not None else math.nan,
            "none": math.nan,
        }
        for weight in settings["lambdas"]:
            channel = rb.Mixture(calibrated, mismatched, float(weight))
            check = check_statistic(rng, channel, p01, p10, int(settings["check_shots"]))
            for family, pair, high, low in candidates:
                observed = [rb.apply_channel(high, channel), rb.apply_channel(low, channel)]
                target = int(settings["target_shots"])
                counts = [
                    rng.multinomial(shots, dist).astype(float)
                    for dist, shots in zip(
                        observed, (target - target // 2, target // 2), strict=True
                    )
                ]
                for method in settings["methods"]:
                    if method == "full_matrix" and full is None:
                        continue
                    results = [
                        _correct(method, c, p01, p10, blocks, full, conditions, signs)
                        for c in counts
                    ]
                    rows.append(
                        {
                            "width": width,
                            "scenario": name,
                            "lambda": float(weight),
                            "distribution": family,
                            "candidate_pair": pair,
                            "method": method,
                            "repeat": repeat,
                            "estimate_high": round(results[0][0], 7),
                            "estimate_low": round(results[1][0], 7),
                            "standard_error_high": round(
                                results[0][2].get("standard_error", math.nan), 7
                            ),
                            "status": results[0][1]
                            if results[0][1] != "complete"
                            else results[1][1],
                            "condition_number": conditions[method],
                            "negative_mass_high": results[0][2].get("negative_mass", math.nan),
                            "check_z": round(check, 6),
                        }
                    )
    return rows


def _correct(method, counts, p01, p10, blocks, full, conditions, signs):
    if method == "none":
        total = counts.sum()
        estimate = float(counts @ signs) / total
        return (
            estimate,
            "complete",
            {"standard_error": math.sqrt(max(1.0 - estimate**2, 1.0 / total) / total)},
        )
    if method == "tensor_product":
        return rb.correct_tensor(counts, p01, p10)
    if method == "pairwise_local":
        return rb.correct_pairs(counts, blocks)
    return rb.correct_full(counts, full, conditions["full_matrix"])


def _task(item):
    width, name, config = item
    return run_cell(width, name, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    check_config_hash()
    settings = CONFIG["mismatch"]
    items = [
        (width, name, CONFIG) for width in settings["widths"] for name in settings["scenarios"]
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_task, items))
    else:
        results = [_task(item) for item in items]
    frame = pd.DataFrame([row for result in results for row in result])
    frame.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.7g")


if __name__ == "__main__":
    main()
