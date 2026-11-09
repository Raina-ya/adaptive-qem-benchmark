"""Run new independent repeats for the frozen exact-engine comparisons and holdout tasks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.density_sim import derive_seed
from src.power_analysis_v1 import PLAN_HASH_PATH, PLAN_PATH
from src.sequential_decision import (
    CellSpec,
    load_config,
    score_policy,
    simulate_trajectory,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/confirmatory_v1.yaml"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
RAW_PATH = ROOT / "results/raw/confirmatory_exact_v1.csv"


def manifest_digest() -> str:
    return hashlib.sha256(PLAN_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_manifest_hash() -> None:
    frozen = PLAN_HASH_PATH.read_text(encoding="utf-8").strip()
    if manifest_digest() != frozen:
        raise ValueError(f"Confirmatory manifest differs from frozen {frozen}")


def engine_config() -> dict:
    """Add the holdout tasks to the exact engine, leaving its v1 settings unchanged."""

    source = copy.deepcopy(load_config())
    source["tasks"].update(CONFIG["holdout_tasks"])
    return source


def sample_outcomes(
    group: str, spec: CellSpec, repeats: int, source: dict
) -> list[dict[str, object]]:
    """Score fixed-budget forced decisions from seeds unrelated to any v1 seed."""

    rows = []
    for repeat in range(repeats):
        seed = derive_seed(
            "confirmatory-v1",
            spec.task,
            spec.margin,
            spec.noise,
            spec.max_total_shots,
            spec.method,
            repeat,
        )
        trajectory = simulate_trajectory(spec, seed, source)
        scored = score_policy("fixed_budget", trajectory, spec.margin, 0.05)
        rows.append(
            {
                "group": group,
                "task": spec.task,
                "margin": spec.margin,
                "noise": spec.noise,
                "budget": spec.max_total_shots,
                "method": spec.method,
                "repeat": repeat,
                "seed": seed,
                "decision_error": bool(scored["forced_error"]),
                "gap_estimate": round(float(scored["gap_estimate"]), 8),
                "status": trajectory.status,
                "result_label": CONFIG["evidence_label"],
            }
        )
    return rows


def planned_runs(plan: pd.DataFrame) -> dict[tuple, int]:
    """Give each method in a cell the most repeats any of its comparisons needs."""

    runs: dict[tuple, int] = {}
    exact = plan.loc[plan.engine == "exact_channel_sampling"]
    for row in exact.itertuples():
        for method in (row.first, row.second):
            key = ("confirmatory", row.task, float(row.margin), row.noise, int(row.budget), method)
            runs[key] = max(runs.get(key, 0), int(row.pairs_planned))
    return runs


def holdout_runs() -> dict[tuple, int]:
    runs = {}
    methods = sorted({method for pair in CONFIG["holdout_comparisons"] for method in pair})
    for task, definition in CONFIG["holdout_tasks"].items():
        for margin in definition["margins"]:
            for noise in definition["noise"]:
                for budget in CONFIG["holdout_budgets"]:
                    for method in methods:
                        key = ("holdout", task, float(margin), noise, int(budget), method)
                        runs[key] = int(CONFIG["holdout_repeats"])
    return runs


def main() -> None:
    check_manifest_hash()
    source = engine_config()
    runs = {**planned_runs(pd.read_csv(PLAN_PATH)), **holdout_runs()}
    rows = []
    for (group, task, margin, noise, budget, method), repeats in runs.items():
        spec = CellSpec(task, margin, noise, budget, method)
        rows.extend(sample_outcomes(group, spec, repeats, source))
    pd.DataFrame(rows).to_csv(RAW_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
