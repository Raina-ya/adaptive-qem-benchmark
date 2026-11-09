"""Run new QAOA selection repeats with the original Aer engine. Each group is saved as it ends."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.applications.qaoa_selection import (
    benchmark_graphs,
    run_candidate_selection_group,
    select_benchmark_candidates,
)
from src.confirmatory_v1 import check_manifest_hash
from src.density_sim import derive_seed
from src.generate_qaoa_selection import SIMULATOR_NOISE
from src.power_analysis_v1 import PLAN_PATH

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/confirmatory_qaoa_v1.csv"
REPEAT_OFFSET = 100000
COLUMNS = [
    "graph_id",
    "method",
    "total_selection_budget",
    "repeat",
    "seed",
    "candidate_id",
    "exact_ideal_energy",
    "corrected_or_extrapolated_estimate",
    "status",
    "failure_reason",
    "result_label",
]


def planned_groups(plan: pd.DataFrame, roles: set[str]) -> dict[tuple[str, int, str], int]:
    runs: dict[tuple[str, int, str], int] = {}
    selected = plan.loc[(plan.engine == "aer") & plan.role.isin(roles)]
    for row in selected.itertuples():
        for method in (row.first, row.second):
            key = (row.task, int(row.budget), method)
            runs[key] = max(runs.get(key, 0), int(row.pairs_planned))
    return runs


def finished_groups() -> set[tuple[str, int, str, int]]:
    if not RAW_PATH.exists():
        return set()
    done = pd.read_csv(
        RAW_PATH, usecols=["graph_id", "total_selection_budget", "method", "repeat"]
    ).drop_duplicates()
    return set(done.itertuples(index=False, name=None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", default=["core_counterexample", "rank_uncertain"])
    args = parser.parse_args()
    check_manifest_hash()
    plan = pd.read_csv(PLAN_PATH)
    graphs = {graph.graph_id: graph for graph in benchmark_graphs()}
    done = finished_groups()
    started, completed = time.perf_counter(), 0
    for (graph_id, budget, method), repeats in planned_groups(plan, set(args.roles)).items():
        graph = graphs[graph_id]
        candidates = select_benchmark_candidates(graph)
        for index in range(repeats):
            repeat = REPEAT_OFFSET + index
            if (graph_id, budget, method, repeat) in done:
                continue
            seeds = [
                derive_seed(
                    "confirmatory-qaoa-v1", graph_id, budget, method, repeat, candidate.candidate_id
                )
                for candidate in candidates
            ]
            rows = run_candidate_selection_group(
                graph,
                candidates,
                method=method,
                total_selection_budget=budget,
                repeat=repeat,
                manifest_seeds=seeds,
                noise=SIMULATOR_NOISE,
                collect_checkpoints=False,
            )
            frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
            frame.to_csv(
                RAW_PATH, mode="a", header=not RAW_PATH.exists(), index=False, lineterminator="\n"
            )
            completed += 1
            if completed % 20 == 0:
                elapsed = time.perf_counter() - started
                print(f"groups={completed} elapsed_seconds={elapsed:.0f}", flush=True)


if __name__ == "__main__":
    main()
