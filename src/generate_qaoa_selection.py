"""Generate the deliberately small Phase F-C QAOA selection smoke artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.applications.qaoa_selection import (
    CANDIDATE_MANIFEST_PATH,
    benchmark_graphs,
    run_candidate_selection_group,
    select_benchmark_candidates,
    summarize_selection_results,
)
from src.noise_models import noise_spec

ROOT = Path(__file__).resolve().parents[1]
SMOKE_OUTPUT = ROOT / "results/raw/qaoa_selection_smoke_v1.csv"
SMOKE_SUMMARY_OUTPUT = ROOT / "results/processed/qaoa_selection_smoke_summary_v1.csv"
CONFIG_PATH = ROOT / "experiments/configs/qaoa_selection_v1.yaml"
SMOKE_GRAPH_ID = "ring4"
SMOKE_BUDGET = 4096
SMOKE_REPEAT = 0
SMOKE_NOISE = noise_spec("mixed", "medium")


def generate_smoke() -> pd.DataFrame:
    """Run one graph, one repeat, and all three frozen methods."""

    manifest = pd.read_csv(CANDIDATE_MANIFEST_PATH)
    graph = next(graph for graph in benchmark_graphs() if graph.graph_id == SMOKE_GRAPH_ID)
    candidates = select_benchmark_candidates(graph)
    rows: list[dict[str, object]] = []
    for method in ("raw", "readout", "zne"):
        frozen = manifest[
            (manifest.graph_id == graph.graph_id)
            & (manifest.method == method)
            & (manifest.total_selection_budget == SMOKE_BUDGET)
            & (manifest.repeat == SMOKE_REPEAT)
        ]
        seed_by_candidate = dict(zip(frozen.candidate_id, frozen.seed, strict=True))
        rows.extend(
            run_candidate_selection_group(
                graph,
                candidates,
                method=method,
                total_selection_budget=SMOKE_BUDGET,
                repeat=SMOKE_REPEAT,
                manifest_seeds=[
                    seed_by_candidate[candidate.candidate_id] for candidate in candidates
                ],
                noise=SMOKE_NOISE,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the frozen one-instance, one-repeat validation slice",
    )
    args = parser.parse_args()
    if not args.smoke:
        parser.error("Phase F-C permits only the tiny smoke run; pass --smoke")
    frame = generate_smoke()
    SMOKE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(SMOKE_OUTPUT, index=False, lineterminator="\n", float_format="%.17g")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    summary = summarize_selection_results(frame, top_k=int(config["top_k"]))
    SMOKE_SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        SMOKE_SUMMARY_OUTPUT,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    print(f"wrote {len(frame)} rows to {SMOKE_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {len(summary)} rows to {SMOKE_SUMMARY_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
