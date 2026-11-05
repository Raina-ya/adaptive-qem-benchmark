"""Generate opt-in QAOA selection pilot, full artifacts, summaries, and figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.applications.qaoa_selection import (
    CANDIDATE_MANIFEST_PATH,
    benchmark_graphs,
    run_candidate_selection_group,
    select_benchmark_candidates,
    selection_metrics_frame,
    summarize_selection_results,
    wilson_interval,
)
from src.noise_models import noise_spec

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/qaoa_selection_v1.yaml"
MANIFEST_HASH_PATH = ROOT / "experiments/manifests/qaoa_selection_v1_manifest.sha256"
SMOKE_OUTPUT = ROOT / "results/raw/qaoa_selection_smoke_v1.csv"
SMOKE_SUMMARY_OUTPUT = ROOT / "results/processed/qaoa_selection_smoke_summary_v1.csv"
PILOT_OUTPUT = ROOT / "results/raw/qaoa_selection_pilot_v1.csv"
FULL_OUTPUT = ROOT / "results/raw/qaoa_selection_v1.csv"
SUMMARY_OUTPUT = ROOT / "results/processed/qaoa_selection_summary_v1.csv"
ERROR_FIGURE = ROOT / "figures/qaoa_selection_error.png"
REGRET_FIGURE = ROOT / "figures/qaoa_simple_regret.png"
CERTIFICATION_FIGURE = ROOT / "figures/qaoa_shots_to_decision.png"
SMOKE_GRAPH_ID = "ring4"
SMOKE_BUDGET = 4096
SMOKE_REPEAT = 0
SIMULATOR_NOISE = noise_spec("mixed", "medium")
METHOD_COLORS = {"raw": "#4C78A8", "readout": "#F58518", "zne": "#54A24B"}


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def manifest_sha256() -> str:
    return hashlib.sha256(CANDIDATE_MANIFEST_PATH.read_bytes()).hexdigest()


def freeze_manifest_hash() -> None:
    MANIFEST_HASH_PATH.write_text(
        f"{manifest_sha256()}  {CANDIDATE_MANIFEST_PATH.name}\n", encoding="ascii"
    )


def _run_manifest_slice(*, repeats: int) -> pd.DataFrame:
    manifest = pd.read_csv(CANDIDATE_MANIFEST_PATH)
    config = _config()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    completed_groups = 0
    total_groups = (
        len(config["graph_ids"])
        * len(config["methods"])
        * len(config["total_selection_budgets"])
        * repeats
    )
    for graph in benchmark_graphs():
        candidates = select_benchmark_candidates(graph)
        for method in config["methods"]:
            for budget in config["total_selection_budgets"]:
                for repeat in range(repeats):
                    frozen = manifest[
                        (manifest.graph_id == graph.graph_id)
                        & (manifest.method == method)
                        & (manifest.total_selection_budget == budget)
                        & (manifest.repeat == repeat)
                    ]
                    seed_by_candidate = dict(
                        zip(frozen.candidate_id, frozen.seed, strict=True)
                    )
                    rows.extend(
                        run_candidate_selection_group(
                            graph,
                            candidates,
                            method=str(method),
                            total_selection_budget=int(budget),
                            repeat=repeat,
                            manifest_seeds=[
                                seed_by_candidate[candidate.candidate_id]
                                for candidate in candidates
                            ],
                            noise=SIMULATOR_NOISE,
                            collect_checkpoints=True,
                        )
                    )
                    completed_groups += 1
                    if completed_groups % 10 == 0 or completed_groups == total_groups:
                        elapsed = time.perf_counter() - started
                        print(
                            f"groups={completed_groups}/{total_groups} "
                            f"elapsed_seconds={elapsed:.1f}",
                            flush=True,
                        )
    return pd.DataFrame(rows)


def generate_smoke() -> pd.DataFrame:
    manifest = pd.read_csv(CANDIDATE_MANIFEST_PATH)
    graph = next(graph for graph in benchmark_graphs() if graph.graph_id == SMOKE_GRAPH_ID)
    candidates = select_benchmark_candidates(graph)
    rows: list[dict[str, object]] = []
    for method in _config()["methods"]:
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
                method=str(method),
                total_selection_budget=SMOKE_BUDGET,
                repeat=SMOKE_REPEAT,
                manifest_seeds=[
                    seed_by_candidate[candidate.candidate_id] for candidate in candidates
                ],
                noise=SIMULATOR_NOISE,
                collect_checkpoints=True,
            )
        )
    return pd.DataFrame(rows)


def audit_raw_results(frame: pd.DataFrame, *, repeats: int) -> None:
    config = _config()
    expected_rows = (
        len(config["graph_ids"])
        * len(config["methods"])
        * len(config["total_selection_budgets"])
        * int(config["selected_candidates_per_graph"])
        * repeats
    )
    if len(frame) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(frame)}")
    if not frame.status.isin({"complete", "failed", "not_applicable"}).all():
        raise RuntimeError("Raw results contain an undeclared status")
    if not frame.seed.is_unique:
        raise RuntimeError("Raw result seeds are not unique")
    group_columns = ["graph_id", "method", "total_selection_budget", "repeat"]
    for _, group in frame.groupby(group_columns, sort=False):
        selection_shots = int(group.selection_shots.iloc[0])
        charged = int(group.candidate_target_shots.sum())
        if group.method.iloc[0] == "readout":
            charged += int(group.calibration_shots.max())
        if charged != selection_shots:
            raise RuntimeError("Selection shot budget does not reconcile")
        if len(group) != int(config["selected_candidates_per_graph"]):
            raise RuntimeError("A candidate row was silently dropped")
        completed = group.status == "complete"
        if completed.any() and group.loc[completed, "raw_counts"].eq("").any():
            raise RuntimeError("A completed row is missing raw counts")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _aggregate_decisions(
    decisions: pd.DataFrame, regime: str, method: str, budget: int
) -> tuple[float, float, float, int]:
    cells = decisions[
        (decisions.candidate_gap_regime == regime)
        & (decisions.method == method)
        & (decisions.total_selection_budget == budget)
        & decisions.resolved
    ]
    if cells.empty:
        return math.nan, math.nan, math.nan, 0
    errors = int(cells.decision_error.sum())
    low, high = wilson_interval(errors, len(cells))
    return errors / len(cells), low, high, len(cells)


def _plot_selection_error(decisions: pd.DataFrame) -> None:
    regimes = ("clear", "near_tie")
    methods = tuple(METHOD_COLORS)
    budgets = sorted(decisions.total_selection_budget.unique())
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    width = 0.12
    base = np.arange(len(methods), dtype=float)
    for axis, regime in zip(axes, regimes, strict=True):
        for budget_index, budget in enumerate(budgets):
            values, lower, upper = [], [], []
            for method in methods:
                value, low, high, _ = _aggregate_decisions(
                    decisions, regime, method, int(budget)
                )
                values.append(value)
                lower.append(value - low)
                upper.append(high - value)
            positions = base + (budget_index - (len(budgets) - 1) / 2) * width
            axis.bar(
                positions,
                values,
                width,
                yerr=np.asarray([lower, upper]),
                capsize=3,
                label=f"B={budget}",
                alpha=0.85,
            )
        axis.set_title(f"{regime.replace('_', ' ').title()} decisions")
        axis.set_xticks(base, methods)
        axis.set_ylim(0.0, 1.05)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Selection-error probability (95% Wilson CI)")
    axes[1].legend(frameon=False)
    figure.suptitle("Mitigation does not uniformly reduce QAOA selection errors")
    figure.tight_layout()
    ERROR_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ERROR_FIGURE, dpi=180, metadata={"Software": "adaptive-qem-benchmark"})
    plt.close(figure)


def _plot_simple_regret(decisions: pd.DataFrame) -> None:
    cells = decisions[(decisions.candidate_gap_regime == "overall") & decisions.resolved]
    grouped = cells.groupby(["method", "total_selection_budget"]).simple_regret
    means = grouped.mean()
    errors = grouped.sem()
    methods = tuple(METHOD_COLORS)
    budgets = sorted(cells.total_selection_budget.unique())
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    width = 0.22
    base = np.arange(len(methods), dtype=float)
    for budget_index, budget in enumerate(budgets):
        values = [means.get((method, budget), math.nan) for method in methods]
        uncertainty = [errors.get((method, budget), math.nan) for method in methods]
        positions = base + (budget_index - (len(budgets) - 1) / 2) * width
        axis.bar(
            positions,
            values,
            width,
            yerr=uncertainty,
            capsize=3,
            label=f"B={budget}",
        )
    axis.set_xticks(base, methods)
    axis.set_ylabel("Mean simple regret (standard error)")
    axis.set_title("Wrong selections can have small application regret")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(REGRET_FIGURE, dpi=180, metadata={"Software": "adaptive-qem-benchmark"})
    plt.close(figure)


def _plot_certification(decisions: pd.DataFrame) -> None:
    cells = decisions[decisions.candidate_gap_regime == "overall"]
    methods = tuple(METHOD_COLORS)
    budgets = sorted(cells.total_selection_budget.unique())
    labels = [f"{method}\nB={budget}" for method in methods for budget in budgets]
    medians, unresolved = [], []
    for method in methods:
        for budget in budgets:
            group = cells[(cells.method == method) & (cells.total_selection_budget == budget)]
            certified = pd.to_numeric(
                group.loc[
                    group.certification_status == "certified",
                    "shots_to_certified_decision",
                ],
                errors="coerce",
            ).dropna()
            medians.append(certified.median() if not certified.empty else 0.0)
            unresolved.append(1.0 - len(certified) / len(group) if len(group) else math.nan)
    figure, axis = plt.subplots(figsize=(9.2, 4.6))
    positions = np.arange(len(labels))
    axis.bar(positions, medians, color=[METHOD_COLORS[label.split("\n")[0]] for label in labels])
    axis.set_ylabel("Median shots to certified decision (0 = none)")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", alpha=0.25)
    second = axis.twinx()
    second.plot(positions, unresolved, color="#B22222", marker="o", label="Unresolved fraction")
    second.set_ylim(0.0, 1.05)
    second.set_ylabel("Unresolved fraction")
    axis.set_title("Certification cost and unresolved decisions remain visible")
    second.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(
        CERTIFICATION_FIGURE,
        dpi=180,
        metadata={"Software": "adaptive-qem-benchmark"},
    )
    plt.close(figure)


def process_results(frame: pd.DataFrame) -> pd.DataFrame:
    config = _config()
    summary = summarize_selection_results(frame, top_k=int(config["top_k"]))
    _write_frame(summary, SUMMARY_OUTPUT)
    decisions = selection_metrics_frame(frame, top_k=int(config["top_k"]))
    _plot_selection_error(decisions)
    _plot_simple_regret(decisions)
    _plot_certification(decisions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--process-only", action="store_true")
    args = parser.parse_args()
    config = _config()
    if args.smoke:
        frame = generate_smoke()
        audit_raw_results(frame, repeats=1)
        _write_frame(frame, SMOKE_OUTPUT)
        _write_frame(
            summarize_selection_results(frame, top_k=int(config["top_k"])),
            SMOKE_SUMMARY_OUTPUT,
        )
        return
    if args.pilot:
        repeats = int(config["pilot_repeats"])
        frame = _run_manifest_slice(repeats=repeats)
        audit_raw_results(frame, repeats=repeats)
        _write_frame(frame, PILOT_OUTPUT)
        print(f"pilot_rows={len(frame)} statuses={frame.status.value_counts().to_dict()}")
        return
    if args.process_only:
        frame = pd.read_csv(FULL_OUTPUT, keep_default_na=False)
        audit_raw_results(frame, repeats=int(config["full_repeats"]))
        process_results(frame)
        return
    if args.full or not (args.smoke or args.pilot or args.process_only):
        freeze_manifest_hash()
        repeats = int(config["full_repeats"])
        frame = _run_manifest_slice(repeats=repeats)
        audit_raw_results(frame, repeats=repeats)
        _write_frame(frame, FULL_OUTPUT)
        summary = process_results(frame)
        print(f"full_rows={len(frame)} summary_rows={len(summary)}")
        return


if __name__ == "__main__":
    main()
