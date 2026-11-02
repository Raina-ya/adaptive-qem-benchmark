"""Build the deterministic Day 8-12 runtime report from its recorded ledger."""

# ruff: noqa: E501

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results/processed/day08_12_runtime.csv"
SUMMARY_PATH = ROOT / "results/processed/day08_12_runtime_summary.md"
BREAKDOWN_PATH = ROOT / "figures/day08_12_runtime_breakdown.png"
SCATTER_PATH = ROOT / "figures/day08_12_simulations_vs_time.png"
PDF_PATH = ROOT / "docs/day08_12_runtime_report.pdf"
REPORT_DATE = "2026-11-02"


def load_runtime() -> pd.DataFrame:
    """Load and validate the immutable runtime ledger."""

    frame = pd.read_csv(CSV_PATH, skipinitialspace=True)
    required = {
        "stage",
        "day",
        "description",
        "start_time",
        "end_time",
        "wall_seconds",
        "timing_kind",
        "cells",
        "repeats",
        "total_simulations",
        "notes",
    }
    if set(frame.columns) != required:
        raise ValueError("Runtime CSV schema does not match the report contract")
    if (frame.wall_seconds < 0).any() or frame.stage.duplicated().any():
        raise ValueError("Runtime stages must be unique with non-negative durations")
    if not frame.timing_kind.isin({"measured", "estimated"}).all():
        raise ValueError("Every runtime row must be classified as measured or estimated")
    return frame


def source_revision() -> str:
    """Return the commit that last changed the runtime ledger."""

    try:
        return subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(CSV_PATH.relative_to(ROOT))],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def create_breakdown(frame: pd.DataFrame) -> None:
    """Create a descending horizontal runtime chart."""

    ordered = frame.sort_values("wall_seconds")
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    bars = axis.barh(ordered.stage, ordered.wall_seconds, color="#326fa8")
    axis.set_xscale("log")
    axis.set(xlabel="Wall time (seconds, log scale)", ylabel="Stage")
    axis.set_title("Day 8-12 wall time by recorded stage")
    for bar, seconds in zip(bars, ordered.wall_seconds, strict=True):
        axis.annotate(
            f"{seconds:.1f}s",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(BREAKDOWN_PATH, dpi=180)
    plt.close(figure)


def create_simulation_scatter(frame: pd.DataFrame) -> None:
    """Compare known logical simulation counts with wall time."""

    known = frame.dropna(subset=["total_simulations"]).copy()
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    axis.scatter(known.total_simulations, known.wall_seconds, s=58, color="#b14a3b")
    for row in known.itertuples():
        axis.annotate(
            row.stage,
            (row.total_simulations, row.wall_seconds),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xscale("symlog", linthresh=10)
    axis.set_yscale("log")
    axis.set(xlabel="Known logical sampling executions", ylabel="Wall time (seconds, log scale)")
    axis.set_title("Simulation workload versus recorded wall time")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(SCATTER_PATH, dpi=180)
    plt.close(figure)


def runtime_fractions(frame: pd.DataFrame) -> tuple[float, float, float]:
    """Return simulation, preparation, and testing shares of recorded wall time."""

    simulation_stages = {
        "day09_artifacts",
        "day11_first_run",
        "day11_rerun",
        "day12_aborted",
        "day12_final",
    }
    testing_stages = {"pytest", "ruff"}
    total = frame.wall_seconds.sum()
    simulation = frame.loc[frame.stage.isin(simulation_stages), "wall_seconds"].sum()
    testing = frame.loc[frame.stage.isin(testing_stages), "wall_seconds"].sum()
    preparation = total - simulation - testing
    return simulation / total, preparation / total, testing / total


def build_summary(frame: pd.DataFrame) -> None:
    """Write the concise Markdown handoff summary."""

    sim_share, prep_share, test_share = runtime_fractions(frame)
    known_simulations = int(frame.total_simulations.fillna(0).sum())
    first = float(frame.loc[frame.stage == "day11_first_run", "wall_seconds"].iloc[0])
    rerun = float(frame.loc[frame.stage == "day11_rerun", "wall_seconds"].iloc[0])
    savings = 1.0 - rerun / first
    table_lines = [
        "| Stage | Day | Wall seconds | Timing | Cells | Simulations |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for row in frame.itertuples():
        cells = "" if pd.isna(row.cells) else str(int(row.cells))
        simulations = "" if pd.isna(row.total_simulations) else str(int(row.total_simulations))
        table_lines.append(
            f"| {row.stage} | {row.day} | {row.wall_seconds:.3f} | {row.timing_kind} | {cells} | {simulations} |"
        )
    table = "\n".join(table_lines)
    text = f"""# Day 8-12 Runtime Summary

{table}

- The final Day 12 GHZ run dominated measured wall time at 942.0 seconds.
- Readout calibration optimization reduced the comparable Day 11 pilot from 103.8 to 28.0 seconds ({savings:.1%} lower wall time).
- Known completed workloads account for {known_simulations:,} logical sampling executions; the aborted Day 12 attempt is excluded because it wrote no checkpoint.
- Recorded wall time was {sim_share:.1%} simulation, {prep_share:.1%} artifact/code preparation, and {test_share:.1%} testing and linting.
- ZNE charges three folded-circuit samples per result while preserving the same total shot budget.
- The final benchmark retained 5,372 complete rows and 28 explicit failures; all failures were width-6, 1,024-shot readout corrections with ill-conditioned sampled assignment matrices (25 high-noise and 3 medium-noise).
- Calibration acceleration changed execution strategy, not the declared readout channel, seeds, shot accounting, or failure policy.
"""
    SUMMARY_PATH.write_text(text, encoding="utf-8")


def add_footer(figure: plt.Figure, page: int, revision: str) -> None:
    figure.text(
        0.02,
        0.015,
        f"Generated by src/generate_runtime_report.py | source revision {revision}",
        fontsize=7,
        color="0.35",
    )
    figure.text(0.98, 0.015, f"Page {page}", ha="right", fontsize=7, color="0.35")


def build_pdf(frame: pd.DataFrame) -> None:
    """Build a compact three-page PDF with tables, figures, and interpretation."""

    revision = source_revision()
    sim_share, prep_share, test_share = runtime_fractions(frame)
    with PdfPages(
        PDF_PATH,
        metadata={
            "Title": "Day 8-12 Runtime and Performance Report",
            "Author": "adaptive-qem-benchmark",
            "CreationDate": datetime(2026, 11, 2, tzinfo=UTC),
            "ModDate": datetime(2026, 11, 2, tzinfo=UTC),
        },
    ) as pdf:
        page1 = plt.figure(figsize=(8.27, 11.69))
        page1.text(
            0.07, 0.95, "Day 8-12 Runtime and Performance Report", fontsize=20, weight="bold"
        )
        page1.text(
            0.07, 0.918, f"Project: adaptive-qem-benchmark | Batch date: {REPORT_DATE}", fontsize=10
        )
        summary = (
            "The final 50-repeat GHZ benchmark was the dominant measured stage. Readout "
            "calibration optimization cut the comparable Day 11 pilot wall time by about "
            "73%, while retaining fixed shot budgets, deterministic seeds, raw rows, and "
            "explicit failure statuses. Times marked estimated use visible session intervals "
            "or artifact timestamps rather than unavailable per-stage timers."
        )
        page1.text(0.07, 0.865, summary, fontsize=10, va="top", wrap=True)
        display = frame[
            ["stage", "day", "wall_seconds", "timing_kind", "cells", "total_simulations"]
        ].copy()
        display["timing_kind"] = display.timing_kind.replace({"estimated": "estimated*"})
        display.columns = ["Stage", "Day", "Seconds", "Timing", "Cells", "Simulations"]
        display = display.fillna("")
        table_axis = page1.add_axes((0.06, 0.24, 0.88, 0.55))
        table_axis.axis("off")
        table = table_axis.table(
            cellText=display.values, colLabels=display.columns, loc="center", cellLoc="left"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.55)
        page1.text(
            0.07,
            0.17,
            f"Recorded allocation: {sim_share:.1%} simulation | {prep_share:.1%} preparation | {test_share:.1%} testing/lint",
            fontsize=10,
            weight="bold",
        )
        page1.text(
            0.07,
            0.12,
            "* Estimated durations are identified per row. The aborted Day 12 attempt has unknown completed-cell and simulation counts because no checkpoint was written.",
            fontsize=9,
            wrap=True,
        )
        add_footer(page1, 1, revision)
        pdf.savefig(page1)
        plt.close(page1)

        page2 = plt.figure(figsize=(8.27, 11.69))
        page2.text(0.06, 0.96, "Measured Runtime and Workload", fontsize=18, weight="bold")
        image1 = plt.imread(BREAKDOWN_PATH)
        image2 = plt.imread(SCATTER_PATH)
        axis1 = page2.add_axes((0.06, 0.52, 0.88, 0.39))
        axis1.imshow(image1)
        axis1.axis("off")
        axis2 = page2.add_axes((0.06, 0.08, 0.88, 0.39))
        axis2.imshow(image2)
        axis2.axis("off")
        add_footer(page2, 2, revision)
        pdf.savefig(page2)
        plt.close(page2)

        page3 = plt.figure(figsize=(8.27, 11.69))
        page3.text(0.07, 0.95, "Where time went", fontsize=18, weight="bold")
        where = (
            "Sampling dominated the batch. Linear ZNE executes factors 1, 3, and 5, so each "
            "ZNE row requires three circuit samples even though their shots sum to the same "
            "budget as competing methods. The frozen GHZ grid then multiplies this work by "
            "three widths, four noise levels, three budgets, three methods, and 50 repeats. "
            "Deeper folded circuits increase transpilation and simulation cost. Before the "
            "optimization, readout calibration launched each basis-state circuit separately; "
            "afterward, the same independent assignment channel was sampled with deterministic "
            "multinomial draws, sharply reducing process-launch overhead."
        )
        page3.text(0.07, 0.89, where, fontsize=11, va="top", wrap=True, linespacing=1.45)
        page3.text(0.07, 0.60, "Failure context", fontsize=16, weight="bold")
        failures = (
            "The final raw table contains 5,372 complete cells and 28 failed cells. Every "
            "failure is an explicit ill_conditioned readout status at width 6 and a 1,024-shot "
            "total budget: 25 under high noise and 3 under medium noise. These rows remain in "
            "the coverage audit and were not clipped, retried, or silently discarded."
        )
        page3.text(0.07, 0.55, failures, fontsize=11, va="top", wrap=True, linespacing=1.45)
        page3.text(0.07, 0.36, "What this does not include", fontsize=16, weight="bold")
        exclusions = (
            "No quantum-hardware queue or execution time is included. No LLM or token time is "
            "included. Unreported code-writing latency is not reconstructed. The aborted run's "
            "completed workload is unknown. Rows labeled estimated use session wait intervals, "
            "combined-command timing, or filesystem timestamps and must not be interpreted as "
            "instrumented profiler measurements."
        )
        page3.text(0.07, 0.31, exclusions, fontsize=11, va="top", wrap=True, linespacing=1.45)
        add_footer(page3, 3, revision)
        pdf.savefig(page3)
        plt.close(page3)


def main() -> None:
    frame = load_runtime()
    BREAKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    create_breakdown(frame)
    create_simulation_scatter(frame)
    build_summary(frame)
    build_pdf(frame)


if __name__ == "__main__":
    main()
