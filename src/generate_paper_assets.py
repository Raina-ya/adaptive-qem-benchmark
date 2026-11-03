"""Generate paper assets from the authoritative adaptive-benchmark results."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"


def load_core_metrics() -> dict[str, float]:
    """Load the authoritative scalar metrics used in the paper."""

    frame = pd.read_csv(ROOT / "results/processed/core_v1_summary.csv")
    return dict(zip(frame["metric"], frame["value"], strict=True))


def write_metrics() -> None:
    """Write LaTeX macros derived from processed evidence."""

    metrics = load_core_metrics()
    regimes = pd.read_csv(ROOT / "results/processed/regime_labels_v1.csv")
    holdout = pd.read_csv(ROOT / "results/processed/selector_holdout_evaluation.csv")
    shifts = pd.read_csv(ROOT / "results/processed/noise_shift_regret.csv")
    summaries = pd.read_csv(ROOT / "results/processed/benchmark_summary_v1.csv")
    hardware = pd.read_csv(
        ROOT / "results/raw/hardware_efficient_v1_repeats-50.csv"
    )
    labels = regimes["oracle_label"].value_counts()
    macros = {
        "OperatingCells": int(metrics["operating_cells"]),
        "PracticalTies": int(labels.get("practical_tie", 0)),
        "ReadoutWinners": int(labels.get("readout", 0)),
        "ZNEWinners": int(labels.get("zne", 0)),
        "InconclusiveCells": int(labels.get("inconclusive", 0)),
        "HoldoutCells": int(metrics["holdout_cells"]),
        "HoldoutCompatibility": (
            f"{100 * metrics['holdout_selector_accuracy']:.1f}\\%"
        ),
        "HoldoutMeanRegret": f"{holdout['rmse_regret'].mean():.4f}",
        "HoldoutMedianRegret": f"{holdout['rmse_regret'].median():.6f}",
        "ShiftCompatibility": (
            f"{100 * metrics['noise_shift_selector_accuracy']:.1f}\\%"
        ),
        "ShiftMeanRegret": f"{shifts['rmse_regret'].mean():.5f}",
        "ShiftMedianRegret": f"{shifts['rmse_regret'].median():.0f}",
        "HardwareFailures": int((hardware["status"] == "failed").sum()),
        "NegativeGains": int((summaries["rmse_gain_vs_raw"] < 0).sum()),
        "SummaryRows": len(summaries),
    }
    text = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    (GENERATED / "adaptive_metrics.tex").write_text(text + "\n", encoding="utf-8")


def write_regime_table() -> None:
    """Write the cost-aware oracle-label counts."""

    frame = pd.read_csv(ROOT / "results/processed/regime_labels_v1.csv")
    counts = frame["oracle_label"].value_counts()
    labels = [
        ("Practical tie", "practical_tie"),
        ("Readout", "readout"),
        ("Linear ZNE", "zne"),
        ("Inconclusive", "inconclusive"),
    ]
    rows = [f"{label} & {int(counts.get(key, 0))} \\\\" for label, key in labels]
    (GENERATED / "adaptive_regime_rows.tex").write_text(
        "\n".join([*rows, r"\bottomrule"]) + "\n", encoding="utf-8"
    )


def plot_circuit_families() -> None:
    """Draw compact conceptual diagrams for the two simulated circuit families."""

    figure, axes = plt.subplots(2, 1, figsize=(7.4, 3.8), constrained_layout=True)
    for axis in axes:
        axis.set_xlim(-0.5, 6.5)
        axis.set_ylim(-0.6, 3.6)
        axis.axis("off")
        for qubit in range(3):
            axis.plot([-0.2, 6.2], [qubit, qubit], color="black", linewidth=1)
            axis.text(-0.35, qubit, f"q{qubit}", ha="right", va="center")
    axes[0].set_title("GHZ family: H followed by a CNOT chain", loc="left")
    axes[0].text(0.5, 0, "H", ha="center", va="center", bbox={"fc": "white"})
    for x, control, target in [(2.0, 0, 1), (3.5, 1, 2)]:
        axes[0].plot([x, x], [control, target], color="#4c78a8", linewidth=2)
        axes[0].scatter([x], [control], color="#4c78a8", s=25)
        axes[0].scatter([x], [target], facecolors="white", edgecolors="#4c78a8", s=90)
        axes[0].plot([x - 0.13, x + 0.13], [target, target], color="#4c78a8")
        axes[0].plot([x, x], [target - 0.13, target + 0.13], color="#4c78a8")
    axes[1].set_title(
        "Hardware-efficient family: fixed rotations and nearest-neighbor entanglers",
        loc="left",
    )
    for layer, x in enumerate((0.7, 3.2)):
        for qubit in range(3):
            axes[1].text(
                x,
                qubit,
                f"R{('y', 'z')[(layer + qubit) % 2]}",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"fc": "white", "ec": "#59a14f"},
            )
        for offset, (control, target) in enumerate(((0, 1), (1, 2))):
            cx = x + 1.0 + 0.55 * offset
            axes[1].plot([cx, cx], [control, target], color="#e15759", linewidth=2)
            axes[1].scatter([cx], [control], color="#e15759", s=22)
            axes[1].scatter([cx], [target], marker="x", color="#e15759", s=45)
    figure.savefig(FIGURES / "circuit_families.png", dpi=220)
    plt.close(figure)


def copy_repository_figures() -> None:
    """Copy current generated figures into the self-contained paper package."""

    for name in (
        "overhead_vs_rmse_gain.png",
        "operating_regime_map_v1.png",
        "selector_rmse_regret.png",
        "regret_under_noise_shift.png",
        "decision_error_probability.png",
    ):
        shutil.copyfile(ROOT / "figures" / name, FIGURES / name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    write_metrics()
    write_regime_table()
    plot_circuit_families()
    copy_repository_figures()


if __name__ == "__main__":
    main()
