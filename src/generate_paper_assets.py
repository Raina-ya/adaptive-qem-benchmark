"""Generate paper figures and LaTeX tables from authoritative result CSV files."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/day12_ghz_volumetric_v1_repeats-50.csv"
DAY09_PATH = ROOT / "results/day09_method_validation_budget-4096_repeats-20.csv"
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"
METHODS = ("raw", "readout", "zne")
NOISE_ORDER = ("zero", "low", "medium", "high")


def load_results() -> pd.DataFrame:
    """Load the frozen GHZ benchmark and enforce its repeat-level contract."""

    frame = pd.read_csv(RAW_PATH)
    if len(frame) != 5400:
        raise ValueError(f"Expected 5,400 applicable rows, found {len(frame):,}")
    if not frame.seed.is_unique:
        raise ValueError("Benchmark seeds are not unique")
    if not set(frame.status).issubset({"complete", "failed"}):
        raise ValueError("Applicable rows must be complete or failed")
    return frame


def method_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute repeat-weighted metrics over completed rows only."""

    complete = frame[frame.status == "complete"].copy()
    complete["error"] = complete.estimate - complete.analytic_reference
    return (
        complete.groupby("method")
        .agg(
            completed=("estimate", "size"),
            bias=("error", "mean"),
            variance=("estimate", "var"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mae=("error", lambda values: float(np.mean(np.abs(values)))),
        )
        .reindex(METHODS)
        .reset_index()
    )


def noise_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute completed-row metrics for every noise-method pair."""

    complete = frame[frame.status == "complete"].copy()
    complete["error"] = complete.estimate - complete.analytic_reference
    summary = (
        complete.groupby(["noise", "method"], sort=False)
        .agg(
            completed=("estimate", "size"),
            bias=("error", "mean"),
            rmse=("error", lambda values: float(np.sqrt(np.mean(values**2)))),
        )
        .reset_index()
    )
    summary["noise"] = pd.Categorical(summary.noise, NOISE_ORDER, ordered=True)
    return summary.sort_values(["noise", "method"]).reset_index(drop=True)


def condition_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute one RMSE per width-noise-budget-method operating cell."""

    complete = frame[frame.status == "complete"].copy()
    complete["squared_error"] = (complete.estimate - complete.analytic_reference) ** 2
    return (
        complete.groupby(["width", "noise", "shots", "method"])
        .agg(
            completed=("estimate", "size"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
        )
        .reset_index()
    )


def regime_winners(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select the minimum-RMSE method only where all methods have 50 repeats."""

    summary = condition_summary(frame)
    counts = summary.pivot(
        index=["width", "noise", "shots"], columns="method", values="completed"
    )
    rmse = summary.pivot(
        index=["width", "noise", "shots"], columns="method", values="rmse"
    )
    complete_mask = (counts.reindex(columns=METHODS) == 50).all(axis=1)
    winners = pd.DataFrame(index=rmse.index)
    winners["status"] = np.where(complete_mask, "complete", "incomplete")
    winners["winner"] = "incomplete"
    winners.loc[complete_mask, "winner"] = (
        rmse.loc[complete_mask].reindex(columns=METHODS).idxmin(axis=1)
    )
    win_counts = {
        method: int((winners.winner == method).sum()) for method in METHODS
    }
    return winners.reset_index(), win_counts


def write_metrics(frame: pd.DataFrame) -> None:
    """Write auditable LaTeX macros and compact generated tables."""

    methods = method_summary(frame)
    noise = noise_summary(frame)
    _, wins = regime_winners(frame)
    failures = frame[frame.status == "failed"]
    readout_total = int((frame.method == "readout").sum())
    readout_failures = int(((frame.method == "readout") & (frame.status == "failed")).sum())
    zne = frame[(frame.method == "zne") & (frame.status == "complete")]
    zne_outside = int(((zne.estimate < -1.0 - 1e-12) | (zne.estimate > 1.0 + 1e-12)).sum())
    macros = {
        "TotalRows": f"{len(frame):,}",
        "CompleteRows": f"{int((frame.status == 'complete').sum()):,}",
        "FailedRows": str(len(failures)),
        "ReadoutFailures": str(readout_failures),
        "ReadoutFailureRate": f"{100 * readout_failures / readout_total:.2f}\\%",
        "CompleteOperatingCells": "34",
        "RawWins": str(wins["raw"]),
        "ReadoutWins": str(wins["readout"]),
        "ZNEWins": str(wins["zne"]),
        "ZNEOutOfRange": str(zne_outside),
    }
    for row in methods.itertuples():
        prefix = {"raw": "Raw", "readout": "Readout", "zne": "ZNE"}[row.method]
        macros[f"{prefix}RMSE"] = f"{row.rmse:.4f}"
        macros[f"{prefix}Bias"] = f"{row.bias:.4f}"
    metric_text = "\n".join(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    (GENERATED / "metrics.tex").write_text(metric_text + "\n", encoding="utf-8")

    rows = []
    for row in noise.itertuples():
        label = {"raw": "Raw", "readout": "Readout", "zne": "Linear ZNE"}[row.method]
        rows.append(
            f"{str(row.noise).title()} & {label} & {row.completed} & "
            f"{row.bias:.4f} & {row.rmse:.4f} \\\\"
        )
    table = "\n".join(rows + [r"\bottomrule"])
    (GENERATED / "noise_method_rows.tex").write_text(table + "\n", encoding="utf-8")

    failure_rows = (
        failures.groupby(["width", "noise", "shots", "method", "error"])
        .size()
        .rename("count")
        .reset_index()
    )
    failure_text = "\n".join(
        f"{row.width} & {str(row.noise).title()} & {row.shots:,} & "
        f"{row.method.title()} & {row.count} & Ill-conditioned assignment matrix \\\\"
        for row in failure_rows.itertuples()
    )
    failure_text += "\n\\bottomrule"
    (GENERATED / "failure_rows.tex").write_text(failure_text + "\n", encoding="utf-8")


def plot_rmse_by_noise(frame: pd.DataFrame) -> None:
    """Plot completed-row RMSE for each method and noise level."""

    summary = noise_summary(frame)
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    colors = {"raw": "#4c78a8", "readout": "#59a14f", "zne": "#e15759"}
    for method in METHODS:
        group = summary[summary.method == method].set_index("noise").reindex(NOISE_ORDER)
        axis.plot(
            NOISE_ORDER,
            group.rmse,
            marker="o",
            linewidth=2,
            color=colors[method],
            label={"raw": "Raw", "readout": "Readout", "zne": "Linear ZNE"}[method],
        )
    axis.set(xlabel="Simulated mixed-noise level", ylabel="RMSE of $X^{\\otimes w}$")
    axis.set_title("Readout mitigation has the lowest aggregate RMSE beyond zero noise")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "rmse_by_noise.png", dpi=220)
    plt.close(figure)


def plot_operating_regimes(frame: pd.DataFrame) -> None:
    """Plot the best completed method for every declared operating condition."""

    winners, _ = regime_winners(frame)
    columns = [(noise, shots) for noise in NOISE_ORDER for shots in (1024, 4096, 16384)]
    code = {"raw": 0, "readout": 1, "zne": 2, "incomplete": 3}
    matrix = np.empty((3, len(columns)), dtype=int)
    labels = np.empty_like(matrix, dtype=object)
    for row_index, width in enumerate((2, 4, 6)):
        indexed = winners[winners.width == width].set_index(["noise", "shots"])
        for column_index, condition in enumerate(columns):
            winner = indexed.loc[condition, "winner"]
            matrix[row_index, column_index] = code[winner]
            labels[row_index, column_index] = {
                "raw": "Raw",
                "readout": "RO",
                "zne": "ZNE",
                "incomplete": "Fail",
            }[winner]
    figure, axis = plt.subplots(figsize=(10.2, 3.3))
    axis.imshow(
        matrix,
        aspect="auto",
        cmap=ListedColormap(["#4c78a8", "#59a14f", "#e15759", "#bab0ac"]),
        vmin=-0.5,
        vmax=3.5,
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, labels[row, column], ha="center", va="center", fontsize=8)
    axis.set_yticks(range(3), ["w=2", "w=4", "w=6"])
    axis.set_xticks(
        range(len(columns)),
        [f"{noise[0].upper()}\n{shots // 1024}k" for noise, shots in columns],
    )
    axis.set(xlabel="Noise level (Z/L/M/H) and total shot budget", ylabel="GHZ width")
    axis.set_title("Lowest-RMSE method varies across operating conditions")
    figure.tight_layout()
    figure.savefig(FIGURES / "operating_regime_map.png", dpi=220)
    plt.close(figure)


def copy_repository_figures() -> None:
    """Copy selected generated repository figures into the paper package."""

    for name in (
        "day08_depth_vs_noise_factor.png",
        "day09_zne_extrapolation_lambda-1-3-5.png",
    ):
        shutil.copyfile(ROOT / "figures" / name, FIGURES / name)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    frame = load_results()
    write_metrics(frame)
    plot_rmse_by_noise(frame)
    plot_operating_regimes(frame)
    copy_repository_figures()


if __name__ == "__main__":
    main()
