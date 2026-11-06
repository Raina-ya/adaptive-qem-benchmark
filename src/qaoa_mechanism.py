"""Explain how QAOA energy accuracy and candidate selection come apart.

This reanalyzes committed repeat rows and runs no circuits. Every result here is exploratory:
medium_dense8 was chosen after its counterexample was seen, and the other graphs were also seen
before this analysis, so they check consistency but are not a holdout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/qaoa_selection_v1.csv"
PROCESSED = ROOT / "results/processed"
FIGURES = ROOT / "figures"
CELLS_PATH = PROCESSED / "qaoa_mechanism_cells_v1.csv"
COMPARISONS_PATH = PROCESSED / "qaoa_mechanism_comparisons_v1.csv"
PAIRS_PATH = PROCESSED / "qaoa_mechanism_pairs_v1.csv"
TRANSITIONS_PATH = PROCESSED / "qaoa_mechanism_transitions_v1.csv"
CORRELATION_PATH = PROCESSED / "qaoa_mechanism_correlation_v1.csv"
FOLD_GAPS_PATH = PROCESSED / "qaoa_mechanism_fold_gaps_v1.csv"
COUNTERFACTUAL_PATH = PROCESSED / "qaoa_mechanism_counterfactuals_v1.csv"
CONTRIBUTION_PATH = PROCESSED / "qaoa_mechanism_mse_contributions_v1.csv"
BOOTSTRAP_SEED = 20261106
BOOTSTRAP_RESAMPLES = 2000
ANALYSIS_ROLE = "exploratory_post_hoc"
GRAPHS = ("medium_dense8", "sparse6", "ring4")
GRAPH_LABELS = {
    "medium_dense8": "Dense, width 8",
    "sparse6": "Sparse, width 6",
    "ring4": "Ring, width 4",
}
BUDGETS = (4096, 16384)
METHODS = ("raw", "readout", "zne")
METHOD_LABELS = {"raw": "Raw", "readout": "Readout", "zne": "Linear ZNE"}
METHOD_COLORS = {"raw": "#2a78d6", "readout": "#eb6834", "zne": "#1baf7a"}
GRAPH_MARKERS = {"medium_dense8": "o", "sparse6": "s", "ring4": "^"}
BUDGET_FILL = {4096: False, 16384: True}
FOLDS = (1, 3, 5)
INK = "#3D3D3A"
MUTED = "#73726C"
GRID = "#E8E6DC"
CELL = ["graph_id", "total_selection_budget", "method"]
GROUP = [*CELL, "repeat"]


def load_rows() -> pd.DataFrame:
    """Return completed candidate rows with e, the common part b, and the contrast part d."""

    columns = [
        "graph_id",
        "total_selection_budget",
        "method",
        "repeat",
        "candidate_id",
        "candidate_role",
        "exact_ideal_energy",
        "corrected_or_extrapolated_estimate",
        "factor_level_estimates",
        "status",
    ]
    frame = pd.read_csv(RAW_PATH, usecols=columns, low_memory=False, keep_default_na=False)
    frame = frame.loc[frame.status == "complete"].copy()
    frame["estimate"] = frame.corrected_or_extrapolated_estimate.astype(float)
    frame["ideal"] = frame.exact_ideal_energy.astype(float)
    return decompose(frame)


def decompose(frame: pd.DataFrame) -> pd.DataFrame:
    """Split each error into its mean over candidates in the repeat and the remainder."""

    output = frame.copy()
    output["error"] = output.estimate - output.ideal
    output["common"] = output.groupby(GROUP).error.transform("mean")
    output["contrast"] = output.error - output.common
    return output


def contenders(frame: pd.DataFrame) -> tuple[str, str]:
    """Return the two candidates with the highest ideal energy, fixed without noisy data."""

    ideal = frame.drop_duplicates("candidate_id").sort_values(
        ["ideal", "candidate_id"], ascending=[False, True]
    )
    return str(ideal.candidate_id.iloc[0]), str(ideal.candidate_id.iloc[1])


def decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one selection per repeat, with correctness, regret, and error components."""

    rows = []
    for keys, group in frame.groupby(GROUP, sort=True):
        ordered = group.sort_values(["estimate", "candidate_id"], ascending=[False, True])
        selected = ordered.iloc[0]
        best = group.ideal.max()
        rows.append(
            {
                **dict(zip(GROUP, keys, strict=True)),
                "selected": selected.candidate_id,
                "correct": bool(best - selected.ideal <= 1e-12),
                "regret": float(best - selected.ideal),
                "mse": float(np.mean(group.error**2)),
                "common": float(group.common.iloc[0]),
                "contrast_mse": float(np.mean(group.contrast**2)),
            }
        )
    return pd.DataFrame(rows)


def gap_statistics(group: pd.DataFrame) -> dict[str, float]:
    """Describe the estimated gap between the two ideal leaders across repeats."""

    best, runner = contenders(group)
    pivot = group.pivot(index="repeat", columns="candidate_id", values="estimate")
    gaps = pivot[best] - pivot[runner]
    ideal = group.drop_duplicates("candidate_id").set_index("candidate_id").ideal
    mean, spread = float(gaps.mean()), float(gaps.std(ddof=1))
    score = mean / spread if spread > 0.0 else math.inf
    return {
        "ideal_gap": float(ideal[best] - ideal[runner]),
        "mean_estimated_gap": mean,
        "gap_sd": spread,
        "gap_signal_to_noise": score,
        "normal_predicted_inversion": float(norm.sf(score)),
        "observed_inversion": float((gaps < 0.0).mean()),
    }


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize each graph, budget, and method, using the identity MSE = b^2 + d^2."""

    rows = []
    selected = decisions(frame)
    for keys, group in frame.groupby(CELL, sort=True):
        chosen = selected
        for column, value in zip(CELL, keys, strict=True):
            chosen = chosen.loc[chosen[column] == value]
        by_candidate = group.groupby("candidate_id")
        systematic = by_candidate.contrast.mean()
        sampling = by_candidate.contrast.var(ddof=0)
        leaders = contenders(group)
        squared = group.assign(squared=group.error**2)
        share = squared.loc[~squared.candidate_id.isin(leaders), "squared"].sum() / (
            squared.squared.sum()
        )
        lead = group.loc[group.candidate_id.isin(leaders)]
        rows.append(
            {
                **dict(zip(CELL, keys, strict=True)),
                "repeats": len(chosen),
                "energy_rmse": math.sqrt(float(np.mean(group.error**2))),
                "common_bias_mean": float(chosen.common.mean()),
                "common_bias_rms": math.sqrt(float(np.mean(chosen.common**2))),
                "contrast_rmse": math.sqrt(float(np.mean(group.contrast**2))),
                "contrast_systematic_rms": math.sqrt(float(np.mean(systematic**2))),
                "contrast_sampling_rms": math.sqrt(float(sampling.mean())),
                "mean_estimate_sd": float(by_candidate.estimate.std(ddof=1).mean()),
                "top1_error": float(1.0 - chosen.correct.mean()),
                "mean_simple_regret": float(chosen.regret.mean()),
                "contender_rmse": math.sqrt(float(np.mean(lead.error**2))),
                "non_contender_mse_share": float(share),
                **gap_statistics(group),
                "analysis_role": ANALYSIS_ROLE,
            }
        )
    return pd.DataFrame(rows)


def mcnemar_exact(first_only: int, second_only: int) -> float:
    """Return the two-sided exact McNemar p-value from the discordant counts."""

    total = first_only + second_only
    if total == 0:
        return 1.0
    return float(binomtest(first_only, total, 0.5).pvalue)


def holm(values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in the input order."""

    order = np.argsort(values)
    adjusted = [0.0] * len(values)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (len(values) - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each mitigation with raw on repeats that share an index."""

    selected = decisions(frame)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for graph in sorted(frame.graph_id.unique()):
        for budget in sorted(frame.total_selection_budget.unique()):
            base = selected.loc[
                (selected.graph_id == graph)
                & (selected.total_selection_budget == budget)
                & (selected.method == "raw")
            ].set_index("repeat")
            for method in ("readout", "zne"):
                other = selected.loc[
                    (selected.graph_id == graph)
                    & (selected.total_selection_budget == budget)
                    & (selected.method == method)
                ].set_index("repeat")
                shared = base.index.intersection(other.index)
                if not len(shared):
                    continue
                rows.append(
                    _compare(graph, budget, method, base.loc[shared], other.loc[shared], rng)
                )
    table = pd.DataFrame(rows)
    table["mcnemar_holm_p"] = holm(table.mcnemar_exact_p.tolist())
    return table


def _compare(graph, budget, method, base, other, rng) -> dict[str, object]:
    raw_only = int((base.correct & ~other.correct).sum())
    other_only = int((~base.correct & other.correct).sum())
    samples = rng.integers(0, len(base), (BOOTSTRAP_RESAMPLES, len(base)))

    def rms_difference(column: str) -> tuple[float, float, float]:
        left, right = other[column].to_numpy(), base[column].to_numpy()
        value = math.sqrt(float(left.mean())) - math.sqrt(float(right.mean()))
        draws = np.sqrt(left[samples].mean(axis=1)) - np.sqrt(right[samples].mean(axis=1))
        return (value, *_interval(draws))

    correct = other.correct.to_numpy(float) - base.correct.to_numpy(float)
    regret = other.regret.to_numpy() - base.regret.to_numpy()
    rate, rate_low, rate_high = correct.mean(), *_interval(correct[samples].mean(axis=1))
    mean_regret, regret_low, regret_high = regret.mean(), *_interval(regret[samples].mean(axis=1))
    energy, energy_low, energy_high = rms_difference("mse")
    contrast, contrast_low, contrast_high = rms_difference("contrast_mse")
    common_left = other.common.to_numpy() ** 2
    common_right = base.common.to_numpy() ** 2
    common = math.sqrt(common_left.mean()) - math.sqrt(common_right.mean())
    common_draws = np.sqrt(common_left[samples].mean(axis=1)) - np.sqrt(
        common_right[samples].mean(axis=1)
    )
    common_low, common_high = _interval(common_draws)
    return {
        "graph_id": graph,
        "total_selection_budget": int(budget),
        "method": method,
        "paired_repeats": len(base),
        "raw_only_correct": raw_only,
        "method_only_correct": other_only,
        "mcnemar_exact_p": mcnemar_exact(raw_only, other_only),
        "correct_rate_difference": float(rate),
        "correct_rate_ci_low": rate_low,
        "correct_rate_ci_high": rate_high,
        "regret_difference": float(mean_regret),
        "regret_ci_low": regret_low,
        "regret_ci_high": regret_high,
        "energy_rmse_difference": energy,
        "energy_rmse_ci_low": energy_low,
        "energy_rmse_ci_high": energy_high,
        "contrast_rmse_difference": contrast,
        "contrast_rmse_ci_low": contrast_low,
        "contrast_rmse_ci_high": contrast_high,
        "common_bias_rms_difference": common,
        "common_bias_ci_low": common_low,
        "common_bias_ci_high": common_high,
        "analysis_role": ANALYSIS_ROLE,
    }


def pair_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Report gap error and inversion probability for every candidate pair in each cell."""

    rows = []
    for keys, group in frame.groupby(CELL, sort=True):
        pivot = group.pivot(index="repeat", columns="candidate_id", values="estimate")
        ideal = group.drop_duplicates("candidate_id").set_index("candidate_id").ideal
        ordered = list(ideal.sort_values(ascending=False).index)
        for position, better in enumerate(ordered):
            for worse in ordered[position + 1 :]:
                gaps = pivot[better] - pivot[worse]
                exact = float(ideal[better] - ideal[worse])
                rows.append(
                    {
                        **dict(zip(CELL, keys, strict=True)),
                        "better_candidate": better,
                        "worse_candidate": worse,
                        "ideal_gap": exact,
                        "mean_gap_error": float((gaps - exact).mean()),
                        "gap_error_sd": float(gaps.std(ddof=1)),
                        "inversion_probability": float((gaps < 0.0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def transition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Count which candidate raw and ZNE choose in the same repeat."""

    selected = decisions(frame)
    keys = ["graph_id", "total_selection_budget", "repeat"]
    raw = selected.loc[selected.method == "raw", [*keys, "selected", "correct"]]
    zne = selected.loc[selected.method == "zne", [*keys, "selected", "correct"]]
    merged = raw.merge(zne, on=keys, suffixes=("_raw", "_zne"), validate="one_to_one")
    counts = merged.groupby(
        ["graph_id", "total_selection_budget", "selected_raw", "selected_zne"]
    ).size()
    return counts.rename("repeats").reset_index()


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Report across-repeat correlations of candidate errors within each cell."""

    rows = []
    for keys, group in frame.groupby(CELL, sort=True):
        matrix = group.pivot(index="repeat", columns="candidate_role", values="error").corr()
        roles = list(matrix.columns)
        for left in range(len(roles)):
            for right in range(left + 1, len(roles)):
                rows.append(
                    {
                        **dict(zip(CELL, keys, strict=True)),
                        "first_role": roles[left],
                        "second_role": roles[right],
                        "error_correlation": float(matrix.iloc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def zne_variance_factor() -> float:
    """Return the variance ratio of the fold fit to one estimate using all the shots."""

    x = np.array(FOLDS, dtype=float)
    weights = 1.0 / len(x) - x.mean() * (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    return float(len(x) * np.sum(weights**2))


def fold_gap_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Follow the leading pair's gap through each ZNE noise factor."""

    rows = []
    zne = frame.loc[frame.method == "zne"]
    for (graph, budget), group in zne.groupby(["graph_id", "total_selection_budget"], sort=True):
        best, runner = contenders(group)
        ideal = group.drop_duplicates("candidate_id").set_index("candidate_id").ideal
        factors = {
            candidate: np.array(
                [json.loads(item) for item in part.sort_values("repeat").factor_level_estimates]
            )
            for candidate, part in group.groupby("candidate_id")
        }
        gaps = factors[best] - factors[runner]
        for index, fold in enumerate(FOLDS):
            rows.append(
                {
                    "graph_id": graph,
                    "total_selection_budget": int(budget),
                    "fold": fold,
                    "ideal_gap": float(ideal[best] - ideal[runner]),
                    "mean_gap": float(gaps[:, index].mean()),
                    "gap_sd": float(gaps[:, index].std(ddof=1)),
                }
            )
    return pd.DataFrame(rows)


def counterfactual_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Swap the deciding gap's mean and spread between raw and ZNE to see which one matters."""

    rows = []
    for (graph, budget), group in cells.groupby(["graph_id", "total_selection_budget"], sort=True):
        indexed = group.set_index("method")
        raw, zne = indexed.loc["raw"], indexed.loc["zne"]
        rows.append(
            {
                "graph_id": graph,
                "total_selection_budget": int(budget),
                "ideal_gap": float(raw.ideal_gap),
                "raw_mean_gap": float(raw.mean_estimated_gap),
                "raw_gap_sd": float(raw.gap_sd),
                "zne_mean_gap": float(zne.mean_estimated_gap),
                "zne_gap_sd": float(zne.gap_sd),
                "observed_sd_ratio": float(zne.gap_sd / raw.gap_sd),
                "observed_mean_gap_ratio": float(zne.mean_estimated_gap / raw.mean_estimated_gap),
                "largest_possible_mean_gap_ratio": float(raw.ideal_gap / raw.mean_estimated_gap),
                "theoretical_sd_ratio": math.sqrt(zne_variance_factor()),
                "zne_mean_gap_standard_error": float(zne.gap_sd / math.sqrt(zne.repeats)),
                "raw_mean_gap_standard_error": float(raw.gap_sd / math.sqrt(raw.repeats)),
                "predicted_inversion_raw": float(norm.sf(raw.mean_estimated_gap / raw.gap_sd)),
                "predicted_inversion_raw_mean_zne_spread": float(
                    norm.sf(raw.mean_estimated_gap / zne.gap_sd)
                ),
                "predicted_inversion_zne_mean_raw_spread": float(
                    norm.sf(zne.mean_estimated_gap / raw.gap_sd)
                ),
                "predicted_inversion_zne": float(norm.sf(zne.mean_estimated_gap / zne.gap_sd)),
                "observed_error_raw": float(raw.top1_error),
                "observed_error_zne": float(zne.top1_error),
                "analysis_role": ANALYSIS_ROLE,
            }
        )
    return pd.DataFrame(rows)


def contribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Attribute each cell's energy MSE to candidates, split into bias and variance."""

    rows = []
    for keys, group in frame.groupby(CELL, sort=True):
        leaders = contenders(group)
        for candidate, part in group.groupby("candidate_id"):
            rows.append(
                {
                    **dict(zip(CELL, keys, strict=True)),
                    "candidate_id": candidate,
                    "candidate_role": part.candidate_role.iloc[0],
                    "contender": candidate in leaders,
                    "ideal_energy": float(part.ideal.iloc[0]),
                    "mean_error": float(part.error.mean()),
                    "estimate_sd": float(part.estimate.std(ddof=1)),
                    "mse_contribution": float(np.mean(part.error**2))
                    / group.candidate_id.nunique(),
                }
            )
    return pd.DataFrame(rows)


def _style(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, length=0, labelsize=8)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.xaxis.label.set(color=INK, size=9)
    axis.yaxis.label.set(color=INK, size=9)


def _scatter(cells: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, path: Path):
    figure, axis = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for row in cells.itertuples():
        color = METHOD_COLORS[row.method]
        axis.scatter(
            getattr(row, x),
            getattr(row, y),
            s=62,
            marker=GRAPH_MARKERS[row.graph_id],
            facecolor=color if BUDGET_FILL[row.total_selection_budget] else "white",
            edgecolor=color,
            linewidth=1.6,
            zorder=3,
        )
    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for m in METHODS
    ]
    handles += [
        plt.Line2D(
            [], [], linestyle="", marker=GRAPH_MARKERS[g], color=MUTED, label=GRAPH_LABELS[g]
        )
        for g in GRAPHS
    ]
    handles += [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markerfacecolor="white",
            color=MUTED,
            label="4,096 shots",
        ),
        plt.Line2D([], [], linestyle="", marker="o", color=MUTED, label="16,384 shots"),
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
    )
    _style(axis)
    axis.set(xlabel=xlabel, ylabel=ylabel)
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_gaps(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(6.4, 6.2), constrained_layout=True)
    budget = 16384
    for axis, graph in zip(axes, GRAPHS, strict=True):
        data = frame.loc[(frame.graph_id == graph) & (frame.total_selection_budget == budget)]
        best, runner = contenders(data)
        ideal = data.drop_duplicates("candidate_id").set_index("candidate_id").ideal
        exact = float(ideal[best] - ideal[runner])
        for row, method in enumerate(("raw", "zne")):
            part = data.loc[data.method == method]
            pivot = part.pivot(index="repeat", columns="candidate_id", values="estimate")
            gaps = (pivot[best] - pivot[runner]).sort_values()
            offsets = np.linspace(-0.22, 0.22, len(gaps))
            axis.scatter(
                gaps, row + offsets, s=14, color=METHOD_COLORS[method], alpha=0.8, zorder=3
            )
            axis.plot([gaps.mean()] * 2, [row - 0.32, row + 0.32], color=INK, linewidth=2, zorder=4)
        axis.axvline(0.0, color=MUTED, linewidth=1.2)
        axis.axvline(exact, color=INK, linewidth=1.2, linestyle=":")
        axis.set_yticks([0, 1], [METHOD_LABELS["raw"], METHOD_LABELS["zne"]])
        axis.grid(False, axis="x")
        axis.set_ylim(-0.6, 1.6)
        _style(axis)
        axis.grid(False, axis="x")
        axis.set_title(GRAPH_LABELS[graph], loc="left", fontsize=9, color=INK)
    axes[-1].set_xlabel(
        "Estimated gap between the two ideal leaders (below 0 is wrong; dotted is ideal)"
    )
    figure.suptitle(
        "ZNE spreads the deciding gap across zero (16,384 shots; bars mark means)",
        x=0.02,
        ha="left",
        fontsize=10,
        color=INK,
    )
    figure.savefig(FIGURES / "qaoa_mechanism_gap_distribution.png", dpi=180)
    plt.close(figure)


def _plot_correlation(frame: pd.DataFrame) -> None:
    data = frame.loc[(frame.graph_id == "medium_dense8") & (frame.total_selection_budget == 16384)]
    order = [
        "incumbent",
        "near_tie_challenger",
        "clear_challenger",
        "median_control",
        "low_control",
    ]
    labels = ["Leader", "Near tie", "Clear", "Median", "Low"]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    image = None
    for axis, method in zip(axes, ("raw", "zne"), strict=True):
        matrix = data.loc[data.method == method].pivot(
            index="repeat", columns="candidate_role", values="error"
        )
        values = matrix[order].corr().to_numpy()
        image = axis.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r")
        for row in range(5):
            for column in range(5):
                shade = "white" if abs(values[row, column]) > 0.6 else INK
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=shade,
                )
        axis.set_xticks(range(5), labels, fontsize=7.5, rotation=30)
        axis.set_yticks(range(5), labels, fontsize=7.5)
        axis.tick_params(length=0, colors=MUTED)
        axis.spines[:].set_visible(False)
        axis.set_title(METHOD_LABELS[method], loc="left", fontsize=9, color=INK)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Error correlation across 50 repeats")
    figure.savefig(FIGURES / "qaoa_mechanism_error_correlation.png", dpi=180)
    plt.close(figure)


def _plot_transitions(frame: pd.DataFrame) -> None:
    table = transition_table(frame)
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 3.3), constrained_layout=True)
    data = table.loc[table.total_selection_budget == 16384]
    for axis, graph in zip(axes, GRAPHS, strict=True):
        part = data.loc[data.graph_id == graph]
        ideal = frame.loc[frame.graph_id == graph].drop_duplicates("candidate_id")
        ids = list(ideal.sort_values("ideal", ascending=False).candidate_id)[:2]
        matrix = np.zeros((2, 2))
        for row in part.itertuples():
            if row.selected_raw in ids and row.selected_zne in ids:
                matrix[ids.index(row.selected_raw), ids.index(row.selected_zne)] = row.repeats
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=50)
        for left in range(2):
            for right in range(2):
                shade = "white" if matrix[left, right] > 28 else INK
                axis.text(
                    right,
                    left,
                    f"{int(matrix[left, right])}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=shade,
                )
        ticks = ["Leader", "2nd"]
        axis.set_xticks(range(2), ticks, fontsize=8)
        axis.set_yticks(range(2), ticks, fontsize=8)
        axis.tick_params(length=0, colors=MUTED)
        axis.spines[:].set_visible(False)
        axis.set_xlabel("ZNE selects", fontsize=8.5, color=INK)
        axis.set_ylabel("Raw selects", fontsize=8.5, color=INK)
        axis.set_title(GRAPH_LABELS[graph], loc="left", fontsize=9, color=INK)
    figure.savefig(FIGURES / "qaoa_mechanism_rank_transitions.png", dpi=180)
    plt.close(figure)


def _plot_error_shares(frame: pd.DataFrame) -> None:
    data = frame.loc[(frame.graph_id == "medium_dense8") & (frame.total_selection_budget == 16384)]
    order = [
        "incumbent",
        "near_tie_challenger",
        "clear_challenger",
        "median_control",
        "low_control",
    ]
    labels = ["Leader", "Near tie", "Clear", "Median", "Low"]
    figure, axis = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    width = 0.36
    for index, method in enumerate(("raw", "zne")):
        part = (
            data.loc[data.method == method]
            .groupby("candidate_role")
            .error.apply(lambda v: float(np.mean(v**2)))
        )
        positions = np.arange(5) + (index - 0.5) * (width + 0.03)
        axis.bar(
            positions,
            part[order] / 5.0,
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.set_xticks(range(5), labels)
    axis.legend(frameon=False, fontsize=8)
    _style(axis)
    axis.grid(False, axis="x")
    axis.set_ylabel("Contribution to energy MSE")
    axis.set_title(
        "A candidate that never contends drives the RMSE gap (dense, 16,384 shots)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    figure.savefig(FIGURES / "qaoa_mechanism_mse_by_candidate.png", dpi=180)
    plt.close(figure)


def plot_all(frame: pd.DataFrame, cells: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    _scatter(
        cells,
        "contrast_rmse",
        "energy_rmse",
        "Contrast RMSE",
        "Energy RMSE",
        "Energy error is almost entirely contrast error",
        FIGURES / "qaoa_mechanism_rmse_vs_contrast.png",
    )
    _scatter(
        cells,
        "contrast_rmse",
        "top1_error",
        "Contrast RMSE",
        "Top-1 selection error",
        "Lower total contrast error does not mean fewer wrong selections",
        FIGURES / "qaoa_mechanism_contrast_vs_selection.png",
    )
    _scatter(
        cells,
        "common_bias_rms",
        "top1_error",
        "Common-mode bias (RMS)",
        "Top-1 selection error",
        "Common bias cannot change a selection",
        FIGURES / "qaoa_mechanism_common_bias_vs_selection.png",
    )
    _plot_gaps(frame)
    _plot_correlation(frame)
    _plot_transitions(frame)
    _plot_error_shares(frame)


def main() -> None:
    frame = load_rows()
    cells = cell_table(frame)
    cells.to_csv(CELLS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    comparison_table(frame).to_csv(
        COMPARISONS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    pair_table(frame).to_csv(PAIRS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    transition_table(frame).to_csv(TRANSITIONS_PATH, index=False, lineterminator="\n")
    correlation_table(frame).to_csv(
        CORRELATION_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    fold_gap_table(frame).to_csv(
        FOLD_GAPS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    counterfactual_table(cells).to_csv(
        COUNTERFACTUAL_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    contribution_table(frame).to_csv(
        CONTRIBUTION_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    plot_all(frame, cells)


if __name__ == "__main__":
    main()
