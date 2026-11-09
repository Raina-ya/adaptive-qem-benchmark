"""Build the figures, macros, and tables of the revised paper from committed processed results.

Every value here is read from a committed CSV under results/processed; nothing is simulated.
Figures share one style: STIX serif text, a fixed method-to-color map, thin marks, and legends.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
FIGURES = ROOT / "paper/figures"
GENERATED = ROOT / "paper/generated"
MACROS_PATH = GENERATED / "boundaries_metrics.tex"
SENSITIVITY_PATH = GENERATED / "threshold_sensitivity_rows.tex"
SENSITIVITY_THRESHOLDS = (0.002, 0.005, 0.01, 0.02)
COLUMN = 3.4
DOUBLE = 7.0
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
METHOD_COLORS = {
    "raw": SLOTS[0],
    "readout": SLOTS[1],
    "zne": SLOTS[2],
    "combined": SLOTS[3],
}
METHOD_LABELS = {"raw": "Raw", "readout": "Readout", "zne": "ZNE", "combined": "Combined"}
MARKERS = {"raw": "o", "readout": "s", "zne": "^", "combined": "D"}
FIGURE_NAMES = (
    "fig_regime_shares.png",
    "fig_rmse_vs_selection.png",
    "fig_signal_bands.png",
    "fig_shots_to_reliability.png",
    "fig_zne_gate_holdout.png",
    "fig_certification.png",
    "fig_readout_width.png",
    "fig_mismatch_dose.png",
    "fig_drift_policy.png",
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "legend.frameon": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "svg.hashsalt": "adaptive-qem",
        }
    )


def save(figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, metadata={"Software": "adaptive-qem-benchmark"})
    plt.close(figure)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED / f"{name}.csv")


def fig_regime_shares() -> None:
    labels = read("regime_labels_v2")
    order = ["raw", "practical_tie", "readout", "zne", "inconclusive"]
    names = ["Raw wins", "Practical tie", "Readout wins", "ZNE wins", "Inconclusive"]
    colors = [SLOTS[0], "#b9b8b3", SLOTS[1], SLOTS[2], "#6b6a66"]
    noises = ["zero", "low", "medium", "high"]
    counts = labels.groupby(["noise", "oracle_label"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=noises, columns=order, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    figure, axis = plt.subplots(figsize=(COLUMN, 1.7))
    left = np.zeros(len(noises))
    for column, name, color in zip(order, names, colors, strict=True):
        values = share[column].to_numpy()
        axis.barh(
            noises, values, left=left, color=color, edgecolor=SURFACE, linewidth=1.0, label=name
        )
        for row, (start, width) in enumerate(zip(left, values, strict=True)):
            if width >= 0.12:
                axis.text(
                    start + width / 2,
                    row,
                    f"{int(counts.iloc[row][column])}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=SURFACE if column != "practical_tie" else INK,
                )
        left += values
    axis.set_xlim(0, 1)
    axis.set_xlabel("Share of operating cells (counts printed)")
    axis.set_ylabel("Noise level")
    axis.grid(axis="y", visible=False)
    axis.legend(
        ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8
    )
    save(figure, "fig_regime_shares.png")


def fig_rmse_vs_selection() -> None:
    cells = read("qaoa_mechanism_cells_v1")
    figure, axes = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    axis = axes[0]
    for method, group in cells.groupby("method"):
        axis.scatter(
            group.energy_rmse,
            group.top1_error,
            s=22,
            marker=MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor=SURFACE,
            linewidth=0.6,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    for (_graph, _budget), group in cells.groupby(["graph_id", "total_selection_budget"]):
        ordered = group.set_index("method")
        if {"raw", "zne"} <= set(ordered.index):
            axis.annotate(
                "",
                xy=(ordered.loc["zne", "energy_rmse"], ordered.loc["zne", "top1_error"]),
                xytext=(ordered.loc["raw", "energy_rmse"], ordered.loc["raw", "top1_error"]),
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.6},
                zorder=2,
            )
    axis.set_xlabel("Candidate-energy RMSE")
    axis.set_ylabel("Top-1 selection error")
    axis.set_title("(a) Lower RMSE does not lower selection error", loc="left")
    axis.legend(loc="upper left")
    axis = axes[1]
    dense = cells.loc[cells.graph_id == "medium_dense8"].copy()
    dense["label"] = (
        dense.method.map(METHOD_LABELS) + "\n" + dense.total_selection_budget.astype(str)
    )
    position = np.arange(len(dense))
    common = dense.common_bias_rms.to_numpy() ** 2
    contrast = dense.contrast_rmse.to_numpy() ** 2
    axis.bar(
        position,
        common,
        color="#b9b8b3",
        edgecolor=SURFACE,
        linewidth=1.0,
        label="Common-mode part",
    )
    axis.bar(
        position,
        contrast,
        bottom=common,
        color=SLOTS[0],
        edgecolor=SURFACE,
        linewidth=1.0,
        label="Contrast part",
    )
    axis.set_xticks(position, dense.label)
    axis.set_ylabel("Energy MSE")
    axis.set_title("(b) MSE split, medium/dense graph", loc="left")
    axis.grid(axis="x", visible=False)
    axis.set_ylim(0, 0.85)
    axis.legend(loc="upper right", ncol=2, handlelength=1.0)
    save(figure, "fig_rmse_vs_selection.png")


def fig_signal_bands() -> None:
    bands = read("signal_v2_bands")
    order = ["near_zero", "weak", "medium", "strong", "boundary_adjacent"]
    names = ["near-zero", "weak", "medium", "strong", "boundary"]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.9), sharey=True)
    for axis, budget in zip(axes, (4096, 16384), strict=True):
        part = bands.loc[bands.total_shot_budget == budget].set_index("signal_band").reindex(order)
        x = np.arange(len(order))
        axis.plot(
            x, part.fit_better_share, marker="^", color=METHOD_COLORS["zne"], label="ZNE better"
        )
        axis.plot(
            x, part.raw_better_share, marker="o", color=METHOD_COLORS["raw"], label="Raw better"
        )
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(f"{budget:,} shots", loc="left")
    axes[0].set_ylabel("Share of conditions")
    axes[1].legend(loc="upper right")
    save(figure, "fig_signal_bands.png")


def fig_shots_to_reliability() -> None:
    ladder = read("shots_reliability_ladder_v1")
    cells = [
        ("symmetric_gap", 0.05, "medium", "(a) No differential bias"),
        ("readout_imbalanced", 0.05, "medium", "(b) Differential readout bias"),
        ("gate_imbalanced", 0.10, "medium", "(c) Differential gate-noise bias"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=True)
    for axis, (task, margin, noise, title) in zip(axes, cells, strict=True):
        part = ladder.loc[
            (ladder.task == task) & np.isclose(ladder.margin, margin) & (ladder.noise == noise)
        ]
        for method in ("raw", "readout", "zne", "combined"):
            line = part.loc[part.method == method].sort_values("budget")
            axis.plot(
                line.budget,
                line.decision_error,
                marker=MARKERS[method],
                markersize=3,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
        axis.axhline(0.10, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
        axis.set_xscale("log", base=2)
        axis.set_xticks(
            [2**10, 2**13, 2**16, 2**19], ["$2^{10}$", "$2^{13}$", "$2^{16}$", "$2^{19}$"]
        )
        axis.set_title(title, loc="left")
        axis.set_xlabel("Total shots, all overhead charged")
    axes[0].set_ylabel("Decision error")
    axes[2].text(2**18, 0.12, "10% target", fontsize=6.5, color=MUTED, ha="center")
    axes[0].legend(loc="upper right")
    save(figure, "fig_shots_to_reliability.png")


def fig_zne_gate_holdout() -> None:
    table = read("zne_gate_v2_holdout")
    names = ["v1 gate (reads exact value)", "v1 runtime checks", "v2 selected gate"]
    colors = [MUTED, SLOTS[1], SLOTS[0]]
    figure, axis = plt.subplots(figsize=(COLUMN, 2.0))
    for row, name, color in zip(table.itertuples(), names, colors, strict=True):
        axis.errorbar(
            row.false_accept_rate,
            row.false_reject_rate,
            xerr=[
                [row.false_accept_rate - row.fa_cluster_low],
                [row.fa_cluster_high - row.false_accept_rate],
            ],
            yerr=[
                [row.false_reject_rate - row.fr_cluster_low],
                [row.fr_cluster_high - row.false_reject_rate],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2,
            markersize=5,
        )
        axis.annotate(
            name,
            (row.false_accept_rate, row.false_reject_rate),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6.5,
            color=INK,
        )
    axis.set_xlim(-0.02, 0.8)
    axis.set_ylim(-0.03, 0.7)
    axis.set_xlabel("False acceptance rate (holdout)")
    axis.set_ylabel("False rejection rate")
    save(figure, "fig_zne_gate_holdout.png")


def fig_certification() -> None:
    groups = read("certification_v2_groups")
    part = groups.loc[groups.grouping == "gap_class"]
    classes = ["clear", "moderate", "near_tie"]
    rules = [
        ("both_layers", "Both layers", SLOTS[0]),
        ("layer_one_only", "Layer 1 only", SLOTS[1]),
        ("naive_independent_z", "Naive independent z", SLOTS[2]),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.8))
    width = 0.26
    x = np.arange(len(classes))
    for offset, (rule, name, color) in zip((-width, 0, width), rules, strict=True):
        values = part.loc[part.rule == rule].set_index("value").reindex(classes)
        axes[0].bar(
            x + offset,
            values.certified_rate,
            width,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.8,
            label=name,
        )
        axes[1].bar(
            x + offset,
            values.false_certification_given_certified,
            width,
            color=color,
            edgecolor=SURFACE,
            linewidth=0.8,
        )
    for axis, title in zip(axes, ("(a) Certified rate", "(b) Wrong, given certified"), strict=True):
        axis.set_xticks(x, ["clear", "moderate", "near tie"])
        axis.set_title(title, loc="left")
        axis.grid(axis="x", visible=False)
    axes[0].legend(loc="lower center", bbox_to_anchor=(1.1, 1.08), ncol=3, handlelength=1.0)
    save(figure, "fig_certification.png")


def fig_readout_width() -> None:
    summary = read("readout_feasibility_v2_summary")
    part = summary.loc[
        (summary.distribution == "product") & (summary.candidate_pair == "margin_0.10")
    ]
    panels = [
        ("fixed_total", 16384, "independent", "(a) Shared budget, independent"),
        ("fixed_total", 16384, "pair_correlated", "(b) Shared budget, pair-correlated"),
        ("fixed_quality", 1024, "pair_correlated", "(c) 1,024 shots per calibration circuit"),
    ]
    methods = [
        ("none", "No correction", SLOTS[0], "o"),
        ("full_matrix", "Full matrix", SLOTS[1], "s"),
        ("tensor_product", "Tensor product", SLOTS[2], "^"),
        ("pairwise_local", "Pairwise local", SLOTS[3], "D"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=True)
    for axis, (regime, setting, model, title) in zip(axes, panels, strict=True):
        cell = part.loc[
            (part.regime == regime) & (part.regime_setting == setting) & (part.model == model)
        ]
        for method, name, color, marker in methods:
            line = cell.loc[cell.method == method].sort_values("width")
            axis.plot(line.width, line.rmse, marker=marker, color=color, label=name, markersize=3.5)
            failed = line.loc[line.failure_probability >= 0.5]
            for width in failed.width:
                axis.annotate(
                    "full matrix\nfailed", (width, 0.255), ha="center", fontsize=6, color=MUTED
                )
        axis.set_xticks([4, 6, 8, 10])
        axis.set_ylim(0, 0.3)
        axis.set_xlabel("Width (qubits)")
        axis.set_title(title, loc="left")
    axes[0].set_ylabel("RMSE of the higher candidate")
    axes[0].legend(loc="upper left")
    save(figure, "fig_readout_width.png")


def fig_mismatch_dose() -> None:
    summary = read("readout_mismatch_dose_summary_v1")
    part = summary.loc[(summary.width == 6) & (summary.distribution == "product")]
    scenarios = [
        ("asymmetric_drift", "(a) Asymmetric drift"),
        ("independent_to_correlated", "(b) Independent to correlated"),
        ("pair_correlated", "(c) Added pair flips"),
    ]
    methods = [
        ("none", "No correction", SLOTS[0], "o"),
        ("tensor_product", "Tensor product", SLOTS[2], "^"),
        ("pairwise_local", "Pairwise local", SLOTS[3], "D"),
        ("full_matrix", "Full matrix", SLOTS[1], "s"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.0), sharey=True)
    for axis, (scenario, title) in zip(axes, scenarios, strict=True):
        cell = part.loc[part.scenario == scenario]
        for method, name, color, marker in methods:
            line = cell.loc[cell.method == method].sort_values("lambda")
            axis.plot(
                line["lambda"],
                line.rmse,
                marker=marker,
                markevery=4,
                markersize=3.5,
                color=color,
                label=name,
            )
        axis.set_xlabel(r"Mismatch dose $\lambda$")
        axis.set_title(title, loc="left")
    axes[0].set_ylabel("RMSE of the higher candidate")
    axes[0].legend(loc="upper left")
    save(figure, "fig_mismatch_dose.png")


def fig_drift_policy() -> None:
    summary = read("drift_policy_summary_v1")
    summary["name"] = summary.policy + summary.policy_parameter.map(
        lambda v: "" if v == 0 else f" {v:g}"
    )
    order = [
        "never",
        "fixed_interval 50",
        "fixed_interval 10",
        "triggered 4",
        "triggered 3",
        "triggered 2",
        "every_decision",
    ]
    pretty = {
        "never": "Never",
        "every_decision": "Every decision",
        "fixed_interval 10": "Every 10th",
        "fixed_interval 50": "Every 50th",
        "triggered 2": "Trigger, z = 2",
        "triggered 3": "Trigger, z = 3",
        "triggered 4": "Trigger, z = 4",
    }
    colors = {
        "never": MUTED,
        "every_decision": SLOTS[1],
        "fixed_interval": SLOTS[3],
        "triggered": SLOTS[0],
    }
    figure, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0), sharey=True)
    for axis, shape in zip(axes, ("ramp", "sinusoid"), strict=True):
        part = summary.loc[(summary["shape"] == shape) & np.isclose(summary.amplitude, 0.06)]
        means = (
            part.groupby(["policy", "name"])[["shots_per_decision", "decision_error"]]
            .mean()
            .reset_index()
        )
        means = means.set_index("name").reindex(order)
        y = np.arange(len(order))
        axis.hlines(y, 0.11, means.decision_error, color=GRID, linewidth=1.0, zorder=1)
        axis.scatter(
            means.decision_error,
            y,
            s=20,
            c=[colors[p] for p in means.policy],
            zorder=3,
            edgecolor=SURFACE,
            linewidth=0.6,
        )
        for row, value, shots in zip(
            y, means.decision_error, means.shots_per_decision, strict=True
        ):
            axis.text(value + 0.003, row, f"{shots:,.0f}", va="center", fontsize=5.8, color=MUTED)
        axis.set_xlim(0.11, 0.215)
        axis.set_title(f"{shape.capitalize()}, amplitude 0.06", loc="left")
        axis.set_xlabel("Mean decision error")
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(order)), [pretty[name] for name in order])
    save(figure, "fig_drift_policy.png")


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def macros() -> dict[str, str]:
    """Return every new number the revised paper prints, read from committed CSVs."""

    values: dict[str, str] = {}
    confirm = read("confirmatory_results_v1")
    core = confirm.loc[
        (confirm.role == "core_counterexample") & confirm.cell.str.endswith("16384")
    ].iloc[0]
    values.update(
        CoreRawError=_fmt(core.first_bad_rate),
        CoreZneError=_fmt(core.second_bad_rate),
        CorePairs=str(int(core.paired_repeats)),
        CorePValue=f"{core.p_value:.3f}",
        ConfirmatoryCount=str(int((confirm.group == "confirmatory").sum())),
        HoldoutComparisons=str(int((confirm.group == "holdout").sum())),
    )
    for group, prefix in (("confirmatory", "Confirm"), ("holdout", "Holdout")):
        part = confirm.loc[confirm.group == group].classification.value_counts()
        values[f"{prefix}Resolved"] = str(int(part.get("resolved_difference", 0)))
        values[f"{prefix}Close"] = str(int(part.get("close_within_margin", 0)))
        values[f"{prefix}Underpowered"] = str(int(part.get("underpowered", 0)))
    targets = read("shots_reliability_targets_v1")
    ten = targets.loc[np.isclose(targets.target_error, 0.10)]
    values["ReliabilityCells"] = str(int(ten.loc[ten.method == "raw"].shape[0]))
    for method, name in (
        ("raw", "Raw"),
        ("readout", "Readout"),
        ("zne", "Zne"),
        ("combined", "Combined"),
    ):
        status = ten.loc[ten.method == method].status.value_counts()
        values[f"Reached{name}"] = str(int(status.get("reached", 0)))
        values[f"Blocked{name}"] = str(int(status.get("blocked_by_bias", 0)))
    bands = read("signal_v2_bands").set_index(["signal_band", "total_shot_budget"])
    values["NearZeroRawBetter"] = _pct(bands.loc[("near_zero", 4096), "raw_better_share"])
    values["BoundaryFitBetter"] = _pct(bands.loc[("boundary_adjacent", 4096), "fit_better_share"])
    contrast = read("signal_v2_band_contrast").iloc[0]
    values["BandContrast"] = _fmt(contrast.estimate)
    values["BandContrastLow"] = _fmt(contrast.cluster_low)
    values["BandContrastHigh"] = _fmt(contrast.cluster_high)
    gate = read("zne_gate_v2_holdout").set_index("gate")
    rows = list(gate.index)
    for key, name in zip(rows, ("GateVOneFull", "GateVOneRuntime", "GateVTwo"), strict=True):
        values[f"{name}FA"] = _fmt(gate.loc[key, "false_accept_rate"])
        values[f"{name}FR"] = _fmt(gate.loc[key, "false_reject_rate"])
    holdout = read("selector_v3_holdout").set_index("policy")
    values["ThreeRegretSelector"] = f"{holdout.loc['selector', 'forced_regret']:.4f}"
    values["ReadoutThreeRegret"] = f"{holdout.loc['always_readout', 'forced_regret']:.4f}"
    values["RawThreeRegret"] = f"{holdout.loc['always_raw', 'forced_regret']:.4f}"
    values["ThreeCoverageSelector"] = _fmt(holdout.loc["selector", "coverage"], 2)
    ratio = (
        holdout.loc["selector", "mean_gate_shots"]
        / holdout.loc["always_readout", "mean_gate_shots"]
    )
    values["ThreeCostRatioSelector"] = f"{ratio:.2f}"
    cert = read("certification_v2_groups")
    near = cert.loc[
        (cert.grouping == "gap_class") & (cert.value == "near_tie") & (cert.rule == "both_layers")
    ].iloc[0]
    over = cert.loc[
        (cert.grouping == "noise")
        & (cert.value == "coherent_overrotation")
        & (cert.rule == "both_layers")
    ].iloc[0]
    values["CertWrongNearTie"] = _pct(near.false_certification_given_certified)
    values["CertWrongOverrotation"] = _pct(over.false_certification_given_certified)
    summary = read("certification_v2_summary").set_index("rule")
    values["BothLayersCertified"] = _fmt(summary.loc["both_layers", "certified_rate"])
    values["LayerOneCertified"] = _fmt(summary.loc["layer_one_only", "certified_rate"])
    values["NaiveCertified"] = _fmt(summary.loc["naive_independent_z", "certified_rate"])
    values["BothLayersFalse"] = f"{summary.loc['both_layers', 'false_certification_per_run']:.4f}"
    values["NaiveFalse"] = (
        f"{summary.loc['naive_independent_z', 'false_certification_per_run']:.4f}"
    )
    feasibility = read("readout_feasibility_v2_summary")
    width8 = feasibility.loc[
        (feasibility.width == 8)
        & (feasibility.method == "full_matrix")
        & (feasibility.regime == "fixed_total")
    ]
    low = width8.loc[width8.regime_setting == 4096].failure_probability
    values["FullFailLow"] = _pct(low.min())
    values["FullFailHigh"] = _pct(low.max())
    quality = feasibility.loc[
        (feasibility.method == "full_matrix")
        & (feasibility.regime == "fixed_quality")
        & (feasibility.width >= 8)
    ]
    values["FullQualityShareLow"] = _pct(quality.calibration_share.min())
    values["FullQualityShareHigh"] = _pct(quality.calibration_share.max())
    thresholds = read("readout_mismatch_thresholds_v1")
    check = thresholds.warns_in_time_check_drift
    values["CheckWarnShare"] = _pct(check.mean())
    values["ConditionWarnShare"] = _pct(thresholds.warns_in_time_condition_number.mean())
    return values


def grid_labels_at(threshold: float) -> pd.Series:
    """Main-grid label counts if the frozen practical threshold had been another value.

    The label rule is reused unchanged; only its threshold is swapped for the duration of the call.
    """

    from src import regime_labels_v2

    summary = read("benchmark_summary_v1")
    frozen = regime_labels_v2.PRACTICAL
    regime_labels_v2.PRACTICAL = threshold
    try:
        labels = regime_labels_v2.build_labels_v2(summary).oracle_label
    finally:
        regime_labels_v2.PRACTICAL = frozen
    return labels.value_counts()


def signal_shares_at(threshold: float) -> tuple[float, float]:
    """Near-zero raw-better and boundary fit-better shares at 4,096 shots for one threshold."""

    truth = read("signal_v2_truth")
    manifest = pd.read_csv(ROOT / "experiments/manifests/signal_v2_manifest.csv")
    frame = truth.loc[
        (truth.design == "global_135")
        & (truth.model == "linear")
        & (truth.noise != "none")
        & (truth.total_shot_budget == 4096)
    ].merge(manifest[["circuit_id", "signal_band"]], on="circuit_id")
    gain = frame.raw_rmse - frame.fit_rmse
    near = frame.signal_band == "near_zero"
    edge = frame.signal_band == "boundary_adjacent"
    return float((gain[near] < -threshold).mean()), float((gain[edge] > threshold).mean())


def threshold_sensitivity() -> pd.DataFrame:
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        counts = grid_labels_at(threshold)
        near, edge = signal_shares_at(threshold)
        rows.append(
            {
                "threshold": threshold,
                **{
                    label: int(counts.get(label, 0))
                    for label in ("raw", "practical_tie", "readout", "zne", "inconclusive")
                },
                "near_zero_raw_better": near,
                "boundary_fit_better": edge,
            }
        )
    return pd.DataFrame(rows)


def write_sensitivity(table: pd.DataFrame) -> dict[str, str]:
    lines = []
    for row in table.itertuples():
        mark = r"\textbf{%s}" if row.threshold == 0.005 else "%s"
        cells = [
            mark % f"{row.threshold:.3f}",
            *(str(getattr(row, key)) for key in ("raw", "practical_tie", "readout", "zne")),
            str(row.inconclusive),
            _pct(row.near_zero_raw_better),
            _pct(row.boundary_fit_better),
        ]
        lines.append(" & ".join(cells) + r" \\")
    # The rule must close the table from inside the input file; LaTeX rejects it after \input.
    lines.append(r"\bottomrule")
    SENSITIVITY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    at = table.set_index("threshold")
    return {
        "ThresholdRawTen": str(at.loc[0.01, "raw"]),
        "ThresholdTieTen": str(at.loc[0.01, "practical_tie"]),
        "ThresholdRawTwenty": str(at.loc[0.02, "raw"]),
        "ThresholdNearZeroTwenty": _pct(at.loc[0.02, "near_zero_raw_better"]),
        "ThresholdReadoutLow": str(table.readout.min()),
        "ThresholdReadoutHigh": str(table.readout.max()),
    }


def write_macros() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by src/generate_paper_v2_assets.py from committed processed results."]
    values = {**macros(), **write_sensitivity(threshold_sensitivity())}
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(values.items())]
    MACROS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    style()
    for builder in (
        fig_regime_shares,
        fig_rmse_vs_selection,
        fig_signal_bands,
        fig_shots_to_reliability,
        fig_zne_gate_holdout,
        fig_certification,
        fig_readout_width,
        fig_mismatch_dose,
        fig_drift_policy,
    ):
        builder()
    write_macros()


if __name__ == "__main__":
    main()
