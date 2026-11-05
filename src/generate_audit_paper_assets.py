"""Generate paper macros, tables, and figures for the later research tracks."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
RAW = ROOT / "results/raw"
GENERATED = ROOT / "paper/generated"
PAPER_FIGURES = ROOT / "paper/figures"
SOURCE_FIGURES = ROOT / "figures"
METRICS_PATH = GENERATED / "audit_metrics.tex"
REGIME_ROWS_PATH = GENERATED / "regime_rows_v2.tex"
AUDIT_ROWS_PATH = GENERATED / "audit_category_rows.tex"
ENVELOPE_ROWS_PATH = GENERATED / "envelope_rows.tex"
SEQUENTIAL_ROWS_PATH = GENERATED / "sequential_rows.tex"
LIFETIME_ROWS_PATH = GENERATED / "calibration_lifetime_rows.tex"
REGIME_FIGURE = PAPER_FIGURES / "operating_regime_map_v2.png"
COPIED_FIGURES = (
    "calibration_age_vs_decision_error.png",
    "shots_to_certification.png",
    "certification_error_tradeoff.png",
    "selector_regret_categories.png",
)
CATEGORY_NAMES = {
    "correct_materially_useful": "Correct, useful",
    "correct_practically_tied": "Correct, tied",
    "incorrect_negligible_regret": "Incorrect, negligible",
    "incorrect_material_regret": "Incorrect, material",
    "catastrophic_regret": "Catastrophic",
    "abstention_avoided_failure": "Abstained, avoided failure",
    "unnecessary_abstention": "Abstained, unnecessary",
    "inconclusive": "Inconclusive",
}
POLICIES = ("selector", "always_raw", "always_readout", "always_zne")
POLICY_NAMES = {
    "fixed_budget": "Fixed budget",
    "wilson_stopping": "Wilson",
    "gap_confidence_stopping": "Gap confidence",
    "variance_aware_stopping": "Variance aware",
    "naive_wald_peeking": "Naive peeking",
}
FEATURE_NAMES = {
    "noise": "noise",
    "absolute_reference": r"$|\mathrm{ref}|$",
    "decision_margin": "margin",
    "width": "width",
}
LABEL_ORDER = ("practical_tie", "raw", "readout", "zne", "inconclusive")
LABEL_NAMES = {
    "practical_tie": "Practical tie",
    "raw": "Raw",
    "readout": "Readout",
    "zne": "Linear ZNE",
    "inconclusive": "Inconclusive",
}
LABEL_COLORS = {
    "practical_tie": "#1baf7a",
    "raw": "#2a78d6",
    "readout": "#eb6834",
    "zne": "#6250d6",
    "inconclusive": "#9a9a9a",
}


def _percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}\\%"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def regime_macros() -> dict[str, object]:
    """Return corrected-label and selector-audit values."""

    labels = pd.read_csv(PROCESSED / "regime_labels_v2.csv")
    counts = labels.oracle_label.value_counts()
    evaluation = pd.read_csv(PROCESSED / "selector_evaluation_v2.csv")
    holdout = evaluation.loc[evaluation.evaluation_set == "holdout"]
    shift = evaluation.loc[evaluation.evaluation_set == "noise_shift"]
    audit = pd.read_csv(PROCESSED / "selector_audit_v1.csv")
    selector = audit.loc[audit.policy == "selector"]
    references = selector.drop_duplicates("circuit_id").set_index("circuit_id").reference
    raw_wins = labels.loc[labels.oracle_label == "raw"]
    low_signal = raw_wins.circuit_id.map(references).abs() < 0.06
    catastrophic = selector.loc[selector.audit_category == "catastrophic_regret"]
    fixed = audit.pivot_table(
        index="evaluation_set", columns="policy", values="rmse_regret", aggfunc="mean"
    )
    harmful = {"catastrophic_regret", "incorrect_material_regret"}

    def harm(policy: str) -> int:
        return int(audit.loc[audit.policy == policy, "audit_category"].isin(harmful).sum())

    strong = audit.loc[~audit.low_signal_descriptive]

    def strong_harm(policy: str) -> int:
        return int(strong.loc[strong.policy == policy, "audit_category"].isin(harmful).sum())

    return {
        "VTwoPracticalTies": int(counts["practical_tie"]),
        "VTwoRawWinners": int(counts["raw"]),
        "VTwoReadoutWinners": int(counts["readout"]),
        "VTwoZNEWinners": int(counts["zne"]),
        "VTwoInconclusive": int(counts["inconclusive"]),
        "RawWinsAccuracy": int((raw_wins.raw_win_basis == "accuracy").sum()),
        "RawWinsCostOnly": int((raw_wins.raw_win_basis == "cost_only").sum()),
        "RawWinsLowSignal": int(low_signal.sum()),
        "HoldoutCompatibilityVTwo": _percent(float(holdout.decision_compatible.mean())),
        "HoldoutStrictVTwo": _percent(float(holdout.strict_label_match.mean())),
        "ShiftCompatibilityVTwo": _percent(float(shift.decision_compatible.mean())),
        "ShiftStrictVTwo": _percent(float(shift.strict_label_match.mean())),
        "HoldoutRegretMean": f"{holdout.rmse_regret.mean():.4f}",
        "HoldoutRegretMedian": f"{holdout.rmse_regret.median():.4f}",
        "HoldoutRegretMax": f"{holdout.rmse_regret.max():.4f}",
        "ShiftRegretMean": f"{shift.rmse_regret.mean():.4f}",
        "HoldoutAlwaysRaw": f"{fixed.loc['holdout', 'always_raw']:.4f}",
        "HoldoutAlwaysReadout": f"{fixed.loc['holdout', 'always_readout']:.4f}",
        "ShiftAlwaysRaw": f"{fixed.loc['noise_shift', 'always_raw']:.4f}",
        "ShiftAlwaysReadout": f"{fixed.loc['noise_shift', 'always_readout']:.4f}",
        "SelectorCatastrophic": len(catastrophic),
        "SelectorCatastrophicGHZ": int((catastrophic.family == "ghz").sum()),
        "SelectorCatastrophicLowSignal": int(catastrophic.low_signal_descriptive.sum()),
        "SelectorAbstentions": int((selector.selected_method == "abstain").sum()),
        "SelectorUnnecessaryAbstentions": int(
            (selector.audit_category == "unnecessary_abstention").sum()
        ),
        "SelectorHarm": harm("selector"),
        "AlwaysRawHarm": harm("always_raw"),
        "StrongSignalCells": int((strong.policy == "selector").sum()),
        "StrongHarmSelector": strong_harm("selector"),
        "StrongHarmRaw": strong_harm("always_raw"),
        "StrongHarmReadout": strong_harm("always_readout"),
    }


def sequential_macros(raw: pd.DataFrame) -> dict[str, object]:
    """Return optional-stopping and bias values from repeat-level rows."""

    null = raw.loc[raw.margin == 0.0]
    rate = null.groupby(["method", "policy"]).wrong_certified.mean()
    near = raw.loc[
        (raw.task == "symmetric_gap")
        & (raw.margin == 0.02)
        & (raw.noise != "zero")
        & (raw.max_total_shots == 16384)
        & (raw.policy == "wilson_stopping")
    ]
    share = near.groupby("method").total_shots_used.mean() / 16384.0
    unresolved = near.groupby("method").status.apply(lambda item: (item == "unresolved").mean())
    biased = raw.loc[
        (raw.task == "gate_imbalanced")
        & (raw.noise == "high")
        & (raw.margin == 0.05)
        & (raw.max_total_shots == 16384)
        & (raw.policy == "wilson_stopping")
    ].groupby("method")
    wilson = rate.xs("wilson_stopping", level="policy")
    naive = rate.xs("naive_wald_peeking", level="policy")
    return {
        "SequentialRows": f"{len(raw):,}".replace(",", "{,}"),
        "NullWilsonMin": _percent(float(wilson.min())),
        "NullWilsonMax": _percent(float(wilson.max())),
        "NullNaiveMin": _percent(float(naive.min())),
        "NullNaiveMax": _percent(float(naive.max())),
        "NearTieShareMin": _percent(float(share.min())),
        "NearTieShareMax": _percent(float(share.max())),
        "NearTieUnresolvedMin": _percent(float(unresolved.min())),
        "NearTieUnresolvedMax": _percent(float(unresolved.max())),
        "BiasedRawWrong": _percent(float(biased.wrong_certified.mean()["raw"]), 0),
        "BiasedRawShots": f"{biased.total_shots_used.mean()['raw']:,.0f}".replace(",", "{,}"),
        "BiasedZNEWrong": _percent(float(biased.wrong_certified.mean()["zne"]), 0),
    }


def write_sequential_rows(raw: pd.DataFrame) -> None:
    """Write the symmetric-task cost table at the largest budget."""

    frame = raw.loc[
        (raw.task == "symmetric_gap")
        & (raw.margin > 0.0)
        & (raw.noise != "zero")
        & (raw.max_total_shots == 16384)
    ].copy()
    frame["unresolved"] = frame.status == "unresolved"
    lines = []
    for method, policies in (
        ("raw", tuple(POLICY_NAMES)),
        ("readout", ("wilson_stopping",)),
        ("zne", ("wilson_stopping",)),
    ):
        for policy in policies:
            group = frame.loc[(frame.method == method) & (frame.policy == policy)]
            name = {"raw": "Raw", "readout": "Readout", "zne": "ZNE"}[method]
            lines.append(
                f"{name} & {POLICY_NAMES[policy]} & "
                f"{group.total_shots_used.mean():,.0f} & "
                f"{_percent(float(group.unresolved.mean()))} & "
                f"{_percent(float(group.wrong_certified.mean()), 2)} \\\\".replace(",", "{,}")
            )
    _write_lines(SEQUENTIAL_ROWS_PATH, [*lines, r"\bottomrule"])


def lifetime_values() -> tuple[dict[str, object], list[str]]:
    """Return calibration-lifetime macros and table rows from the raw decisions."""

    raw = pd.read_csv(RAW / "calibration_lifetime_v1.csv")
    workload = raw.loc[raw.workload_length == 100]
    summary = workload.groupby("policy").agg(
        shots=("amortized_shots_per_decision", "mean"),
        error=("decision_error", "mean"),
    )
    names = (
        ("every_decision", "Every decision"),
        ("fixed_interval", "Every 10 decisions"),
        ("uncertainty_triggered", "Uncertainty trigger"),
        ("never", "Never"),
    )
    rows = [
        f"{name} & {summary.loc[key, 'shots']:,.0f} & {summary.loc[key, 'error']:.3f} \\\\".replace(
            ",", "{,}"
        )
        for key, name in names
    ]
    never = (
        workload.loc[workload.policy == "never"].groupby("drift_model").decision_error.mean()
    )
    trigger = summary.loc["uncertainty_triggered", "shots"] / summary.loc["every_decision", "shots"]
    macros = {
        "LifetimeRows": f"{len(raw):,}".replace(",", "{,}"),
        "NeverAbrupt": f"{never['abrupt_change']:.3f}",
        "NeverAsymmetric": f"{never['asymmetric']:.3f}",
        "NeverSlow": f"{never['slow_monotonic']:.3f}",
        "NeverRandom": f"{never['random_bounded']:.3f}",
        "TriggerShotSaving": _percent(1.0 - float(trigger), 0),
    }
    return macros, [*rows, r"\bottomrule"]


def write_audit_rows() -> None:
    """Write category counts for the selector and fixed policies."""

    audit = pd.read_csv(PROCESSED / "selector_audit_v1.csv")
    counts = audit.pivot_table(
        index="audit_category", columns="policy", values="width", aggfunc="size", fill_value=0
    )
    lines = []
    for key, name in CATEGORY_NAMES.items():
        values = " & ".join(
            str(int(counts.loc[key, policy])) if key in counts.index else "0"
            for policy in POLICIES
        )
        lines.append(f"{name} & {values} \\\\")
    _write_lines(AUDIT_ROWS_PATH, [*lines, r"\bottomrule"])


def write_envelope_rows() -> None:
    """Write every supported level with its counts and resampling stability."""

    envelopes = pd.read_csv(PROCESSED / "method_reliability_envelopes_v1.csv")
    supported = envelopes.loc[envelopes.status == "supported"]
    names = {"raw": "Raw", "readout": "Readout"}
    lines = []
    for row in supported.itertuples():
        level = str(row.level).replace(">=", r"$\geq$").replace("[0, 0.06)", r"$<0.06$")
        lines.append(
            f"{names[row.method]} & {FEATURE_NAMES[row.feature]} {level} & "
            f"{int(row.reliable_cells)}/{int(row.cells)} & "
            f"{100.0 * row.bootstrap_supported_frequency:.0f}\\% \\\\"
        )
    _write_lines(ENVELOPE_ROWS_PATH, [*lines, r"\bottomrule"])


def write_regime_rows() -> None:
    """Write corrected oracle-label counts."""

    counts = pd.read_csv(PROCESSED / "regime_labels_v2.csv").oracle_label.value_counts()
    lines = [f"{LABEL_NAMES[key]} & {int(counts.get(key, 0))} \\\\" for key in LABEL_ORDER]
    _write_lines(REGIME_ROWS_PATH, [*lines, r"\bottomrule"])


def plot_regime_map(path: Path = REGIME_FIGURE) -> None:
    """Plot corrected labels by main-grid noise level."""

    labels = pd.read_csv(PROCESSED / "regime_labels_v2.csv")
    counts = labels.groupby(["noise", "oracle_label"]).size().unstack(fill_value=0)
    levels = ["zero", "low", "medium", "high"]
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    bottom = [0] * len(levels)
    for key in LABEL_ORDER:
        values = [int(counts.loc[level].get(key, 0)) for level in levels]
        axis.bar(
            levels,
            values,
            bottom=bottom,
            color=LABEL_COLORS[key],
            edgecolor="white",
            linewidth=1.5,
            label=LABEL_NAMES[key],
        )
        for position, value in enumerate(values):
            if value >= 3:
                axis.text(
                    position,
                    bottom[position] + value / 2.0,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if key not in {"practical_tie", "inconclusive"} else "#2B2B2B",
                    fontsize=9,
                )
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    axis.set_xlabel("Main-grid noise level", color="#6B6B6B")
    axis.set_ylabel("Operating cells", color="#6B6B6B")
    axis.tick_params(colors="#6B6B6B", length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#E4E4E4")
    axis.legend(frameon=False, fontsize=8, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    sequential = pd.read_csv(RAW / "sequential_decision_v1.csv", low_memory=False)
    lifetime_macros, lifetime_rows = lifetime_values()
    macros = {**regime_macros(), **sequential_macros(sequential), **lifetime_macros}
    _write_lines(
        METRICS_PATH, [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()]
    )
    write_regime_rows()
    write_audit_rows()
    write_envelope_rows()
    write_sequential_rows(sequential)
    _write_lines(LIFETIME_ROWS_PATH, lifetime_rows)
    plot_regime_map()
    for name in COPIED_FIGURES:
        shutil.copyfile(SOURCE_FIGURES / name, PAPER_FIGURES / name)


if __name__ == "__main__":
    main()
