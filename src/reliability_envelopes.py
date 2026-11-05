"""Empirical per-method reliability ranges on the frozen simulator grid."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import derive_seed
from src.metrics import CONDITION_COLUMNS, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/method_envelopes_v1.yaml"
PROCESSED = ROOT / "results/processed"
RAW = ROOT / "results/raw"
CELLS_PATH = PROCESSED / "method_envelope_cells_v1.csv"
ENVELOPE_PATH = PROCESSED / "method_reliability_envelopes_v1.csv"
VALIDATION_PATH = PROCESSED / "method_envelope_validation_v1.csv"
LEVEL_VALIDATION_PATH = PROCESSED / "method_envelope_level_validation_v1.csv"
GATE_PATH = PROCESSED / "selector_gate_stability_v1.csv"
FIGURE_PATH = ROOT / "figures/method_reliability_envelopes.png"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
METHODS = ("raw", "readout", "zne")
NEGLIGIBLE = float(CONFIG["negligible_regret_max"])
COMPLETE = int(CONFIG["complete_repeats"])
SUPPORT = CONFIG["support_rule"]
STABILITY = CONFIG["stability_rule"]
Z_95 = float(norm.ppf(0.975))
FEATURES = (
    "width",
    "depth",
    "noise",
    "shots",
    "execution_overhead",
    "absolute_reference",
    "condition_number",
    "fit_residual",
    "signal_retention",
)
FEATURE_TITLES = {
    "width": "Width",
    "depth": "Depth",
    "noise": "Noise",
    "shots": "Total shots",
    "execution_overhead": "Circuit executions",
    "absolute_reference": "|Reference|",
    "condition_number": "Condition number",
    "fit_residual": "ZNE fit residual",
    "signal_retention": "Signal retention",
    "decision_margin": "Decision margin (w2, d3)",
}


def bin_label(value: float, edges: list[float]) -> str | None:
    """Return a half-open interval label, with the last bin unbounded."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    for low, high in zip(edges, edges[1:], strict=False):
        if low <= value < high:
            return f"[{low:g}, {high:g})"
    if value >= edges[-1]:
        return f">= {edges[-1]:g}"
    return f"< {edges[0]:g}"


def signal_retention(factor_estimates: list[float], factor_shots: list[int]) -> float:
    """Return |E(5)|/|E(1)|, or NaN when factor one is within two standard errors of zero."""

    first = float(factor_estimates[0])
    variance = max(0.0, 1.0 - first**2) / int(factor_shots[0])
    if abs(first) <= 2.0 * math.sqrt(variance) or first == 0.0:
        return float("nan")
    return abs(float(factor_estimates[-1])) / abs(first)


def _payload(text: object) -> dict[str, object]:
    if not isinstance(text, str) or not text.startswith("{"):
        return {}
    return json.loads(text)


def _ghz_repeats() -> pd.DataFrame:
    frame = pd.read_csv(
        RAW / "day12_ghz_volumetric_v1_repeats-50.csv",
        usecols=[
            "width",
            "depth",
            "noise",
            "shots",
            "method",
            "repeat",
            "status",
            "estimate",
            "raw_payload_json",
        ],
        low_memory=False,
    )
    frame = frame.loc[frame.status != "not_applicable"].copy()
    frame["family"] = "ghz"
    frame["circuit_id"] = frame.width.map(lambda width: f"ghz_w{int(width)}")
    frame["reference"] = 1.0
    payloads = frame.raw_payload_json.map(_payload)
    frame["condition_number"] = payloads.map(lambda item: item.get("condition_number"))
    frame["fit_residual_sum_squares"] = payloads.map(
        lambda item: item.get("residual_sum_squares")
    )
    frame["execution_overhead"] = [
        1.0 if method == "raw" else 3.0 if method == "zne" else 2.0**width + 1.0
        for method, width in zip(frame.method, frame.width, strict=True)
    ]
    return frame


def _circuit_repeats(path: Path) -> pd.DataFrame:
    columns = [
        "family",
        "circuit_id",
        "width",
        "depth",
        "noise",
        "shots",
        "method",
        "repeat",
        "status",
        "estimate",
        "noiseless_statevector_reference",
        "condition_number",
        "fit_residual_sum_squares",
        "total_circuit_executions",
        "raw_payload_json",
    ]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    return frame.rename(
        columns={
            "noiseless_statevector_reference": "reference",
            "total_circuit_executions": "execution_overhead",
        }
    )


def load_repeats() -> pd.DataFrame:
    """Return one row per repeat with errors and diagnostics, keeping failed rows."""

    frames = [
        _ghz_repeats(),
        _circuit_repeats(RAW / "hardware_efficient_v1_repeats-50.csv"),
        _circuit_repeats(RAW / "noise_shift_holdout_v1_repeats-50.csv"),
    ]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["error"] = pd.to_numeric(frame.estimate, errors="coerce") - frame.reference
    frame.loc[frame.status != "complete", "error"] = np.nan
    frame["condition_number"] = pd.to_numeric(frame.condition_number, errors="coerce")
    retention = []
    for method, text in zip(frame.method, frame.raw_payload_json, strict=True):
        if method != "zne":
            retention.append(np.nan)
            continue
        payload = _payload(text)
        retention.append(
            signal_retention(payload["factor_estimates"], payload["shots_by_factor"])
            if payload
            else np.nan
        )
    frame["signal_retention"] = retention
    return frame.drop(columns="raw_payload_json")


def repeat_bootstrap_rates(errors: np.ndarray, seed: int) -> np.ndarray:
    """Return each method's reliable share when repeats are resampled within one cell."""

    resamples = int(CONFIG["repeat_bootstrap_resamples"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, errors.shape[1], size=(resamples, errors.shape[1]))
    complete = ~np.isnan(errors).any(axis=1)
    squares = errors**2
    rmse = np.full((resamples, errors.shape[0]), np.nan)
    for row in range(errors.shape[0]):
        drawn = squares[row][indices]
        counts = np.sum(~np.isnan(drawn), axis=1)
        totals = np.nansum(drawn, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rmse[:, row] = np.sqrt(totals / counts)
    best = np.nanmin(rmse, axis=1, keepdims=True)
    reliable = (rmse - best) <= NEGLIGIBLE
    rates = reliable.mean(axis=0)
    rates[~complete] = 0.0
    return rates


def build_cell_table(repeats: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    """Return one row per method and cell with reliability and binned features."""

    bins = CONFIG["bins"]
    sets = evaluation[[*CONDITION_COLUMNS, "evaluation_set"]].drop_duplicates()
    rows: list[dict[str, object]] = []
    for cell_index, (condition, group) in enumerate(
        repeats.groupby(CONDITION_COLUMNS, sort=True)
    ):
        methods = {method: group.loc[group.method == method] for method in METHODS}
        errors = np.vstack(
            [
                methods[method].sort_values("repeat").error.to_numpy(dtype=float)
                for method in METHODS
            ]
        )
        rates = repeat_bootstrap_rates(errors, derive_seed("envelope-repeats", cell_index))
        rmse = np.sqrt(np.nanmean(errors**2, axis=1))
        best = float(np.nanmin(rmse))
        for position, method in enumerate(METHODS):
            frame = methods[method]
            complete = int((frame.status == "complete").sum())
            regret = float(rmse[position] - best)
            row: dict[str, object] = dict(zip(CONDITION_COLUMNS, condition, strict=True))
            reference = float(frame.reference.iloc[0])
            row.update(
                method=method,
                complete_repeats=complete,
                rmse=float(rmse[position]),
                rmse_regret=regret,
                reliable=bool(complete == COMPLETE and regret <= NEGLIGIBLE),
                repeat_bootstrap_reliable_rate=float(rates[position]),
                execution_overhead=float(frame.execution_overhead.iloc[0]),
                absolute_reference=bin_label(abs(reference), bins["absolute_reference"]),
                condition_number=None,
                fit_residual=None,
                signal_retention=None,
            )
            if method == "readout":
                values = frame.condition_number.where(frame.status == "complete", np.inf)
                row["condition_number"] = bin_label(
                    float(np.median(values.to_numpy(dtype=float))), bins["condition_number"]
                )
            if method == "zne":
                fit = pd.to_numeric(frame.fit_residual_sum_squares, errors="coerce")
                row["fit_residual"] = bin_label(float(fit.median()), bins["fit_residual"])
                ratio = frame.signal_retention.dropna()
                row["signal_retention"] = (
                    bin_label(float(ratio.median()), bins["signal_retention"])
                    if len(ratio) >= len(frame) / 2.0
                    else "unidentifiable"
                )
            rows.append(row)
    cells = pd.DataFrame(rows).merge(sets, on=CONDITION_COLUMNS, how="left", validate="many_to_one")
    if cells.evaluation_set.isna().any():
        raise ValueError("Every cell must belong to one evaluation set")
    return cells


def support_status(cells: int, reliable: int) -> str:
    """Apply the frozen support rule to one feature level."""

    if cells < int(SUPPORT["min_cells"]):
        return "insufficient_cells"
    low, _ = wilson_interval(reliable, cells)
    passes = reliable / cells >= float(SUPPORT["min_reliable_rate"])
    return "supported" if passes and low >= float(SUPPORT["min_wilson_low"]) else "unsupported"


def _supported_frequency(rates: np.ndarray, seed: int) -> float:
    resamples = int(CONFIG["cell_bootstrap_resamples"])
    count = len(rates)
    if count < int(SUPPORT["min_cells"]):
        return 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(resamples, count))
    successes = (rng.random((resamples, count)) < rates[indices]).sum(axis=1)
    share = successes / count
    denominator = 1.0 + Z_95**2 / count
    center = (share + Z_95**2 / (2.0 * count)) / denominator
    radius = (
        Z_95 * np.sqrt(share * (1.0 - share) / count + Z_95**2 / (4.0 * count**2)) / denominator
    )
    passed = (share >= float(SUPPORT["min_reliable_rate"])) & (
        center - radius >= float(SUPPORT["min_wilson_low"])
    )
    return float(passed.mean())


def stability_label(frequency: float) -> str:
    """Name how often a level keeps its support status under resampling."""

    if frequency >= float(STABILITY["stable_supported_frequency"]):
        return "stable_supported"
    if frequency <= float(STABILITY["stable_unsupported_frequency"]):
        return "stable_unsupported"
    return "unstable"


def _level_order(value: object) -> tuple[float, str]:
    text = str(value)
    if text == "unidentifiable":
        return (float("inf"), text)
    digits = text.lstrip("[>= <").split(",")[0]
    try:
        return (float(digits), text)
    except ValueError:
        order = {"zero": 0, "low": 1, "medium": 2, "high": 3}
        return (float(order.get(text, 4)), text)


def build_envelopes(cells: pd.DataFrame, decision: pd.DataFrame) -> pd.DataFrame:
    """Return support for every observed method and feature level on the design set."""

    design = cells.loc[cells.evaluation_set == CONFIG["design_set"]]
    rows: list[dict[str, object]] = []

    def add(
        method: str, feature: str, level: object, flags: np.ndarray, rates: np.ndarray, source: str
    ) -> None:
        count, reliable = len(flags), int(flags.sum())
        low, high = wilson_interval(reliable, count)
        frequency = _supported_frequency(
            rates, derive_seed("envelope-cells", len(rows), METHODS.index(method))
        )
        rows.append(
            {
                "method": method,
                "feature": feature,
                "level": str(level),
                "source": source,
                "cells": count,
                "reliable_cells": reliable,
                "reliable_rate": reliable / count,
                "wilson_low": low,
                "wilson_high": high,
                "status": support_status(count, reliable),
                "bootstrap_supported_frequency": frequency,
                "stability": stability_label(frequency),
            }
        )

    for method in METHODS:
        frame = design.loc[design.method == method]
        for feature in FEATURES:
            present = frame.dropna(subset=[feature])
            levels = sorted(present[feature].unique(), key=_level_order)
            for level in levels:
                group = present.loc[present[feature] == level]
                add(
                    method,
                    feature,
                    level,
                    group.reliable.to_numpy(dtype=bool),
                    group.repeat_bootstrap_reliable_rate.to_numpy(dtype=float),
                    "rule_design_cells",
                )
        margins = decision.loc[
            (decision.method == method) & (decision.record_mode == "independent_records")
        ]
        for margin, group in margins.groupby("margin", sort=True):
            flags = (group.status == "complete").to_numpy(dtype=bool)
            add(
                method,
                "decision_margin",
                f"{margin:g}",
                flags,
                flags.astype(float),
                "decision_benchmark_w2_d3",
            )
    return pd.DataFrame(rows)


def supported_levels(envelopes: pd.DataFrame) -> dict[tuple[str, str], set[str]]:
    """Return the supported level set per method and cell-level feature."""

    output: dict[tuple[str, str], set[str]] = {}
    for row in envelopes.itertuples():
        if row.feature == "decision_margin":
            continue
        key = (row.method, row.feature)
        output.setdefault(key, set())
        if row.status == "supported":
            output[key].add(row.level)
    return output


def inside_envelope(
    row: dict[str, object],
    support: dict[tuple[str, str], set[str]],
    *,
    ignore: tuple[str, ...] = (),
) -> bool:
    """Require every applicable marginal feature to be supported for this method."""

    for feature in FEATURES:
        if feature in ignore or pd.isna(row[feature]):
            continue
        level = str(row[feature])
        if level not in support.get((str(row["method"]), feature), set()):
            return False
    return True


def validate_envelopes(cells: pd.DataFrame, envelopes: pd.DataFrame) -> pd.DataFrame:
    """Score design-set envelopes on cells that did not define them."""

    support = supported_levels(envelopes)
    rows: list[dict[str, object]] = []
    variants = {"all_features": (), "ignoring_noise_name": ("noise",)}
    for name in CONFIG["validation_sets"]:
        for method in METHODS:
            frame = cells.loc[(cells.evaluation_set == name) & (cells.method == method)]
            for variant, ignore in variants.items():
                flags = [
                    inside_envelope(row, support, ignore=ignore) for row in frame.to_dict("records")
                ]
                inside = pd.Series(flags, index=frame.index, dtype=bool)
                for region, mask in (("inside", inside), ("outside", ~inside)):
                    group = frame.loc[mask]
                    count, reliable = len(group), int(group.reliable.sum())
                    low, high = wilson_interval(reliable, count) if count else (np.nan, np.nan)
                    rows.append(
                        {
                            "evaluation_set": name,
                            "method": method,
                            "variant": variant,
                            "region": region,
                            "cells": count,
                            "reliable_cells": reliable,
                            "reliable_rate": reliable / count if count else np.nan,
                            "wilson_low": low,
                            "wilson_high": high,
                        }
                    )
    return pd.DataFrame(rows)


def validate_supported_levels(cells: pd.DataFrame, envelopes: pd.DataFrame) -> pd.DataFrame:
    """Score each supported single level on cells that did not define it."""

    rows: list[dict[str, object]] = []
    supported = envelopes.loc[
        (envelopes.status == "supported") & (envelopes.feature != "decision_margin")
    ]
    for item in supported.itertuples():
        for name in CONFIG["validation_sets"]:
            frame = cells.loc[(cells.evaluation_set == name) & (cells.method == item.method)]
            group = frame.loc[frame[item.feature].astype(str) == item.level]
            count, reliable = len(group), int(group.reliable.sum())
            low, high = wilson_interval(reliable, count) if count else (np.nan, np.nan)
            rows.append(
                {
                    "method": item.method,
                    "feature": item.feature,
                    "level": item.level,
                    "stability": item.stability,
                    "design_reliable_rate": item.reliable_rate,
                    "evaluation_set": name,
                    "cells": count,
                    "reliable_cells": reliable,
                    "reliable_rate": reliable / count if count else np.nan,
                    "wilson_low": low,
                    "wilson_high": high,
                }
            )
    return pd.DataFrame(rows)


def gate_stability(repeats: pd.DataFrame) -> pd.DataFrame:
    """Resample repeats to show how often each selector gate would change its answer."""

    resamples = int(CONFIG["gate_bootstrap_resamples"])
    low, high = CONFIG["gate_unstable_range"]
    sources = (
        ("condition_number", "readout", "condition_number"),
        ("fit_residual", "zne", "fit_residual_sum_squares"),
    )
    rows: list[dict[str, object]] = []
    for diagnostic, method, column in sources:
        frame = repeats.loc[(repeats.method == method) & (repeats.status == "complete")]
        for cell_index, (condition, group) in enumerate(
            frame.groupby(CONDITION_COLUMNS, sort=True)
        ):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float)
            rng = np.random.default_rng(derive_seed("gate-stability", cell_index, len(diagnostic)))
            means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
            for threshold in CONFIG["selector_gates"][diagnostic]:
                share = float((means > threshold).mean())
                rows.append(
                    {
                        **dict(zip(CONDITION_COLUMNS, condition, strict=True)),
                        "diagnostic": diagnostic,
                        "threshold": threshold,
                        "completed_repeats": len(values),
                        "observed_mean": float(values.mean()),
                        "observed_above_threshold": bool(values.mean() > threshold),
                        "bootstrap_above_share": share,
                        "unstable": bool(low < share < high),
                    }
                )
    return pd.DataFrame(rows)


def plot_envelopes(envelopes: pd.DataFrame, path: Path = FIGURE_PATH) -> None:
    """Draw the reliable rate of each level, marking support status and sample size."""

    order = [*FEATURES, "decision_margin"]
    layout: list[tuple[str, str]] = []
    for feature in order:
        levels = envelopes.loc[envelopes.feature == feature, "level"].unique().tolist()
        layout.extend((feature, level) for level in sorted(levels, key=_level_order))
    figure, axis = plt.subplots(figsize=(8.8, 0.31 * len(layout) + 1.7))
    colormap = plt.get_cmap("Blues")
    for row_index, (feature, level) in enumerate(layout):
        for column, method in enumerate(METHODS):
            match = envelopes.loc[
                (envelopes.feature == feature)
                & (envelopes.level == level)
                & (envelopes.method == method)
            ]
            if match.empty:
                axis.add_patch(
                    plt.Rectangle((column, row_index), 1, 1, color="#F2F2F2", linewidth=0)
                )
                axis.text(column + 0.5, row_index + 0.5, "n/a", ha="center", va="center",
                          fontsize=7.5, color="#8A8A8A")
                continue
            item = match.iloc[0]
            rate = float(item.reliable_rate)
            insufficient = item.status == "insufficient_cells"
            face = "#EDEDED" if insufficient else colormap(0.10 + 0.72 * rate)
            axis.add_patch(
                plt.Rectangle((column, row_index), 1, 1, facecolor=face, edgecolor="white")
            )
            mark = {"supported": "✓ ", "unstable": ""}.get(item.status, "")
            shade = "white" if not insufficient and rate >= 0.62 else "#2B2B2B"
            axis.text(
                column + 0.5,
                row_index + 0.5,
                f"{mark}{rate:.0%}  (n={int(item.cells)})",
                ha="center",
                va="center",
                fontsize=8,
                color=shade,
                fontweight="bold" if item.status == "supported" else "normal",
            )
    axis.set_xlim(0, len(METHODS))
    axis.set_ylim(len(layout), 0)
    axis.set_xticks(np.arange(len(METHODS)) + 0.5, ["Raw", "Readout", "ZNE"])
    axis.xaxis.tick_top()
    axis.set_yticks(
        np.arange(len(layout)) + 0.5,
        [f"{FEATURE_TITLES[feature]}: {level}" for feature, level in layout],
    )
    axis.tick_params(length=0, labelsize=8.5, colors="#3A3A3A")
    for side in axis.spines.values():
        side.set_visible(False)
    figure.suptitle(
        "Reliable rate by method and level on rule-design cells\n"
        "✓ = supported (at least 10 cells, rate at least 80%, Wilson low at least 50%); "
        "gray = under 10 cells\n"
        "Decision-margin rows come from the separate width-2, depth-3 decision benchmark",
        fontsize=9.5,
        x=0.01,
        ha="left",
        color="#2B2B2B",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    repeats = load_repeats()
    evaluation = pd.read_csv(PROCESSED / "selector_evaluation_v2.csv")
    cells = build_cell_table(repeats, evaluation)
    decision = pd.read_csv(PROCESSED / "decision_margin_summary_v1.csv")
    envelopes = build_envelopes(cells, decision)
    validation = validate_envelopes(cells, envelopes)
    gates = gate_stability(repeats)
    for frame, path in (
        (cells, CELLS_PATH),
        (envelopes, ENVELOPE_PATH),
        (validation, VALIDATION_PATH),
        (validate_supported_levels(cells, envelopes), LEVEL_VALIDATION_PATH),
        (gates, GATE_PATH),
    ):
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    plot_envelopes(envelopes)


if __name__ == "__main__":
    main()
