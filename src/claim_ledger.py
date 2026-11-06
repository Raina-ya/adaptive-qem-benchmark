"""Recompute paper claims from raw rows and trace every printed number to its source.

This module deliberately imports nothing from the project. Each computed claim repeats its
aggregation here, so a mistake in a generator or helper cannot silently confirm itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
PROCESSED = ROOT / "results/processed"
GENERATED = ROOT / "paper/generated"
PAPER_PATH = ROOT / "paper/main.tex"
CONFIG_PATH = ROOT / "experiments/configs/claim_ledger_v1.yaml"
LEDGER_PATH = PROCESSED / "claim_ledger_v1.csv"
PROVENANCE_PATH = PROCESSED / "paper_number_provenance_v1.csv"
AUDIT_PATH = PROCESSED / "claim_audit_sample_v1.csv"
COVERAGE_PATH = PROCESSED / "claim_coverage_summary_v1.csv"
DERIVED_PATH = PROCESSED / "derived_field_reconstruction_v1.csv"
SENSITIVITY_PATH = PROCESSED / "label_bootstrap_sensitivity_v1.csv"
REGENERATED_FIGURES = {"zne_signal_retention_map.png"}
MACRO_FILES = {
    "adaptive_metrics.tex": "src/generate_paper_assets.py",
    "audit_metrics.tex": "src/generate_audit_paper_assets.py",
    "final_metrics.tex": "src/final_synthesis.py",
}
TABLE_SOURCES = {
    "regime_rows_v2.tex": ("src/generate_audit_paper_assets.py", "regime_labels_v2.csv"),
    "audit_category_rows.tex": ("src/generate_audit_paper_assets.py", "selector_audit_v1.csv"),
    "envelope_rows.tex": (
        "src/generate_audit_paper_assets.py",
        "method_reliability_envelopes_v1.csv",
    ),
    "sequential_rows.tex": ("src/generate_audit_paper_assets.py", "sequential_decision_v1.csv"),
    "calibration_lifetime_rows.tex": (
        "src/generate_audit_paper_assets.py",
        "calibration_lifetime_v1.csv",
    ),
    "selector_v2_rows.tex": ("src/generate_audit_paper_assets.py", "selector_v2_cells.csv"),
    "hardware_cell_rows.tex": ("src/final_synthesis.py", "hardware_manifest_v1.csv"),
    "evidence_matrix_rows.tex": ("src/final_synthesis.py", "final_evidence_matrix_v1.csv"),
}
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PRACTICAL = float(CONFIG["practical_rmse_threshold"])
PENALTY = float(CONFIG["overhead_penalty"])
NEGLIGIBLE = float(CONFIG["negligible_regret"])
CATASTROPHIC = float(CONFIG["catastrophic_regret"])
LOW_SIGNAL = float(CONFIG["low_signal_reference"])
CELL = ["family", "circuit_id", "width", "depth", "noise", "shots"]
ANY_REPEATS = 50
DIAGNOSTICS = [
    "noiseless_statevector_reference",
    "total_circuit_executions",
    "condition_number",
    "fit_residual_sum_squares",
    "folded_two_qubit_gate_count",
]
RENAMED = {
    "noiseless_statevector_reference": "reference",
    "total_circuit_executions": "executions",
    "fit_residual_sum_squares": "fit_residual",
    "folded_two_qubit_gate_count": "gates",
}
ADEQUATE_BIN = 10
Z_95 = 1.959963984540054
METHODS = ("raw", "readout", "zne")
ROOT_SEED = 20261102
SOURCE_ONLY = "source_only_not_recomputed"


def _raw(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{name}.csv", usecols=columns, low_memory=False)


def _rmse(values: pd.Series) -> float:
    return math.sqrt(float(np.mean(np.square(values)))) if len(values) else math.nan


def _count(value: float) -> str:
    return str(int(value))


def _thousands(value: float) -> str:
    return f"{int(round(value)):,}"


def _percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%"


def _wilson_low(successes: int, total: int) -> float:
    rate = successes / total
    scale = 1.0 + Z_95**2 / total
    center = rate + Z_95**2 / (2.0 * total)
    margin = Z_95 * math.sqrt(rate * (1.0 - rate) / total + Z_95**2 / (4.0 * total**2))
    return (center - margin) / scale


def _payload(text: object) -> dict[str, object]:
    return json.loads(text) if isinstance(text, str) and text.startswith("{") else {}


@lru_cache(maxsize=1)
def benchmark_rows() -> pd.DataFrame:
    """Return the main GHZ and hardware-efficient repeat rows with one error column."""

    base = ["width", "depth", "noise", "shots", "method", "repeat", "status", "estimate"]
    ghz = _raw(
        "day12_ghz_volumetric_v1_repeats-50",
        [*base, "family", "analytic_reference", "raw_payload_json"],
    )
    ghz = ghz.loc[ghz.status != "not_applicable"].copy()
    ghz["circuit_id"] = "ghz_w" + ghz.width.astype(str)
    ghz["reference"] = ghz.analytic_reference
    payloads = ghz.raw_payload_json.map(_payload)
    ghz["condition_number"] = payloads.map(lambda item: item.get("condition_number"))
    ghz["fit_residual"] = payloads.map(lambda item: item.get("residual_sum_squares"))
    ghz["gates"] = ghz.width - 1.0
    overhead = {"raw": 1.0, "zne": 3.0}
    ghz["executions"] = [
        overhead.get(method, 2.0**width + 1.0)
        for method, width in zip(ghz.method, ghz.width, strict=True)
    ]
    efficient = _raw(
        "hardware_efficient_v1_repeats-50",
        [
            *base,
            "family",
            "circuit_id",
            *DIAGNOSTICS,
        ],
    ).rename(columns=RENAMED)
    columns = [*CELL, "method", "repeat", "status", "estimate", "reference", "executions"]
    columns = [*columns, "condition_number", "fit_residual", "gates"]
    frame = pd.concat([ghz[columns], efficient[columns]], ignore_index=True)
    frame["error"] = frame.estimate - frame.reference
    return frame


@lru_cache(maxsize=1)
def shift_rows() -> pd.DataFrame:
    frame = _raw(
        "noise_shift_holdout_v1_repeats-50",
        [*CELL, "method", "repeat", "status", "estimate", *DIAGNOSTICS],
    ).rename(columns=RENAMED)
    frame["error"] = frame.estimate - frame.reference
    return frame


def cell_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return RMSE, completed repeats, and mean executions per cell and method."""

    complete = frame.loc[frame.status == "complete"]
    table = complete.groupby([*CELL, "method"]).agg(
        rmse=("error", _rmse),
        repeats=("error", "size"),
        overhead=("executions", "mean"),
        reference=("reference", "first"),
        condition_number=("condition_number", "mean"),
        fit_residual=("fit_residual", "mean"),
        gates=("gates", "mean"),
    )
    return table.reset_index()


def gain_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return paired percentile intervals for the RMSE gain over raw.

    The generators use a BCa bootstrap. A different interval here tests whether a label depends
    on that choice, so a label that flips is reported instead of being hidden by shared code.
    """

    complete = frame.loc[frame.status == "complete"]
    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]))
    resamples = int(CONFIG["bootstrap_resamples"])
    rows = []
    for keys, group in complete.groupby(CELL, sort=True):
        pivot = group.pivot_table(index="repeat", columns="method", values="error")
        for method in pivot.columns.drop("raw", errors="ignore"):
            if "raw" not in pivot:
                continue
            pairs = pivot[["raw", method]].dropna().to_numpy()
            if len(pairs) < 3:
                continue
            draws = pairs[rng.integers(0, len(pairs), (resamples, len(pairs)))]
            rmse = np.sqrt(np.mean(draws**2, axis=1))
            gain = rmse[:, 0] - rmse[:, 1]
            low, high = np.percentile(gain, [2.5, 97.5])
            rows.append((*keys, method, float(low), float(high)))
    columns = [*CELL, "method", "rmse_gain_ci_low", "rmse_gain_ci_high"]
    return pd.DataFrame(rows, columns=columns).set_index([*CELL, "method"])


def label_cells(table: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Apply the stated cost-aware oracle to recomputed RMSE values."""

    rows = []
    for keys, group in table.groupby(CELL, sort=True):
        row = dict(zip(CELL, keys, strict=True))
        indexed = group.set_index("method")
        row["reference"] = float(group.reference.iloc[0])
        if set(indexed.index) != set(METHODS) or (indexed.repeats != ANY_REPEATS).any():
            rows.append({**row, "label": "inconclusive", "basis": "incomplete", "equivalent": ()})
            continue
        utility = indexed.rmse + PENALTY * indexed.overhead / max(indexed.overhead.max(), 1.0)
        ordered = utility.sort_index().sort_values(kind="mergesort")
        winner, runner = ordered.index[0], ordered.index[1]
        gap = float(ordered.iloc[1] - ordered.iloc[0])
        equivalent = tuple(sorted(ordered.index[ordered <= ordered.iloc[0] + PRACTICAL]))
        basis = ""
        if gap < PRACTICAL:
            label = "practical_tie"
        elif winner == "raw":
            label = "raw"
            high = float(intervals.loc[(*keys, runner), "rmse_gain_ci_high"])
            basis = "accuracy" if high < 0.0 else "cost_only"
        else:
            low = float(intervals.loc[(*keys, winner), "rmse_gain_ci_low"])
            high = float(intervals.loc[(*keys, winner), "rmse_gain_ci_high"])
            gain = float(indexed.rmse["raw"] - indexed.rmse[winner])
            if low <= 0.0 <= high:
                label, basis = "inconclusive", "interval"
            elif gain < PRACTICAL:
                label = "practical_tie"
            else:
                label = str(winner)
        rows.append({**row, "label": label, "basis": basis, "equivalent": equivalent})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def benchmark_labels() -> pd.DataFrame:
    frame = benchmark_rows()
    return label_cells(cell_table(frame), gain_intervals(frame))


def claim_grid() -> dict[str, str]:
    frame = benchmark_rows()
    efficient = frame.loc[frame.family == "hardware_efficient"]
    return {
        "OperatingCells": _count(len(frame.drop_duplicates(CELL))),
        "SummaryRows": _count(len(frame.drop_duplicates([*CELL, "method"]))),
        "HardwareFailures": _count((efficient.status == "failed").sum()),
    }


def claim_negative_gains() -> dict[str, str]:
    table = cell_table(benchmark_rows())
    raw = table.loc[table.method == "raw"].set_index(CELL).rmse
    others = table.set_index(CELL)
    gain = raw.reindex(others.index).to_numpy() - others.rmse.to_numpy()
    return {"NegativeGains": _count((gain < 0.0).sum())}


def claim_regimes() -> dict[str, str]:
    labels = benchmark_labels()
    counts = labels.label.value_counts()
    wins = labels.loc[labels.label == "raw"]
    return {
        "VTwoPracticalTies": _count(counts["practical_tie"]),
        "VTwoRawWinners": _count(counts["raw"]),
        "VTwoReadoutWinners": _count(counts["readout"]),
        "VTwoZNEWinners": _count(counts["zne"]),
        "VTwoInconclusive": _count(counts["inconclusive"]),
        "RawWinsAccuracy": _count((wins.basis == "accuracy").sum()),
        "RawWinsCostOnly": _count((wins.basis == "cost_only").sum()),
        "RawWinsLowSignal": _count((wins.reference.abs() < LOW_SIGNAL).sum()),
        "incomplete_inconclusive": _word((labels.basis == "incomplete").sum()),
        "raw_win_family": str(sorted(wins.family.unique())),
    }


def _word(value: int) -> str:
    words = {8: "eight", 9: "Nine"}
    return words.get(int(value), str(int(value)))


def _derive_seed(*parts: object) -> int:
    """Repeat the published seed rule: SHA-256 of the colon-joined parts, first four bytes."""

    text = ":".join(str(part) for part in (ROOT_SEED, *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def holdout_keys(cells: pd.DataFrame) -> set[tuple[object, ...]]:
    """Return the lowest-ranked fifth of cells under the stated hash rule."""

    scored = cells.drop_duplicates(CELL).copy()
    scored["score"] = [
        _derive_seed(
            "regime-holdout",
            row.width,
            row.depth,
            row.shots,
            sum(ord(character) for character in f"{row.family}:{row.noise}"),
        )
        for row in scored.itertuples()
    ]
    count = max(1, math.ceil(0.20 * len(scored)))
    chosen = scored.nsmallest(count, "score")
    return {tuple(row) for row in chosen[CELL].itertuples(index=False)}


def frozen_rule(
    noise: str,
    shots: int,
    width: int,
    depth: int,
    gates: float,
    condition: float | None,
    residual: float | None,
) -> str:
    """Restate the frozen selector from its published rule order."""

    if condition is not None and condition > 8.0:
        return "abstain"
    if residual is not None and residual > 0.03:
        return "abstain"
    if noise == "zero":
        return "raw"
    if shots <= 1024 and width >= 6:
        return "raw"
    if noise in {"asymmetric_readout", "low", "medium"} and (condition is None or condition <= 3.0):
        return "readout"
    if noise in {"coherent_shift", "correlated_shift", "high"} and depth >= 3:
        if residual is None or residual <= 0.01:
            return "zne"
    if gates > 100 and shots < 4096:
        return "inconclusive"
    return "raw"


def _diagnostic(indexed: pd.DataFrame, method: str, column: str) -> float | None:
    if method not in indexed.index or pd.isna(indexed.loc[method, column]):
        return None
    return float(indexed.loc[method, column])


def _scored_policies() -> pd.DataFrame:
    """Rebuild selections, labels, and regret for every policy from raw rows."""

    main, shift = benchmark_rows(), shift_rows()
    tables = {"main": cell_table(main), "noise_shift": cell_table(shift)}
    labels = pd.concat(
        [
            benchmark_labels(),
            label_cells(tables["noise_shift"], gain_intervals(shift)),
        ],
        ignore_index=True,
    ).set_index(CELL)
    holdout = holdout_keys(main)
    combined = pd.concat(tables.values(), ignore_index=True).set_index([*CELL, "method"])
    best = combined.rmse.groupby(level=CELL).min()
    rows = []
    for name, table in tables.items():
        for keys, group in table.groupby(CELL, sort=True):
            indexed = group.set_index("method")

            noise, shots, width, depth = keys[4], int(keys[5]), int(keys[2]), int(keys[3])
            gates = int(indexed.gates["raw"]) if "raw" in indexed.index else 0
            condition = _diagnostic(indexed, "readout", "condition_number")
            residual = _diagnostic(indexed, "zne", "fit_residual")
            selected = frozen_rule(noise, shots, width, depth, gates, condition, residual)
            evaluation = (
                "noise_shift"
                if name == "noise_shift"
                else "holdout"
                if keys in holdout
                else "rule_design"
            )
            label = labels.loc[keys]
            fallback = frozen_rule(
                noise,
                shots,
                width,
                depth,
                gates,
                None if condition is None else min(condition, 8.0),
                None if residual is None else min(residual, 0.03),
            )
            policies = {"selector": selected, "always_raw": "raw"}
            policies.update(always_readout="readout", always_zne="zne")
            for policy, choice in policies.items():
                chosen = (*keys, choice)
                regret = (
                    float(combined.rmse[chosen] - best[keys])
                    if chosen in combined.index
                    else math.nan
                )
                if label.label == "practical_tie":
                    compatible = choice in label.equivalent
                elif label.label == "inconclusive":
                    compatible = choice in ("abstain", "inconclusive")
                else:
                    compatible = choice == label.label
                if choice == "abstain":
                    category = "abstain"
                elif choice == "inconclusive" or label.label == "inconclusive":
                    category = "inconclusive"
                elif compatible:
                    category = "correct"
                elif regret >= CATASTROPHIC or regret >= abs(float(label.reference)):
                    category = "catastrophic"
                elif regret <= NEGLIGIBLE:
                    category = "negligible"
                else:
                    category = "material"
                backup = (*keys, fallback)
                complete = backup in combined.index and (
                    combined.repeats[backup] >= ANY_REPEATS
                )
                backup_regret = (
                    float(combined.rmse[backup] - best[keys])
                    if backup in combined.index
                    else math.inf
                )
                rows.append(
                    {
                        **dict(zip(CELL, keys, strict=True)),
                        "policy": policy,
                        "evaluation_set": evaluation,
                        "selected": choice,
                        "regret": regret,
                        "compatible": compatible,
                        "strict": choice == label.label,
                        "category": category,
                        "reference": float(label.reference),
                        "unnecessary": complete and backup_regret <= NEGLIGIBLE,
                    }
                )
    return pd.DataFrame(rows)


def claim_selector_v1() -> dict[str, str]:
    scored = _scored_policies()
    selector = scored.loc[scored.policy == "selector"]
    holdout = selector.loc[selector.evaluation_set == "holdout"]
    shift = selector.loc[selector.evaluation_set == "noise_shift"]
    fixed = scored.groupby(["evaluation_set", "policy"]).regret.mean()
    catastrophic = selector.loc[selector.category == "catastrophic"]
    harmful = scored.category.isin(("catastrophic", "material"))
    harm = scored.loc[harmful].groupby("policy").size()
    strong = scored.loc[scored.reference.abs() >= LOW_SIGNAL]
    strong_harm = strong.loc[strong.category.isin(("catastrophic", "material"))]
    strong_counts = strong_harm.groupby("policy").size()
    abstained = selector.loc[selector.selected == "abstain"]
    return {
        "HoldoutCells": _count(len(holdout)),
        "HoldoutCompatibilityVTwo": _percent(holdout.compatible.mean()),
        "HoldoutStrictVTwo": _percent(holdout.strict.mean()),
        "ShiftCompatibilityVTwo": _percent(shift.compatible.mean()),
        "ShiftStrictVTwo": _percent(shift.strict.mean()),
        "HoldoutRegretMean": f"{holdout.regret.mean():.4f}",
        "HoldoutRegretMedian": f"{holdout.regret.median():.4f}",
        "HoldoutRegretMax": f"{holdout.regret.max():.4f}",
        "ShiftRegretMean": f"{shift.regret.mean():.4f}",
        "HoldoutAlwaysRaw": f"{fixed['holdout', 'always_raw']:.4f}",
        "HoldoutAlwaysReadout": f"{fixed['holdout', 'always_readout']:.4f}",
        "ShiftAlwaysRaw": f"{fixed['noise_shift', 'always_raw']:.4f}",
        "ShiftAlwaysReadout": f"{fixed['noise_shift', 'always_readout']:.4f}",
        "SelectorCatastrophic": _count(len(catastrophic)),
        "SelectorCatastrophicGHZ": _count((catastrophic.family == "ghz").sum()),
        "SelectorCatastrophicLowSignal": _count(
            (catastrophic.reference.abs() < LOW_SIGNAL).sum()
        ),
        "SelectorAbstentions": _count(len(abstained)),
        "SelectorUnnecessaryAbstentions": _count(
            abstained.unnecessary.sum()
        ),
        "SelectorHarm": _count(harm["selector"]),
        "AlwaysRawHarm": _count(harm["always_raw"]),
        "StrongSignalCells": _count((strong.policy == "selector").sum()),
        "StrongHarmSelector": _count(strong_counts["selector"]),
        "StrongHarmRaw": _count(strong_counts["always_raw"]),
        "StrongHarmReadout": _count(strong_counts["always_readout"]),
        "evaluated_cells": _count(len(selector)),
    }


def claim_zne_probability() -> dict[str, str]:
    factors = _raw(
        "zne_probability_diagnostics_v1_repeats-50",
        ["circuit_id", "noise", "total_shot_budget", "repeat"],
    )
    classified = pd.read_csv(PROCESSED / "zne_probability_diagnostics_v1.csv")
    counts = classified.validity_classification.value_counts()
    return {
        "ZNEProbabilityRepeats": _count(len(factors.drop_duplicates())),
        "ZNEUnstableRepeats": _count(counts["extrapolation_unstable"]),
        "ZNEDestroyedRepeats": _count(counts["signal_destroyed"]),
    }


def _gate_failures(
    genuine: pd.DataFrame,
    control_gain: np.ndarray,
    retention: np.ndarray,
    overshoot: np.ndarray,
) -> list[str]:
    """Apply the six stated gate checks with a separately written normal interval."""

    def low(values: np.ndarray) -> float:
        return float(values.mean() - Z_95 * values.std(ddof=1) / math.sqrt(len(values)))

    gains = genuine.true_gain.to_numpy(dtype=float)
    classes = genuine.validity_classification
    checks = {
        "signal_retention": low(retention) > 0.10,
        "physical_bounds": overshoot.max() <= 1e-12,
        "negative_mass": genuine.negative_mass.max() <= 0.05,
        "garbage_control": low(gains - control_gain) > 0.0,
        "gain_uncertainty": low(gains) > 0.0,
        "trend": (classes == "signal_destroyed").mean() == 0.0
        and (classes == "extrapolation_unstable").mean() <= 0.05,
    }
    return [name for name, passed in checks.items() if not passed]


def rebuilt_true_gain(frame: pd.DataFrame) -> pd.Series:
    """Rebuild true gain as baseline error minus extrapolated error."""

    baseline = frame.factor_estimates_json.map(lambda item: float(json.loads(item)[0]))
    return (baseline - frame.reference).abs() - (frame.estimate - frame.reference).abs()


def zne_gate_table() -> pd.DataFrame:
    frame = _raw(
        "zne_falsification_v1",
        [
            "circuit_id",
            "noise",
            "total_shot_budget",
            "repeat",
            "method",
            "estimate",
            "reference",
            "factor_estimates_json",
            "true_gain",
            "negative_mass",
            "signal_retention_json",
            "validity_classification",
        ],
    )
    frame = frame.assign(true_gain=rebuilt_true_gain(frame))
    rows = []
    keys = ["circuit_id", "noise", "total_shot_budget"]
    for values, group in frame.groupby(keys, sort=True):
        genuine = group.loc[group.method == "genuine_folding"].sort_values("repeat")
        control = group.loc[group.method == "garbage_folding"].sort_values("repeat")
        retention = np.array(
            [_last_retention(item) for item in genuine.signal_retention_json], dtype=float
        )
        overshoot = np.maximum(genuine.estimate.abs().to_numpy() - 1.0, 0.0)
        failures = _gate_failures(
            genuine, control.true_gain.to_numpy(dtype=float), retention, overshoot
        )
        rows.append({**dict(zip(keys, values, strict=True)), "failures": failures})
    return pd.DataFrame(rows)


def _last_retention(payload: str) -> float:
    values = json.loads(payload)
    if isinstance(values, dict):
        ordered = sorted(values.items(), key=lambda item: float(item[0]))
        return float(ordered[-1][1])
    return float(values[-1])


def claim_zne_gate() -> dict[str, str]:
    table = zne_gate_table()
    credible = table.failures.map(len) == 0
    efficient = table.circuit_id.str.startswith("hardware_efficient")
    control = table.loc[(table.circuit_id == "bell_xx") & (table.noise == "low")]
    status = {
        int(row.total_shot_budget): "credible" if not row.failures else "rejected"
        for row in control.itertuples()
    }
    return {
        "ZNEValidityCells": _count(len(table)),
        "ZNECredibleCells": _count(credible.sum()),
        "ZNERejectedCells": _count((~credible).sum()),
        "ZNEHardwareCells": _count(efficient.sum()),
        "ZNEHardwareCredibleCells": _count((efficient & credible).sum()),
        "control_16384": status[16384],
        "control_4096": status[4096],
    }


@lru_cache(maxsize=1)
def combined_rows() -> pd.DataFrame:
    return _raw(
        "combined_qem_v1",
        [
            "cell_id",
            "kind",
            "noise",
            "total_shot_budget",
            "allocation_policy",
            "method",
            "repeat",
            "status",
            "record_kind",
            "estimate_a",
            "reference_a",
            "estimated_gap",
            "exact_gap",
            "signed_error",
            "absolute_error",
            "decision_error",
            "true_gain",
            "negative_mass",
            "physical_overshoot",
            "final_signal_retention",
            "zne_validity_classification",
            "calibration_executions",
            "folded_executions",
        ],
    ).rename(columns={"zne_validity_classification": "validity_classification"})


def combined_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """Rebuild signed errors and decision errors from stored estimates and exact values."""

    decision = frame.kind == "decision"
    signed = (frame.estimated_gap - frame.exact_gap).where(
        decision, frame.estimate_a - frame.reference_a
    )
    wrong = np.sign(frame.estimated_gap) != np.sign(frame.exact_gap)
    return frame.assign(
        rebuilt_signed=signed,
        rebuilt_absolute=signed.abs(),
        rebuilt_decision=wrong.where(decision, False).astype(bool),
    )


def claim_combined() -> dict[str, str]:
    frame = combined_errors(combined_rows())
    primary = frame.loc[(frame.record_kind == "primary") & (frame.status == "complete")]
    keys = ["cell_id", "total_shot_budget", "allocation_policy"]
    expectation = primary.loc[primary.kind == "expectation"]
    rmse = expectation.groupby([*keys, "method"]).rebuilt_signed.apply(_rmse).unstack("method")
    near = primary.loc[(primary.cell_id == "decision_near_tie") & (primary.method == "combined")]
    near_rates = near.groupby(keys).rebuilt_decision.mean()
    pivot = primary.pivot_table(
        index=[*keys, "repeat"], columns="method", values="rebuilt_absolute"
    )
    interaction = pivot.combined - pivot.readout - pivot.zne + pivot.raw
    means = interaction.groupby(level=keys).mean()
    expectation_all = frame.loc[(frame.kind == "expectation") & (frame.status == "complete")]
    rejected = 0
    cells = 0
    for _, group in expectation_all.groupby(keys, sort=True):
        genuine = group.loc[group.method == "combined"].sort_values("repeat")
        control = group.loc[group.method == "combined_control"].sort_values("repeat")
        failures = _gate_failures(
            genuine,
            control.true_gain.to_numpy(dtype=float),
            genuine.final_signal_retention.to_numpy(dtype=float),
            genuine.physical_overshoot.to_numpy(dtype=float),
        )
        cells += 1
        rejected += bool(failures)
    width_two = primary.loc[
        (primary.cell_id == "readout_dominated")
        & (primary.total_shot_budget == 16384)
        & (primary.allocation_policy == "balanced")
    ]
    executions = (
        width_two.calibration_executions + width_two.folded_executions
    ).groupby(width_two.method).mean()
    return {
        "CombinedExpectationCells": _count(len(rmse)),
        "CombinedExpectationWins": _count((rmse.idxmin(axis=1) == "combined").sum()),
        "CombinedValidityCells": _count(cells),
        "CombinedValidityRejected": _count(rejected),
        "CombinedNegativeInteractions": _count((means < 0.0).sum()),
        "CombinedInteractionCells": _count(len(means)),
        "CombinedNearTieMinimum": f"{near_rates.min():.2f}",
        "CombinedNearTieMaximum": f"{near_rates.max():.2f}",
        "executions_combined": _count(executions["combined"]),
        "executions_raw": _count(executions["raw"]),
        "executions_zne": _count(executions["zne"]),
        "executions_readout": _count(executions["readout"]),
    }


def rebuilt_mismatch_decision(frame: pd.DataFrame) -> pd.Series:
    estimated = np.sign(frame.corrected_estimate - frame.decision_threshold)
    exact = np.sign(frame.reference - frame.decision_threshold)
    return pd.Series(estimated != exact, index=frame.index)


def claim_mismatch() -> dict[str, str]:
    frame = _raw(
        "readout_mismatch_v1",
        [
            "width",
            "distribution",
            "target_model",
            "total_shot_budget",
            "method",
            "status",
            "corrected_estimate",
            "reference",
            "decision_threshold",
            "decision_error",
        ],
    )
    frame = frame.assign(rebuilt_decision=rebuilt_mismatch_decision(frame))
    selected = frame.loc[
        (frame.width == 6)
        & (frame.distribution == "broad")
        & (frame.total_shot_budget == 16384)
        & (frame.target_model == "calibration_mismatch")
        & (frame.status == "complete")
    ]
    rates = selected.groupby("method").rebuilt_decision.mean()
    return {
        "MismatchRows": _count(len(frame)),
        "MismatchDecisionMinimum": f"{rates.min():.2f}",
        "MismatchDecisionMaximum": f"{rates.max():.2f}",
    }


def claim_readout_deviation() -> dict[str, str]:
    manifest = pd.read_csv(ROOT / "experiments/manifests/readout_scalability_v1_manifest.csv")
    correlated = manifest.loc[
        manifest.readout_model.isin(("pair_correlated", "calibration_mismatch"))
    ]
    planned = int(correlated.loc[correlated.width > 6, "planned_repeats"].sum())
    run = _raw("readout_mismatch_v1", ["width"])
    scalability = _raw("readout_scalability_v1", ["readout_model"])
    executed = int((run.width > 6).sum()) + int(
        scalability.readout_model.isin(("pair_correlated", "calibration_mismatch")).sum()
    )
    return {"planned_not_run": _count(planned - executed)}


@lru_cache(maxsize=1)
def qaoa_decisions() -> pd.DataFrame:
    """Reduce candidate rows to one decision per regime and repeat."""

    frame = _raw(
        "qaoa_selection_v1",
        [
            "graph_id",
            "method",
            "total_selection_budget",
            "repeat",
            "candidate_id",
            "candidate_role",
            "exact_ideal_energy",
            "corrected_or_extrapolated_estimate",
            "status",
            "certification_status",
        ],
    ).rename(columns={"corrected_or_extrapolated_estimate": "estimate"})
    rows = []
    keys = ["graph_id", "method", "total_selection_budget", "repeat"]
    for values, group in frame.groupby(keys, sort=True):
        incumbent = group.loc[group.candidate_role == "incumbent"]
        subsets = {"overall": group}
        challenger = group.loc[group.candidate_role == "near_tie_challenger"]
        if len(challenger):
            subsets["near_tie"] = pd.concat([incumbent, challenger])
        for regime, subset in subsets.items():
            resolved = (subset.status == "complete").all() and subset.estimate.notna().all()
            row = {
                **dict(zip(keys, values, strict=True)),
                "regime": regime,
                "resolved": bool(resolved),
                "certified": (group.certification_status.iloc[0] == "certified"),
            }
            if resolved:
                exact_best = subset.exact_ideal_energy.max()
                winners = subset.candidate_id[subset.exact_ideal_energy >= exact_best - 1e-12]
                choice = subset.loc[subset.estimate == subset.estimate.max()]
                selected = sorted(choice.candidate_id)[0]
                chosen_energy = float(
                    subset.loc[subset.candidate_id == selected, "exact_ideal_energy"].iloc[0]
                )
                row.update(
                    error=selected not in set(winners),
                    regret=max(0.0, float(exact_best) - chosen_energy),
                    energy_rmse=_rmse(subset.estimate - subset.exact_ideal_energy),
                )
            rows.append(row)
    return pd.DataFrame(rows)


def claim_qaoa() -> dict[str, str]:
    decisions = qaoa_decisions()
    overall = decisions.loc[decisions.regime == "overall"]
    done = overall.loc[overall.resolved].copy()
    near = decisions.loc[(decisions.regime == "near_tie") & decisions.resolved]
    near_rates = near.groupby(["total_selection_budget", "method"]).error.mean()
    def stratum_rmse(graph: str) -> tuple[float, float]:
        group = done.loc[
            (done.graph_id == graph) & (done.total_selection_budget == 4096)
        ].groupby("method").energy_rmse.apply(lambda item: math.sqrt(np.mean(item**2)))
        return float(group["raw"]), float(group["readout"])

    ring = stratum_rmse("ring4")
    sparse = stratum_rmse("sparse6")
    large = done.loc[done.total_selection_budget == 16384]
    regret = large.groupby("method").regret.mean()
    zne = large.loc[large.method == "zne"].groupby("graph_id").error.mean()
    failures = overall.loc[~overall.resolved]
    failed = failures.groupby("total_selection_budget").size()
    certified = done.loc[done.certified]
    correlation = np.corrcoef(done.energy_rmse.to_numpy(), done.error.to_numpy(dtype=float))[0, 1]
    return {
        "near_4096_raw": f"{near_rates[4096, 'raw']:.2f}",
        "near_4096_readout": f"{near_rates[4096, 'readout']:.2f}",
        "near_4096_zne": f"{near_rates[4096, 'zne']:.2f}",
        "near_16384_raw": f"{near_rates[16384, 'raw']:.2f}",
        "near_16384_readout": f"{near_rates[16384, 'readout']:.3f}",
        "near_16384_zne": f"{near_rates[16384, 'zne']:.2f}",
        "ring_raw_rmse": f"{ring[0]:.3f}",
        "ring_readout_rmse": f"{ring[1]:.3f}",
        "sparse_raw_rmse": f"{sparse[0]:.3f}",
        "sparse_readout_rmse": f"{sparse[1]:.3f}",
        "correlation": f"{correlation:.3f}",
        "regret_raw": f"{regret['raw']:.4f}",
        "regret_readout": f"{regret['readout']:.4f}",
        "regret_zne": f"{regret['zne']:.4f}",
        "failed_4096": _count(failed[4096]),
        "failed_16384": _count(failed[16384]),
        "failed_total": _count(len(failures)),
        "zne_ring": f"{zne['ring4']:.2f}",
        "zne_sparse": f"{zne['sparse6']:.2f}",
        "zne_dense": f"{zne['medium_dense8']:.2f}",
        "completed": _count(len(done)),
        "certified": _count(len(certified)),
        "unresolved": _count(len(done) - len(certified)),
        "candidate_rows": _thousands(len(_raw("qaoa_selection_v1", ["status"]))),
        "groups": _count(len(overall)),
    }


@lru_cache(maxsize=1)
def sequential_rows() -> pd.DataFrame:
    return _raw(
        "sequential_decision_v1",
        [
            "task",
            "margin",
            "noise",
            "max_total_shots",
            "method",
            "policy",
            "certified_choice",
            "wrong_certified",
            "total_shots_used",
            "status",
        ],
    )


def rebuilt_wrong_certified(frame: pd.DataFrame) -> pd.Series:
    """A certified choice is wrong if no winner exists or it is not candidate a."""

    certified = frame.certified_choice.fillna("").isin(("a", "b"))
    return certified & ((frame.certified_choice != "a") | (frame.margin == 0.0))


def claim_sequential() -> dict[str, str]:
    raw = sequential_rows()
    raw = raw.assign(wrong_certified=rebuilt_wrong_certified(raw))
    null = raw.loc[raw.margin == 0.0]
    rates = null.groupby(["method", "policy"]).wrong_certified.mean()
    wilson = rates.xs("wilson_stopping", level="policy")
    naive = rates.xs("naive_wald_peeking", level="policy")
    near = raw.loc[
        (raw.task == "symmetric_gap")
        & (raw.margin == 0.02)
        & (raw.noise != "zero")
        & (raw.max_total_shots == 16384)
        & (raw.policy == "wilson_stopping")
    ]
    share = near.groupby("method").total_shots_used.mean() / 16384.0
    unresolved = near.status.eq("unresolved").groupby(near.method).mean()
    biased = raw.loc[
        (raw.task == "gate_imbalanced")
        & (raw.noise == "high")
        & (raw.margin == 0.05)
        & (raw.max_total_shots == 16384)
        & (raw.policy == "wilson_stopping")
    ]
    wrong = biased.groupby("method").wrong_certified.mean()
    raw_shots = biased.loc[biased.method == "raw", "total_shots_used"].mean()
    return {
        "SequentialRows": _thousands(len(raw)),
        "NullWilsonMin": _percent(wilson.min()),
        "NullWilsonMax": _percent(wilson.max()),
        "NullNaiveMin": _percent(naive.min()),
        "NullNaiveMax": _percent(naive.max()),
        "NearTieShareMin": _percent(share.min()),
        "NearTieShareMax": _percent(share.max()),
        "NearTieUnresolvedMin": _percent(unresolved.min()),
        "NearTieUnresolvedMax": _percent(unresolved.max()),
        "BiasedRawWrong": _percent(wrong["raw"], 0),
        "BiasedRawShots": _thousands(raw_shots),
        "BiasedZNEWrong": _percent(wrong["zne"], 0),
        "null_bound": f"{math.ceil(100 * wilson.max()):d}%",
        "naive_range": f"{round(100 * naive.min()):d}--{round(100 * naive.max()):d}",
        "biased_rounded": _thousands(round(raw_shots, -3)),
    }


def rebuilt_lifetime_decision(frame: pd.DataFrame) -> pd.Series:
    corrected = frame.corrected_probability_one > frame.decision_threshold
    return corrected != (frame.true_probability_one > frame.decision_threshold)


def claim_lifetime() -> dict[str, str]:
    raw = _raw(
        "calibration_lifetime_v1",
        [
            "workload_length",
            "drift_model",
            "policy",
            "true_probability_one",
            "corrected_probability_one",
            "decision_threshold",
            "decision_error",
            "workflow_total_shots",
            "amortized_shots_per_decision",
        ],
    )
    raw = raw.assign(
        decision_error=rebuilt_lifetime_decision(raw),
        amortized_shots_per_decision=raw.workflow_total_shots / raw.workload_length,
    )
    workload = raw.loc[raw.workload_length == 100]
    never = workload.loc[workload.policy == "never"].groupby("drift_model").decision_error.mean()
    shots = workload.groupby("policy").amortized_shots_per_decision.mean()
    return {
        "LifetimeRows": _thousands(len(raw)),
        "NeverAbrupt": f"{never['abrupt_change']:.3f}",
        "NeverAsymmetric": f"{never['asymmetric']:.3f}",
        "NeverSlow": f"{never['slow_monotonic']:.3f}",
        "NeverRandom": f"{never['random_bounded']:.3f}",
        "TriggerShotSaving": _percent(
            1.0 - shots["uncertainty_triggered"] / shots["every_decision"], 0
        ),
    }


def _harmful(regret: float, reference: float) -> bool:
    return (not math.isnan(regret)) and (regret >= CATASTROPHIC or regret >= reference)


def claim_selector_v2() -> dict[str, str]:
    raw = _raw(
        "selector_holdout_v2",
        ["circuit_id", "noise", "shots", "method", "status", "decision", "estimate"],
    )
    references = (
        pd.read_csv(PROCESSED / "selector_v2_cells.csv")
        .groupby("circuit_id")
        .reference.first()
    )
    raw["error"] = raw.estimate - raw.circuit_id.map(references)
    rows = []
    decided_share = float(CONFIG["selector_decided_share"])
    for (circuit, _, _), group in raw.groupby(["circuit_id", "noise", "shots"]):
        rmse = {
            method: _rmse(
                group.loc[(group.method == method) & (group.status == "complete"), "error"]
            )
            for method in METHODS
        }
        best = np.nanmin(list(rmse.values()))
        policy = group.loc[group.method == "selector_v2"]
        chosen = policy.loc[(policy.decision == "readout") & (policy.status == "complete")]
        share = len(chosen) / len(policy)
        counted = share >= decided_share
        regret = _rmse(chosen.error) - best if len(chosen) else math.nan
        reference = abs(float(references[circuit]))
        rows.append(
            {
                "decided": counted,
                "selector": regret,
                "raw": rmse["raw"] - best,
                "readout": rmse["readout"] - best,
                "reliable": counted and regret <= NEGLIGIBLE,
                "avoided": (not counted) and rmse["raw"] - best > NEGLIGIBLE,
                "catastrophic_selector": counted and _harmful(regret, reference),
                "catastrophic_raw": _harmful(rmse["raw"] - best, reference),
                "catastrophic_readout": _harmful(rmse["readout"] - best, reference),
                "reference": reference,
            }
        )
    cells = pd.DataFrame(rows)
    decided = cells.loc[cells.decided]
    safe = cells.loc[~cells.decided]
    reliable, avoided = int(decided.reliable.sum()), int(safe.avoided.sum())
    shares = raw.loc[raw.method == "selector_v2", "decision"].value_counts(normalize=True)
    weak = cells.loc[cells.reference < 0.06]
    strong = cells.loc[cells.reference >= 0.5]
    return {
        "VTwoCells": _count(len(cells)),
        "VTwoDecidedCells": _count(len(decided)),
        "VTwoNoSafeCells": _count(len(safe)),
        "VTwoReliableCells": _count(reliable),
        "VTwoHOneRate": _percent(reliable / len(decided)),
        "VTwoHOneLow": _percent(_wilson_low(reliable, len(decided))),
        "VTwoAvoidedCells": _count(avoided),
        "VTwoHTwoRate": _percent(avoided / len(safe)),
        "VTwoHTwoLow": _percent(_wilson_low(avoided, len(safe))),
        "VTwoRegretSelector": f"{decided.selector.mean():.4f}",
        "VTwoRegretRaw": f"{decided.raw.mean():.4f}",
        "VTwoRegretReadout": f"{decided.readout.mean():.4f}",
        "VTwoCatastrophicSelector": _count(decided.catastrophic_selector.sum()),
        "VTwoCatastrophicRaw": _count(decided.catastrophic_raw.sum()),
        "VTwoCatastrophicReadout": _count(decided.catastrophic_readout.sum()),
        "VTwoReadoutDecisionShare": _percent(shares["readout"], 0),
        "VTwoRequestShare": _percent(shares["request_more_shots"], 0),
        "VTwoUnresolvedShare": _percent(shares["unresolved"], 0),
        "raw_rows": _thousands(len(raw)),
        "weak_reliable": _count((weak.raw <= NEGLIGIBLE).sum()),
        "weak_total": _count(len(weak)),
        "strong_reliable": _count((strong.readout <= NEGLIGIBLE).sum()),
        "strong_total": _count(len(strong)),
    }


def claim_envelope_diagnostics() -> dict[str, str]:
    cells = pd.read_csv(PROCESSED / "method_envelope_cells_v1.csv")
    design = cells.loc[(cells.evaluation_set == "rule_design") & (cells.method == "readout")]
    rates = design.groupby("condition_number").reliable.mean()
    counts = design.groupby("condition_number").size()
    gates = pd.read_csv(PROCESSED / "selector_gate_stability_v1.csv")
    unstable = gates.loc[(gates.diagnostic == "condition_number") & (gates.threshold == 8.0)]
    means = sorted(unstable.loc[unstable.unstable, "observed_mean"])
    zne = cells.loc[(cells.evaluation_set == "rule_design") & (cells.method == "zne")]
    bins = zne.groupby("fit_residual").reliable.agg(["mean", "size"])
    residual = bins.loc[bins["size"] >= ADEQUATE_BIN, "mean"]
    return {
        "rate_low": _percent(rates["[1, 1.5)"], 0),
        "rate_mid": _percent(rates["[1.5, 3)"], 0),
        "rate_high": _percent(rates["[3, 8)"], 0),
        "count_top": _count(counts[">= 8"]),
        "top_reliable": _count(design.loc[design.condition_number == ">= 8", "reliable"].sum()),
        "flip_means": f"{means[0]:.2f}, {means[1]:.2f}, and {means[2]:.2f}",
        "residual_rates": f"{100 * residual.min():.0f}--{100 * residual.max():.0f}",
    }


def claim_hardware_shots() -> dict[str, str]:
    manifest = pd.read_csv(ROOT / "experiments/hardware/hardware_manifest_v1.csv")
    config = json.loads(
        (ROOT / "experiments/configs/hardware_readiness_v1.yaml").read_text(encoding="utf-8")
    )
    frame = _raw(
        "combined_qem_v1",
        [
            "cell_id",
            "total_shot_budget",
            "allocation_policy",
            "method",
            "status",
            "record_kind",
            "signed_error",
            "exact_gap",
        ],
    )
    selected = frame.loc[
        (frame.cell_id == "decision_near_tie")
        & (frame.total_shot_budget == 16384)
        & (frame.allocation_policy == "balanced")
        & (frame.method == "raw")
        & (frame.record_kind == "primary")
        & (frame.status == "complete")
    ]
    reference = float(selected.exact_gap.iloc[0])
    gap = abs(reference + float(selected.signed_error.mean()))
    # Normal quantiles for a two-sided 5% test and 80% power.
    z_total = Z_95 + 0.8416212335729143
    if config["alpha"] != 0.05 or config["power"] != 0.8:
        raise ValueError("The stated power calculation uses alpha 0.05 and power 0.80")
    variance = 2.0 * (1.0 - (reference / 2.0) ** 2)
    candidate = int(z_total**2 * variance / gap**2) + 1
    return {
        "HardwareTotalShots": f"{manifest.required_total_shots.sum() / 1e6:.1f}",
        "HardwareNearTieShots": _thousands(2 * candidate),
    }


RECOMPUTE: dict[str, Callable[[], dict[str, str]]] = {
    "grid": claim_grid,
    "negative_gains": claim_negative_gains,
    "regimes": claim_regimes,
    "selector_v1": claim_selector_v1,
    "zne_probability": claim_zne_probability,
    "zne_gate": claim_zne_gate,
    "combined": claim_combined,
    "mismatch": claim_mismatch,
    "readout_deviation": claim_readout_deviation,
    "qaoa": claim_qaoa,
    "sequential": claim_sequential,
    "lifetime": claim_lifetime,
    "selector_v2": claim_selector_v2,
    "envelope": claim_envelope_diagnostics,
    "hardware_shots": claim_hardware_shots,
}


def printed_macros() -> dict[str, str]:
    """Read macro values as printed, with LaTeX escapes removed."""

    values: dict[str, str] = {}
    for name in MACRO_FILES:
        text = (GENERATED / name).read_text(encoding="utf-8")
        for key, value in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text):
            values[key] = value.replace("{,}", ",").replace("\\%", "%")
    return values


def paper_text() -> str:
    """Return the paper body with collapsed whitespace."""

    return " ".join(PAPER_PATH.read_text(encoding="utf-8").split())


def compare_claim(
    claim: dict[str, object],
    macros: dict[str, str],
    text: str,
    computed: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    """Return compared items and mismatches between recomputed and printed values."""

    values = computed[str(claim["recompute"])]
    compared: list[str] = []
    mismatches: list[str] = []
    for macro in claim.get("macros", []):
        recomputed = values.get(macro)
        printed = macros.get(macro)
        compared.append(f"{macro}={recomputed}")
        if recomputed != printed:
            mismatches.append(f"{macro}: printed {printed}, recomputed {recomputed}")
    for item in claim.get("literals", []):
        recomputed = values.get(item["name"])
        compared.append(f"{item['name']}={recomputed}")
        if recomputed != item["value"]:
            mismatches.append(
                f"{item['name']}: printed {item['value']}, recomputed {recomputed}"
            )
        if " ".join(item["context"].split()) not in text:
            mismatches.append(f"{item['name']}: context not found in paper")
        if item["value"] not in item["context"].replace("\\%", "%"):
            mismatches.append(f"{item['name']}: value not in its context")
    return compared, mismatches


def build_ledger() -> pd.DataFrame:
    """Return one row per claim with recomputation status and its source."""

    macros = printed_macros()
    text = paper_text()
    computed = {name: function() for name, function in RECOMPUTE.items()}
    rows = []
    for claim in CONFIG["claims"]:
        row = {
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_type": claim["claim_type"],
            "source_raw_files": ";".join(claim.get("source_raw_files", [])),
            "source_processed_inputs": ";".join(claim.get("source_processed_inputs", [])),
            "evidence_origin": claim.get("evidence_origin", "not_applicable"),
            "derived_fields": ";".join(claim.get("derived_fields", [])),
            "processed_input_trace": claim.get("processed_input_trace", ""),
            "processing_script": claim.get("processing_script", ""),
            "metric_definition": claim.get("metric_definition", ""),
            "figure_or_table": claim.get("figure_or_table", ""),
            "applicable_scope": claim["applicable_scope"],
            "known_counterexample": claim.get("known_counterexample", "none recorded"),
            "preregistered_before_result": claim["preregistered_before_result"],
            "commit_hash": claim.get("commit_hash", ""),
        }
        if claim["claim_type"] == "computed":
            compared, mismatches = compare_claim(claim, macros, text, computed)
            row["verification_status"] = "mismatch" if mismatches else "matches_printed_value"
            row["recomputed_values"] = "; ".join(compared)
            row["mismatches"] = "; ".join(mismatches)
        else:
            row["verification_status"] = f"not_computable_{claim['claim_type']}"
            row["recomputed_values"] = ""
            row["mismatches"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def used_macros(macros: dict[str, str]) -> set[str]:
    """Return macros the paper body actually references."""

    source = PAPER_PATH.read_text(encoding="utf-8")
    return {name for name in macros if re.search(rf"\\{name}(?![A-Za-z])", source)}


def provenance_table(ledger: pd.DataFrame) -> pd.DataFrame:
    """Trace every macro, generated table, and figure to its script and claim."""

    macros = printed_macros()
    used = used_macros(macros)
    owner: dict[str, str] = {}
    for claim in CONFIG["claims"]:
        for macro in claim.get("macros", []):
            owner[macro] = claim["claim_id"]
    status = ledger.set_index("claim_id").verification_status
    origin = ledger.set_index("claim_id").evidence_origin
    rows = []
    for file_name, script in MACRO_FILES.items():
        text = (GENERATED / file_name).read_text(encoding="utf-8")
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", text):
            claim_id = owner.get(name, "")
            computed = bool(claim_id) and origin[claim_id] != "not_applicable"
            rows.append(
                _provenance_row(
                    "macro",
                    name,
                    name in used,
                    script,
                    f"paper/generated/{file_name}",
                    claim_id=claim_id,
                    printed=macros[name],
                    status=status[claim_id] if claim_id else "not_in_ledger",
                    origin=origin[claim_id] if computed else SOURCE_ONLY,
                    recomputed=computed,
                )
            )
    source = PAPER_PATH.read_text(encoding="utf-8")
    for file_name, (script, data) in TABLE_SOURCES.items():
        rows.append(
            _provenance_row(
                "generated_table",
                file_name,
                f"generated/{file_name}" in source,
                script,
                f"paper/generated/{file_name}",
                data=data,
            )
        )
    scripts = {
        path.relative_to(ROOT).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src").rglob("*.py"))
        if path.name not in {"verify_generated_artifacts.py", "claim_ledger.py"}
    }
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source)
    for figure in sorted(set(figures)):
        stem = Path(figure).stem
        direct = [
            path
            for path, body in scripts.items()
            if stem in body and not path.endswith("generate_paper_assets.py")
        ]
        makers = direct or [path for path, body in scripts.items() if stem in body]
        rows.append(
            _provenance_row(
                "figure",
                figure,
                True,
                ";".join(makers),
                f"paper/figures/{figure}",
                regenerated=figure in REGENERATED_FIGURES,
                inspected=figure in REGENERATED_FIGURES,
            )
        )
    return pd.DataFrame(rows)


def _provenance_row(
    kind: str,
    item: str,
    used: bool,
    script: str,
    generated: str,
    *,
    claim_id: str = "",
    printed: str = "",
    data: str = "",
    status: str = "traced_not_recomputed",
    origin: str = SOURCE_ONLY,
    recomputed: bool = False,
    regenerated: bool = False,
    inspected: bool = False,
) -> dict[str, object]:
    visual = kind == "figure"
    return {
        "item_kind": kind,
        "item": item,
        "printed_value": printed,
        "used_in_paper": used,
        "generating_script": script,
        "generated_file": generated,
        "source_data": data,
        "claim_id": claim_id,
        "verification_status": status,
        "evidence_origin": origin,
        "source_located": bool(script),
        "values_recomputed": recomputed,
        "visual_artifact_regenerated": regenerated if visual else "not_applicable",
        "visual_artifact_inspected": inspected if visual else "not_applicable",
    }


def derived_field_table() -> pd.DataFrame:
    """Compare every rebuilt derived field with its stored column, and list the others."""

    combined = combined_errors(combined_rows())
    primary = combined.loc[(combined.record_kind == "primary") & (combined.status == "complete")]
    decision = primary.loc[primary.kind == "decision"]
    zne = _raw(
        "zne_falsification_v1",
        ["estimate", "reference", "factor_estimates_json", "true_gain"],
    )
    mismatch = _raw(
        "readout_mismatch_v1",
        ["status", "corrected_estimate", "reference", "decision_threshold", "decision_error"],
    )
    mismatch = mismatch.loc[mismatch.status == "complete"]
    lifetime = _raw(
        "calibration_lifetime_v1",
        [
            "workload_length",
            "true_probability_one",
            "corrected_probability_one",
            "decision_threshold",
            "decision_error",
            "workflow_total_shots",
            "amortized_shots_per_decision",
        ],
    )
    sequential = sequential_rows()
    checks = [
        ("combined_qem_v1", "signed_error", primary.rebuilt_signed, primary.signed_error),
        ("combined_qem_v1", "absolute_error", primary.rebuilt_absolute, primary.absolute_error),
        ("combined_qem_v1", "decision_error", decision.rebuilt_decision, decision.decision_error),
        ("zne_falsification_v1", "true_gain", rebuilt_true_gain(zne), zne.true_gain),
        (
            "readout_mismatch_v1",
            "decision_error",
            rebuilt_mismatch_decision(mismatch),
            mismatch.decision_error,
        ),
        (
            "calibration_lifetime_v1",
            "decision_error",
            rebuilt_lifetime_decision(lifetime),
            lifetime.decision_error,
        ),
        (
            "calibration_lifetime_v1",
            "amortized_shots_per_decision",
            lifetime.workflow_total_shots / lifetime.workload_length,
            lifetime.amortized_shots_per_decision,
        ),
        (
            "sequential_decision_v1",
            "wrong_certified",
            rebuilt_wrong_certified(sequential),
            sequential.wrong_certified,
        ),
    ]
    rows = []
    for file_name, field, rebuilt, stored in checks:
        left = pd.to_numeric(rebuilt.astype(float))
        right = pd.to_numeric(stored.astype(float))
        different = ~(np.isclose(left, right, rtol=1e-9, atol=1e-12) | (left.isna() & right.isna()))
        rows.append(
            {
                "raw_file": file_name,
                "field": field,
                "status": "reconstructed",
                "rows_compared": len(left),
                "rows_different": int(different.sum()),
            }
        )
    consumers: dict[str, list[str]] = {}
    for claim in CONFIG["claims"]:
        for item in claim.get("derived_fields", []):
            if item.endswith(":not_reconstructed"):
                consumers.setdefault(item.split(":")[0], []).append(claim["claim_id"])
    for field, claim_ids in sorted(consumers.items()):
        rows.append(
            {
                "raw_file": "used by " + ";".join(claim_ids),
                "field": field,
                "status": "not_reconstructed",
                "rows_compared": 0,
                "rows_different": 0,
            }
        )
    return pd.DataFrame(rows)


def label_sensitivity() -> pd.DataFrame:
    """Compare authoritative BCa labels with the percentile-bootstrap sensitivity labels."""

    authority = pd.read_csv(PROCESSED / "regime_labels_v2.csv")
    columns = [*CELL, "oracle_label", "raw_win_basis"]
    merged = authority[columns].merge(benchmark_labels()[[*CELL, "label", "basis"]], on=CELL)
    merged = merged.rename(columns={"oracle_label": "bca_label", "label": "percentile_label"})
    merged["bca_raw_win_basis"] = merged.raw_win_basis.fillna("")
    merged["percentile_raw_win_basis"] = merged.basis.where(merged.percentile_label == "raw", "")
    merged["label_changed"] = merged.bca_label != merged.percentile_label
    merged["analysis_role"] = "sensitivity_only_bca_is_authoritative"
    return merged[
        [
            *CELL,
            "bca_label",
            "percentile_label",
            "bca_raw_win_basis",
            "percentile_raw_win_basis",
            "label_changed",
            "analysis_role",
        ]
    ]


def coverage_summary(ledger: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    """State what has and has not been checked, with counts taken from the outputs."""

    computed = ledger.loc[ledger.claim_type == "computed"]
    printed = sum(
        len(claim.get("macros", [])) + len(claim.get("literals", []))
        for claim in CONFIG["claims"]
        if claim["claim_type"] == "computed"
    )
    tables = provenance.loc[provenance.item_kind == "generated_table"]
    figures = provenance.loc[provenance.item_kind == "figure"]
    rows = [
        ("computed_claims_checked", len(computed)),
        ("printed_numbers_checked", printed),
        ("printed_number_mismatches", int((computed.verification_status == "mismatch").sum())),
        ("sampled_claims_final_raw_recomputed", int(CONFIG["audit_size"])),
        ("sampled_claims_initial_raw_recomputed", 3),
        ("generated_tables", len(tables)),
        ("generated_tables_values_recomputed", int(tables.values_recomputed.sum())),
        ("figures", len(figures)),
        ("figures_values_recomputed", int(figures.values_recomputed.sum())),
        ("figures_regenerated_in_this_audit", int(figures.visual_artifact_regenerated.sum())),
        ("figures_inspected_in_this_audit", int(figures.visual_artifact_inspected.sum())),
        ("paper_independently_verified_as_a_whole", "no"),
    ]
    for name in CONFIG["evidence_origins"]:
        rows.append((f"computed_claims_{name}", int((computed.evidence_origin == name).sum())))
    return pd.DataFrame(rows, columns=["metric", "value"])


def audit_sample(ledger: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Draw computed claims with a declared seed and report how each was verified."""

    computed = sorted(ledger.loc[ledger.claim_type == "computed", "claim_id"])
    chosen = random.Random(seed).sample(computed, size)
    columns = [
        "claim_id",
        "initial_audit_result",
        "final_audit_result",
        "seed",
        "evidence_origin",
        "verification_status",
        "source_raw_files",
        "source_processed_inputs",
        "derived_fields",
        "recomputed_values",
        "mismatches",
        "audit_note",
    ]
    sample = ledger.set_index("claim_id").loc[chosen].reset_index()
    sample["initial_audit_result"] = sample.claim_id.map(CONFIG["audit_initial_results"])
    sample["final_audit_result"] = sample.evidence_origin.map(
        lambda origin: "raw" if origin == "raw_recomputed" else "not_raw"
    )
    sample["seed"] = seed
    sample["audit_note"] = CONFIG["audit_note"]
    return sample[columns]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-size", type=int, default=int(CONFIG["audit_size"]))
    parser.add_argument("--audit-seed", type=int, default=int(CONFIG["audit_seed"]))
    arguments = parser.parse_args(argv)
    ledger = build_ledger()
    ledger.to_csv(LEDGER_PATH, index=False, lineterminator="\n")
    provenance = provenance_table(ledger)
    provenance.to_csv(PROVENANCE_PATH, index=False, lineterminator="\n")
    coverage_summary(ledger, provenance).to_csv(COVERAGE_PATH, index=False, lineterminator="\n")
    derived_field_table().to_csv(DERIVED_PATH, index=False, lineterminator="\n")
    label_sensitivity().to_csv(SENSITIVITY_PATH, index=False, lineterminator="\n")
    audit_sample(ledger, arguments.audit_size, arguments.audit_seed).to_csv(
        AUDIT_PATH, index=False, lineterminator="\n"
    )
    failed = ledger.loc[ledger.verification_status == "mismatch"]
    if len(failed):
        raise SystemExit(f"{len(failed)} claims disagree with the paper:\n{failed.to_string()}")


if __name__ == "__main__":
    main()
