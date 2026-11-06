"""Stratify committed v1 results by signal strength. This is post-hoc and changes no v1 file."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
CELLS_PATH = PROCESSED / "signal_strata_v1_cells.csv"
SUMMARY_PATH = PROCESSED / "signal_strata_v1_summary.csv"
CIRCUITS_PATH = PROCESSED / "signal_strata_v1_circuits.csv"
MARGIN_PATH = PROCESSED / "signal_strata_v1_margins.csv"
ATTENUATION_PATH = PROCESSED / "signal_strata_v1_attenuation.csv"
ANALYSIS_ROLE = "post_hoc_descriptive"
BAND_EDGES = (0.06, 0.2, 0.5, 0.9)
BANDS = ("near_zero", "weak", "medium", "strong", "boundary_adjacent")
ATTENUATION_EDGES = (0.01, 0.05, 0.2)
ATTENUATIONS = ("below_0.01", "0.01_to_0.05", "0.05_to_0.2", "at_least_0.2")
CELL = ["family", "circuit_id", "width", "depth", "noise", "shots"]


def signal_band(reference: float) -> str:
    """Name the band of an ideal expectation value, using its absolute size."""

    return BANDS[int(np.searchsorted(BAND_EDGES, abs(reference), side="right"))]


def normalized_error(error: float, reference: float) -> float:
    """Divide by the reference, except near zero where a relative error is meaningless."""

    return math.nan if signal_band(reference) == "near_zero" else abs(error) / abs(reference)


def _references() -> dict[str, float]:
    frame = pd.read_csv(
        ROOT / "results/raw/hardware_efficient_v1_repeats-50.csv",
        usecols=["circuit_id", "noiseless_statevector_reference"],
    )
    values = frame.groupby("circuit_id").noiseless_statevector_reference.first().to_dict()
    values.update({f"ghz_w{width}": 1.0 for width in (2, 4, 6)})
    return values


def build_cells() -> pd.DataFrame:
    """Join summaries with references, and test whether raw error is shrinkage toward zero."""

    summary = pd.read_csv(PROCESSED / "benchmark_summary_v1.csv")
    labels = pd.read_csv(PROCESSED / "regime_labels_v2.csv")[[*CELL, "oracle_label"]]
    references = _references()
    frame = summary.merge(labels, on=CELL, validate="many_to_one")
    frame["reference"] = frame.circuit_id.map(references)
    frame["signal_band"] = frame.reference.map(signal_band)
    frame["normalized_rmse"] = [
        normalized_error(rmse, reference)
        for rmse, reference in zip(frame.rmse, frame.reference, strict=True)
    ]
    frame["attenuation"] = frame.bias.abs()
    frame["attenuation_band"] = frame.attenuation.map(
        lambda value: ATTENUATIONS[int(np.searchsorted(ATTENUATION_EDGES, value, side="right"))]
    )
    # Bias toward zero has the opposite sign to the reference.
    frame["shrinks_toward_zero"] = np.sign(frame.bias) == -np.sign(frame.reference)
    columns = [
        *CELL,
        "method",
        "complete_repeats",
        "reference",
        "signal_band",
        "bias",
        "rmse",
        "normalized_rmse",
        "attenuation",
        "attenuation_band",
        "shrinks_toward_zero",
        "oracle_label",
    ]
    return frame[columns].assign(analysis_role=ANALYSIS_ROLE)


def summarize_bands(cells: pd.DataFrame) -> pd.DataFrame:
    """Report absolute and normalized error, and oracle labels, by band and method.

    Noise-free cells are reported apart, because raw cannot lose them on bias.
    """

    rows = []
    for scope, subset in (
        ("noisy", cells.loc[cells.noise != "zero"]),
        ("noise_free", cells.loc[cells.noise == "zero"]),
    ):
        rows.extend(_band_rows(scope, subset))
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def _band_rows(scope: str, cells: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    unique = cells.drop_duplicates(CELL)
    for band in BANDS:
        group = cells.loc[cells.signal_band == band]
        labeled = unique.loc[unique.signal_band == band]
        for method in ("raw", "readout", "zne"):
            part = group.loc[group.method == method]
            rows.append(
                {
                    "noise_scope": scope,
                    "signal_band": band,
                    "method": method,
                    "cells": len(labeled),
                    "independent_circuits": int(labeled.circuit_id.nunique()),
                    "families": ";".join(sorted(labeled.family.unique())),
                    "summaries_available": len(part),
                    "median_rmse": float(part.rmse.median()) if len(part) else math.nan,
                    "median_normalized_rmse": float(part.normalized_rmse.median())
                    if len(part)
                    else math.nan,
                    "shrinkage_share": float(part.shrinks_toward_zero.mean())
                    if len(part)
                    else math.nan,
                    "oracle_wins": int((labeled.oracle_label == method).sum()),
                    "oracle_ties": int((labeled.oracle_label == "practical_tie").sum()),
                    "oracle_inconclusive": int((labeled.oracle_label == "inconclusive").sum()),
                }
            )
    return rows


def circuit_variation(cells: pd.DataFrame) -> pd.DataFrame:
    """Show how much results vary between circuits inside a band."""

    rows = []
    noisy = cells.loc[cells.noise != "zero"]
    for (band, circuit, method), group in noisy.groupby(
        ["signal_band", "circuit_id", "method"], sort=True
    ):
        labels = noisy.loc[noisy.circuit_id == circuit].drop_duplicates(CELL)
        rows.append(
            {
                "signal_band": band,
                "circuit_id": circuit,
                "reference": float(group.reference.iloc[0]),
                "method": method,
                "median_rmse": float(group.rmse.median()),
                "min_rmse": float(group.rmse.min()),
                "max_rmse": float(group.rmse.max()),
                "raw_win_share": float((labels.oracle_label == "raw").mean()),
            }
        )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def attenuation_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Report how each method's error depends on the raw estimator's attenuation."""

    noisy = cells.loc[cells.noise != "zero"]
    raw = noisy.loc[noisy.method == "raw", [*CELL, "attenuation_band"]].rename(
        columns={"attenuation_band": "raw_attenuation_band"}
    )
    joined = noisy.merge(raw, on=CELL, validate="many_to_one")
    rows = []
    for (band, method), group in joined.groupby(["raw_attenuation_band", "method"], sort=True):
        rows.append(
            {
                "raw_attenuation_band": band,
                "method": method,
                "summaries": len(group),
                "median_rmse": float(group.rmse.median()),
                "oracle_wins": int((group.drop_duplicates(CELL).oracle_label == method).sum()),
            }
        )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def margin_summary() -> pd.DataFrame:
    """Report decision error by ideal margin from the separate decision benchmark."""

    frame = pd.read_csv(PROCESSED / "decision_margin_summary_v1.csv")
    frame = frame.loc[frame.status == "complete"]
    grouped = frame.groupby(["margin", "method"], sort=True).agg(
        conditions=("decision_error_probability", "size"),
        median_decision_error=("decision_error_probability", "median"),
        max_decision_error=("decision_error_probability", "max"),
    )
    return grouped.reset_index().assign(analysis_role=ANALYSIS_ROLE)


def main() -> None:
    cells = build_cells()
    for frame, path in (
        (cells, CELLS_PATH),
        (summarize_bands(cells), SUMMARY_PATH),
        (circuit_variation(cells), CIRCUITS_PATH),
        (attenuation_summary(cells), ATTENUATION_PATH),
        (margin_summary(), MARGIN_PATH),
    ):
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


if __name__ == "__main__":
    main()
