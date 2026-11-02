"""Manifest construction and auditable GHZ benchmark execution."""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import pandas as pd

from src.circuits import ghz_circuit
from src.config import DEPTHS, NOISE_LEVELS, SHOT_BUDGETS, WIDTHS, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.zne import run_linear_zne
from src.noise_models import noise_spec

METHODS = ("raw", "readout", "zne")
MANIFEST_COLUMNS = ("width", "depth", "noise", "shots", "method", "repeat", "status", "seed")


def build_manifest(repeats: int = 50) -> pd.DataFrame:
    """Create the complete declared grid, retaining inapplicable GHZ depths."""

    rows = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for noise_index, noise in enumerate(NOISE_LEVELS):
                for shots in SHOT_BUDGETS:
                    for method_index, method in enumerate(METHODS):
                        for repeat in range(repeats):
                            applicable = depth == 1
                            rows.append(
                                {
                                    "width": width,
                                    "depth": depth,
                                    "noise": noise,
                                    "shots": shots,
                                    "method": method,
                                    "repeat": repeat,
                                    "status": "pending" if applicable else "not_applicable",
                                    "seed": derive_seed(
                                        "volumetric-v1",
                                        width,
                                        depth,
                                        noise_index,
                                        shots,
                                        method_index,
                                        repeat,
                                    ),
                                }
                            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def execute_row(row: pd.Series) -> dict[str, object]:
    """Execute one applicable manifest row and preserve method-level raw data."""

    output = row.to_dict()
    output.update(
        {
            "family": "ghz",
            "observable_q0_first": "X" * int(row.width),
            "analytic_reference": 1.0,
            "result_label": "noisy_simulated",
            "estimate": None,
            "raw_payload_json": None,
            "runtime_seconds": 0.0,
            "error": None,
        }
    )
    if row.status == "not_applicable":
        return output
    started = time.perf_counter()
    try:
        circuit = ghz_circuit(int(row.width))
        spec = noise_spec("mixed", str(row.noise))
        if row.method == "raw":
            result = estimate_pauli(
                circuit,
                "X" * int(row.width),
                shots=int(row.shots),
                seed=int(row.seed),
                noise_spec=spec,
            )
            output["estimate"] = result.estimate
            output["raw_payload_json"] = result.raw_counts_json
        elif row.method == "readout":
            result = run_readout_mitigation(
                circuit,
                "X" * int(row.width),
                total_shot_budget=int(row.shots),
                noise_spec=spec,
                seed=int(row.seed),
            )
            output["estimate"] = result.corrected_estimate
            output["raw_payload_json"] = json.dumps(asdict(result), separators=(",", ":"))
            if result.status != "ok":
                raise RuntimeError(f"readout mitigation status: {result.status}")
        elif row.method == "zne":
            result = run_linear_zne(
                circuit,
                "X" * int(row.width),
                total_shot_budget=int(row.shots),
                noise_spec=spec,
                seed=int(row.seed),
            )
            output["estimate"] = result.extrapolated_estimate
            output["raw_payload_json"] = json.dumps(asdict(result), separators=(",", ":"))
        else:
            raise ValueError(f"Unknown method: {row.method}")
        output["status"] = "complete"
    except Exception as error:  # row-level audit must retain failures
        output["status"] = "failed"
        output["error"] = f"{type(error).__name__}: {error}"
    output["runtime_seconds"] = time.perf_counter() - started
    return output


def run_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute every row without silently dropping failures or N/A cells."""

    return pd.DataFrame(execute_row(row) for _, row in manifest.iterrows())


def coverage_audit(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize coverage by status and verify repeat-level accounting."""

    return (
        results.groupby(["width", "depth", "noise", "shots", "method", "status"], dropna=False)
        .size()
        .rename("row_count")
        .reset_index()
    )
