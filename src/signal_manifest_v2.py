"""Freeze the v2 circuit set using noiseless values only, before any noisy benchmark exists."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import (
    derive_seed,
    hardware_efficient_gates,
    ideal_expectation,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/signal_v2.yaml"
MANIFEST_PATH = ROOT / "experiments/manifests/signal_v2_manifest.csv"
HASH_PATH = ROOT / "experiments/manifests/signal_v2_manifest.sha256"
COVERAGE_PATH = ROOT / "results/processed/signal_v2_coverage.csv"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SPLITS = ("train", "validation", "holdout")


def observables(width: int) -> tuple[str, ...]:
    """List observables in a fixed order: full-support alternating, then two, then one qubit."""

    alternating = "".join("X" if qubit % 2 == 0 else "Z" for qubit in range(width))
    two = "ZZ" + "I" * (width - 2)
    one = "Z" + "I" * (width - 1)
    return tuple(dict.fromkeys((alternating, two, one)))


def band_of(value: float) -> str:
    edges, bands = CONFIG["band_edges"], CONFIG["bands"]
    return bands[int(np.searchsorted(edges, abs(value), side="right"))]


def _select_cell(width: int, depth: int) -> list[dict[str, object]]:
    """Scan seeds in order, using each seed at most once so circuits stay independent."""

    wanted = int(CONFIG["circuits_per_cell"])
    counts = {band: 0 for band in CONFIG["bands"]}
    rows = []
    for index in range(int(CONFIG["seed_search_limit"])):
        seed = derive_seed("signal-v2", width, depth, index)
        gates = hardware_efficient_gates(width, depth, seed)
        chosen = None
        for pauli in observables(width):
            value = ideal_expectation(gates, pauli)
            band = band_of(value)
            if counts[band] < wanted:
                chosen = (pauli, value, band)
                break
        if chosen is None:
            if all(count >= wanted for count in counts.values()):
                break
            continue
        pauli, value, band = chosen
        counts[band] += 1
        rows.append(
            {
                "circuit_id": f"sv2_w{width}_d{depth}_s{index}",
                "width": width,
                "depth": depth,
                "seed_index": index,
                "parameter_seed": seed,
                "observable_q0_first": pauli,
                "ideal_expectation": round(value, 12),
                "signal_band": band,
            }
        )
    return rows


def _assign_splits(frame: pd.DataFrame) -> pd.Series:
    """Assign splits by hash rank within each band, never using a result."""

    fractions = CONFIG["split_fractions"]
    output = pd.Series(index=frame.index, dtype=object)
    for _, group in frame.groupby("signal_band", sort=False):
        ranks = group.circuit_id.map(lambda name: derive_seed("signal-v2-split", name))
        ordered = ranks.sort_values(kind="mergesort").index
        train = int(round(len(ordered) * fractions["train"]))
        validation = int(round(len(ordered) * fractions["validation"]))
        output.loc[ordered[:train]] = "train"
        output.loc[ordered[train : train + validation]] = "validation"
        output.loc[ordered[train + validation :]] = "holdout"
    return output


def build_manifest() -> pd.DataFrame:
    rows = [
        row
        for width in CONFIG["widths"]
        for depth in CONFIG["depths"]
        for row in _select_cell(width, depth)
    ]
    frame = pd.DataFrame(rows)
    frame["split"] = _assign_splits(frame)
    return frame


def coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Report circuits by band, split, width, and depth, including shortfalls."""

    rows = []
    for band in CONFIG["bands"]:
        group = frame.loc[frame.signal_band == band]
        row = {
            "signal_band": band,
            "circuits": len(group),
            "meets_minimum": len(group) >= int(CONFIG["minimum_circuits_per_band"]),
        }
        row.update({split: int((group.split == split).sum()) for split in SPLITS})
        for width in CONFIG["widths"]:
            row[f"width_{width}"] = int((group.width == width).sum())
        for depth in CONFIG["depths"]:
            row[f"depth_{depth}"] = int((group.depth == depth).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frame = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(MANIFEST_PATH, index=False, lineterminator="\n", float_format="%.15g")
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    HASH_PATH.write_text(digest + "\n", encoding="utf-8", newline="\n")
    coverage(frame).to_csv(COVERAGE_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
