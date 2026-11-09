"""Freeze the v3 decision tasks across four circuit families, reading noiseless values only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import Gate, derive_seed, hardware_efficient_gates, ideal_expectation

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/selector_v3.yaml"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
MANIFEST_PATH = ROOT / "experiments/manifests/selector_v3_manifest.csv"
HASH_PATH = ROOT / "experiments/manifests/selector_v3_manifest.sha256"
COVERAGE_PATH = ROOT / "results/processed/selector_v3_coverage.csv"


def _rzz(left: int, right: int, angle: float) -> list[Gate]:
    return [Gate("cx", (left, right)), Gate("rz", (right,), angle), Gate("cx", (left, right))]


def ghz_rotated_gates(width: int, depth: int, seed: int) -> list[Gate]:
    """Prepare GHZ, idle through cancelling pairs, then rotate slightly by seeded angles."""

    rng = np.random.default_rng(seed)
    gates = [Gate("h", (0,))]
    chain = [Gate("cx", (qubit, qubit + 1)) for qubit in range(width - 1)]
    gates.extend(chain)
    for _ in range(depth - 1):
        gates.extend(reversed(chain))
        gates.extend(chain)
    gates.extend(Gate("rz", (qubit,), float(rng.uniform(-0.5, 0.5))) for qubit in range(width))
    return gates


def tfim_trotter_gates(width: int, depth: int, seed: int) -> list[Gate]:
    """Evolve a chain under a transverse-field Ising model with seeded couplings."""

    rng = np.random.default_rng(seed)
    coupling, field = rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)
    step = float(rng.uniform(0.2, 0.8))
    gates: list[Gate] = []
    for _ in range(depth):
        for qubit in range(width - 1):
            gates.extend(_rzz(qubit, qubit + 1, float(2.0 * coupling * step)))
        gates.extend(Gate("rx", (qubit,), float(2.0 * field * step)) for qubit in range(width))
    return gates


def qaoa_ring_gates(width: int, depth: int, seed: int) -> list[Gate]:
    """Build ring MaxCut QAOA with seeded angles. A width of two has one edge."""

    rng = np.random.default_rng(seed)
    edges = [(0, 1)] if width == 2 else [(q, (q + 1) % width) for q in range(width)]
    gates = [Gate("h", (qubit,)) for qubit in range(width)]
    for _ in range(depth):
        gamma, beta = rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi / 2.0, np.pi / 2.0)
        for left, right in edges:
            gates.extend(_rzz(left, right, float(gamma)))
        gates.extend(Gate("rx", (qubit,), float(2.0 * beta)) for qubit in range(width))
    return gates


BUILDERS = {
    "hardware_efficient": hardware_efficient_gates,
    "ghz_rotated": ghz_rotated_gates,
    "tfim_trotter": tfim_trotter_gates,
    "qaoa_ring": qaoa_ring_gates,
}


def observable(family: str, width: int) -> str:
    """Return one fixed observable per family, so candidates in a task are comparable."""

    if family == "ghz_rotated":
        return "X" * width
    if family == "hardware_efficient":
        return "".join("X" if qubit % 2 == 0 else "Z" for qubit in range(width))
    return "ZZ" + "I" * (width - 2)


def _select_task(family: str, width: int, depth: int, gap_class: str, count: int) -> list[dict]:
    """Slide a window of consecutive seeds until its top gap falls in the wanted class.

    Each class scans its own seeds, so no circuit can appear in two tasks or two splits.
    """

    low, high = CONFIG["gap_classes"][gap_class]
    pauli = observable(family, width)
    limit = int(CONFIG["seed_search_limit"])
    values = []
    for index in range(limit):
        seed = derive_seed("selector-v3", family, width, depth, gap_class, index)
        values.append((seed, ideal_expectation(BUILDERS[family](width, depth, seed), pauli)))
        if len(values) < count:
            continue
        window = values[-count:]
        ordered = sorted((value for _, value in window), reverse=True)
        if low <= ordered[0] - ordered[1] < high:
            return [
                {
                    "task_id": f"v3_{family}_w{width}_d{depth}_{gap_class}",
                    "family": family,
                    "width": width,
                    "depth": depth,
                    "gap_class": gap_class,
                    "candidate": position,
                    "seed_index": index - count + 1 + position,
                    "parameter_seed": seed,
                    "observable_q0_first": pauli,
                    "ideal_expectation": round(value, 12),
                    "ideal_top_gap": round(ordered[0] - ordered[1], 12),
                }
                for position, (seed, value) in enumerate(window)
            ]
    return []


def _assign_splits(frame: pd.DataFrame) -> pd.Series:
    fractions = CONFIG["split_fractions"]
    tasks = frame.drop_duplicates("task_id")
    output: dict[str, str] = {}
    for _, group in tasks.groupby("family", sort=True):
        ordered = sorted(group.task_id, key=lambda name: derive_seed("selector-v3-split", name))
        train = int(round(len(ordered) * fractions["train"]))
        validation = int(round(len(ordered) * fractions["validation"]))
        for position, name in enumerate(ordered):
            if position < train:
                output[name] = "train"
            elif position < train + validation:
                output[name] = "validation"
            else:
                output[name] = "holdout"
    return frame.task_id.map(output)


def build_manifest() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coverage = [], []
    for family in CONFIG["families"]:
        for width in CONFIG["widths"]:
            for depth in CONFIG["depths"]:
                for gap_class in CONFIG["gap_classes"]:
                    selected = _select_task(
                        family, width, depth, gap_class, int(CONFIG["candidates_per_task"])
                    )
                    rows.extend(selected)
                    coverage.append(
                        {
                            "family": family,
                            "width": width,
                            "depth": depth,
                            "gap_class": gap_class,
                            "found": bool(selected),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame["split"] = _assign_splits(frame)
    return frame, pd.DataFrame(coverage)


def main() -> None:
    frame, coverage = build_manifest()
    frame.to_csv(MANIFEST_PATH, index=False, lineterminator="\n", float_format="%.15g")
    digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    HASH_PATH.write_text(digest + "\n", encoding="utf-8", newline="\n")
    coverage.to_csv(COVERAGE_PATH, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
