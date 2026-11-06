"""Audit the v1 ZNE gate on its committed repeats. This is post-hoc and leaves v1 unchanged.

Each cell's repeats split into even and odd halves. The gate reads one half and the "really
improved" verdict reads the other, then the halves swap, so a verdict never sees its own decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.zne_validity_gate import ZNEGateResult, evaluate_zne_gate

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
CELLS_PATH = PROCESSED / "zne_gate_audit_v1_cells.csv"
CHECKS_PATH = PROCESSED / "zne_gate_audit_v1_checks.csv"
OVERLAP_PATH = PROCESSED / "zne_gate_audit_v1_trend_overlap.csv"
ANALYSIS_ROLE = "post_hoc_descriptive"
PRACTICAL = 0.005
Z_95 = 1.959963984540054
CHECKS = (
    "signal_retention",
    "physical_bounds",
    "negative_mass",
    "garbage_control",
    "gain_uncertainty",
    "trend",
)
# The v1 gate reads the exact reference through these two checks.
REFERENCE_CHECKS = {"gain_uncertainty", "garbage_control"}


def really_improved(gains: np.ndarray) -> tuple[bool, bool]:
    """Return whether the mean gain clears zero, and whether it also clears the practical size."""

    mean = float(gains.mean())
    low = mean - Z_95 * float(gains.std(ddof=1)) / np.sqrt(len(gains))
    return bool(low > 0.0), bool(low > 0.0 and mean > PRACTICAL)


def _final_retention(payload: object) -> float:
    return float(json.loads(payload)[-1]) if isinstance(payload, str) else float("nan")


def load_inputs() -> pd.DataFrame:
    """Return one row per genuine repeat, in one shape, for both v1 gate applications."""

    frames = []
    zne = pd.read_csv(ROOT / "results/raw/zne_falsification_v1.csv")
    keys = ["circuit_id", "noise", "total_shot_budget"]
    genuine = zne.loc[zne.method == "genuine_folding"]
    control = zne.loc[zne.method == "garbage_folding"]
    paired = genuine.merge(control, on=[*keys, "repeat"], suffixes=("", "_garbage"))
    frames.append(
        pd.DataFrame(
            {
                "study": "zne_falsification",
                "cell": paired.circuit_id + "|" + paired.noise,
                "circuit_id": paired.circuit_id,
                "noise": paired.noise,
                "total_shot_budget": paired.total_shot_budget,
                "repeat": paired.repeat,
                "true_gain": paired.true_gain,
                "garbage_gain": paired.true_gain_garbage,
                "retention": paired.signal_retention_json.map(_final_retention),
                "negative_mass": paired.negative_mass,
                "overshoot": (paired.estimate.abs() - 1.0).clip(lower=0.0),
                "classification": paired.validity_classification,
            }
        )
    )
    combined = pd.read_csv(ROOT / "results/raw/combined_qem_v1.csv")
    combined = combined.loc[(combined.kind == "expectation") & (combined.status == "complete")]
    keys = ["cell_id", "circuit_id", "noise", "total_shot_budget", "allocation_policy"]
    genuine = combined.loc[combined.method == "combined"]
    control = combined.loc[combined.method == "combined_control"]
    paired = genuine.merge(control, on=[*keys, "repeat"], suffixes=("", "_garbage"))
    frames.append(
        pd.DataFrame(
            {
                "study": "combined_qem",
                "cell": paired.cell_id + "|" + paired.circuit_id + "|" + paired.allocation_policy,
                "circuit_id": paired.circuit_id,
                "noise": paired.noise,
                "total_shot_budget": paired.total_shot_budget,
                "repeat": paired.repeat,
                "true_gain": paired.true_gain,
                "garbage_gain": paired.true_gain_garbage,
                "retention": paired.final_signal_retention,
                "negative_mass": paired.negative_mass,
                "overshoot": paired.physical_overshoot,
                "classification": paired.zne_validity_classification,
            }
        )
    )
    return pd.concat(frames, ignore_index=True)


def apply_gate(group: pd.DataFrame) -> ZNEGateResult:
    return evaluate_zne_gate(
        group.true_gain.to_numpy(float),
        group.garbage_gain.to_numpy(float),
        group.retention.to_numpy(float),
        group.negative_mass.to_numpy(float),
        group.overshoot.to_numpy(float),
        group.classification.eq("signal_destroyed").to_numpy(),
        group.classification.eq("extrapolation_unstable").to_numpy(),
    )


def audit_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the gate to each half and judge it on the other half."""

    rows = []
    keys = ["study", "cell", "circuit_id", "noise", "total_shot_budget"]
    for key, group in frame.groupby(keys, sort=True):
        group = group.sort_values("repeat")
        odd = group.repeat % 2 == 1
        for name, gated, held in (
            ("even", group[~odd], group[odd]),
            ("odd", group[odd], group[~odd]),
        ):
            result = apply_gate(gated)
            improved, practical = really_improved(held.true_gain.to_numpy(float))
            row = dict(zip(keys, key, strict=True))
            row.update(
                gate_half=name,
                held_gain_mean=float(held.true_gain.mean()),
                really_improved=improved,
                practically_improved=practical,
                accepted=result.validity_status == "credible",
                failure_reasons_json=json.dumps(result.failure_reasons),
            )
            row.update({f"{check}_pass": getattr(result, f"{check}_pass") for check in CHECKS})
            rows.append(row)
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def _confusion(passed: pd.Series, truth: pd.Series) -> dict[str, int]:
    return {
        "true_accept": int((passed & truth).sum()),
        "false_accept": int((passed & ~truth).sum()),
        "true_reject": int((~passed & ~truth).sum()),
        "false_reject": int((~passed & truth).sum()),
    }


def check_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Tabulate each check alone, the full gate, and the gate without reference-reading checks."""

    runtime = [check for check in CHECKS if check not in REFERENCE_CHECKS]
    decisions = {f"check_{check}": cells[f"{check}_pass"] for check in CHECKS}
    decisions["full_gate"] = cells.accepted
    decisions["runtime_checks_only"] = cells[[f"{check}_pass" for check in runtime]].all(axis=1)
    rows = []
    for study, subset in (("all", cells), *cells.groupby("study", sort=True)):
        for truth_name in ("really_improved", "practically_improved"):
            for decision, values in decisions.items():
                rows.append(
                    {
                        "study": study,
                        "truth_definition": truth_name,
                        "decision": decision,
                        "half_assessments": len(subset),
                        "independent_circuits": int(subset.circuit_id.nunique()),
                        "uses_exact_reference": decision.removeprefix("check_") in REFERENCE_CHECKS
                        or decision == "full_gate",
                        **_confusion(values.loc[subset.index], subset[truth_name]),
                    }
                )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def trend_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    """Measure how much of the v1 trend check is the physical-bounds check counted again."""

    rows = []
    for (study, cell, budget), group in frame.groupby(
        ["study", "cell", "total_shot_budget"], sort=True
    ):
        unstable = group.classification.eq("extrapolation_unstable")
        over = group.overshoot > 1e-12
        rows.append(
            {
                "study": study,
                "cell": cell,
                "total_shot_budget": int(budget),
                "repeats": len(group),
                "unstable_repeats": int(unstable.sum()),
                "unstable_with_overshoot": int((unstable & over).sum()),
                "unstable_from_residual_only": int((unstable & ~over).sum()),
                "destroyed_repeats": int(group.classification.eq("signal_destroyed").sum()),
            }
        )
    return pd.DataFrame(rows).assign(analysis_role=ANALYSIS_ROLE)


def main() -> None:
    frame = load_inputs()
    cells = audit_cells(frame)
    for table, path in (
        (cells, CELLS_PATH),
        (check_table(cells), CHECKS_PATH),
        (trend_overlap(frame), OVERLAP_PATH),
    ):
        table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


if __name__ == "__main__":
    main()
