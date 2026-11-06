"""Freeze simulator predictions and requirements for a possible hardware study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/hardware_readiness_v1.yaml"
MANIFEST_PATH = ROOT / "experiments/hardware/hardware_manifest_v1.csv"
PROCESSED = ROOT / "results/processed"
RAW = ROOT / "results/raw"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
METHODS = ("raw", "readout", "zne", "combined")
NOT_APPLICABLE = "not_applicable"
NOT_ASSESSED = "not_assessed"


def required_candidate_shots(gap: float, mean: float) -> int:
    """Return shots per candidate to detect a noisy gap with the frozen power."""

    if gap == 0.0:
        raise ValueError("A zero gap cannot be powered")
    alpha = float(CONFIG["alpha"])
    power = float(CONFIG["power"])
    z_total = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    variance = 2.0 * max(1.0 - mean**2, 0.0)
    return int(z_total**2 * variance / gap**2) + 1


def required_repeats(effect: float, spread: float) -> int:
    """Return paired repeats that detect a mean effect with the frozen power."""

    if effect == 0.0:
        raise ValueError("A zero effect cannot be powered")
    alpha = float(CONFIG["alpha"])
    power = float(CONFIG["power"])
    z_total = float(norm.ppf(1.0 - alpha / 2.0) + norm.ppf(power))
    return max(int((z_total * spread / abs(effect)) ** 2) + 1, 3)


def _decision_text(reference: float, kind: str) -> str:
    if kind == "decision":
        return "candidate_a"
    return "negative" if reference < 0.0 else "positive"


def _validity(cell: dict[str, object], zne_table: pd.DataFrame, budget: int) -> str:
    source = cell["zne_validity_source"]
    if source is None:
        return NOT_ASSESSED
    circuit, noise = source
    row = zne_table.loc[
        (zne_table.circuit_id == circuit)
        & (zne_table.noise == noise)
        & (zne_table.total_shot_budget == budget)
    ]
    return str(row.validity_status.iloc[0]) if len(row) else NOT_ASSESSED


def _paired_power(
    raw: pd.DataFrame, source: str, budget: int, policy: str, method: str
) -> tuple[float, float]:
    frame = raw.loc[
        (raw.cell_id == source)
        & (raw.total_shot_budget == budget)
        & (raw.allocation_policy == policy)
        & (raw.status == "complete")
    ]
    baseline = frame.loc[frame.method == "raw"].set_index("repeat").absolute_error
    other = frame.loc[frame.method == method].set_index("repeat").absolute_error
    difference = (baseline - other).dropna()
    return float(difference.mean()), float(difference.std(ddof=1))


def _combined_row(
    cell: dict[str, object],
    summary: pd.DataFrame,
    validity: pd.DataFrame,
    zne_table: pd.DataFrame,
    raw: pd.DataFrame,
) -> dict[str, object]:
    budget = int(CONFIG["reference_budget"])
    policy = str(CONFIG["reference_allocation"])
    source = str(cell["source_cell"])
    rows = summary.loc[
        (summary.cell_id == source)
        & (summary.total_shot_budget == budget)
        & (summary.allocation_policy == policy)
    ].set_index("method")
    kind = str(rows.kind.iloc[0])
    identity = raw.loc[(raw.cell_id == source) & (raw.status == "complete")].iloc[0]
    reference = (
        float(identity.exact_gap) if kind == "decision" else float(identity.reference_a)
    )
    output: dict[str, object] = {"reference_value": reference}
    for method in METHODS:
        output[f"predicted_{method}"] = reference + float(rows.loc[method, "bias"])
        output[f"uncertainty_{method}"] = float(rows.loc[method, "variance"]) ** 0.5
    check = validity.loc[
        (validity.cell_id == source)
        & (validity.total_shot_budget == budget)
        & (validity.allocation_policy == policy)
    ]
    output.update(
        expected_decision=_decision_text(reference, kind),
        expected_zne_validity=_validity(cell, zne_table, budget),
        expected_combined_validity=(
            str(check.validity_status.iloc[0]) if len(check) else NOT_ASSESSED
        ),
        max_decision_error_wilson_high=(
            float(rows.decision_error_wilson_high.max()) if kind == "decision" else float("nan")
        ),
    )
    raw_rows = rows.loc["raw"]
    if kind == "decision":
        gap = abs(reference + float(raw_rows.bias))
        candidate = required_candidate_shots(gap, reference / 2.0)
        output.update(
            powered_quantity="raw noisy gap resolved in one repeat",
            required_shots_per_repeat=2 * candidate,
            required_repeats=1,
        )
    else:
        target = "zne" if cell["cell_id"] != "readout_stable" else "readout"
        effect, spread = _paired_power(raw, source, budget, policy, target)
        output.update(
            powered_quantity=f"paired absolute-error gain of {target} over raw",
            required_shots_per_repeat=budget,
            required_repeats=required_repeats(effect, spread),
        )
    planned = int(CONFIG["planned_repeats"])
    per_repeat = max(int(output["required_shots_per_repeat"]), budget)
    executed_methods = len(METHODS) + (1 if cell["needs_garbage_control"] else 0)
    output["planned_repeats"] = planned
    output["required_total_shots"] = per_repeat * planned * executed_methods
    return output


def _lifetime_row(cell: dict[str, object]) -> dict[str, object]:
    settings = CONFIG["calibration_reuse"]
    columns = [
        "workload_length",
        "drift_model",
        "policy",
        "true_probability_one",
        "observed_probability_one",
        "corrected_probability_one",
        "decision_error",
        "workflow_total_shots",
        "repeat",
    ]
    raw = pd.read_csv(RAW / "calibration_lifetime_v1.csv", usecols=columns)
    frame = raw.loc[
        (raw.workload_length == int(settings["workload_length"]))
        & (raw.drift_model == cell["reference_noise"])
    ]
    never = frame.loc[frame.policy == "never"]
    every = frame.loc[frame.policy == "every_decision"]
    workflow = frame.drop_duplicates(["policy", "repeat"]).groupby("policy").workflow_total_shots
    return {
        "reference_value": float(frame.true_probability_one.iloc[0]),
        "predicted_raw": float(never.observed_probability_one.mean()),
        "uncertainty_raw": float(never.observed_probability_one.std(ddof=1)),
        "predicted_readout": float(every.corrected_probability_one.mean()),
        "uncertainty_readout": float(every.corrected_probability_one.std(ddof=1)),
        "predicted_zne": float("nan"),
        "uncertainty_zne": float("nan"),
        "predicted_combined": float("nan"),
        "uncertainty_combined": float("nan"),
        "expected_decision": "above_threshold",
        "expected_zne_validity": NOT_APPLICABLE,
        "expected_combined_validity": NOT_APPLICABLE,
        "max_decision_error_wilson_high": float("nan"),
        "powered_quantity": "drift direction unknown before hardware; not powerable",
        "required_shots_per_repeat": int(workflow.first().sum()),
        "required_repeats": int(settings["repeats"]),
        "planned_repeats": int(settings["repeats"]),
        "required_total_shots": int(workflow.first().sum()) * int(settings["repeats"]),
    }


def build_manifest() -> pd.DataFrame:
    """Return one frozen prediction row per proposed cell, with empty job metadata."""

    summary = pd.read_csv(PROCESSED / "combined_qem_summary_v1.csv")
    validity = pd.read_csv(PROCESSED / "combined_qem_validity_v1.csv")
    zne_table = pd.read_csv(PROCESSED / "zne_validity_v1.csv")
    raw = pd.read_csv(
        RAW / "combined_qem_v1.csv",
        usecols=[
            "cell_id",
            "total_shot_budget",
            "allocation_policy",
            "method",
            "repeat",
            "status",
            "absolute_error",
            "exact_gap",
            "reference_a",
            "record_kind",
        ],
    )
    raw = raw.loc[raw.record_kind == "primary"]
    rows = []
    for cell in CONFIG["cells"]:
        values = (
            _combined_row(cell, summary, validity, zne_table, raw)
            if cell["source_cell"]
            else _lifetime_row(cell)
        )
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "role": cell["role"],
                "circuit": cell["circuit"],
                "observable": cell["observable"],
                "prediction_noise_condition": cell["reference_noise"],
                "prediction_evidence": CONFIG["evidence_label"],
                **values,
                "falsifiable_hypothesis": cell["hypothesis"],
                "falsifier": cell["falsifier"],
                "submission_status": "not_submitted",
                **{field: "" for field in CONFIG["required_metadata"]},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_manifest().to_csv(MANIFEST_PATH, index=False, lineterminator="\n", float_format="%.12g")


if __name__ == "__main__":
    main()
