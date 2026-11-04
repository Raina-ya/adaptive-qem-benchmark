"""Generate the preregistered combined readout-mitigation and ZNE study."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from functools import cache
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuits import ghz_circuit
from src.config import derive_seed
from src.decision_instances import controlled_decision_instance
from src.decision_metrics import decision_error_interval, evaluate_decision
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.mitigation.combined import ALLOCATION_POLICIES, run_combined_mitigation
from src.mitigation.folding import folded_metadata
from src.mitigation.garbage_folding import garbage_fold
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.zne import run_linear_zne
from src.noise_models import noise_spec
from src.observables import exact_expectation
from src.resource_accounting import split_shots
from src.zne_validity_gate import evaluate_zne_gate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/manifests/combined_qem_v1_manifest.csv"
RAW_PATH = ROOT / "results/raw/combined_qem_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/combined_qem_summary_v1.csv"
INTERACTIONS_PATH = ROOT / "results/processed/combined_qem_interactions_v1.csv"
VALIDITY_PATH = ROOT / "results/processed/combined_qem_validity_v1.csv"
FIGURE_DIR = ROOT / "figures"

METHODS = ("raw", "readout", "zne", "combined")
BUDGETS = (4096, 16384)
REPEATS = 50
FACTORS = (1, 3, 5)
CELLS = (
    {
        "cell_id": "readout_dominated",
        "kind": "expectation",
        "circuit_id": "ghz_w2",
        "noise": "high",
        "margin": np.nan,
    },
    {
        "cell_id": "gate_noise_dominated",
        "kind": "expectation",
        "circuit_id": "he_w4_d3",
        "noise": "medium",
        "margin": np.nan,
    },
    {
        "cell_id": "practical_tie",
        "kind": "expectation",
        "circuit_id": "ghz_w2",
        "noise": "low",
        "margin": np.nan,
    },
    {
        "cell_id": "zne_invalid",
        "kind": "expectation",
        "circuit_id": "he_w2_d3",
        "noise": "medium",
        "margin": np.nan,
    },
    {
        "cell_id": "decision_near_tie",
        "kind": "decision",
        "circuit_id": "controlled_margin_002",
        "noise": "medium",
        "margin": 0.02,
    },
    {
        "cell_id": "decision_larger_margin",
        "kind": "decision",
        "circuit_id": "controlled_margin_020",
        "noise": "medium",
        "margin": 0.20,
    },
)


@cache
def _expectation_circuit(circuit_id: str):
    if circuit_id == "ghz_w2":
        circuit = ghz_circuit(2)
        observable = "XX"
    elif circuit_id == "he_w2_d3":
        circuit = hardware_efficient_circuit(2, 3)
        observable = hardware_efficient_observable(2)
    elif circuit_id == "he_w4_d3":
        circuit = hardware_efficient_circuit(4, 3)
        observable = hardware_efficient_observable(4)
    else:
        raise ValueError(f"Unknown expectation circuit: {circuit_id}")
    return circuit, observable, exact_expectation(circuit, observable)


def build_manifest(repeats: int = REPEATS) -> pd.DataFrame:
    """Freeze every primary four-method row before combined outcomes are read."""

    rows: list[dict[str, object]] = []
    for cell_index, cell in enumerate(CELLS):
        for budget in BUDGETS:
            for policy in ALLOCATION_POLICIES:
                for method_index, method in enumerate(METHODS):
                    for repeat in range(repeats):
                        pair_seed = derive_seed("combined-qem-pair", cell_index, budget, repeat)
                        rows.append(
                            {
                                **cell,
                                "total_shot_budget": budget,
                                "allocation_policy": policy,
                                "method": method,
                                "repeat": repeat,
                                "pair_seed": pair_seed,
                                "seed": derive_seed("combined-qem-method", pair_seed, method_index),
                                "status": "pending",
                            }
                        )
    return pd.DataFrame(rows)


@cache
def _fold_resources(circuit_id: str, method: str) -> tuple[int, int]:
    circuit, _, _ = _expectation_circuit(circuit_id)
    factors = FACTORS if method in {"zne", "combined", "combined_control"} else (1,)
    metadata = [
        folded_metadata(
            circuit,
            factor,
            seed=derive_seed("combined-resource", circuit_id, method, factor),
        )
        for factor in factors
    ]
    return len(factors), sum(item["transpiled_two_qubit_gate_count"] for item in metadata)


def _estimate_one(circuit, observable: str, method: str, budget: int, seed: int, spec, policy):
    """Return one estimate and its complete method diagnostics."""

    if method == "raw":
        result = estimate_pauli(circuit, observable, shots=budget, seed=seed, noise_spec=spec)
        return result.estimate, {
            "status": "ok",
            "total_shots_used": budget,
            "calibration_shots": 0,
            "calibration_executions": 0,
            "folded_executions": 1,
            "raw_counts": result.raw_counts_json,
        }
    if method == "readout":
        result = run_readout_mitigation(
            circuit,
            observable,
            total_shot_budget=budget,
            noise_spec=spec,
            seed=seed,
        )
        return result.corrected_estimate, {
            **asdict(result),
            "total_shots_used": result.total_shot_budget,
            "calibration_executions": result.calibration_circuit_executions,
            "folded_executions": result.target_circuit_executions,
        }
    if method == "zne":
        result = run_linear_zne(
            circuit,
            observable,
            total_shot_budget=budget,
            noise_spec=spec,
            seed=seed,
        )
        return result.extrapolated_estimate, {
            **asdict(result),
            "status": "ok",
            "total_shots_used": result.total_shot_budget,
            "calibration_shots": 0,
            "calibration_executions": 0,
            "folded_executions": len(result.factors),
        }
    folder = garbage_fold if method == "combined_control" else None
    kwargs = {"folder": folder} if folder is not None else {}
    result = run_combined_mitigation(
        circuit,
        observable,
        total_shot_budget=budget,
        noise_spec=spec,
        seed=seed,
        allocation_policy=policy,
        **kwargs,
    )
    return result.corrected_extrapolated_estimate, {
        **asdict(result),
        "total_shots_used": result.total_shot_budget,
        "calibration_executions": result.calibration_circuit_executions,
        "folded_executions": result.folded_circuit_executions,
    }


def _execute_row(item: pd.Series, method_override: str | None = None) -> dict[str, object]:
    method = method_override or str(item.method)
    output = item.to_dict()
    output.update(
        method=method,
        record_kind="validation_control" if method_override else "primary",
        result_label="noisy_simulated",
        error="",
        estimate_a=np.nan,
        estimate_b=np.nan,
        reference_a=np.nan,
        reference_b=np.nan,
        estimated_gap=np.nan,
        exact_gap=np.nan,
        decision_error=np.nan,
        signed_error=np.nan,
        absolute_error=np.nan,
        total_shots_used=0,
        calibration_shots=0,
        calibration_executions=0,
        folded_executions=0,
        folded_two_qubit_gate_count=0,
        diagnostics_a_json="",
        diagnostics_b_json="",
        zne_validity_classification="not_applicable",
        negative_mass=np.nan,
        physical_overshoot=np.nan,
        final_signal_retention=np.nan,
        factor_one_corrected_estimate=np.nan,
        true_gain=np.nan,
    )
    spec = noise_spec("mixed", str(item.noise))
    try:
        if item.kind == "expectation":
            circuit, observable, reference = _expectation_circuit(str(item.circuit_id))
            estimate, diagnostics = _estimate_one(
                circuit,
                observable,
                method,
                int(item.total_shot_budget),
                int(item.seed),
                spec,
                str(item.allocation_policy),
            )
            if diagnostics["status"] != "ok" or estimate is None:
                raise RuntimeError(f"estimator status: {diagnostics['status']}")
            output.update(
                estimate_a=estimate,
                reference_a=reference,
                signed_error=estimate - reference,
                absolute_error=abs(estimate - reference),
                total_shots_used=int(diagnostics["total_shots_used"]),
                calibration_shots=int(diagnostics.get("calibration_shots", 0)),
                calibration_executions=int(diagnostics["calibration_executions"]),
                folded_executions=int(diagnostics["folded_executions"]),
                diagnostics_a_json=json.dumps(diagnostics, sort_keys=True, separators=(",", ":")),
            )
            _, two_qubit_count = _fold_resources(str(item.circuit_id), method)
            output["folded_two_qubit_gate_count"] = two_qubit_count
            if method in {"combined", "combined_control"}:
                output.update(
                    zne_validity_classification=diagnostics["validity_classification"],
                    negative_mass=diagnostics["negative_mass"],
                    physical_overshoot=diagnostics["physical_overshoot"],
                    final_signal_retention=diagnostics["final_signal_retention"],
                    factor_one_corrected_estimate=diagnostics[
                        "factor_corrected_estimates"
                    ][0],
                    true_gain=(
                        abs(diagnostics["factor_corrected_estimates"][0] - reference)
                        - abs(estimate - reference)
                    ),
                )
        else:
            instance = controlled_decision_instance(float(item.margin), depth=3)
            candidate_budgets = split_shots(int(item.total_shot_budget), 2)
            estimates: list[float] = []
            diagnostics_by_candidate: list[dict[str, object]] = []
            for index, (candidate, candidate_budget) in enumerate(
                zip(
                    (instance.candidate_a, instance.candidate_b),
                    candidate_budgets,
                    strict=True,
                )
            ):
                estimate, diagnostics = _estimate_one(
                    candidate.circuit,
                    candidate.observable,
                    method,
                    candidate_budget,
                    derive_seed("combined-decision-candidate", int(item.seed), index),
                    spec,
                    str(item.allocation_policy),
                )
                if diagnostics["status"] != "ok" or estimate is None:
                    raise RuntimeError(f"candidate {index} status: {diagnostics['status']}")
                estimates.append(estimate)
                diagnostics_by_candidate.append(diagnostics)
            outcome = evaluate_decision(
                estimates[0],
                estimates[1],
                instance.candidate_a.exact_expectation,
                instance.candidate_b.exact_expectation,
            )
            output.update(
                estimate_a=estimates[0],
                estimate_b=estimates[1],
                reference_a=instance.candidate_a.exact_expectation,
                reference_b=instance.candidate_b.exact_expectation,
                estimated_gap=outcome.estimated_gap,
                exact_gap=instance.margin,
                decision_error=outcome.decision_error,
                signed_error=outcome.estimated_gap - instance.margin,
                absolute_error=abs(outcome.estimated_gap - instance.margin),
                total_shots_used=sum(
                    int(diagnostics["total_shots_used"])
                    for diagnostics in diagnostics_by_candidate
                ),
                calibration_shots=sum(
                    int(diagnostics.get("calibration_shots", 0))
                    for diagnostics in diagnostics_by_candidate
                ),
                calibration_executions=sum(
                    int(diagnostics["calibration_executions"])
                    for diagnostics in diagnostics_by_candidate
                ),
                folded_executions=sum(
                    int(diagnostics["folded_executions"])
                    for diagnostics in diagnostics_by_candidate
                ),
                diagnostics_a_json=json.dumps(
                    diagnostics_by_candidate[0], sort_keys=True, separators=(",", ":")
                ),
                diagnostics_b_json=json.dumps(
                    diagnostics_by_candidate[1], sort_keys=True, separators=(",", ":")
                ),
            )
            if method == "combined":
                classifications = {
                    str(diagnostics["validity_classification"])
                    for diagnostics in diagnostics_by_candidate
                }
                output["zne_validity_classification"] = (
                    classifications.pop() if len(classifications) == 1 else "mixed"
                )
        if output["total_shots_used"] != int(item.total_shot_budget):
            raise RuntimeError("shot ledger does not equal the declared budget")
        output["status"] = "complete"
    except Exception as error:
        output["status"] = "failed"
        output["error"] = f"{type(error).__name__}: {error}"
    return output


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute primary rows and matched-cost expectation controls."""

    rows: list[dict[str, object]] = []
    for index, item in manifest.iterrows():
        rows.append(_execute_row(item))
        if item.method == "combined" and item.kind == "expectation":
            rows.append(_execute_row(item, method_override="combined_control"))
        if (index + 1) % 100 == 0:
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(RAW_PATH, index=False)
    return pd.DataFrame(rows)


SUMMARY_GROUPS = [
    "cell_id",
    "kind",
    "circuit_id",
    "noise",
    "total_shot_budget",
    "allocation_policy",
    "method",
]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate primary method performance without hiding failures."""

    rows: list[dict[str, object]] = []
    primary = raw.loc[raw.record_kind == "primary"]
    for keys, all_rows in primary.groupby(SUMMARY_GROUPS, sort=True, dropna=False):
        complete = all_rows.loc[all_rows.status == "complete"]
        row = dict(zip(SUMMARY_GROUPS, keys, strict=True))
        errors = complete.signed_error.to_numpy(dtype=float)
        row.update(
            expected_repeats=len(all_rows),
            complete_repeats=len(complete),
            failed_repeats=int((all_rows.status == "failed").sum()),
            status="complete" if len(complete) == len(all_rows) else "failed",
            bias=float(errors.mean()) if len(errors) else np.nan,
            variance=float(errors.var(ddof=1)) if len(errors) > 1 else np.nan,
            rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan,
            execution_overhead=float(
                (complete.calibration_executions + complete.folded_executions).mean()
            )
            if len(complete)
            else np.nan,
            calibration_overhead=float(complete.calibration_shots.mean())
            if len(complete)
            else np.nan,
            folded_two_qubit_count=float(complete.folded_two_qubit_gate_count.mean())
            if len(complete)
            else np.nan,
            failure_probability=1.0 - len(complete) / len(all_rows),
            decision_error_probability=np.nan,
            decision_error_wilson_low=np.nan,
            decision_error_wilson_high=np.nan,
        )
        if len(complete) and row["kind"] == "decision":
            probability, low, high = decision_error_interval(
                complete.decision_error.astype(bool).to_numpy()
            )
            row.update(
                decision_error_probability=probability,
                decision_error_wilson_low=low,
                decision_error_wilson_high=high,
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    raw_rmse = summary.loc[summary.method == "raw", SUMMARY_GROUPS[:-1] + ["rmse"]].rename(
        columns={"rmse": "raw_rmse"}
    )
    summary = summary.merge(raw_rmse, on=SUMMARY_GROUPS[:-1], how="left", validate="many_to_one")
    summary["rmse_gain_vs_raw"] = summary.raw_rmse - summary.rmse
    return summary


def summarize_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Estimate paired 2x2 interaction effects for continuous and decision error."""

    primary = raw.loc[(raw.record_kind == "primary") & (raw.status == "complete")].copy()
    identity = [
        "cell_id",
        "kind",
        "circuit_id",
        "noise",
        "total_shot_budget",
        "allocation_policy",
        "repeat",
    ]
    error_pivot = primary.pivot(index=identity, columns="method", values="absolute_error")
    decision_pivot = primary.pivot(index=identity, columns="method", values="decision_error")
    rows: list[dict[str, object]] = []
    group_columns = identity[:-1]
    for keys, group in error_pivot.dropna(subset=list(METHODS)).groupby(
        level=group_columns, sort=True
    ):
        interactions = (
            group["combined"] - group["readout"] - group["zne"] + group["raw"]
        ).to_numpy(dtype=float)
        mean = float(interactions.mean())
        standard_error = float(interactions.std(ddof=1) / math.sqrt(len(interactions)))
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            matched_repeats=len(interactions),
            interaction_mean=mean,
            interaction_ci_low=mean - 1.959963984540054 * standard_error,
            interaction_ci_high=mean + 1.959963984540054 * standard_error,
            negative_interaction_rate=float((interactions < 0.0).mean()),
            decision_interaction_mean=np.nan,
        )
        if row["kind"] == "decision":
            decision_group = decision_pivot.xs(keys, level=group_columns).dropna(
                subset=list(METHODS)
            )
            values = (
                decision_group["combined"].astype(float)
                - decision_group["readout"].astype(float)
                - decision_group["zne"].astype(float)
                + decision_group["raw"].astype(float)
            )
            row["decision_interaction_mean"] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_validity(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the full frozen C5 gate to combined expectation estimates."""

    rows: list[dict[str, object]] = []
    groups = [
        "cell_id",
        "circuit_id",
        "noise",
        "total_shot_budget",
        "allocation_policy",
    ]
    expectation = raw.loc[(raw.kind == "expectation") & (raw.status == "complete")]
    for keys, frame in expectation.groupby(groups, sort=True):
        genuine = frame.loc[frame.method == "combined"].sort_values("repeat")
        control = frame.loc[frame.method == "combined_control"].sort_values("repeat")
        if len(genuine) != REPEATS or len(control) != REPEATS:
            continue
        result = evaluate_zne_gate(
            genuine.true_gain.to_numpy(dtype=float),
            control.true_gain.to_numpy(dtype=float),
            genuine.final_signal_retention.to_numpy(dtype=float),
            genuine.negative_mass.to_numpy(dtype=float),
            genuine.physical_overshoot.to_numpy(dtype=float),
            genuine.zne_validity_classification.eq("signal_destroyed").to_numpy(),
            genuine.zne_validity_classification.eq("extrapolation_unstable").to_numpy(),
        )
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                **asdict(result),
                "failure_reasons_json": json.dumps(result.failure_reasons),
                "result_label": "noisy_simulated",
            }
        )
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame, interactions: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    primary = summary.loc[summary.status == "complete"]

    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    for method, group in primary.groupby("method", sort=True):
        axis.scatter(group.execution_overhead, group.rmse_gain_vs_raw, label=method, alpha=0.75)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Mean circuit executions per repeat",
        ylabel="RMSE gain versus raw",
        title="Combined overhead does not guarantee estimation gain",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "combined_overhead_vs_gain.png", dpi=180)
    plt.close(figure)

    ordered = interactions.sort_values(
        ["cell_id", "total_shot_budget", "allocation_policy"]
    ).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(ordered))
    axis.errorbar(
        x,
        ordered.interaction_mean,
        yerr=np.vstack(
            (
                ordered.interaction_mean - ordered.interaction_ci_low,
                ordered.interaction_ci_high - ordered.interaction_mean,
            )
        ),
        fmt="o",
        capsize=3,
    )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(
        x,
        [
            f"{row.cell_id}\n{row.total_shot_budget}/{row.allocation_policy}"
            for row in ordered.itertuples()
        ],
        rotation=70,
        ha="right",
    )
    axis.set(
        ylabel="Mean factorial interaction in absolute error",
        title="Negative and positive combined-method interactions remain visible",
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "combined_interaction_effect.png", dpi=180)
    plt.close(figure)

    decisions = primary.loc[primary.kind == "decision"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for axis, policy in zip(axes, ALLOCATION_POLICIES, strict=True):
        subset = decisions.loc[decisions.allocation_policy == policy]
        positions = np.arange(len(METHODS))
        width = 0.35
        for offset, (cell_id, group) in enumerate(subset.groupby("cell_id", sort=True)):
            values = group.set_index("method").loc[list(METHODS)]
            x = positions + (offset - 0.5) * width
            axis.bar(x, values.decision_error_probability, width=width, label=cell_id)
            axis.errorbar(
                x,
                values.decision_error_probability,
                yerr=np.vstack(
                    (
                        values.decision_error_probability - values.decision_error_wilson_low,
                        values.decision_error_wilson_high - values.decision_error_probability,
                    )
                ),
                fmt="none",
                color="black",
                capsize=2,
            )
        axis.set_xticks(positions, METHODS)
        axis.set(title=policy, xlabel="Method")
    axes[0].set_ylabel("Decision-error probability with 95% Wilson interval")
    axes[-1].legend(fontsize=8)
    figure.suptitle("Combined QEM is evaluated separately on near and larger margins")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "combined_decision_error.png", dpi=180)
    plt.close(figure)


def main() -> None:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    raw = execute_manifest(manifest)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    summary = summarize(raw)
    interactions = summarize_interactions(raw)
    validity = summarize_validity(raw)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    interactions.to_csv(INTERACTIONS_PATH, index=False)
    validity.to_csv(VALIDITY_PATH, index=False)
    _plot(summary, interactions)


if __name__ == "__main__":
    main()
