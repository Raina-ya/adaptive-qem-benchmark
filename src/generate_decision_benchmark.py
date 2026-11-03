"""Generate the preregistered controlled decision-reliability benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import derive_seed
from src.decision_instances import SUPPORTED_MARGINS, controlled_decision_instance
from src.decision_metrics import (
    decision_error_interval,
    evaluate_decision,
    paired_gap_statistics,
)
from src.estimators.unmitigated import estimate_pauli
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.zne import run_linear_zne
from src.noise_models import noise_spec
from src.resource_accounting import split_shots

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/decision_margin_benchmark_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/decision_margin_summary_v1.csv"
UNRESOLVED_PATH = ROOT / "results/processed/decision_unresolved_cells_v1.csv"
MANIFEST_PATH = ROOT / "experiments/manifests/decision_benchmark_v1_manifest.csv"
FIGURE_DIR = ROOT / "figures"
MARGINS = SUPPORTED_MARGINS
BUDGETS = (1024, 4096, 16384)
METHODS = ("raw", "readout", "zne")
RECORD_MODES = ("independent_records", "shared_records")
NOISE_LEVELS = ("zero", "low", "medium", "high")
MAX_ERROR = 0.20


def build_manifest(repeats: int = 100) -> pd.DataFrame:
    """Build the complete preregistered decision grid."""

    rows = []
    for margin in MARGINS:
        for shots in BUDGETS:
            for method in METHODS:
                for record_mode in RECORD_MODES:
                    for noise in NOISE_LEVELS:
                        for repeat in range(repeats):
                            seed = derive_seed(
                                "decision",
                                int(margin * 1000),
                                shots,
                                METHODS.index(method),
                                RECORD_MODES.index(record_mode),
                                NOISE_LEVELS.index(noise),
                                repeat,
                            )
                            rows.append(
                                {
                                    "margin": margin,
                                    "width": 2,
                                    "depth": 3,
                                    "noise": noise,
                                    "shots": shots,
                                    "method": method,
                                    "record_mode": record_mode,
                                    "repeat": repeat,
                                    "status": "pending",
                                    "seed": seed,
                                }
                            )
    return pd.DataFrame(rows)


def _candidate_seed(base_seed: int, label: str, record_mode: str) -> int:
    if record_mode == "shared_records":
        return base_seed
    return derive_seed("decision-candidate", base_seed, 0 if label == "a" else 1)


def _estimate(candidate, method: str, shots: int, seed: int, noise):
    if method == "raw":
        result = estimate_pauli(
            candidate.circuit, candidate.observable, shots=shots, seed=seed, noise_spec=noise
        )
        return result.estimate, result.raw_counts_json, {"executions": 1}
    if method == "readout":
        result = run_readout_mitigation(
            candidate.circuit,
            candidate.observable,
            total_shot_budget=shots,
            noise_spec=noise,
            seed=seed,
        )
        if result.status != "ok" or result.corrected_estimate is None:
            raise RuntimeError(f"readout correction status: {result.status}")
        return (
            result.corrected_estimate,
            result.raw_counts_json,
            {
                "executions": result.total_circuit_executions,
                "condition_number": result.condition_number,
                "calibration_counts": result.calibration_counts_json,
            },
        )
    result = run_linear_zne(
        candidate.circuit,
        candidate.observable,
        total_shot_budget=shots,
        noise_spec=noise,
        seed=seed,
    )
    return (
        result.extrapolated_estimate,
        "",
        {
            "executions": len(result.factors),
            "factors": result.factors,
            "factor_shots": result.shots_by_factor,
            "factor_estimates": result.factor_estimates,
            "fit_residual": result.residual_sum_squares,
            "out_of_range": result.out_of_range,
        },
    )


def execute_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Execute all manifest rows, retaining failures and repeat-level diagnostics."""

    rows: list[dict[str, object]] = []
    identity_columns = [column for column in manifest.columns if column != "status"]
    if RAW_PATH.exists():
        checkpoint = pd.read_csv(RAW_PATH)
        prefix = manifest.iloc[: len(checkpoint)].reset_index(drop=True)
        observed = checkpoint[identity_columns].reset_index(drop=True)
        prefix = prefix[identity_columns]
        if len(checkpoint) > len(manifest) or not prefix.equals(observed):
            raise RuntimeError("Existing decision checkpoint does not match the manifest prefix")
        rows = checkpoint.to_dict("records")
    for index, item in manifest.iloc[len(rows) :].iterrows():
        row = item.to_dict()
        instance = controlled_decision_instance(float(item.margin), depth=int(item.depth))
        candidate_budget = split_shots(int(item.shots), 2)
        spec = noise_spec("mixed", str(item.noise))
        try:
            estimate_a, raw_a, diagnostics_a = _estimate(
                instance.candidate_a,
                str(item.method),
                candidate_budget[0],
                _candidate_seed(int(item.seed), "a", str(item.record_mode)),
                spec,
            )
            estimate_b, raw_b, diagnostics_b = _estimate(
                instance.candidate_b,
                str(item.method),
                candidate_budget[1],
                _candidate_seed(int(item.seed), "b", str(item.record_mode)),
                spec,
            )
            outcome = evaluate_decision(
                estimate_a,
                estimate_b,
                instance.candidate_a.exact_expectation,
                instance.candidate_b.exact_expectation,
            )
            row.update(
                status="complete",
                error="",
                reference_a=instance.candidate_a.exact_expectation,
                reference_b=instance.candidate_b.exact_expectation,
                estimate_a=estimate_a,
                estimate_b=estimate_b,
                estimated_gap=outcome.estimated_gap,
                decision_error=outcome.decision_error,
                simple_regret=outcome.regret,
                practical_tie=abs(outcome.estimated_gap) <= float(item.margin) / 2.0,
                candidate_shots_a=candidate_budget[0],
                candidate_shots_b=candidate_budget[1],
                total_shots_used=sum(candidate_budget),
                raw_counts_a=raw_a,
                raw_counts_b=raw_b,
                diagnostics_a=json.dumps(diagnostics_a, sort_keys=True),
                diagnostics_b=json.dumps(diagnostics_b, sort_keys=True),
                result_label="noisy_simulated",
            )
        except Exception as error:  # preserve explicit failed cells
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        rows.append(row)
        if (index + 1) % 500 == 0:
            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(RAW_PATH, index=False)
    return pd.DataFrame(rows)


GROUPS = ["margin", "noise", "shots", "method", "record_mode"]


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate complete repeats without hiding incomplete cells."""

    rows = []
    for keys, all_rows in raw.groupby(GROUPS, sort=True):
        group = all_rows.loc[all_rows.status == "complete"].copy()
        row = dict(zip(GROUPS, keys, strict=True))
        row["expected_repeats"] = len(all_rows)
        row["complete_repeats"] = len(group)
        row["failed_repeats"] = int((all_rows.status == "failed").sum())
        if group.empty:
            row.update(status="failed")
            rows.append(row)
            continue
        errors = group.decision_error.astype(bool).to_numpy()
        rate, low, high = decision_error_interval(errors)
        covariance, gap_variance = paired_gap_statistics(
            group.estimate_a.to_numpy(), group.estimate_b.to_numpy()
        )
        exact_gap = float(keys[0])
        gap_errors = group.estimated_gap.to_numpy() - exact_gap
        interval_low, interval_high = np.quantile(group.estimated_gap, [0.025, 0.975])
        unresolved = interval_low <= 0.0 <= interval_high or high > MAX_ERROR
        row.update(
            status="unresolved" if unresolved else "complete",
            gap_bias=float(gap_errors.mean()),
            gap_variance=gap_variance,
            candidate_covariance=covariance,
            decision_error_probability=rate,
            decision_error_wilson_low=low,
            decision_error_wilson_high=high,
            simple_regret_mean=float(group.simple_regret.mean()),
            unresolved_probability=float(group.practical_tie.astype(bool).mean()),
            expectation_rmse_a=float(np.sqrt(np.mean((group.estimate_a - group.reference_a) ** 2))),
            expectation_rmse_b=float(np.sqrt(np.mean((group.estimate_b - group.reference_b) ** 2))),
            marginal_expectation_rmse=float(
                np.sqrt(
                    np.mean(
                        np.concatenate(
                            (
                                (group.estimate_a - group.reference_a) ** 2,
                                (group.estimate_b - group.reference_b) ** 2,
                            )
                        )
                    )
                )
            ),
            gap_rmse=float(np.sqrt(np.mean(gap_errors**2))),
            argmax_accuracy=1.0 - rate,
            empirical_gap_interval_low=float(interval_low),
            empirical_gap_interval_high=float(interval_high),
            total_shots_mean=float(group.total_shots_used.mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    palette = {"raw": "#4c78a8", "readout": "#f58518", "zne": "#54a24b"}
    selected = summary.query("noise == 'medium' and record_mode == 'independent_records'")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, (shots, frame) in zip(axes, selected.groupby("shots"), strict=True):
        for method, group in frame.groupby("method"):
            axis.errorbar(
                group.margin,
                group.decision_error_probability,
                yerr=[
                    group.decision_error_probability - group.decision_error_wilson_low,
                    group.decision_error_wilson_high - group.decision_error_probability,
                ],
                marker="o",
                label=method,
                color=palette[method],
            )
        axis.set_title(f"budget={shots}")
        axis.set_xlabel("Exact margin")
    axes[0].set_ylabel("Decision-error probability (95% Wilson)")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "decision_margin_vs_error.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for method, group in summary.groupby("method"):
        axis.scatter(
            group.marginal_expectation_rmse,
            group.decision_error_probability,
            s=18,
            alpha=0.55,
            label=method,
            color=palette[method],
        )
    axis.set(xlabel="Marginal expectation RMSE", ylabel="Decision-error probability")
    axis.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "rmse_vs_decision_error.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5))
    for method, group in summary.query("shots == 4096 and noise == 'medium'").groupby("method"):
        for mode, mode_group in group.groupby("record_mode"):
            axis.plot(
                mode_group.margin,
                mode_group.gap_variance,
                marker="o",
                label=f"{method}: {mode.replace('_records', '')}",
            )
    axis.set_yscale("log")
    axis.set(xlabel="Exact margin", ylabel="Gap variance")
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "gap_variance_vs_margin.png", dpi=160)
    plt.close(fig)

    paired = summary.pivot_table(
        index=["margin", "noise", "shots", "method"],
        columns="record_mode",
        values="decision_error_probability",
    ).reset_index()
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(paired.independent_records, paired.shared_records, s=18, alpha=0.6)
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set(
        xlabel="Independent-record error",
        ylabel="Shared-record error",
        xlim=(-0.02, 1),
        ylim=(-0.02, 1),
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shared_vs_independent_decisions.png", dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args(argv)
    if args.repeats <= 1:
        raise ValueError("At least two repeats are required")
    manifest = build_manifest(args.repeats)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_PATH, index=False)
    raw = execute_manifest(manifest)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    summary = summarize(raw)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    summary.loc[summary.status != "complete"].to_csv(UNRESOLVED_PATH, index=False)
    _plot(summary)


if __name__ == "__main__":
    main()
