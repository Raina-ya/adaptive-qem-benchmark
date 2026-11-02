"""Generate the reproducible Day 8 through Day 12 benchmark artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.benchmark_runner import build_manifest, coverage_audit, run_manifest
from src.circuits import bell_circuit, ghz_circuit
from src.config import derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import hardware_efficient_circuit
from src.mitigation.folding import folded_metadata
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.zne import PREREGISTERED_MODEL, run_linear_zne
from src.noise_models import noise_spec

RESULTS = Path("results")
RAW = RESULTS / "raw"
PROCESSED = RESULTS / "processed"
FIGURES = Path("figures")
MANIFESTS = Path("experiments/manifests")


def prepare() -> None:
    for directory in (RESULTS, RAW, PROCESSED, FIGURES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)


def generate_day08() -> None:
    targets = (
        ("bell", bell_circuit()),
        ("ghz_w4", ghz_circuit(4)),
        ("he_w4_d3", hardware_efficient_circuit(4, 3)),
    )
    rows = []
    for target_index, (circuit_id, circuit) in enumerate(targets):
        for factor in (1, 3, 5):
            rows.append(
                {
                    "circuit_id": circuit_id,
                    "width": circuit.num_qubits,
                    "noise_factor": factor,
                    "result_label": "noiseless_simulated_resource_validation",
                    **folded_metadata(
                        circuit, factor, seed=derive_seed("day08", target_index, factor)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "day08_fold_validation_lambda-1-3-5.csv", index=False)
    figure, axis = plt.subplots(figsize=(6.8, 4.2))
    for circuit_id, group in frame.groupby("circuit_id", sort=False):
        axis.plot(group.noise_factor, group.transpiled_depth, marker="o", label=circuit_id)
    axis.set(xlabel="Noise factor", ylabel="Transpiled depth", xticks=(1, 3, 5))
    axis.set_title("Global folding increases transpiled circuit depth")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "day08_depth_vs_noise_factor.png", dpi=180)
    plt.close(figure)


def generate_day09(*, write_figure: bool = True) -> None:
    circuit = bell_circuit()
    spec = noise_spec("mixed", "medium")
    rows = []
    for repeat in range(20):
        for method_index, method in enumerate(("raw", "readout", "zne")):
            seed = derive_seed("day09", method_index, repeat)
            base = {
                "repeat": repeat,
                "method": method,
                "total_shot_budget": 4096,
                "noise_name": spec.name,
                "seed": seed,
                "analytic_reference": 1.0,
                "result_label": "noisy_simulated",
                "preregistered_model": PREREGISTERED_MODEL if method == "zne" else None,
            }
            if method == "raw":
                result = estimate_pauli(circuit, "XX", shots=4096, seed=seed, noise_spec=spec)
                base.update(estimate=result.estimate, raw_payload_json=result.raw_counts_json)
            elif method == "readout":
                result = run_readout_mitigation(
                    circuit, "XX", total_shot_budget=4096, noise_spec=spec, seed=seed
                )
                base.update(
                    estimate=result.corrected_estimate,
                    raw_payload_json=json.dumps(asdict(result), separators=(",", ":")),
                )
            else:
                result = run_linear_zne(
                    circuit, "XX", total_shot_budget=4096, noise_spec=spec, seed=seed
                )
                base.update(
                    estimate=result.extrapolated_estimate,
                    raw_payload_json=json.dumps(asdict(result), separators=(",", ":")),
                    factor_1_estimate=result.factor_estimates[0],
                    factor_3_estimate=result.factor_estimates[1],
                    factor_5_estimate=result.factor_estimates[2],
                    fit_residual_sum_squares=result.residual_sum_squares,
                    out_of_range=result.out_of_range,
                )
            rows.append(base)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "day09_method_validation_budget-4096_repeats-20.csv", index=False)
    if not write_figure:
        return
    zne = frame[frame.method == "zne"].iloc[0]
    factors = np.array([1, 3, 5])
    estimates = np.array([zne.factor_1_estimate, zne.factor_3_estimate, zne.factor_5_estimate])
    coefficients = np.polyfit(factors, estimates, 1)
    x = np.linspace(0, 5, 100)
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.scatter(factors, estimates, label="factor-level estimates")
    axis.plot(x, np.polyval(coefficients, x), label="preregistered linear fit")
    axis.scatter([0], [zne.estimate], marker="*", s=120, label="zero-noise extrapolate")
    axis.axhline(1.0, color="black", linestyle="--", label="analytic reference")
    axis.set(xlabel="Noise factor", ylabel="XX expectation")
    axis.set_title("Linear ZNE retains factor-level observations")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "day09_zne_extrapolation_lambda-1-3-5.png", dpi=180)
    plt.close(figure)


def generate_day10() -> None:
    build_manifest(repeats=50).to_csv(MANIFESTS / "volumetric_v1_manifest.csv", index=False)


def _plot_rmse_heatmap(results: pd.DataFrame, path: Path) -> None:
    completed = results[results.status == "complete"].copy()
    completed["squared_error"] = (completed.estimate - completed.analytic_reference) ** 2
    table = completed.pivot_table(
        index="method",
        columns=["width", "noise"],
        values="squared_error",
        aggfunc=lambda values: float(np.sqrt(np.mean(values))),
    )
    figure, axis = plt.subplots(figsize=(8.5, 3.6))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xticks(range(len(table.columns)), [f"w{w}\n{n}" for w, n in table.columns])
    axis.set(xlabel="Endpoint cell", ylabel="Method")
    axis.set_title("Pilot RMSE under fixed total shot budgets")
    figure.colorbar(image, ax=axis, label="RMSE")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def generate_day11(*, write_figure: bool = True) -> None:
    manifest = build_manifest(repeats=20)
    pilot = manifest[
        (manifest.depth == 1)
        & manifest.width.isin([2, 6])
        & manifest.noise.isin(["zero", "high"])
        & (manifest.shots == 1024)
    ].reset_index(drop=True)
    results = run_manifest(pilot)
    results.to_csv(RAW / "day11_pilot_repeats-20.csv", index=False)
    audit = coverage_audit(results)
    audit["raw_file_bytes"] = results.to_csv(index=False).encode("utf-8").__len__()
    audit.to_csv(PROCESSED / "day11_pilot_audit.csv", index=False)
    if write_figure:
        _plot_rmse_heatmap(results, FIGURES / "day11_pilot_rmse_heatmap.png")


def generate_day12(*, write_derived: bool = True) -> None:
    manifest = pd.read_csv(MANIFESTS / "volumetric_v1_manifest.csv")
    applicable = manifest[manifest.status != "not_applicable"].reset_index(drop=True)
    results = run_manifest(applicable)
    results.to_csv(RAW / "day12_ghz_volumetric_v1_repeats-50.csv", index=False)
    if not write_derived:
        return
    coverage = coverage_audit(results)
    coverage.to_csv(PROCESSED / "day12_ghz_coverage.csv", index=False)
    completed = results[results.status == "complete"].copy()
    completed["absolute_error"] = (completed.estimate - completed.analytic_reference).abs()
    summary = completed.groupby(["noise", "method"], sort=False).absolute_error.mean().reset_index()
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    order = ["zero", "low", "medium", "high"]
    for method, group in summary.groupby("method", sort=False):
        values = group.set_index("noise").reindex(order).absolute_error
        axis.plot(order, values, marker="o", label=method)
    axis.set(xlabel="Simulated mixed-noise level", ylabel="Mean absolute error")
    axis.set_title("GHZ mitigation accuracy across the frozen noise grid")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "day12_ghz_error_vs_noise.png", dpi=180)
    plt.close(figure)


def main() -> None:
    prepare()
    generate_day08()
    generate_day09()
    generate_day10()
    generate_day11()
    generate_day12()


if __name__ == "__main__":
    main()
