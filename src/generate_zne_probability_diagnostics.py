"""Generate repeat-level ZNE signed-probability diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.circuits import bell_circuit
from src.config import derive_seed
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.noise_models import noise_spec
from src.observable_floors import pauli_observable_floor
from src.observables import exact_expectation
from src.zne_probability_diagnostics import run_distribution_zne
from src.zne_validity import (
    assess_zne_validity,
    extrapolate_probability_distribution,
    probability_vector_from_counts,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results/raw/zne_probability_diagnostics_v1_repeats-50.csv"
PROCESSED_PATH = ROOT / "results/processed/zne_probability_diagnostics_v1.csv"
FIGURE_PATH = ROOT / "figures/zne_negative_mass_vs_gain.png"
NOISE_LEVELS = ("zero", "low", "medium", "high")
SHOT_BUDGETS = (4096, 16384)
REPEATS = 50


def _circuits():
    return (
        ("bell_xx", bell_circuit(), "XX"),
        (
            "hardware_efficient_w2_d3_xz",
            hardware_efficient_circuit(2, 3),
            hardware_efficient_observable(2),
        ),
    )


def generate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the frozen C3 grid and return raw factor and processed repeat rows."""

    raw_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for circuit_index, (circuit_id, circuit, observable) in enumerate(_circuits()):
        reference = exact_expectation(circuit, observable)
        floor = pauli_observable_floor(observable, reference)
        for noise_index, level in enumerate(NOISE_LEVELS):
            spec = noise_spec("mixed", level)
            for budget in SHOT_BUDGETS:
                for repeat in range(REPEATS):
                    seed = derive_seed(
                        "zne-probability-diagnostics",
                        circuit_index,
                        noise_index,
                        budget,
                        repeat,
                    )
                    result = run_distribution_zne(
                        circuit,
                        observable,
                        total_shot_budget=budget,
                        noise_spec=spec,
                        seed=seed,
                    )
                    probabilities = tuple(
                        probability_vector_from_counts(payload, circuit.num_qubits)
                        for payload in result.factor_counts_json
                    )
                    probability_diagnostic = extrapolate_probability_distribution(
                        result.factors, probabilities
                    )
                    validity = assess_zne_validity(
                        result.factors,
                        result.factor_estimates,
                        result.factor_standard_errors,
                        floor,
                    )
                    for factor, shots, estimate, uncertainty, counts in zip(
                        result.factors,
                        result.shots_by_factor,
                        result.factor_estimates,
                        result.factor_standard_errors,
                        result.factor_counts_json,
                        strict=True,
                    ):
                        raw_rows.append(
                            {
                                "circuit_id": circuit_id,
                                "width": circuit.num_qubits,
                                "observable": observable,
                                "noise": level,
                                "total_shot_budget": budget,
                                "repeat": repeat,
                                "seed": seed,
                                "noise_factor": factor,
                                "factor_shots": shots,
                                "factor_estimate": estimate,
                                "factor_standard_error": uncertainty,
                                "raw_counts_json": counts,
                                "reference": reference,
                                "result_label": "noisy_simulated",
                                "status": "complete",
                            }
                        )
                    scale_one_error = abs(result.factor_estimates[0] - reference)
                    extrapolated_error = abs(result.extrapolated_estimate - reference)
                    diagnostic_rows.append(
                        {
                            "circuit_id": circuit_id,
                            "width": circuit.num_qubits,
                            "observable": observable,
                            "noise": level,
                            "total_shot_budget": budget,
                            "repeat": repeat,
                            "seed": seed,
                            "reference": reference,
                            "scale_one_estimate": result.factor_estimates[0],
                            "extrapolated_estimate": result.extrapolated_estimate,
                            "apparent_gain": scale_one_error - extrapolated_error,
                            "negative_mass": probability_diagnostic.negative_mass,
                            "above_one_mass": probability_diagnostic.above_one_mass,
                            "normalization_error": probability_diagnostic.normalization_error,
                            "unphysical_mass_flag": (probability_diagnostic.unphysical_mass_flag),
                            "extrapolation_weights_json": json.dumps(
                                probability_diagnostic.extrapolation_weights,
                                separators=(",", ":"),
                            ),
                            "extrapolated_probabilities_json": json.dumps(
                                probability_diagnostic.extrapolated_probabilities,
                                separators=(",", ":"),
                            ),
                            "factor_estimates_json": json.dumps(
                                result.factor_estimates, separators=(",", ":")
                            ),
                            "factor_uncertainties_json": json.dumps(
                                result.factor_standard_errors, separators=(",", ":")
                            ),
                            "signal_retention_json": json.dumps(
                                validity.signal_retention_ratios, separators=(",", ":")
                            ),
                            "fit_residual": result.fit_residual,
                            "validity_classification": validity.validity_classification,
                            "result_label": "noisy_simulated",
                            "status": "complete",
                        }
                    )
    return pd.DataFrame(raw_rows), pd.DataFrame(diagnostic_rows)


def _plot(frame: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    markers = {"bell_xx": "o", "hardware_efficient_w2_d3_xz": "s"}
    colors = {"zero": "#4c78a8", "low": "#72b7b2", "medium": "#f58518", "high": "#e45756"}
    for (circuit_id, noise), group in frame.groupby(["circuit_id", "noise"]):
        axis.scatter(
            group.apparent_gain,
            group.negative_mass,
            marker=markers[circuit_id],
            color=colors[noise],
            alpha=0.55,
            s=24,
            label=f"{circuit_id}, {noise}",
        )
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="mass threshold")
    axis.axvline(0.0, color="grey", linestyle=":", linewidth=1)
    axis.set_yscale("symlog", linthresh=1e-5, linscale=0.5)
    axis.set(
        xlabel="Apparent absolute-error gain vs factor 1",
        ylabel="Total negative probability mass",
        title="Apparent ZNE gain can coexist with unphysical signed mass",
    )
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    raw, diagnostics = generate()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False)
    diagnostics.to_csv(PROCESSED_PATH, index=False)
    _plot(diagnostics)


if __name__ == "__main__":
    main()
