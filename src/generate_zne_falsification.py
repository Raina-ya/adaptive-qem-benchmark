"""Generate the matched-cost genuine-versus-garbage folding experiment."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.circuits import bell_circuit
from src.config import derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.mitigation.folding import fold_global
from src.mitigation.garbage_folding import garbage_fold
from src.mitigation.zne import linear_extrapolate
from src.noise_models import noise_spec
from src.observable_floors import pauli_observable_floor
from src.observables import exact_expectation
from src.resource_accounting import split_shots
from src.zne_validity import (
    assess_zne_validity,
    extrapolate_probability_distribution,
    probability_vector_from_counts,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results/raw/zne_falsification_v1.csv"
FIGURE_PATH = ROOT / "figures/zne_genuine_vs_garbage_folding.png"
FACTORS = (1, 3, 5)
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


def _folded_result(
    circuit: QuantumCircuit,
    observable: str,
    folder: Callable[[QuantumCircuit, int], QuantumCircuit],
    *,
    budget: int,
    spec,
    seed: int,
) -> dict[str, object]:
    factor_shots = split_shots(budget, len(FACTORS))
    estimates: list[float] = []
    uncertainties: list[float] = []
    counts: list[str] = []
    operation_counts: list[int] = []
    for factor, shots in zip(FACTORS, factor_shots, strict=True):
        folded = folder(circuit, factor)
        result = estimate_pauli(
            folded,
            observable,
            shots=shots,
            seed=derive_seed("zne-falsification-factor", seed, factor),
            noise_spec=spec,
        )
        estimates.append(result.estimate)
        uncertainties.append(result.analytic_standard_error)
        counts.append(result.raw_counts_json)
        operation_counts.append(len(folded.data))
    extrapolated, _, residual = linear_extrapolate(FACTORS, tuple(estimates))
    return {
        "estimate": extrapolated,
        "factor_shots": tuple(factor_shots),
        "factor_estimates": tuple(estimates),
        "factor_uncertainties": tuple(uncertainties),
        "factor_counts": tuple(counts),
        "factor_operation_counts": tuple(operation_counts),
        "fit_residual": residual,
    }


def generate() -> pd.DataFrame:
    """Execute every preregistered raw and folding-control cell."""

    rows: list[dict[str, object]] = []
    methods = (
        ("genuine_folding", fold_global),
        ("garbage_folding", garbage_fold),
    )
    for circuit_index, (circuit_id, circuit, observable) in enumerate(_circuits()):
        reference = exact_expectation(circuit, observable)
        floor = pauli_observable_floor(observable, reference)
        for noise_index, level in enumerate(NOISE_LEVELS):
            spec = noise_spec("mixed", level)
            for budget in SHOT_BUDGETS:
                for repeat in range(REPEATS):
                    seed = derive_seed(
                        "zne-falsification",
                        circuit_index,
                        noise_index,
                        budget,
                        repeat,
                    )
                    raw = estimate_pauli(
                        circuit,
                        observable,
                        shots=budget,
                        seed=derive_seed("zne-falsification-raw", seed),
                        noise_spec=spec,
                    )
                    rows.append(
                        {
                            "circuit_id": circuit_id,
                            "width": circuit.num_qubits,
                            "observable": observable,
                            "noise": level,
                            "total_shot_budget": budget,
                            "repeat": repeat,
                            "seed": seed,
                            "method": "raw",
                            "estimate": raw.estimate,
                            "reference": reference,
                            "apparent_gain": 0.0,
                            "true_gain": 0.0,
                            "factor_shots_json": json.dumps((budget,)),
                            "factor_estimates_json": json.dumps((raw.estimate,)),
                            "factor_uncertainties_json": json.dumps((raw.analytic_standard_error,)),
                            "factor_counts_json": json.dumps((raw.raw_counts_json,)),
                            "factor_operation_counts_json": json.dumps((len(circuit.data),)),
                            "negative_mass": 0.0,
                            "above_one_mass": 0.0,
                            "signal_retention_json": "",
                            "fit_residual": 0.0,
                            "validity_classification": "not_applicable",
                            "result_label": "noisy_simulated",
                            "status": "complete",
                        }
                    )
                    for method, folder in methods:
                        result = _folded_result(
                            circuit,
                            observable,
                            folder,
                            budget=budget,
                            spec=spec,
                            seed=seed,
                        )
                        probabilities = tuple(
                            probability_vector_from_counts(payload, circuit.num_qubits)
                            for payload in result["factor_counts"]
                        )
                        probability_diagnostic = extrapolate_probability_distribution(
                            FACTORS, probabilities
                        )
                        validity = assess_zne_validity(
                            FACTORS,
                            result["factor_estimates"],
                            result["factor_uncertainties"],
                            floor,
                        )
                        estimate = float(result["estimate"])
                        factor_one = float(result["factor_estimates"][0])
                        rows.append(
                            {
                                "circuit_id": circuit_id,
                                "width": circuit.num_qubits,
                                "observable": observable,
                                "noise": level,
                                "total_shot_budget": budget,
                                "repeat": repeat,
                                "seed": seed,
                                "method": method,
                                "estimate": estimate,
                                "reference": reference,
                                "apparent_gain": abs(estimate - factor_one),
                                "true_gain": (
                                    abs(factor_one - reference) - abs(estimate - reference)
                                ),
                                "factor_shots_json": json.dumps(result["factor_shots"]),
                                "factor_estimates_json": json.dumps(result["factor_estimates"]),
                                "factor_uncertainties_json": json.dumps(
                                    result["factor_uncertainties"]
                                ),
                                "factor_counts_json": json.dumps(result["factor_counts"]),
                                "factor_operation_counts_json": json.dumps(
                                    result["factor_operation_counts"]
                                ),
                                "negative_mass": probability_diagnostic.negative_mass,
                                "above_one_mass": probability_diagnostic.above_one_mass,
                                "signal_retention_json": json.dumps(
                                    validity.signal_retention_ratios
                                ),
                                "fit_residual": result["fit_residual"],
                                "validity_classification": (validity.validity_classification),
                                "result_label": "noisy_simulated",
                                "status": "complete",
                            }
                        )
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame) -> None:
    folded = frame.loc[frame.method != "raw"].copy()
    summary = (
        folded.groupby(["method", "noise", "total_shot_budget"], sort=True)
        .agg(
            true_gain_mean=("true_gain", "mean"),
            true_gain_standard_error=(
                "true_gain",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))),
            ),
        )
        .reset_index()
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for axis, budget in zip(axes, SHOT_BUDGETS, strict=True):
        subset = summary.loc[summary.total_shot_budget == budget]
        for method, group in subset.groupby("method"):
            group = group.set_index("noise").loc[list(NOISE_LEVELS)].reset_index()
            axis.errorbar(
                group.noise,
                group.true_gain_mean,
                yerr=group.true_gain_standard_error,
                marker="o",
                label=method.replace("_", " "),
            )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"Total budget={budget}", xlabel="Noise level")
    axes[0].set_ylabel("Mean true gain vs factor 1")
    axes[-1].legend()
    figure.suptitle("Matched-cost garbage folding tests whether extrapolation manufactures gain")
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    frame = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_PATH, index=False)
    _plot(frame)


if __name__ == "__main__":
    main()
