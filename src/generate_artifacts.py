"""Generate the deterministic Phase 0 through readout-mitigation artifacts."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.circuits import bell_circuit
from src.config import NOISE_LEVELS, SHOT_BUDGETS, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.minimal_example import run_minimal_example
from src.mitigation.readout import run_readout_mitigation
from src.noise_models import noise_spec
from src.references import ghz_reference_table, hardware_efficient_reference_table

RESULTS = Path("results")
FIGURES = Path("figures")


def _prepare_directories() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def generate_minimal_example() -> None:
    rows = [run_minimal_example(repeat=repeat) for repeat in range(20)]
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "day02_bell_xx_shots-1024_seed-20261102.csv", index=False)

    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.hist(frame["estimate"], bins=np.linspace(0.9, 1.01, 12), color="#2474b5")
    axis.axvline(1.0, color="#b22222", linestyle="--", label="analytic reference")
    axis.set(xlabel="Estimated XX expectation", ylabel="Repeat count")
    axis.set_title("Bell-state XX estimates, 1,024 shots per repeat")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "day02_bell_xx_estimate_distribution.png", dpi=180)
    plt.close(figure)


def generate_reference_tables() -> None:
    ghz_reference_table().to_csv(
        RESULTS / "day03_ghz_references_widths-2-4-6.csv", index=False
    )
    hardware_efficient_reference_table().to_csv(
        RESULTS / "day04_he_references_w-2-4-6_d-1-3-6_seed-20261102.csv", index=False
    )


def generate_noise_response() -> None:
    rows: list[dict[str, object]] = []
    circuit = bell_circuit()
    for kind_index, kind in enumerate(("readout", "one_qubit", "two_qubit", "mixed")):
        for level_index, level in enumerate(NOISE_LEVELS):
            spec = noise_spec(kind, level)
            result = estimate_pauli(
                circuit,
                "XX",
                shots=4096,
                seed=derive_seed("noise-response", kind_index, level_index),
                noise_spec=spec,
            )
            rows.append(
                {
                    "noise_kind": kind,
                    "noise_level": level,
                    **asdict(spec),
                    **asdict(result),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "day05_noise_response_bell_xx_shots-4096.csv", index=False)

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(NOISE_LEVELS))
    for kind, group in frame.groupby("noise_kind", sort=False):
        ordered = group.set_index("noise_level").loc[list(NOISE_LEVELS)]
        axis.plot(x, ordered["estimate"], marker="o", label=kind)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="exact reference")
    axis.set_xticks(x, NOISE_LEVELS)
    axis.set(xlabel="Simulated noise level", ylabel="XX expectation")
    axis.set_title("Bell-state response to idealized simulator noise")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURES / "day05_error_vs_noise_by_family.png", dpi=180)
    plt.close(figure)


def generate_unmitigated_results() -> None:
    rows: list[dict[str, object]] = []
    circuit = bell_circuit()
    spec = noise_spec("mixed", "medium")
    for budget_index, shots in enumerate(SHOT_BUDGETS):
        for repeat in range(30):
            result = estimate_pauli(
                circuit,
                "XX",
                shots=shots,
                seed=derive_seed("unmitigated-artifact", budget_index, repeat),
                noise_spec=spec,
            )
            rows.append(
                {
                    "repeat": repeat,
                    "total_shot_budget": shots,
                    "target_shots": shots,
                    "calibration_shots": 0,
                    "noise_name": spec.name,
                    **asdict(result),
                }
            )
    pd.DataFrame(rows).to_csv(
        RESULTS / "day06_unmitigated_shots-1024-4096-16384_repeats-30.csv", index=False
    )


def generate_readout_mitigation_results() -> None:
    rows: list[dict[str, object]] = []
    spec = noise_spec("readout", "medium")
    for repeat in range(30):
        result = run_readout_mitigation(
            bell_circuit(),
            "XX",
            total_shot_budget=4096,
            noise_spec=spec,
            seed=derive_seed("readout-artifact", repeat),
        )
        rows.append({"repeat": repeat, "noise_name": spec.name, **asdict(result)})
    frame = pd.DataFrame(rows)
    frame.to_csv(
        RESULTS / "day07_readout_mitigation_budget-4096_repeats-30.csv", index=False
    )
    frame[
        [
            "repeat",
            "noise_name",
            "seed",
            "calibration_shots",
            "calibration_circuit_executions",
            "condition_number",
            "status",
            "calibration_counts_json",
            "assignment_matrix_json",
        ]
    ].to_csv(RESULTS / "day07_readout_calibration_budget-4096_repeats-30.csv", index=False)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.plot(frame["repeat"], frame["raw_estimate"], "o", label="raw")
    axis.plot(frame["repeat"], frame["corrected_estimate"], "o", label="corrected")
    axis.axhline(1.0, color="black", linestyle="--", label="exact reference")
    axis.set(xlabel="Independent repeat", ylabel="XX expectation", ylim=(0.8, 1.02))
    axis.set_title("Readout mitigation under a fixed 4,096-shot budget")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "day07_readout_raw_vs_corrected.png", dpi=180)
    plt.close(figure)


def main() -> None:
    _prepare_directories()
    generate_minimal_example()
    generate_reference_tables()
    generate_noise_response()
    generate_unmitigated_results()
    generate_readout_mitigation_results()


if __name__ == "__main__":
    main()
