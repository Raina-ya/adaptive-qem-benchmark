"""Nontrivial circuit-family and distribution-shift benchmark execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import cache

import pandas as pd

from src.benchmark_runner import MANIFEST_COLUMNS, METHODS, coverage_audit
from src.config import DEPTHS, NOISE_LEVELS, SHOT_BUDGETS, WIDTHS, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.mitigation.folding import folded_metadata
from src.mitigation.readout import run_readout_mitigation
from src.mitigation.zne import run_linear_zne
from src.noise_models import NoiseSpec, noise_spec
from src.observables import exact_expectation


@cache
def _circuit_reference(width: int, depth: int) -> tuple[str, str, float]:
    circuit = hardware_efficient_circuit(width, depth)
    observable = hardware_efficient_observable(width)
    return f"he_w{width}_d{depth}", observable, exact_expectation(circuit, observable)


@cache
def _resource_metadata(width: int, depth: int, method: str) -> dict[str, int]:
    circuit = hardware_efficient_circuit(width, depth)
    factors = (1, 3, 5) if method == "zne" else (1,)
    folded = [
        folded_metadata(
            circuit,
            factor,
            seed=derive_seed("adaptive-resource", width, depth, factor),
        )
        for factor in factors
    ]
    return {
        "logical_depth": circuit.depth(),
        "transpiled_depth": max(item["transpiled_depth"] for item in folded),
        "folded_two_qubit_gate_count": sum(
            item["transpiled_two_qubit_gate_count"] for item in folded
        ),
        "target_circuit_executions": len(factors),
    }


def build_hardware_efficient_manifest(repeats: int = 50) -> pd.DataFrame:
    """Create the complete all-applicable hardware-efficient grid."""

    rows: list[dict[str, object]] = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for noise_index, noise in enumerate(NOISE_LEVELS):
                for shots in SHOT_BUDGETS:
                    for method_index, method in enumerate(METHODS):
                        for repeat in range(repeats):
                            rows.append(
                                {
                                    "width": width,
                                    "depth": depth,
                                    "noise": noise,
                                    "shots": shots,
                                    "method": method,
                                    "repeat": repeat,
                                    "status": "pending",
                                    "seed": derive_seed(
                                        "hardware-efficient-volumetric-v1",
                                        width,
                                        depth,
                                        noise_index,
                                        shots,
                                        method_index,
                                        repeat,
                                    ),
                                }
                            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def noise_shift_specs() -> dict[str, NoiseSpec]:
    """Return preregistered holdout channels absent from the design grid."""

    return {
        "intermediate_mixed": NoiseSpec(
            "intermediate_mixed",
            readout_p01=0.0125,
            readout_p10=0.0100,
            one_qubit_depolarizing=0.003,
            two_qubit_depolarizing=0.020,
        ),
        "asymmetric_readout": NoiseSpec(
            "asymmetric_readout", readout_p01=0.040, readout_p10=0.005
        ),
        "coherent_shift": NoiseSpec(
            "coherent_shift",
            readout_p01=0.005,
            readout_p10=0.004,
            coherent_overrotation=0.025,
        ),
        "correlated_shift": NoiseSpec(
            "correlated_shift",
            readout_p01=0.005,
            readout_p10=0.004,
            correlated_zz=0.020,
        ),
    }


def build_noise_shift_manifest(repeats: int = 50) -> pd.DataFrame:
    """Create a bounded out-of-distribution holdout grid."""

    rows: list[dict[str, object]] = []
    for width in WIDTHS:
        for depth in DEPTHS:
            for noise_index, noise in enumerate(noise_shift_specs()):
                for method_index, method in enumerate(METHODS):
                    for repeat in range(repeats):
                        rows.append(
                            {
                                "width": width,
                                "depth": depth,
                                "noise": noise,
                                "shots": 4096,
                                "method": method,
                                "repeat": repeat,
                                "status": "pending",
                                "seed": derive_seed(
                                    "noise-shift-holdout-v1",
                                    width,
                                    depth,
                                    noise_index,
                                    method_index,
                                    repeat,
                                ),
                            }
                        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def execute_hardware_efficient_row(
    row: pd.Series, *, noise_override: NoiseSpec | None = None
) -> dict[str, object]:
    """Execute one row while retaining diagnostics and resource accounting."""

    width = int(row.width)
    depth = int(row.depth)
    circuit_id, observable, reference = _circuit_reference(width, depth)
    resources = _resource_metadata(width, depth, str(row.method))
    output = {
        **row.to_dict(),
        "family": "hardware_efficient",
        "circuit_id": circuit_id,
        "observable_q0_first": observable,
        "noiseless_statevector_reference": reference,
        "result_label": "noisy_simulated",
        "estimate": None,
        "raw_payload_json": None,
        "condition_number": None,
        "fit_residual_sum_squares": None,
        "out_of_range": None,
        "calibration_shots": 0,
        "calibration_circuit_executions": 0,
        **resources,
        "error": None,
    }
    try:
        circuit = hardware_efficient_circuit(width, depth)
        spec = noise_override or noise_spec("mixed", str(row.noise))
        if row.method == "raw":
            result = estimate_pauli(
                circuit,
                observable,
                shots=int(row.shots),
                seed=int(row.seed),
                noise_spec=spec,
            )
            output.update(
                estimate=result.estimate,
                raw_payload_json=result.raw_counts_json,
                transpiled_depth=result.transpiled_depth,
            )
        elif row.method == "readout":
            result = run_readout_mitigation(
                circuit,
                observable,
                total_shot_budget=int(row.shots),
                noise_spec=spec,
                seed=int(row.seed),
            )
            output.update(
                estimate=result.corrected_estimate,
                raw_payload_json=json.dumps(asdict(result), separators=(",", ":")),
                condition_number=result.condition_number,
                calibration_shots=result.calibration_shots,
                calibration_circuit_executions=result.calibration_circuit_executions,
                target_circuit_executions=result.target_circuit_executions,
            )
            if result.status != "ok":
                raise RuntimeError(f"readout mitigation status: {result.status}")
        elif row.method == "zne":
            result = run_linear_zne(
                circuit,
                observable,
                total_shot_budget=int(row.shots),
                noise_spec=spec,
                seed=int(row.seed),
            )
            output.update(
                estimate=result.extrapolated_estimate,
                raw_payload_json=json.dumps(asdict(result), separators=(",", ":")),
                fit_residual_sum_squares=result.residual_sum_squares,
                out_of_range=result.out_of_range,
            )
        else:
            raise ValueError(f"Unknown method: {row.method}")
        output["status"] = "complete"
    except Exception as error:
        output["status"] = "failed"
        output["error"] = f"{type(error).__name__}: {error}"
    output["total_circuit_executions"] = (
        int(output["target_circuit_executions"])
        + int(output["calibration_circuit_executions"])
    )
    return output


def run_hardware_efficient_manifest(
    manifest: pd.DataFrame, *, shifted_noise: bool = False
) -> pd.DataFrame:
    """Execute every manifest row without dropping failures."""

    shifts = noise_shift_specs() if shifted_noise else {}
    return pd.DataFrame(
        execute_hardware_efficient_row(
            row,
            noise_override=shifts[str(row.noise)] if shifted_noise else None,
        )
        for _, row in manifest.iterrows()
    )


def hardware_efficient_coverage(results: pd.DataFrame) -> pd.DataFrame:
    """Return the standard explicit coverage audit."""

    return coverage_audit(results)
