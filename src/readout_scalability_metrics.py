"""Frozen resource model and manifest for the scalable readout study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import derive_seed
from src.mitigation.readout import constrained_readout_correction

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/manifests/readout_scalability_v1_manifest.csv"
REFERENCE_PATH = ROOT / "results/processed/readout_scalability_reference_v1.csv"

WIDTHS = (2, 4, 6, 8, 10, 12)
DISTRIBUTIONS = ("sparse", "broad")
METHODS = (
    "full_matrix",
    "tensor_product",
    "observed_subspace",
    "matrix_free_subspace",
)
READOUT_MODELS = (
    "independent",
    "asymmetric",
    "pair_correlated",
    "calibration_mismatch",
)
SHOT_BUDGETS = (4096, 16384)
FLOAT_BYTES = 8
FULL_MATRIX_MAX_BYTES = 512 * 1024**2
FULL_MATRIX_MAX_CALIBRATION_CIRCUITS = 1024
FULL_MATRIX_MAX_SOLVE_DIMENSION = 256
SMALL_WIDTH_REPEATS = 50
LARGE_WIDTH_PILOT_REPEATS = 10


@dataclass(frozen=True)
class ReadoutResourceEstimate:
    """Analytic resource requirements before any correction is executed."""

    width: int
    assignment_dimension: int
    dense_entries: int
    dense_matrix_bytes: int
    full_calibration_circuits: int
    tensor_calibration_circuits: int


def resource_estimate(width: int) -> ReadoutResourceEstimate:
    """Return exact dense-matrix and calibration scaling for one width."""

    if width < 1:
        raise ValueError("Width must be positive")
    dimension = 2**width
    return ReadoutResourceEstimate(
        width=width,
        assignment_dimension=dimension,
        dense_entries=dimension**2,
        dense_matrix_bytes=FLOAT_BYTES * dimension**2,
        full_calibration_circuits=dimension,
        tensor_calibration_circuits=2 * width,
    )


def planned_status(method: str, estimate: ReadoutResourceEstimate) -> tuple[str, str]:
    """Apply the frozen pre-execution resource gates."""

    if method not in METHODS:
        raise ValueError(f"Unknown readout method: {method}")
    if method != "full_matrix":
        return "planned", "within_method_specific_gate"
    reasons = []
    if estimate.dense_matrix_bytes > FULL_MATRIX_MAX_BYTES:
        reasons.append("dense_memory_limit")
    if estimate.full_calibration_circuits > FULL_MATRIX_MAX_CALIBRATION_CIRCUITS:
        reasons.append("calibration_circuit_limit")
    if estimate.assignment_dimension > FULL_MATRIX_MAX_SOLVE_DIMENSION:
        reasons.append("solve_dimension_limit")
    if reasons:
        return "resource_limited", ";".join(reasons)
    return "planned", "within_full_matrix_gate"


def build_manifest() -> pd.DataFrame:
    """Build one deterministic row per preregistered scalability cell."""

    rows: list[dict[str, object]] = []
    model_index = {name: index for index, name in enumerate(READOUT_MODELS)}
    distribution_index = {name: index for index, name in enumerate(DISTRIBUTIONS)}
    method_index = {name: index for index, name in enumerate(METHODS)}
    for width in WIDTHS:
        estimate = resource_estimate(width)
        for distribution in DISTRIBUTIONS:
            for readout_model in READOUT_MODELS:
                for shots in SHOT_BUDGETS:
                    for method in METHODS:
                        status, reason = planned_status(method, estimate)
                        rows.append(
                            {
                                "width": width,
                                "distribution": distribution,
                                "readout_model": readout_model,
                                "shots": shots,
                                "method": method,
                                "planned_repeats": (
                                    SMALL_WIDTH_REPEATS
                                    if width <= 6
                                    else LARGE_WIDTH_PILOT_REPEATS
                                ),
                                "validation_role": (
                                    "exact_reference"
                                    if width in (2, 4)
                                    else "small_width_validation"
                                    if width == 6
                                    else "scalability_pilot"
                                ),
                                "status": status,
                                "status_reason": reason,
                                "seed": derive_seed(
                                    "readout_scalability",
                                    width,
                                    distribution_index[distribution],
                                    model_index[readout_model],
                                    shots,
                                    method_index[method],
                                ),
                                **asdict(estimate),
                            }
                        )
    return pd.DataFrame(rows)


def build_small_width_reference() -> pd.DataFrame:
    """Validate the existing full-matrix solver on exact width-2/4 distributions."""

    rows: list[dict[str, object]] = []
    p01, p10 = 0.03, 0.05
    single_qubit = np.array([[1.0 - p01, p10], [p01, 1.0 - p10]])
    for width in (2, 4):
        assignment = single_qubit.copy()
        for _ in range(width - 1):
            assignment = np.kron(single_qubit, assignment)
        for distribution in DISTRIBUTIONS:
            exact = np.full(2**width, 1.0 / 2**width)
            if distribution == "sparse":
                exact.fill(0.0)
                exact[0] = exact[-1] = 0.5
            observed = assignment @ exact
            corrected, condition, status = constrained_readout_correction(
                assignment, observed
            )
            error = (
                float(np.max(np.abs(corrected - exact)))
                if corrected is not None
                else float("nan")
            )
            rows.append(
                {
                    "width": width,
                    "distribution": distribution,
                    "method": "full_matrix",
                    "input_kind": "analytic_probability_vector",
                    "condition_estimate": condition,
                    "maximum_absolute_probability_error": error,
                    "status": status,
                    "result_label": "analytic",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Write the frozen manifest without executing benchmark cells."""

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_manifest().to_csv(MANIFEST_PATH, index=False, lineterminator="\n")
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_small_width_reference().to_csv(
        REFERENCE_PATH, index=False, lineterminator="\n", float_format="%.17g"
    )


if __name__ == "__main__":
    main()
