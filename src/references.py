"""Reference-table generation for both circuit families."""

from __future__ import annotations

import pandas as pd

from src.circuits import circuit_metadata, ghz_circuit, ghz_observables
from src.config import DEPTHS, WIDTHS
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.observables import exact_expectation


def ghz_reference_table() -> pd.DataFrame:
    """Return analytic and statevector cross-checks for GHZ stabilizers."""

    rows: list[dict[str, object]] = []
    for width in WIDTHS:
        circuit = ghz_circuit(width)
        metadata = circuit_metadata(circuit)
        for observable in ghz_observables(width):
            rows.append(
                {
                    "family": "ghz",
                    "circuit_id": f"ghz_w{width}",
                    "width": width,
                    "depth_parameter": 1,
                    "observable_q0_first": observable,
                    "analytic_reference": 1.0,
                    "noiseless_statevector_reference": exact_expectation(circuit, observable),
                    "result_label": "analytic_and_noiseless_simulated",
                    **metadata,
                }
            )
    return pd.DataFrame(rows)


def hardware_efficient_reference_table() -> pd.DataFrame:
    """Return exact statevector references for deterministic HE circuits."""

    rows: list[dict[str, object]] = []
    for width in WIDTHS:
        observable = hardware_efficient_observable(width)
        for depth in DEPTHS:
            circuit = hardware_efficient_circuit(width, depth)
            rows.append(
                {
                    "family": "hardware_efficient",
                    "circuit_id": f"he_w{width}_d{depth}",
                    "width": width,
                    "depth_parameter": depth,
                    "observable_q0_first": observable,
                    "noiseless_statevector_reference": exact_expectation(circuit, observable),
                    "result_label": "noiseless_simulated",
                    **circuit_metadata(circuit),
                }
            )
    return pd.DataFrame(rows)
