"""Controlled, explicitly labeled Qiskit Aer noise models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
    pauli_error,
)


@dataclass(frozen=True)
class NoiseSpec:
    """Independent idealized error probabilities for one simulator condition."""

    name: str
    readout_p01: float = 0.0
    readout_p10: float = 0.0
    one_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0
    coherent_overrotation: float = 0.0
    correlated_zz: float = 0.0


_STRENGTHS = {
    "zero": (0.0, 0.0, 0.0),
    "low": (0.005, 0.001, 0.01),
    "medium": (0.02, 0.005, 0.03),
    "high": (0.05, 0.01, 0.08),
}


def noise_spec(kind: str, level: str) -> NoiseSpec:
    """Return a named readout, gate, or mixed idealized noise condition."""

    if kind not in {"readout", "one_qubit", "two_qubit", "mixed"}:
        raise ValueError(f"Unsupported noise kind: {kind}")
    if level not in _STRENGTHS:
        raise ValueError(f"Unsupported noise level: {level}")
    readout, one_qubit, two_qubit = _STRENGTHS[level]
    return NoiseSpec(
        name=f"{kind}_{level}",
        readout_p01=readout if kind in {"readout", "mixed"} else 0.0,
        readout_p10=(0.8 * readout) if kind in {"readout", "mixed"} else 0.0,
        one_qubit_depolarizing=one_qubit if kind in {"one_qubit", "mixed"} else 0.0,
        two_qubit_depolarizing=two_qubit if kind in {"two_qubit", "mixed"} else 0.0,
    )


def build_noise_model(spec: NoiseSpec) -> NoiseModel:
    """Build an Aer noise model from an explicit specification."""

    probabilities = (
        spec.readout_p01,
        spec.readout_p10,
        spec.one_qubit_depolarizing,
        spec.two_qubit_depolarizing,
        spec.coherent_overrotation,
        spec.correlated_zz,
    )
    if any(value < 0 for value in probabilities):
        raise ValueError("Noise probabilities must be non-negative")
    model = NoiseModel()
    if spec.readout_p01 or spec.readout_p10:
        assignment = [
            [1.0 - spec.readout_p01, spec.readout_p01],
            [spec.readout_p10, 1.0 - spec.readout_p10],
        ]
        model.add_all_qubit_readout_error(ReadoutError(assignment))
    if spec.one_qubit_depolarizing:
        one_qubit_error = depolarizing_error(spec.one_qubit_depolarizing, 1)
    else:
        one_qubit_error = None
    if spec.coherent_overrotation:
        half_angle = spec.coherent_overrotation / 2.0
        unitary = np.array(
            [
                [math.cos(half_angle), -1j * math.sin(half_angle)],
                [-1j * math.sin(half_angle), math.cos(half_angle)],
            ],
            dtype=complex,
        )
        coherent_error = coherent_unitary_error(unitary)
        one_qubit_error = (
            coherent_error
            if one_qubit_error is None
            else one_qubit_error.compose(coherent_error)
        )
    if one_qubit_error is not None:
        model.add_all_qubit_quantum_error(one_qubit_error, ["x", "sx", "h", "rx", "ry", "rz"])

    two_qubit_error = (
        depolarizing_error(spec.two_qubit_depolarizing, 2)
        if spec.two_qubit_depolarizing
        else None
    )
    if spec.correlated_zz:
        correlated_error = pauli_error(
            [("ZZ", spec.correlated_zz), ("II", 1.0 - spec.correlated_zz)]
        )
        two_qubit_error = (
            correlated_error
            if two_qubit_error is None
            else two_qubit_error.compose(correlated_error)
        )
    if two_qubit_error is not None:
        model.add_all_qubit_quantum_error(two_qubit_error, ["cx"])
    return model


def aer_simulator(spec: NoiseSpec | None = None) -> AerSimulator:
    """Return an Aer simulator with the requested noise model."""

    if spec is None:
        return AerSimulator()
    return AerSimulator(noise_model=build_noise_model(spec))
