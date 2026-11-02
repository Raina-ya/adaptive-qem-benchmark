"""Controlled, explicitly labeled Qiskit Aer noise models."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error


@dataclass(frozen=True)
class NoiseSpec:
    """Independent idealized error probabilities for one simulator condition."""

    name: str
    readout_p01: float = 0.0
    readout_p10: float = 0.0
    one_qubit_depolarizing: float = 0.0
    two_qubit_depolarizing: float = 0.0


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
        model.add_all_qubit_quantum_error(
            one_qubit_error, ["x", "sx", "h", "rx", "ry", "rz"]
        )
    if spec.two_qubit_depolarizing:
        model.add_all_qubit_quantum_error(
            depolarizing_error(spec.two_qubit_depolarizing, 2), ["cx"]
        )
    return model


def aer_simulator(spec: NoiseSpec | None = None) -> AerSimulator:
    """Return an Aer simulator with the requested noise model."""

    if spec is None:
        return AerSimulator()
    return AerSimulator(noise_model=build_noise_model(spec))
