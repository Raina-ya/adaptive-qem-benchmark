"""Controlled two-choice quantum decision instances with analytic margins."""

from __future__ import annotations

import math
from dataclasses import dataclass

from qiskit import QuantumCircuit


@dataclass(frozen=True)
class DecisionCandidate:
    """One candidate circuit and its independently known expectation."""

    label: str
    circuit: QuantumCircuit
    observable: str
    exact_expectation: float


@dataclass(frozen=True)
class DecisionInstance:
    """A two-candidate decision whose exact winner is known analytically."""

    margin: float
    candidate_a: DecisionCandidate
    candidate_b: DecisionCandidate


SUPPORTED_MARGINS = (0.02, 0.05, 0.10, 0.20)


def _candidate(label: str, expectation: float, depth: int) -> DecisionCandidate:
    circuit = QuantumCircuit(2, name=f"decision_{label}_{expectation:+.3f}")
    circuit.ry(math.acos(expectation), 0)
    for _ in range(depth):
        circuit.cx(0, 1)
    return DecisionCandidate(label, circuit, "ZI", expectation)


def controlled_decision_instance(margin: float, *, depth: int = 3) -> DecisionInstance:
    """Return a pair with exact gap ``E[A] - E[B] == margin``."""

    if margin not in SUPPORTED_MARGINS:
        raise ValueError(f"Unsupported decision margin: {margin}")
    if depth <= 0:
        raise ValueError("Depth must be positive")
    half = margin / 2.0
    return DecisionInstance(
        margin=margin,
        candidate_a=_candidate("a", half, depth),
        candidate_b=_candidate("b", -half, depth),
    )
