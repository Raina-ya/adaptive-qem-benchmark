"""Runtime-only selector with an explicit no-safe-method output."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/selector_v2.yaml"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
NO_SAFE_OUTPUTS = frozenset(
    {"abstain", "unresolved", "request_more_shots", "recalibrate", "invalid_zne_scaling"}
)
SIGNIFICANCE = float(CONFIG["significance_multiplier"])
STRONG_SIGNAL = float(CONFIG["strong_signal"])
CONDITION_ABSTAIN = float(CONFIG["condition_abstain"])
CONDITION_RECALIBRATE = float(CONFIG["condition_recalibrate"])
PILOT = CONFIG["pilot"]


@dataclass(frozen=True)
class PilotEvidence:
    """Diagnostics available from one repeat's pilot, with no exact reference."""

    e1: float
    se1: float
    e5: float
    se5: float
    condition_number: float


@dataclass(frozen=True)
class SelectionV2:
    """One decision and the rule that produced it."""

    decision: str
    trace: str


def pauli_standard_error(estimate: float, shots: int) -> float:
    """Return the plug-in standard error of a +-1 mean, never below one count's resolution."""

    if shots <= 0:
        raise ValueError("Shots must be positive")
    return math.sqrt(max(1.0 - estimate**2, 1.0 / shots) / shots)


def select_v2(evidence: PilotEvidence) -> SelectionV2:
    """Apply the preregistered rules in order."""

    e1, se1, e5, se5 = evidence.e1, evidence.se1, evidence.e5, evidence.se5
    condition = evidence.condition_number
    if abs(e1) <= SIGNIFICANCE * se1:
        return SelectionV2("request_more_shots", "pilot cannot distinguish the signal from zero")
    if not math.isfinite(condition) or condition >= CONDITION_RECALIBRATE:
        return SelectionV2("recalibrate", "pilot calibration is singular or too ill-conditioned")
    reversed_sign = (
        e1 * e5 < 0.0
        and abs(e1) > SIGNIFICANCE * se1
        and abs(e5) > SIGNIFICANCE * se5
    )
    grew = abs(e5) - abs(e1) > SIGNIFICANCE * math.hypot(se1, se5)
    if reversed_sign or grew:
        return SelectionV2("invalid_zne_scaling", "folding did not reduce the observed signal")
    if abs(e1) >= STRONG_SIGNAL:
        if condition >= CONDITION_ABSTAIN:
            return SelectionV2("abstain", "strong signal but readout calibration lacks support")
        return SelectionV2("readout", "strong signal with supported calibration")
    return SelectionV2("unresolved", "signal is identified but below the supported level")


def pilot_budget(total_shots: int) -> tuple[int, int, int, int]:
    """Return pilot total, calibration, factor-one, and factor-five shots."""

    pilot = int(total_shots * float(PILOT["fraction_of_total_budget"]))
    calibration = int(pilot * float(PILOT["calibration_fraction_of_pilot"]))
    target = pilot - calibration
    first = target - target // 2
    return pilot, calibration, first, target // 2
