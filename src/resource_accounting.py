"""Resource accounting for fixed-shot-budget comparisons."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import READOUT_CALIBRATION_FRACTION


@dataclass(frozen=True)
class ResourceLedger:
    """Shots and circuit executions charged to one estimator result."""

    total_shot_budget: int
    target_shots: int
    calibration_shots: int = 0
    target_circuit_executions: int = 1
    calibration_circuit_executions: int = 0

    def __post_init__(self) -> None:
        if self.total_shot_budget <= 0:
            raise ValueError("Total shot budget must be positive")
        if self.target_shots + self.calibration_shots != self.total_shot_budget:
            raise ValueError("Target and calibration shots must equal the total budget")

    @property
    def total_circuit_executions(self) -> int:
        return self.target_circuit_executions + self.calibration_circuit_executions


def unmitigated_ledger(total_shot_budget: int) -> ResourceLedger:
    """Charge every shot to the target circuit."""

    return ResourceLedger(total_shot_budget, target_shots=total_shot_budget)


def readout_mitigation_ledger(total_shot_budget: int, width: int) -> ResourceLedger:
    """Allocate 20% calibration and 80% target shots deterministically."""

    if width < 1:
        raise ValueError("Width must be positive")
    calibration_shots = int(total_shot_budget * READOUT_CALIBRATION_FRACTION)
    state_count = 2**width
    if calibration_shots < state_count:
        raise ValueError("Budget is too small to sample every calibration state")
    return ResourceLedger(
        total_shot_budget=total_shot_budget,
        target_shots=total_shot_budget - calibration_shots,
        calibration_shots=calibration_shots,
        calibration_circuit_executions=state_count,
    )


def split_shots(total_shots: int, buckets: int) -> tuple[int, ...]:
    """Divide integer shots deterministically, assigning remainders first."""

    if total_shots < buckets or buckets < 1:
        raise ValueError("Each bucket must receive at least one shot")
    quotient, remainder = divmod(total_shots, buckets)
    return tuple(quotient + (index < remainder) for index in range(buckets))
