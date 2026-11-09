"""Runtime-only selector: readout by default, deviate on strong evidence, else abstain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DRIFT_LIMIT = 3.0


@dataclass(frozen=True)
class Rule:
    """One setting of the frozen selector grid."""

    z_separate: float
    z_deviate: float
    allow_raw: bool

    @property
    def name(self) -> str:
        raw = "raw" if self.allow_raw else "noraw"
        return f"sep{self.z_separate:g}_dev{self.z_deviate:g}_{raw}"


def choose_branch(
    pilot_raw: np.ndarray,
    pilot_raw_se: np.ndarray,
    pilot_readout: np.ndarray,
    pilot_fit: np.ndarray,
    pilot_fit_se: np.ndarray,
    rule: Rule,
) -> np.ndarray:
    """Return the branch per run: 0 readout, 1 combined, 2 raw.

    Combined needs both pilot contenders to show a resolved rise in magnitude after
    extrapolation. Raw needs correction to be unresolved on every candidate.
    """

    runs = len(pilot_readout)
    order = np.argsort(-pilot_readout, axis=1, kind="stable")
    top = order[:, :2]
    index = np.arange(runs)[:, None]
    shift = np.abs(pilot_fit[index, top]) - np.abs(pilot_readout[index, top])
    strong = np.all(shift >= rule.z_deviate * pilot_fit_se[index, top], axis=1)
    branch = np.where(strong, 1, 0)
    if rule.allow_raw:
        unresolved = np.all(np.abs(pilot_readout - pilot_raw) < pilot_raw_se, axis=1)
        branch = np.where(unresolved & ~strong, 2, branch)
    return branch


def apply_rule(
    data: dict[str, np.ndarray], rule: Rule
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return branch, chosen candidate, whether the run abstains, and the abstain reason code."""

    branch = choose_branch(
        data["pilot_raw"],
        data["pilot_raw_se"],
        data["pilot_readout"],
        data["pilot_fit"],
        data["pilot_fit_se"],
        rule,
    )
    runs = len(branch)
    estimates = np.stack([data["branch_readout"], data["branch_combined"], data["branch_raw"]])
    errors = np.stack(
        [data["branch_readout_se"], data["branch_combined_se"], data["branch_raw_se"]]
    )
    index = np.arange(runs)
    chosen_estimates = estimates[branch, index]
    chosen_errors = errors[branch, index]
    pick = np.argmax(chosen_estimates, axis=1)
    ordered = np.sort(chosen_estimates, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    runner = np.argsort(chosen_estimates, axis=1)[:, -2]
    joint = np.hypot(chosen_errors[index, pick], chosen_errors[index, runner])
    drifted = data["drift_statistic"] >= DRIFT_LIMIT
    unseparated = gap < rule.z_separate * joint
    abstain = drifted | unseparated
    reason = np.where(drifted, 1, np.where(unseparated, 2, 0))
    return branch, pick, abstain, reason
