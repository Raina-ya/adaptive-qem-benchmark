"""Frozen, explainable, non-ML QEM selection policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    method: str
    trace: str


def select_qem_method(
    *,
    noise: str,
    shots: int,
    width: int,
    depth: int,
    gate_count: int,
    condition_number: float | None,
    fit_residual: float | None,
) -> Selection:
    """Select from runtime-visible diagnostics without exact-reference access."""

    if condition_number is not None and condition_number > 8.0:
        return Selection("abstain", "calibration condition number exceeds 8")
    if fit_residual is not None and fit_residual > 0.03:
        return Selection("abstain", "ZNE residual exceeds 0.03")
    if noise == "zero":
        return Selection("raw", "declared zero-noise condition needs no mitigation")
    if shots <= 1024 and width >= 6:
        return Selection("raw", "calibration overhead dominates the small wide-circuit budget")
    if noise in {"asymmetric_readout", "low", "medium"} and (
        condition_number is None or condition_number <= 3.0
    ):
        return Selection("readout", "stable calibration and material readout component")
    if noise in {"coherent_shift", "correlated_shift", "high"} and depth >= 3:
        if fit_residual is None or fit_residual <= 0.01:
            return Selection("zne", "deep gate-noise regime with an acceptable ZNE fit")
    if gate_count > 100 and shots < 4096:
        return Selection("inconclusive", "deep circuit lacks enough budget for a stable choice")
    return Selection("raw", "no mitigation rule has sufficient evidence")
