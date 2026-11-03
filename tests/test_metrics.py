import numpy as np
import pandas as pd

from src.metrics import bca_interval, benchmark_summary, paired_rmse_gain_interval, wilson_interval


def test_bca_interval_is_deterministic_and_contains_mean() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = bca_interval(values, np.mean, seed=7, resamples=200)
    second = bca_interval(values, np.mean, seed=7, resamples=200)
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_paired_gain_uses_matching_repeat_positions() -> None:
    raw = np.array([0.4, 0.3, 0.2, 0.1])
    mitigated = np.array([0.2, 0.1, 0.1, 0.0])
    estimate, lower, upper = paired_rmse_gain_interval(raw, mitigated, seed=8, resamples=200)
    assert estimate > 0.0
    assert lower <= estimate <= upper


def test_wilson_interval_is_bounded() -> None:
    lower, upper = wilson_interval(3, 10)
    assert 0.0 <= lower < 0.3 < upper <= 1.0


def test_summary_keeps_negative_mitigation_gain() -> None:
    rows = []
    for repeat, raw, corrected in ((0, 0.1, 0.4), (1, -0.1, -0.4), (2, 0.0, 0.3)):
        for method, estimate in (("raw", raw), ("readout", corrected), ("zne", corrected)):
            rows.append(
                {
                    "family": "hardware_efficient",
                    "circuit_id": "he_w2_d1",
                    "width": 2,
                    "depth": 1,
                    "noise": "low",
                    "shots": 1024,
                    "method": method,
                    "repeat": repeat,
                    "status": "complete",
                    "estimate": estimate,
                    "noiseless_statevector_reference": 0.0,
                    "total_circuit_executions": 1,
                    "folded_two_qubit_gate_count": 1,
                    "condition_number": np.nan,
                    "fit_residual_sum_squares": np.nan,
                    "out_of_range": False,
                }
            )
    summary = benchmark_summary(pd.DataFrame(rows), resamples=100)
    assert (summary.loc[summary.method != "raw", "rmse_gain_vs_raw"] < 0.0).all()
