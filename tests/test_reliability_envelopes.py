import math

import numpy as np
import pandas as pd
import pytest

from src.reliability_envelopes import (
    CELLS_PATH,
    CONFIG,
    ENVELOPE_PATH,
    GATE_PATH,
    LEVEL_VALIDATION_PATH,
    VALIDATION_PATH,
    bin_label,
    inside_envelope,
    repeat_bootstrap_rates,
    signal_retention,
    stability_label,
    support_status,
)


def test_config_is_labeled_post_hoc_and_simulator_only() -> None:
    assert CONFIG["status"] == "post_hoc_analysis_after_inspecting_i1_results"
    assert CONFIG["simulator_only"] is True
    assert CONFIG["design_set"] == "rule_design"
    assert CONFIG["support_rule"] == {
        "min_cells": 10,
        "min_reliable_rate": 0.8,
        "min_wilson_low": 0.5,
    }


def test_bins_are_half_open_with_an_unbounded_last_bin() -> None:
    edges = [1.0, 1.5, 3.0, 8.0]
    assert bin_label(1.0, edges) == "[1, 1.5)"
    assert bin_label(1.5, edges) == "[1.5, 3)"
    assert bin_label(8.0, edges) == ">= 8"
    assert bin_label(math.inf, edges) == ">= 8"
    assert bin_label(float("nan"), edges) is None


def test_signal_retention_requires_an_identifiable_baseline() -> None:
    assert signal_retention([0.8, 0.5, 0.4], [1000, 1000, 1000]) == pytest.approx(0.5)
    assert math.isnan(signal_retention([0.01, 0.5, 0.4], [100, 100, 100]))
    assert math.isnan(signal_retention([0.0, 0.5, 0.4], [100, 100, 100]))


def test_support_rule_requires_cells_rate_and_interval() -> None:
    assert support_status(9, 9) == "insufficient_cells"
    assert support_status(10, 10) == "supported"
    assert support_status(10, 7) == "unsupported"
    assert support_status(50, 39) == "unsupported"
    assert support_status(50, 40) == "supported"


def test_stability_labels_use_frozen_frequencies() -> None:
    assert stability_label(0.9) == "stable_supported"
    assert stability_label(0.1) == "stable_unsupported"
    assert stability_label(0.5) == "unstable"


def test_repeat_bootstrap_marks_failed_methods_unreliable() -> None:
    rng = np.random.default_rng(1)
    errors = rng.normal(0.0, 0.01, size=(3, 50))
    errors[1] += 1.0
    errors[2, 0] = np.nan
    rates = repeat_bootstrap_rates(errors, 7)
    assert rates[1] == 0.0
    assert rates[2] == 0.0
    assert rates[0] == pytest.approx(1.0)
    np.testing.assert_array_equal(rates, repeat_bootstrap_rates(errors, 7))


def test_inside_envelope_requires_every_applicable_feature() -> None:
    row = {
        "method": "raw",
        "width": 2,
        "depth": 1,
        "noise": "coherent_shift",
        "shots": 4096,
        "execution_overhead": 1.0,
        "absolute_reference": "[0, 0.06)",
        "condition_number": None,
        "fit_residual": None,
        "signal_retention": None,
    }
    support = {
        ("raw", "width"): {"2"},
        ("raw", "depth"): {"1"},
        ("raw", "noise"): {"low"},
        ("raw", "shots"): {"4096"},
        ("raw", "execution_overhead"): {"1.0"},
        ("raw", "absolute_reference"): {"[0, 0.06)"},
    }
    assert not inside_envelope(row, support)
    assert inside_envelope(row, support, ignore=("noise",))


def test_committed_envelopes_cover_all_requested_dimensions() -> None:
    if not ENVELOPE_PATH.exists():
        pytest.skip("envelope artifacts have not been generated")
    envelopes = pd.read_csv(ENVELOPE_PATH)
    assert set(envelopes.feature) == {
        "width",
        "depth",
        "noise",
        "shots",
        "decision_margin",
        "condition_number",
        "fit_residual",
        "signal_retention",
        "execution_overhead",
        "absolute_reference",
    }
    assert set(envelopes.status) <= {"supported", "unsupported", "insufficient_cells"}
    assert not envelopes.loc[envelopes.method == "raw", "feature"].isin(
        ["condition_number", "fit_residual", "signal_retention"]
    ).any()
    cells = pd.read_csv(CELLS_PATH)
    assert len(cells) == 180 * 3
    assert (cells.loc[cells.complete_repeats < 50, "reliable"] == False).all()  # noqa: E712
    validation = pd.read_csv(VALIDATION_PATH)
    assert set(validation.evaluation_set) == {"holdout", "noise_shift"}
    totals = validation.groupby(["evaluation_set", "method", "variant"]).cells.sum()
    assert set(totals.unique()) == {29, 36}
    levels = pd.read_csv(LEVEL_VALIDATION_PATH)
    assert set(levels.evaluation_set) == {"holdout", "noise_shift"}
    assert (levels.reliable_cells <= levels.cells).all()
    gates = pd.read_csv(GATE_PATH)
    assert gates.bootstrap_above_share.between(0.0, 1.0).all()
