import pandas as pd
import pytest

from src.readout_scalability_metrics import (
    DISTRIBUTIONS,
    MANIFEST_PATH,
    METHODS,
    READOUT_MODELS,
    REFERENCE_PATH,
    SHOT_BUDGETS,
    WIDTHS,
    build_manifest,
    build_small_width_reference,
    planned_status,
    resource_estimate,
)


@pytest.mark.parametrize(
    ("width", "dimension", "entries", "bytes_required", "calibration_circuits"),
    [
        (2, 4, 16, 128, 4),
        (4, 16, 256, 2048, 16),
        (6, 64, 4096, 32768, 64),
        (8, 256, 65536, 524288, 256),
        (10, 1024, 1048576, 8388608, 1024),
        (12, 4096, 16777216, 134217728, 4096),
    ],
)
def test_dense_resource_formula_is_exact(
    width: int,
    dimension: int,
    entries: int,
    bytes_required: int,
    calibration_circuits: int,
) -> None:
    estimate = resource_estimate(width)

    assert estimate.assignment_dimension == dimension
    assert estimate.dense_entries == entries
    assert estimate.dense_matrix_bytes == bytes_required
    assert estimate.full_calibration_circuits == calibration_circuits
    assert estimate.tensor_calibration_circuits == 2 * width


def test_resource_estimate_rejects_nonpositive_width() -> None:
    with pytest.raises(ValueError, match="positive"):
        resource_estimate(0)


def test_frozen_full_matrix_gate_is_explicit() -> None:
    assert planned_status("full_matrix", resource_estimate(8))[0] == "planned"
    assert planned_status("full_matrix", resource_estimate(10)) == (
        "resource_limited",
        "solve_dimension_limit",
    )
    assert planned_status("full_matrix", resource_estimate(12)) == (
        "resource_limited",
        "calibration_circuit_limit;solve_dimension_limit",
    )
    assert planned_status("tensor_product", resource_estimate(12))[0] == "planned"


def test_manifest_covers_the_frozen_cell_grid() -> None:
    first = build_manifest()
    second = build_manifest()
    expected = (
        len(WIDTHS)
        * len(DISTRIBUTIONS)
        * len(READOUT_MODELS)
        * len(SHOT_BUDGETS)
        * len(METHODS)
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == expected == 384
    assert first.seed.is_unique
    assert not first.duplicated(
        ["width", "distribution", "readout_model", "shots", "method"]
    ).any()
    assert set(first.status) == {"planned", "resource_limited"}


def test_small_width_reference_and_large_width_pilot_are_preregistered() -> None:
    manifest = build_manifest()

    assert set(manifest.loc[manifest.width.isin([2, 4]), "validation_role"]) == {
        "exact_reference"
    }
    assert set(manifest.loc[manifest.width <= 6, "planned_repeats"]) == {50}
    assert set(manifest.loc[manifest.width >= 8, "planned_repeats"]) == {10}
    limited = manifest.loc[manifest.status == "resource_limited"]
    assert set(limited.method) == {"full_matrix"}
    assert set(limited.width) == {10, 12}


def test_small_width_full_matrix_reference_recovers_exact_probabilities() -> None:
    reference = build_small_width_reference()

    assert len(reference) == 4
    assert set(reference.width) == {2, 4}
    assert set(reference.distribution) == {"sparse", "broad"}
    assert set(reference.status) == {"ok"}
    assert set(reference.result_label) == {"analytic"}
    assert (reference.maximum_absolute_probability_error < 1e-12).all()


def test_committed_ea_artifacts_match_the_deterministic_builders() -> None:
    pd.testing.assert_frame_equal(pd.read_csv(MANIFEST_PATH), build_manifest())
    pd.testing.assert_frame_equal(
        pd.read_csv(REFERENCE_PATH),
        build_small_width_reference(),
        check_exact=False,
        rtol=1e-15,
        atol=1e-18,
    )
