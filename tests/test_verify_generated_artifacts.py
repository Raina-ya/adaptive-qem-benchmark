from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.verify_generated_artifacts import (
    EXPENSIVE_GENERATORS,
    GENERATORS,
    PHASE_F_ARTIFACTS,
    compare_artifact_trees,
)


def _write(root: Path, relative: str, content: str | bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_comparator_regenerates_phase_0_through_day_7() -> None:
    assert "src.generate_artifacts" in GENERATORS


def test_comparator_regenerates_adaptive_extension() -> None:
    assert "src.generate_nontrivial_benchmark" in GENERATORS
    assert "src.generate_adaptive_analysis" in GENERATORS
    assert "src.generate_zne_probability_diagnostics" in GENERATORS
    assert "src.generate_zne_falsification" in GENERATORS
    assert "src.generate_zne_validity_gate" in GENERATORS
    assert "src.readout_scalability_metrics" in GENERATORS
    assert "src.generate_readout_tensor_validation" in GENERATORS
    assert "src.generate_readout_subspace_validation" in GENERATORS
    assert "src.generate_readout_matrix_free_validation" in GENERATORS
    assert "src.calibration_drift" in GENERATORS
    assert "src.sequential_decision" in GENERATORS
    assert "src.regime_labels_v2" in GENERATORS
    assert "src.selector_audit" in GENERATORS
    assert "src.reliability_envelopes" in GENERATORS
    assert "src.generate_readout_scalability_benchmark" in EXPENSIVE_GENERATORS
    assert "src.generate_readout_mismatch_benchmark" in EXPENSIVE_GENERATORS
    assert "src.generate_qaoa_selection" in EXPENSIVE_GENERATORS
    assert "src.generate_selector_holdout_v2" in EXPENSIVE_GENERATORS


def test_phase_f_scope_covers_application_artifacts() -> None:
    assert PHASE_F_ARTIFACTS == {
        "experiments/manifests/qaoa_selection_v1_manifest.csv",
        "experiments/manifests/qaoa_selection_v1_manifest.sha256",
        "figures/qaoa_selection_error.png",
        "figures/qaoa_shots_to_decision.png",
        "figures/qaoa_simple_regret.png",
        "results/processed/qaoa_selection_summary_v1.csv",
        "results/raw/qaoa_selection_v1.csv",
    }


def test_float_and_json_noise_within_tolerance_passes(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write(reference, "results/sample.csv", 'id,value,payload\n1,0.5,"[0.5,0.25]"\n')
    _write(
        candidate,
        "results/sample.csv",
        'id,value,payload\n1,0.5000000001,"[0.5000000001,0.25]"\n',
    )
    _write(reference, "figures/sample.png", b"reference")
    _write(candidate, "figures/sample.png", b"different rendering")

    assert compare_artifact_trees(reference, candidate) == []


def test_real_value_change_fails(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write(reference, "results/sample.csv", "id,value\n1,0.5\n")
    _write(candidate, "results/sample.csv", "id,value\n1,0.6\n")

    failures = compare_artifact_trees(reference, candidate)

    assert failures
    assert failures[0].path == "results/sample.csv"


def test_artifact_filter_ignores_unselected_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    selected = "results/selected.csv"
    _write(reference, selected, "id,value\n1,0.5\n")
    _write(candidate, selected, "id,value\n1,0.5\n")
    _write(reference, "results/unselected.csv", "id,value\n1,0.5\n")
    _write(candidate, "results/unselected.csv", "id,value\n1,0.9\n")

    assert compare_artifact_trees(reference, candidate, {selected}) == []


def test_artifact_filter_reports_unknown_path(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"

    failures = compare_artifact_trees(
        reference, candidate, {"results/does_not_exist.csv"}
    )

    assert failures[0].detail == "requested artifact is absent from both trees"


def test_measured_scalability_timing_may_differ_but_must_be_nonnegative(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    relative = "results/raw/readout_scalability_v1.csv"
    _write(reference, relative, "id,solve_wall_seconds\n1,0.1\n")
    _write(candidate, relative, "id,solve_wall_seconds\n1,0.2\n")

    assert compare_artifact_trees(reference, candidate) == []

    _write(candidate, relative, "id,solve_wall_seconds\n1,-0.2\n")
    failures = compare_artifact_trees(reference, candidate)
    assert failures[0].detail == (
        "measured timing column 'solve_wall_seconds' contains invalid values"
    )


def test_missing_row_and_empty_png_fail(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write(reference, "results/sample.csv", "id,value\n1,0.5\n2,0.6\n")
    _write(candidate, "results/sample.csv", "id,value\n1,0.5\n")
    _write(reference, "figures/sample.png", b"reference")
    _write(candidate, "figures/sample.png", b"")

    details = [failure.detail for failure in compare_artifact_trees(reference, candidate)]

    assert "row count differs: 2 != 1" in details
    assert "PNG is empty" in details


def test_runtime_pdf_is_compared_structurally(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    relative = "docs/day08_12_runtime_report.pdf"
    for root, metadata in ((reference, "one"), (candidate, "two")):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"pdf.compression": 0}):
            from matplotlib.backends.backend_pdf import PdfPages

            with PdfPages(path, metadata={"Title": metadata}) as pdf:
                for page_text in (
                    "Day 8-12 Runtime and Performance Report\nWhere time went",
                    "What this does not include",
                ):
                    figure = plt.figure()
                    figure.text(0.1, 0.5, page_text)
                    pdf.savefig(figure)
                    plt.close(figure)

    assert (reference / relative).read_bytes() != (candidate / relative).read_bytes()
    assert compare_artifact_trees(reference, candidate) == []
