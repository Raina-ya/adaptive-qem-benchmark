from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.verify_generated_artifacts import GENERATORS, compare_artifact_trees


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
