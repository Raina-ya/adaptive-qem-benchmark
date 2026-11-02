from pathlib import Path

from src.verify_generated_artifacts import compare_artifact_trees


def _write(root: Path, relative: str, content: str | bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


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
