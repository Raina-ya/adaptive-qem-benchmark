"""Regenerate artifacts in isolation and compare them semantically with HEAD."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

RTOL = 1e-9
ATOL = 1e-12
GENERATORS = (
    "src.generate_artifacts",
    "src.regenerate_readout_affected",
    "src.generate_nontrivial_benchmark",
    "src.generate_adaptive_analysis",
    "src.generate_zne_probability_diagnostics",
    "src.generate_zne_falsification",
    "src.generate_zne_validity_gate",
    "src.readout_scalability_metrics",
    "src.generate_readout_tensor_validation",
    "src.generate_readout_subspace_validation",
    "src.generate_readout_matrix_free_validation",
    "src.calibration_drift",
    "src.sequential_decision",
    "src.regime_labels_v2",
    "src.selector_audit",
    "src.reliability_envelopes",
    "src.generate_audit_manifest",
)
EXPENSIVE_GENERATORS = (
    "src.generate_readout_scalability_benchmark",
    "src.generate_readout_mismatch_benchmark",
    "src.generate_qaoa_selection",
)
PHASE_F_ARTIFACTS = {
    "experiments/manifests/qaoa_selection_v1_manifest.csv",
    "experiments/manifests/qaoa_selection_v1_manifest.sha256",
    "figures/qaoa_selection_error.png",
    "figures/qaoa_shots_to_decision.png",
    "figures/qaoa_simple_regret.png",
    "results/processed/qaoa_selection_summary_v1.csv",
    "results/raw/qaoa_selection_v1.csv",
}
ARTIFACT_ROOTS = (
    "results",
    "experiments/manifests",
    "paper/generated",
    "paper/figures",
    "figures",
)
SINGLE_ARTIFACTS = ("docs/day08_12_runtime_report.pdf",)
AUDIT_MANIFEST = "results/processed/day08_12_audit_manifest.csv"
MEASURED_TIMING_COLUMNS = {
    "results/raw/readout_scalability_v1.csv": {"solve_wall_seconds"},
    "results/processed/readout_scalability_summary_v1.csv": {
        "mean_solve_wall_seconds"
    },
}
PDF_REQUIREMENTS = {
    "docs/day08_12_runtime_report.pdf": (
        2,
        3,
        (
            "Day 8-12 Runtime and Performance Report",
            "Where time went",
            "What this does not include",
        ),
    )
}


@dataclass(frozen=True)
class ComparisonFailure:
    """One actionable artifact mismatch."""

    path: str
    detail: str


def _artifact_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for relative_root in ARTIFACT_ROOTS:
        directory = root / relative_root
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and "results/runs" not in path.as_posix():
                paths.add(path.relative_to(root).as_posix())
    for relative in SINGLE_ARTIFACTS:
        if (root / relative).is_file():
            paths.add(relative)
    return paths


def _is_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _values_equal(reference: Any, candidate: Any) -> bool:
    if _is_float(reference) and _is_float(candidate):
        if math.isnan(reference) and math.isnan(candidate):
            return True
        return math.isclose(reference, candidate, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(reference, int) or isinstance(candidate, int):
        return type(reference) is type(candidate) and reference == candidate
    if isinstance(reference, dict) and isinstance(candidate, dict):
        return reference.keys() == candidate.keys() and all(
            _values_equal(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list) and isinstance(candidate, list):
        return len(reference) == len(candidate) and all(
            _values_equal(left, right) for left, right in zip(reference, candidate, strict=True)
        )
    return type(reference) is type(candidate) and reference == candidate


def _json_values_equal(reference: object, candidate: object) -> bool:
    try:
        left = json.loads(str(reference))
        right = json.loads(str(candidate))
    except (json.JSONDecodeError, TypeError):
        return False
    return _values_equal(left, right)


def _compare_audit_manifest(reference: Path, candidate: Path) -> list[str]:
    left = pd.read_csv(reference, dtype=str, keep_default_na=False)
    right = pd.read_csv(candidate, dtype=str, keep_default_na=False)
    required = ["path", "artifact_kind", "size_bytes", "sha256", "csv_data_rows"]
    if list(left.columns) != required or list(right.columns) != required:
        return ["audit manifest schema changed"]
    stable_columns = ["path", "artifact_kind", "csv_data_rows"]
    if not left[stable_columns].equals(right[stable_columns]):
        return ["audit manifest paths, kinds, or CSV row counts changed"]
    return []


def _compare_csv(reference: Path, candidate: Path, relative: str) -> list[str]:
    if relative == AUDIT_MANIFEST:
        return _compare_audit_manifest(reference, candidate)
    left = pd.read_csv(reference)
    right = pd.read_csv(candidate)
    if list(left.columns) != list(right.columns):
        return [f"columns differ: {list(left.columns)} != {list(right.columns)}"]
    if len(left) != len(right):
        return [f"row count differs: {len(left)} != {len(right)}"]

    failures: list[str] = []
    for column in left.columns:
        left_column = left[column]
        right_column = right[column]
        if column in MEASURED_TIMING_COLUMNS.get(relative, set()):
            if not left_column.isna().equals(right_column.isna()):
                failures.append(f"measured timing column {column!r} missing-value pattern differs")
                continue
            finite_left = left_column.dropna().map(
                lambda value: math.isfinite(value) and value >= 0
            )
            finite_right = right_column.dropna().map(
                lambda value: math.isfinite(value) and value >= 0
            )
            if not finite_left.all() or not finite_right.all():
                failures.append(f"measured timing column {column!r} contains invalid values")
            continue
        if pd.api.types.is_integer_dtype(left_column) and pd.api.types.is_integer_dtype(
            right_column
        ):
            equal = left_column.equals(right_column)
        elif pd.api.types.is_numeric_dtype(left_column) and pd.api.types.is_numeric_dtype(
            right_column
        ):
            equal = all(
                math.isclose(float(a), float(b), rel_tol=RTOL, abs_tol=ATOL)
                if not (pd.isna(a) or pd.isna(b))
                else pd.isna(a) and pd.isna(b)
                for a, b in zip(left_column, right_column, strict=True)
            )
        else:
            equal = all(
                (pd.isna(a) and pd.isna(b))
                or a == b
                or _json_values_equal(a, b)
                for a, b in zip(left_column, right_column, strict=True)
            )
        if not equal:
            failures.append(f"column {column!r} differs")
    return failures


def _validate_pdf(path: Path, relative: str) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return ["PDF is missing or empty"]
    minimum, maximum, required_strings = PDF_REQUIREMENTS[relative]
    try:
        reader = PdfReader(path)
        page_count = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as error:  # pypdf exposes multiple parse exception types
        return [f"PDF cannot be parsed: {error}"]
    failures = []
    if not minimum <= page_count <= maximum:
        failures.append(f"PDF page count {page_count} is outside [{minimum}, {maximum}]")
    missing = [value for value in required_strings if value not in text]
    if missing:
        failures.append(f"PDF is missing required strings: {missing}")
    return failures


def compare_artifact_trees(
    reference_root: Path,
    candidate_root: Path,
    only_artifacts: set[str] | None = None,
) -> list[ComparisonFailure]:
    """Return semantic differences between committed and regenerated artifacts."""

    reference_paths = _artifact_paths(reference_root)
    candidate_paths = _artifact_paths(candidate_root)
    failures = []
    if only_artifacts is not None:
        missing_from_both = only_artifacts - (reference_paths | candidate_paths)
        failures.extend(
            ComparisonFailure(path, "requested artifact is absent from both trees")
            for path in sorted(missing_from_both)
        )
        reference_paths &= only_artifacts
        candidate_paths &= only_artifacts
    failures.extend(
        ComparisonFailure(path, "missing from regenerated artifacts")
        for path in sorted(reference_paths - candidate_paths)
    )
    failures.extend(
        ComparisonFailure(path, "unexpected regenerated artifact")
        for path in sorted(candidate_paths - reference_paths)
    )

    for relative in sorted(reference_paths & candidate_paths):
        reference = reference_root / relative
        candidate = candidate_root / relative
        suffix = reference.suffix.lower()
        if suffix == ".png":
            if reference.stat().st_size == 0 or candidate.stat().st_size == 0:
                failures.append(ComparisonFailure(relative, "PNG is empty"))
        elif suffix == ".csv":
            failures.extend(
                ComparisonFailure(relative, detail)
                for detail in _compare_csv(reference, candidate, relative)
            )
        elif suffix == ".pdf" and relative in PDF_REQUIREMENTS:
            failures.extend(
                ComparisonFailure(relative, f"committed {detail}")
                for detail in _validate_pdf(reference, relative)
            )
            failures.extend(
                ComparisonFailure(relative, f"regenerated {detail}")
                for detail in _validate_pdf(candidate, relative)
            )
        elif reference.read_bytes() != candidate.read_bytes():
            failures.append(ComparisonFailure(relative, "binary or text content differs"))
    return failures


def _extract_head(root: Path, destination: Path) -> None:
    completed = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=BytesIO(completed.stdout), mode="r:") as archive:
        archive.extractall(destination, filter="data")


def _copy_run_logs(generated_root: Path, output: Path) -> None:
    source = generated_root / "results/runs"
    if not source.exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        shutil.copy2(path, output / path.name)


def _regenerate(
    root: Path,
    destination: Path,
    run_log_output: Path | None,
    *,
    include_expensive: bool,
) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(destination)],
        check=True,
    )
    environment = os.environ.copy()
    environment.update({"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(destination / ".mpl")})
    try:
        modules = (*GENERATORS, *EXPENSIVE_GENERATORS) if include_expensive else GENERATORS
        for module in modules:
            subprocess.run(
                [sys.executable, "-m", "src.run_with_log", module],
                cwd=destination,
                env=environment,
                check=True,
            )
    finally:
        if run_log_output is not None:
            _copy_run_logs(destination, run_log_output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current",
        action="store_true",
        help="Compare the working tree with HEAD without running generators",
    )
    parser.add_argument(
        "--run-log-output",
        type=Path,
        help="Copy isolated generation run logs to this directory",
    )
    parser.add_argument(
        "--include-expensive",
        action="store_true",
        help="Also regenerate long-running benchmark artifacts",
    )
    parser.add_argument(
        "--only-artifact",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
        help="With --current, compare only this artifact path (repeatable)",
    )
    parser.add_argument(
        "--phase-f",
        action="store_true",
        help="Compare committed and current Phase F artifacts without regeneration",
    )
    parsed = parser.parse_args(argv)
    if parsed.only_artifact and not parsed.current:
        parser.error("--only-artifact requires --current")
    if parsed.phase_f and (parsed.only_artifact or parsed.include_expensive):
        parser.error("--phase-f cannot be combined with artifact or generation filters")
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="qem-artifact-check-") as temporary:
        temporary_root = Path(temporary)
        reference = temporary_root / "reference"
        reference.mkdir()
        _extract_head(root, reference)
        if parsed.current or parsed.phase_f:
            candidate = root
        else:
            candidate = temporary_root / "generated"
            _regenerate(
                root,
                candidate,
                parsed.run_log_output,
                include_expensive=parsed.include_expensive,
            )
        failures = compare_artifact_trees(
            reference,
            candidate,
            PHASE_F_ARTIFACTS
            if parsed.phase_f
            else set(parsed.only_artifact)
            if parsed.only_artifact
            else None,
        )

    if failures:
        for failure in failures:
            print(f"FAIL {failure.path}: {failure.detail}")
        print(f"Artifact verification failed with {len(failures)} difference(s).")
        return 1
    print(f"Artifact verification passed (rtol={RTOL:g}, atol={ATOL:g}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
