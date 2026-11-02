import csv

import pandas as pd

from src.benchmark_runner import (
    MANIFEST_COLUMNS,
    METHODS,
    build_manifest,
    coverage_audit,
    execute_row,
)
from src.config import DEPTHS, NOISE_LEVELS, SHOT_BUDGETS, WIDTHS
from src.generate_audit_manifest import MANIFEST_COLUMNS as AUDIT_COLUMNS
from src.generate_audit_manifest import OUTPUT_PATH, ROOT, build_audit_manifest


def test_manifest_contains_complete_declared_grid() -> None:
    manifest = build_manifest(repeats=2)
    expected = len(WIDTHS) * len(DEPTHS) * len(NOISE_LEVELS) * len(SHOT_BUDGETS) * len(METHODS) * 2
    assert len(manifest) == expected
    assert tuple(manifest.columns) == MANIFEST_COLUMNS
    assert not manifest.duplicated(["width", "depth", "noise", "shots", "method", "repeat"]).any()
    assert manifest.seed.is_unique


def test_invalid_ghz_depths_are_explicit_not_applicable() -> None:
    manifest = build_manifest(repeats=1)
    assert (manifest.loc[manifest.depth != 1, "status"] == "not_applicable").all()
    assert (manifest.loc[manifest.depth == 1, "status"] == "pending").all()


def test_coverage_audit_accounts_for_every_row() -> None:
    manifest = build_manifest(repeats=1)
    audit = coverage_audit(manifest)
    assert audit.row_count.sum() == len(manifest)


def test_manifest_row_execution_is_byte_reproducible() -> None:
    row = build_manifest(repeats=1).query(
        "width == 2 and depth == 1 and noise == 'low' and shots == 1024 and method == 'zne'"
    ).iloc[0]
    first = execute_row(row)
    second = execute_row(row)
    assert first == second
    assert "runtime_seconds" not in first


def test_audit_manifest_is_derived_from_repository_artifacts() -> None:
    expected = build_audit_manifest()
    committed = pd.read_csv(OUTPUT_PATH)

    pd.testing.assert_frame_equal(committed, expected, check_dtype=False)
    assert tuple(committed.columns) == AUDIT_COLUMNS
    assert committed.path.is_monotonic_increasing
    assert committed.path.is_unique
    assert "results/raw/day12_ghz_volumetric_v1_repeats-50.csv" in set(committed.path)


def test_audit_manifest_csv_row_counts_match_files() -> None:
    manifest = build_audit_manifest().dropna(subset=["csv_data_rows"])
    for row in manifest.itertuples():
        with (ROOT / row.path).open("r", encoding="utf-8", newline="") as handle:
            actual_rows = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
        assert int(row.csv_data_rows) == actual_rows
