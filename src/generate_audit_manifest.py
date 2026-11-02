"""Generate a deterministic inventory of Day 8-12 repository artifacts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results/processed/day08_12_audit_manifest.csv"
SEARCH_ROOTS = ("results", "figures", "docs", "experiments")
DAY_TOKENS = ("day08", "day09", "day10", "day11", "day12", "day08_12", "volumetric_v1")
MANIFEST_COLUMNS = ("path", "artifact_kind", "size_bytes", "sha256", "csv_data_rows")
TEXT_SUFFIXES = {".bib", ".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


def _canonical_bytes(path: Path) -> bytes:
    """Return platform-stable bytes while preserving binary artifacts exactly."""

    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def _csv_data_rows(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _is_audit_artifact(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if relative == OUTPUT_PATH.relative_to(ROOT) or relative.parts[:2] == ("results", "runs"):
        return False
    name = path.name.lower()
    return any(token in name for token in DAY_TOKENS)


def build_audit_manifest() -> pd.DataFrame:
    """Derive paths, hashes, sizes, and CSV row counts from repository files."""

    rows: list[dict[str, object]] = []
    for root_name in SEARCH_ROOTS:
        directory = ROOT / root_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or not _is_audit_artifact(path):
                continue
            canonical = _canonical_bytes(path)
            rows.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "artifact_kind": path.suffix.lower().lstrip(".") or "file",
                    "size_bytes": len(canonical),
                    "sha256": hashlib.sha256(canonical).hexdigest(),
                    "csv_data_rows": _csv_data_rows(path),
                }
            )
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("path").reset_index(drop=True)
    frame["csv_data_rows"] = frame.csv_data_rows.astype("Int64")
    return frame


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_audit_manifest().to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
