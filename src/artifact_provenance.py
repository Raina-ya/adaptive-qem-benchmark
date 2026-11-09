"""Record and check where each generated artifact came from.

`--write` updates the committed provenance table. Without it, the module only checks the
committed tree, which is what routine CI does through the test suite. Neither mode runs a
simulation, and the table is deliberately not part of any regeneration scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "experiments/configs/artifact_registry_v1.yaml"
PROVENANCE_PATH = ROOT / "results/processed/artifact_provenance_v1.csv"
RUNS = ROOT / "results/runs"
TRACKED_ROOTS = ("results", "figures", "paper/generated", "paper/figures", "experiments/manifests")
EXTRA_ROOTS = ("experiments/hardware", "docs/day08_12_runtime_report.pdf")
TEXT_SUFFIXES = {".csv", ".md", ".tex", ".sha256", ".yaml"}
COLUMNS = [
    "artifact",
    "generator",
    "tier",
    "opt_in_command",
    "config_sha256",
    "manifest_sha256",
    "output_sha256",
    "rows",
    "expected_rows",
    "expected_runtime_seconds",
    "runtime_source",
    "run_log",
    "run_log_status",
    "run_log_commit",
    "run_log_tree_dirty",
    "provenance_status",
]


def load_registry() -> dict[str, object]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def canonical_bytes(path: Path) -> bytes:
    """Return platform-stable bytes, normalizing line endings of text artifacts only."""

    content = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES:
        return content.replace(b"\r\n", b"\n")
    return content


def sha256_of(paths: list[str]) -> str:
    """Hash files in order; an empty list has no hash."""

    if not paths:
        return ""
    digest = hashlib.sha256()
    for relative in paths:
        digest.update(canonical_bytes(ROOT / relative))
    return digest.hexdigest()


def tracked_artifacts() -> list[str]:
    """List committed generated artifacts, excluding run logs and this table."""

    listed = subprocess.run(
        ["git", "ls-files", *TRACKED_ROOTS, *EXTRA_ROOTS],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    skipped = PROVENANCE_PATH.relative_to(ROOT).as_posix()
    return sorted(
        path for path in listed if not path.startswith("results/runs/") and path != skipped
    )


def _run_logs() -> list[dict[str, object]]:
    logs = []
    for path in sorted(RUNS.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_file"] = path.name
        logs.append(record)
    return logs


def _matches_log(path: Path, logged_hash: str) -> bool:
    """Compare with a logged hash, which may predate Git's line-ending conversion."""

    content = path.read_bytes()
    variants = {content, content.replace(b"\r\n", b"\n")}
    if path.suffix.lower() in TEXT_SUFFIXES:
        variants.add(content.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    return logged_hash in {hashlib.sha256(item).hexdigest() for item in variants}


def _log_fields(artifact: str, module: str, logs: list[dict[str, object]]) -> dict[str, object]:
    """Find the newest run of this generator that logged a hash equal to the committed file."""

    candidates = [
        log for log in logs if log["script"] == module and artifact in log["output_hashes"]
    ]
    empty = {
        "run_log": "",
        "run_log_status": "",
        "run_log_commit": "",
        "run_log_tree_dirty": "",
    }
    if not candidates:
        return {**empty, "provenance_status": "no_run_log"}
    path = ROOT / artifact
    matching = [
        log for log in candidates if _matches_log(path, str(log["output_hashes"][artifact]))
    ]
    chosen = (matching or candidates)[-1]
    if not matching:
        status = "committed_file_differs_from_every_log"
    elif chosen["status"] == "ok":
        status = "matches_successful_run"
    else:
        status = "matches_run_that_later_failed"
    return {
        "run_log": chosen["_file"],
        "run_log_status": chosen["status"],
        "run_log_commit": str(chosen["git_commit_before"] or "")[:7],
        "run_log_tree_dirty": chosen["git_dirty_before"],
        "provenance_status": status,
    }


def _row_count(path: Path) -> object:
    if path.suffix.lower() != ".csv":
        return ""
    return len(pd.read_csv(path, usecols=[0], low_memory=False))


def build_table(registry: dict[str, object] | None = None) -> pd.DataFrame:
    """Return one row per registered output."""

    registry = load_registry() if registry is None else registry
    logs = _run_logs()
    rows = []
    for generator in registry["generators"]:
        arguments = " ".join(generator["args"])
        command = f"python -m src.run_with_log {generator['module']} {arguments}".strip()
        for artifact in generator["outputs"]:
            expected = generator.get("expected_rows", {}).get(artifact, "")
            rows.append(
                {
                    "artifact": artifact,
                    "generator": generator["module"],
                    "tier": generator["tier"],
                    "opt_in_command": command,
                    "config_sha256": sha256_of(generator["configs"]),
                    "manifest_sha256": sha256_of(generator["manifests"]),
                    "output_sha256": sha256_of([artifact]),
                    "rows": _row_count(ROOT / artifact),
                    "expected_rows": expected,
                    "expected_runtime_seconds": generator.get("expected_runtime_seconds", ""),
                    "runtime_source": generator.get("runtime_source", ""),
                    **_log_fields(artifact, generator["module"], logs),
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("artifact", ignore_index=True)


def coverage_failures(registry: dict[str, object] | None = None) -> list[str]:
    """Require every committed artifact to have exactly one declared origin."""

    registry = load_registry() if registry is None else registry
    owned: list[str] = [
        path for generator in registry["generators"] for path in generator["outputs"]
    ]
    declared = {
        *registry["frozen_inputs"],
        *registry["auxiliary_not_regenerated"],
        *registry["legacy_unreferenced"],
    }
    failures = [f"{path}: listed under two generators" for path in owned if owned.count(path) > 1]
    failures += [f"{path}: both generated and declared" for path in set(owned) & declared]
    known = set(owned) | declared
    tracked = set(tracked_artifacts())
    failures += [f"{path}: no declared origin" for path in sorted(tracked - known)]
    failures += [f"{path}: declared but not committed" for path in sorted(known - tracked)]
    return failures


def paper_failures(registry: dict[str, object] | None = None) -> list[str]:
    """Require every table, macro file, and figure in the paper to be regenerable."""

    registry = load_registry() if registry is None else registry
    owned = {path for generator in registry["generators"] for path in generator["outputs"]}
    source = (ROOT / "paper/main_v2.tex").read_text(encoding="utf-8")
    inputs = [f"paper/{name}" for name in re.findall(r"\\input\{([^}]+)\}", source)]
    pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
    figures = [f"paper/figures/{name}" for name in re.findall(pattern, source)]
    return [
        f"{path}: used by the paper but not regenerable"
        for path in inputs + figures
        if path not in owned
    ]


def committed_failures(committed: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    """Compare the committed table with the tree, reporting every stale field."""

    failures: list[str] = []
    if list(committed.columns) != COLUMNS:
        return ["provenance table schema changed"]
    left = committed.set_index("artifact")
    right = current.set_index("artifact")
    for artifact in sorted(set(left.index) ^ set(right.index)):
        failures.append(f"{artifact}: present in only one of the table and the registry")
    for artifact in sorted(set(left.index) & set(right.index)):
        for column in ("config_sha256", "manifest_sha256", "output_sha256", "rows", "generator"):
            if str(left.loc[artifact, column]) != str(right.loc[artifact, column]):
                failures.append(f"{artifact}: {column} changed; rerun with --write if intended")
    return failures


def row_failures(table: pd.DataFrame) -> list[str]:
    """Check observed row counts against the counts declared for expensive artifacts."""

    stated = table.loc[table.expected_rows.notna() & (table.expected_rows != "")]
    return [
        f"{row.artifact}: expected {row.expected_rows} rows, found {row.rows}"
        for row in stated.itertuples()
        if int(float(row.expected_rows)) != int(float(row.rows))
    ]


def check() -> list[str]:
    """Run every routine check without executing any generator."""

    registry = load_registry()
    current = build_table(registry)
    committed = pd.read_csv(PROVENANCE_PATH, keep_default_na=False)
    return [
        *coverage_failures(registry),
        *paper_failures(registry),
        *committed_failures(committed, current),
        *row_failures(current),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Update the committed table")
    arguments = parser.parse_args(argv)
    if arguments.write:
        build_table().to_csv(PROVENANCE_PATH, index=False, lineterminator="\n")
    failures = check()
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"Provenance check: {len(failures)} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
