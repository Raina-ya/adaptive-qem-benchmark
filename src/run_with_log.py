"""Run a generation module and record compact reproducibility metadata."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.config import ROOT_SEED

PACKAGE_NAMES = ("qiskit", "qiskit-aer", "numpy", "pandas", "matplotlib", "scipy")
REQUIRED_FIELDS = (
    "run_id",
    "script",
    "started_at_utc",
    "finished_at_utc",
    "wall_seconds",
    "python_version",
    "package_versions",
    "git_commit_before",
    "git_dirty_before",
    "git_commit_after",
    "git_dirty_after",
    "host",
    "args",
    "seed_root",
    "output_hashes",
    "status",
    "error",
)


@dataclass(frozen=True)
class _FileState:
    mtime_ns: int
    size: int
    sha256: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_utc(moment: datetime) -> str:
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_outputs(root: Path) -> dict[str, _FileState]:
    snapshot: dict[str, _FileState] = {}
    for directory_name in ("results", "figures"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or (root / "results" / "runs") in path.parents:
                continue
            stat = path.stat()
            snapshot[path.relative_to(root).as_posix()] = _FileState(
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                sha256=_sha256(path),
            )
    return snapshot


def _git_commit(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0 and bool(completed.stdout.strip())


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _log_filename(finished_at: datetime, module: str) -> str:
    timestamp = finished_at.strftime("%Y%m%dT%H%M%S.%fZ")
    script_name = re.sub(r"[^A-Za-z0-9_.-]", "_", module.rsplit(".", 1)[-1])
    return f"{timestamp}_{script_name}.json"


def run_module_with_log(
    module: str,
    args: list[str] | None = None,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[Path, int]:
    """Run ``module`` in a subprocess and return the run-log path and exit code."""

    arguments = list(args or [])
    root = Path.cwd() if cwd is None else Path(cwd)
    started_at = _utc_now()
    started_clock = time.perf_counter()
    before = _snapshot_outputs(root)
    commit_before = _git_commit(root)
    dirty_before = _git_dirty(root)

    completed = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=root,
        env=os.environ.copy() if env is None else env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)

    finished_at = _utc_now()
    after = _snapshot_outputs(root)
    output_hashes = {
        path: state.sha256
        for path, state in after.items()
        if path not in before or before[path] != state
    }
    error = None
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"module exited with code {completed.returncode}"
        error = detail[-4000:]

    record = {
        "run_id": str(uuid.uuid4()),
        "script": module,
        "started_at_utc": _format_utc(started_at),
        "finished_at_utc": _format_utc(finished_at),
        "wall_seconds": round(time.perf_counter() - started_clock, 6),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "git_commit_before": commit_before,
        "git_dirty_before": dirty_before,
        "git_commit_after": _git_commit(root),
        "git_dirty_after": _git_dirty(root),
        "host": platform.node(),
        "args": arguments,
        "seed_root": ROOT_SEED,
        "output_hashes": output_hashes,
        "status": "ok" if completed.returncode == 0 else "failed",
        "error": error,
    }
    assert tuple(record) == REQUIRED_FIELDS

    run_directory = root / "results" / "runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    log_path = run_directory / _log_filename(finished_at, module)
    log_path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return log_path, completed.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Python module path to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the module")
    parsed = parser.parse_args(argv)
    log_path, returncode = run_module_with_log(parsed.module, parsed.args)
    print(f"run_log={log_path.as_posix()}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
