import hashlib
import json
import os
from pathlib import Path

from src.run_with_log import REQUIRED_FIELDS, run_module_with_log


def _environment_with_module_path(tmp_path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(tmp_path), current)))
    return environment


def test_successful_run_records_matching_output_hashes(tmp_path: Path) -> None:
    (tmp_path / "success_module.py").write_text(
        "from pathlib import Path\n"
        "Path('results').mkdir()\n"
        "Path('figures').mkdir()\n"
        "Path('results/example.csv').write_text('value\\n1\\n', encoding='utf-8')\n"
        "Path('figures/example.txt').write_text('figure', encoding='utf-8')\n",
        encoding="utf-8",
    )

    log_path, returncode = run_module_with_log(
        "success_module", ["--example"], cwd=tmp_path, env=_environment_with_module_path(tmp_path)
    )
    record = json.loads(log_path.read_text(encoding="utf-8"))

    assert returncode == 0
    assert tuple(record) == REQUIRED_FIELDS
    assert record["status"] == "ok"
    assert record["error"] is None
    assert record["args"] == ["--example"]
    assert set(record["output_hashes"]) == {"results/example.csv", "figures/example.txt"}
    for relative_path, expected_hash in record["output_hashes"].items():
        actual_hash = hashlib.sha256((tmp_path / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash


def test_failed_run_is_recorded(tmp_path: Path) -> None:
    (tmp_path / "failed_module.py").write_text(
        "raise RuntimeError('intentional test failure')\n", encoding="utf-8"
    )

    log_path, returncode = run_module_with_log(
        "failed_module", cwd=tmp_path, env=_environment_with_module_path(tmp_path)
    )
    record = json.loads(log_path.read_text(encoding="utf-8"))

    assert returncode != 0
    assert tuple(record) == REQUIRED_FIELDS
    assert record["status"] == "failed"
    assert "intentional test failure" in record["error"]
    assert record["output_hashes"] == {}
