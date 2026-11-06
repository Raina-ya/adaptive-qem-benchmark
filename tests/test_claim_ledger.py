import ast
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from src import claim_ledger
from src.claim_ledger import (
    AUDIT_PATH,
    CONFIG,
    LEDGER_PATH,
    PROVENANCE_PATH,
    RECOMPUTE,
    audit_sample,
    build_ledger,
    compare_claim,
    printed_macros,
    provenance_table,
    used_macros,
)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "claim_id",
    "claim_text",
    "claim_type",
    "source_raw_files",
    "processing_script",
    "metric_definition",
    "figure_or_table",
    "applicable_scope",
    "known_counterexample",
    "verification_status",
    "preregistered_before_result",
    "commit_hash",
}
UNSCOPED = {"ZNE", "QEM", "hardware-efficient circuits", "all circuits"}


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    return build_ledger()


def test_every_computed_claim_matches_the_printed_paper(ledger: pd.DataFrame) -> None:
    computed = ledger.loc[ledger.claim_type == "computed"]
    assert len(computed) >= 15
    assert not computed.verification_status.eq("mismatch").any(), computed.mismatches.tolist()
    assert computed.recomputed_values.str.len().gt(0).all()


def test_committed_ledger_is_current(ledger: pd.DataFrame) -> None:
    committed = pd.read_csv(LEDGER_PATH, keep_default_na=False)
    assert list(committed.claim_id) == list(ledger.claim_id)
    assert list(committed.verification_status) == list(ledger.verification_status)
    assert list(committed.recomputed_values) == list(ledger.recomputed_values)


def test_ledger_has_required_fields_and_only_known_types(ledger: pd.DataFrame) -> None:
    assert REQUIRED_COLUMNS <= set(ledger.columns)
    assert set(ledger.claim_type) == set(CONFIG["claim_types"])
    assert ledger.claim_id.is_unique
    assert not ledger.applicable_scope.isin(UNSCOPED).any()
    assert ledger.applicable_scope.str.len().gt(15).all()


def test_judgments_are_never_reported_as_computed(ledger: pd.DataFrame) -> None:
    judged = ledger.loc[ledger.claim_type != "computed"]
    assert judged.verification_status.str.startswith("not_computable_").all()
    assert judged.recomputed_values.eq("").all()
    assert len(ledger.loc[ledger.claim_type == "manual_interpretation"]) == 9
    decision = ledger.loc[ledger.claim_id == "R01"].iloc[0]
    assert decision.claim_type == "review_judgment"


def test_every_macro_in_the_paper_belongs_to_a_claim(ledger: pd.DataFrame) -> None:
    table = provenance_table(ledger)
    macros = table.loc[(table.item_kind == "macro") & table.used_in_paper]
    assert len(macros) == len(used_macros(printed_macros()))
    assert macros.claim_id.ne("").all(), macros.loc[macros.claim_id == "", "item"].tolist()
    figures = table.loc[table.item_kind == "figure"]
    assert figures.generating_script.ne("").all(), figures.item.tolist()
    committed = pd.read_csv(PROVENANCE_PATH, keep_default_na=False)
    assert len(committed) == len(table)


def test_recomputation_shares_no_code_with_the_project() -> None:
    tree = ast.parse(Path(claim_ledger.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.startswith("src") for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("src")


def test_a_changed_printed_value_is_detected() -> None:
    claim = next(item for item in CONFIG["claims"] if item["claim_id"] == "C03")
    computed = {"regimes": RECOMPUTE["regimes"]()}
    macros = printed_macros()
    text = claim_ledger.paper_text()
    assert compare_claim(claim, macros, text, computed)[1] == []
    changed = {**macros, "VTwoRawWinners": "76"}
    assert compare_claim(claim, changed, text, computed)[1]
    assert compare_claim(claim, macros, "", computed)[1]


def test_a_changed_raw_row_changes_the_recomputation(monkeypatch) -> None:
    baseline = RECOMPUTE["lifetime"]()
    original = claim_ledger._raw

    def corrupted(name: str, columns: list[str]) -> pd.DataFrame:
        frame = original(name, columns)
        return frame.assign(decision_error=True) if name.startswith("calibration") else frame

    monkeypatch.setattr(claim_ledger, "_raw", corrupted)
    assert RECOMPUTE["lifetime"]() != baseline


def test_declared_audit_sample_is_repeatable_and_clean(ledger: pd.DataFrame) -> None:
    sample = audit_sample(ledger, int(CONFIG["audit_size"]), int(CONFIG["audit_seed"]))
    assert len(sample) == 5
    assert sample.mismatches.eq("").all()
    committed = pd.read_csv(AUDIT_PATH, keep_default_na=False)
    assert list(committed.claim_id) == list(sample.claim_id)


def test_commit_hashes_exist_when_history_is_available() -> None:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "false":
        pytest.skip("full Git history is unavailable")
    for claim in CONFIG["claims"]:
        commit = claim.get("commit_hash")
        if commit:
            result = subprocess.run(
                ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )
            assert result.returncode == 0, commit
