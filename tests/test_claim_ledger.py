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
    matched = computed.verification_status.eq("matches_printed_value")
    assert matched.all(), computed.mismatches.tolist()
    assert computed.evidence_origin.isin(CONFIG["evidence_origins"]).all()
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
    assert judged.evidence_origin.eq("not_applicable").all()
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
        if name.startswith("calibration"):
            return frame.assign(corrected_probability_one=0.0)
        return frame

    monkeypatch.setattr(claim_ledger, "_raw", corrupted)
    assert RECOMPUTE["lifetime"]() != baseline


def test_declared_audit_keeps_its_seed_and_records_the_first_result(ledger: pd.DataFrame) -> None:
    assert CONFIG["audit_seed"] == 20261106
    sample = audit_sample(ledger, int(CONFIG["audit_size"]), int(CONFIG["audit_seed"]))
    assert set(sample.claim_id) == {"C01", "C02", "C03", "C04", "C11"}
    assert sample.mismatches.eq("").all()
    assert (sample.initial_audit_result == "raw").sum() == 3
    assert (sample.initial_audit_result == "processed_artifacts").sum() == 2
    assert sample.final_audit_result.eq("raw").all()
    assert sample.audit_note.str.contains("not the initial one").all()
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


def test_claims_named_in_the_review_have_the_required_origin(ledger: pd.DataFrame) -> None:
    origin = ledger.set_index("claim_id").evidence_origin
    assert origin["C14"] == "processed_only"
    assert origin["C09"] == "manifest_derived"
    assert origin["C15"] == "manifest_derived"
    for claim_id in ("C05", "C13"):
        assert origin[claim_id] == "processed_recomputed"
    trace = ledger.set_index("claim_id").processed_input_trace
    for claim_id in ("C05", "C09", "C13", "C14", "C15"):
        assert trace[claim_id]


def test_reused_derived_fields_cannot_be_called_raw_recomputed(ledger: pd.DataFrame) -> None:
    """A claim whose named metric is a stored column must rebuild it to count as raw."""

    primary_metrics = {
        "C06": "validity_classification",
        "C07": "validity_classification",
        "C10": "certification_status",
        "C05": "validity_classification",
    }
    rows = ledger.set_index("claim_id")
    for claim_id, field in primary_metrics.items():
        assert f"{field}:not_reconstructed" in rows.derived_fields[claim_id]
        assert rows.evidence_origin[claim_id] != "raw_recomputed"
    raw = ledger.loc[ledger.evidence_origin == "raw_recomputed"]
    for name in ("decision_error", "wrong_certified", "true_gain"):
        assert not raw.derived_fields.str.contains(f"{name}:not_reconstructed").any()


def test_rebuilt_derived_fields_match_their_stored_columns() -> None:
    table = claim_ledger.derived_field_table()
    rebuilt = table.loc[table.status == "reconstructed"]
    assert len(rebuilt) == 8
    assert (rebuilt.rows_compared > 0).all()
    assert rebuilt.rows_different.eq(0).all(), rebuilt.to_string()


def test_bootstrap_comparison_is_labeled_as_sensitivity() -> None:
    table = claim_ledger.label_sensitivity()
    assert len(table) == 144
    assert not table.label_changed.any()
    assert set(table.analysis_role) == {"sensitivity_only_bca_is_authoritative"}
    committed = pd.read_csv(claim_ledger.SENSITIVITY_PATH)
    assert list(committed.percentile_label) == list(table.percentile_label)


def test_coverage_summary_states_what_is_not_verified(ledger: pd.DataFrame) -> None:
    summary = claim_ledger.coverage_summary(ledger, provenance_table(ledger))
    values = dict(zip(summary.metric, summary.value, strict=True))
    assert values["computed_claims_checked"] == 23
    assert values["printed_numbers_checked"] == 195
    assert values["generated_tables"] == 9
    assert values["figures"] == 9
    assert values["generated_tables_values_recomputed"] == 0
    assert values["figures_values_recomputed"] == 0
    assert values["paper_independently_verified_as_a_whole"] == "no"


def test_tables_and_figures_are_not_marked_recomputed(ledger: pd.DataFrame) -> None:
    table = provenance_table(ledger)
    visual = table.loc[table.item_kind.isin(("generated_table", "figure"))]
    assert visual.evidence_origin.eq("source_only_not_recomputed").all()
    assert visual.source_located.all()
    assert not visual.values_recomputed.any()


ALLOWED_PREREGISTRATION = (
    "preregistered in the repository",
    "preregistration in the repository",
    "not an independently timestamped preregistration",
    "new preregistration",
    "backend-specific preregistration",
)


def test_preregistration_language_is_qualified() -> None:
    paths = [ROOT / "README.md", ROOT / "paper/main.tex"]
    paths += sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / "paper/generated").glob("*.tex"))
    for path in paths:
        text = " ".join(path.read_text(encoding="utf-8").split()).lower()
        for phrase in ALLOWED_PREREGISTRATION:
            text = text.replace(phrase, "")
        assert "preregist" not in text and "pre-regist" not in text, path.name


def test_every_paper_v2_macro_is_recomputed_except_the_gate_ratio(ledger: pd.DataFrame) -> None:
    table = provenance_table(ledger)
    owned = table.loc[
        (table.item_kind == "macro")
        & table.used_in_paper
        & table.generated_file.eq("paper/generated/boundaries_metrics.tex")
    ].set_index("item")
    claims = ledger.set_index("claim_id").claim_type
    kinds = owned.claim_id.map(claims)
    assert kinds.drop("ThreeCostRatioSelector").eq("computed").all()
    assert kinds["ThreeCostRatioSelector"] == "source_traced_only"


def test_signal_truth_table_agrees_with_normal_theory() -> None:
    assert claim_ledger.signal_truth_deviation() < 0.002


def test_new_audit_claims_do_not_change_the_declared_sample() -> None:
    population = set(CONFIG["audit_population"])
    assert population == {f"C{index:02d}" for index in range(1, 16)}


def test_rewritten_label_rule_reproduces_the_frozen_labels() -> None:
    summary = pd.read_csv(claim_ledger.PROCESSED / "benchmark_summary_v1.csv")
    frozen = pd.read_csv(claim_ledger.PROCESSED / "regime_labels_v2.csv")
    keys = ["family", "circuit_id", "width", "depth", "noise", "shots"]
    rebuilt = [
        claim_ledger._grid_label(group, 0.005, 0.01)
        for _, group in summary.groupby(keys, sort=True)
    ]
    expected = frozen.sort_values(keys).oracle_label.tolist()
    assert rebuilt == expected
