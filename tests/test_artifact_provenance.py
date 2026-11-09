from pathlib import Path

import pandas as pd
import pytest

from src import artifact_provenance as provenance
from src.verify_generated_artifacts import (
    EXPENSIVE_GENERATORS,
    GENERATORS,
    ComparisonFailure,
    compare_artifact_trees,
    load_registry,
    plan_module,
    plan_scope,
    remove_outputs,
)

REGISTRY = load_registry()
OWNERS = {item["module"]: item for item in REGISTRY["generators"]}


def test_every_committed_artifact_has_one_declared_origin() -> None:
    assert provenance.coverage_failures(REGISTRY) == []


def test_every_paper_table_and_figure_can_be_regenerated() -> None:
    assert provenance.paper_failures(REGISTRY) == []


def test_committed_hashes_match_the_tree() -> None:
    assert provenance.check() == []


def test_table_has_the_required_provenance_fields() -> None:
    table = pd.read_csv(provenance.PROVENANCE_PATH, keep_default_na=False)
    assert list(table.columns) == provenance.COLUMNS
    assert table.artifact.is_unique
    assert table.output_sha256.str.fullmatch("[0-9a-f]{64}").all()
    assert table.opt_in_command.str.startswith("python -m src.run_with_log src.").all()


def test_every_expensive_artifact_has_explicit_provenance() -> None:
    table = pd.read_csv(provenance.PROVENANCE_PATH, keep_default_na=False)
    expensive = table.loc[table.tier == "expensive"]
    assert {"src.generate_combined_qem", "src.generate_decision_benchmark"} <= set(
        expensive.generator
    )
    assert expensive.config_sha256.str.len().eq(64).all()
    assert expensive.expected_runtime_seconds.ne("").all()
    assert expensive.runtime_source.ne("").all()
    data = expensive.loc[expensive.artifact.str.endswith(".csv")]
    assert data.expected_rows.ne("").all()
    assert (data.provenance_status != "").all()


def test_all_verifier_generators_are_registered() -> None:
    assert set(GENERATORS) <= set(OWNERS)
    assert set(EXPENSIVE_GENERATORS) <= set(OWNERS)
    assert all(OWNERS[module]["tier"] == "expensive" for module in EXPENSIVE_GENERATORS)


def test_combined_and_decision_scope_deletes_their_committed_results() -> None:
    plan = plan_scope("combined_decision", REGISTRY)
    assert "results/raw/combined_qem_v1.csv" in plan.remove
    assert "results/raw/decision_margin_benchmark_v1.csv" in plan.remove
    assert "results/raw/combined_qem_v1.csv" in plan.compare
    assert "results/processed/claim_ledger_v1.csv" in plan.compare
    assert "results/processed/claim_ledger_v1.csv" not in plan.remove
    modules = [command[0] for command in plan.commands]
    assert modules[-2:] == ["src.verify_final_project", "src.claim_ledger"]


def test_release_scope_rebuilds_everything_except_declared_inputs() -> None:
    plan = plan_scope("release", REGISTRY)
    owned = {path for item in REGISTRY["generators"] for path in item["outputs"]}
    assert plan.compare == owned
    assert not plan.remove & set(REGISTRY["frozen_inputs"])
    modules = [command[0] for command in plan.commands]
    assert ("src.generate_qaoa_selection", "--full") in plan.commands
    assert modules.index("src.generate_combined_qem") < modules.index("src.hardware_readiness")
    assert modules.index("src.generate_decision_benchmark") < modules.index(
        "src.reliability_envelopes"
    )
    assert modules.index("src.final_synthesis") < modules.index("src.claim_ledger")
    assert modules[-1] == "src.generate_audit_manifest"


def test_frozen_inputs_remain_when_a_scope_clears_outputs(tmp_path: Path) -> None:
    plan = plan_scope("expensive", REGISTRY)
    assert "experiments/manifests/qaoa_candidate_pool_v1.csv" not in plan.remove
    kept = tmp_path / "experiments/manifests/qaoa_candidate_pool_v1.csv"
    kept.parent.mkdir(parents=True)
    kept.write_text("frozen", encoding="utf-8")
    removed = tmp_path / "results/raw/selector_holdout_v2.csv"
    removed.parent.mkdir(parents=True)
    removed.write_text("old", encoding="utf-8")
    remove_outputs(tmp_path, plan.remove)
    assert kept.exists()
    assert not removed.exists()


def test_one_generator_jobs_cover_the_release_scope_without_overlap() -> None:
    release = plan_scope("release", REGISTRY)
    split = [
        plan_module("release", module, REGISTRY)
        for module in REGISTRY["scopes"]["release"]["fresh"]
    ]
    removed = [path for plan in split for path in plan.remove]
    assert len(removed) == len(set(removed))
    assert set(removed) == set(release.remove)
    qaoa = plan_module("release", "src.generate_qaoa_selection", REGISTRY)
    assert qaoa.commands == (("src.generate_qaoa_selection", "--full"),)
    assert qaoa.compare == qaoa.remove
    assert not qaoa.remove & set(REGISTRY["frozen_inputs"])


def test_rebuild_workflow_lists_every_generator_outside_routine_ci() -> None:
    workflow = Path(__file__).resolve().parents[1] / ".github/workflows/rebuild-generators.yml"
    text = workflow.read_text(encoding="utf-8")
    outside = [m for m in REGISTRY["scopes"]["release"]["fresh"] if m not in GENERATORS]
    assert outside
    for module in outside:
        assert f"- {module}\n" in text, module


def test_a_module_outside_its_scope_is_rejected() -> None:
    with pytest.raises(ValueError, match="not regenerated"):
        plan_module("combined_decision", "src.generate_signal_v2", REGISTRY)


def test_unknown_scope_generators_are_rejected() -> None:
    registry = {
        "scopes": {"bad": {"fresh": ["src.missing"], "post_checks": []}},
        "generators": [],
        "frozen_inputs": {},
    }
    with pytest.raises(ValueError, match="unregistered"):
        plan_scope("bad", registry)


def test_comparator_reports_a_missing_regenerated_artifact(tmp_path: Path) -> None:
    reference = tmp_path / "reference/results/raw"
    reference.mkdir(parents=True)
    (reference / "a.csv").write_text("x\n1\n", encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    failures = compare_artifact_trees(
        tmp_path / "reference", candidate, {"results/raw/a.csv"}
    )
    assert failures == [
        ComparisonFailure("results/raw/a.csv", "missing from regenerated artifacts")
    ]


def test_comparator_ignores_line_endings_but_not_text(tmp_path: Path) -> None:
    for name, content in (("reference", b"a\nb\n"), ("candidate", b"a\r\nb\r\n")):
        path = tmp_path / name / "paper/generated/t.tex"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    assert compare_artifact_trees(tmp_path / "reference", tmp_path / "candidate") == []
    (tmp_path / "candidate/paper/generated/t.tex").write_bytes(b"a\r\nc\r\n")
    assert compare_artifact_trees(tmp_path / "reference", tmp_path / "candidate")


def test_stale_hash_and_wrong_row_count_are_detected() -> None:
    current = provenance.build_table(REGISTRY)
    changed = current.copy()
    changed.loc[0, "output_sha256"] = "0" * 64
    assert provenance.committed_failures(changed, current)
    wrong = current.copy()
    row = wrong.index[wrong.expected_rows.ne("")][0]
    wrong.loc[row, "expected_rows"] = int(wrong.loc[row, "rows"]) + 1
    assert provenance.row_failures(wrong)


def test_log_match_accepts_either_line_ending(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "a.csv"
    path.write_bytes(b"x\n1\n")
    crlf = hashlib.sha256(b"x\r\n1\r\n").hexdigest()
    assert provenance._matches_log(path, crlf)
    assert not provenance._matches_log(path, "0" * 64)


def test_differences_report_how_many_rows_and_how_far() -> None:
    from src.verify_generated_artifacts import _difference_detail

    left, right = pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.5, 3.0])
    detail = _difference_detail(left, right, [True, False, True], numeric=True)
    assert detail == " in 1 of 3 rows; largest absolute difference 0.5"
    words = _difference_detail(
        pd.Series(["complete", "failed"]), pd.Series(["complete", "complete"]), [True, False], False
    )
    assert words == " in 1 of 2 rows; first at row 1: 'failed' vs 'complete'"
