from pathlib import Path

import pandas as pd
from pypdf import PdfWriter

from src.final_synthesis import (
    CRITERIA_PATH,
    MATRIX_PATH,
    SUMMARY_PATH,
    qaoa_selection_counts,
    readiness_criteria,
)
from src.hardware_readiness import MANIFEST_PATH
from src.verify_final_project import (
    _compare,
    figure_checks,
    manifest_checks,
    pdf_checks,
    summary_checks,
)

EVIDENCE_TRACKS = {
    "Expectation accuracy",
    "Decision reliability",
    "ZNE validity",
    "Combined mitigation",
    "Readout scalability",
    "Application case study",
    "Calibration drift",
    "Sequential certification",
    "Hardware readiness",
}


def test_evidence_matrix_has_the_required_shape_and_no_hardware_evidence() -> None:
    matrix = pd.read_csv(MATRIX_PATH, keep_default_na=False)
    assert set(matrix.track) == EVIDENCE_TRACKS
    assert list(matrix.columns) == [
        "track",
        "analytic_evidence",
        "noiseless_simulation",
        "noisy_simulation",
        "hardware_evidence",
        "supported_finding",
        "unresolved_question",
        "limitation",
    ]
    assert set(matrix.hardware_evidence) == {"none"}
    assert (matrix != "").all(axis=None)


def test_review_is_no_go_because_triggers_are_evaluated() -> None:
    criteria = readiness_criteria(pd.read_csv(MANIFEST_PATH))
    assert set(criteria.decision) == {"no_go"}
    triggers = criteria.loc[criteria.criterion_type == "no_go"]
    assert len(triggers) == 6
    assert (triggers.status == "triggered").all()
    assert len(criteria) - len(triggers) == 7
    committed = pd.read_csv(CRITERIA_PATH)
    assert list(committed.status) == list(criteria.status)


def test_final_summary_counts_selections_once_and_reports_no_jobs() -> None:
    assert qaoa_selection_counts() == (828, 2)
    summary = pd.read_csv(SUMMARY_PATH).set_index(["track", "metric"])
    assert summary.loc[("application_case_study", "completed_selection_groups"), "value"] == "828"
    assert summary.loc[("hardware_readiness", "jobs_submitted"), "value"] == "0"
    assert summary.loc[("hardware_readiness", "decision"), "value"] == "no_go"
    assert "hardware_derived" not in set(summary.evidence_label)


def test_manifests_summaries_and_figures_account_for_all_rows() -> None:
    for row in (*manifest_checks(), *summary_checks(), *figure_checks()):
        assert row["passed"], row


def test_comparison_reports_missing_and_changed_values() -> None:
    saved = pd.Series([1.0, 2.0], index=["a", "b"])
    assert _compare("same", saved, saved)["passed"]
    assert not _compare("changed", saved, saved + 1.0)["passed"]
    assert not _compare("missing", saved, saved.iloc[:1])["passed"]


def test_pdf_check_rejects_a_blank_document(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    with path.open("wb") as handle:
        writer.write(handle)
    rows = pdf_checks(path)
    assert rows[0]["passed"]
    assert not all(row["passed"] for row in rows)
