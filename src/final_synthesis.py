"""Build the readiness review tables, final summary, and evidence matrix."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.hardware_readiness import CONFIG as HARDWARE_CONFIG
from src.hardware_readiness import MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
GENERATED = ROOT / "paper/generated"
SUMMARY_PATH = PROCESSED / "final_project_summary_v1.csv"
MATRIX_PATH = PROCESSED / "final_evidence_matrix_v1.csv"
CRITERIA_PATH = PROCESSED / "hardware_readiness_criteria_v1.csv"
EVIDENCE_ROWS_PATH = GENERATED / "evidence_matrix_rows.tex"
HARDWARE_ROWS_PATH = GENERATED / "hardware_cell_rows.tex"
FINAL_MACROS_PATH = GENERATED / "final_metrics.tex"
REQUIREMENTS = ROOT / "requirements-ci.txt"
CELL_NAMES = {
    "readout_stable": "Readout-dominated",
    "zne_positive_control": "ZNE control",
    "zne_signal_risk": "ZNE signal risk",
    "decision_near_tie": "Near tie",
    "decision_clear": "Clear margin",
    "calibration_reuse": "Calibration reuse",
}


# Only these statuses follow directly from committed data; a person assigned the others.
COMPUTED_CRITERIA = {
    "no unresolved ZNE validity issue in positive controls",
    "selected cells already inconclusive in simulation",
    "ZNE validity controls incomplete",
}


def paper_macros() -> dict[str, str]:
    """Read every generated paper value so the synthesis cannot drift from the paper."""

    values: dict[str, str] = {}
    for name in ("adaptive_metrics.tex", "audit_metrics.tex"):
        text = (GENERATED / name).read_text(encoding="utf-8")
        values.update(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))
    return {key: value.replace("{,}", ",").replace("\\%", "%") for key, value in values.items()}


def qaoa_selection_counts() -> tuple[int, int]:
    """Count completed and certified selections once each, ignoring regime strata."""

    summary = pd.read_csv(PROCESSED / "qaoa_selection_summary_v1.csv")
    overall = summary.loc[summary.candidate_gap_regime == "overall"]
    certified = (overall.certified_probability * overall.resolved_repeats).fillna(0.0).sum()
    return int(overall.resolved_repeats.sum()), int(round(certified))


def readiness_criteria(manifest: pd.DataFrame) -> pd.DataFrame:
    """Evaluate go requirements and no-go triggers from committed evidence."""

    requirements = REQUIREMENTS.read_text(encoding="utf-8").lower()
    has_provider = "ibm" in requirements or "runtime" in requirements
    zne_cells = manifest.loc[manifest.expected_zne_validity != "not_applicable"]
    not_assessed = int((zne_cells.expected_zne_validity == "not_assessed").sum())
    tolerance = float(HARDWARE_CONFIG["decision_error_tolerance"])
    inconclusive = manifest.loc[manifest.max_decision_error_wilson_high > tolerance, "cell_id"]
    needed = int(manifest.required_total_shots.sum())
    zne_table = pd.read_csv(PROCESSED / "zne_validity_v1.csv")
    control = zne_table.loc[(zne_table.circuit_id == "bell_xx") & (zne_table.noise == "low")]
    control_stable = bool((control.validity_status == "credible").all())
    rows = [
        (
            "go",
            "stable local tests and CI",
            "met",
            "CI is green from the QAOA ranking fix onward, and local tests pass",
        ),
        (
            "go",
            "frozen simulator predictions",
            "partial",
            "values are frozen, but only for named simulator noise rather than a backend",
        ),
        (
            "go",
            "falsifiable hardware hypothesis",
            "partial",
            "five cells have falsifiers; calibration reuse depends on unknown drift direction",
        ),
        (
            "go",
            "declared execution budget",
            "not_met",
            f"none declared; plan needs {needed:,} shots",
        ),
        (
            "go",
            "complete metadata capture",
            "not_met",
            "fields are specified, but no provider dependency or capture code exists"
            if not has_provider
            else "provider dependency present but capture is not implemented",
        ),
        (
            "go",
            "no unresolved ZNE validity issue in positive controls",
            "met" if control_stable else "not_met",
            "the positive control is credible at 16,384 shots but rejected at 4,096"
            if not control_stable
            else "the positive control is credible at both budgets",
        ),
        (
            "go",
            "hardware can discriminate between explanations",
            "not_met",
            "device noise cannot be set to the predicted conditions, and one backend confounds "
            "method, device, and drift",
        ),
        (
            "no_go",
            "hardware would only decorate the project",
            "triggered",
            "without a budget and controls, results could not change a stated conclusion",
        ),
        (
            "no_go",
            "simulator prediction is not falsifiable",
            "triggered",
            "the calibration-reuse prediction assumes a drift direction hardware may not have",
        ),
        (
            "no_go",
            "required shot budget unavailable",
            "triggered",
            "no budget or access is declared",
        ),
        (
            "no_go",
            "calibration or drift cannot be documented",
            "triggered",
            "no code records backend properties, calibration time, or queue time",
        ),
        (
            "no_go",
            "selected cells already inconclusive in simulation",
            "triggered" if len(inconclusive) else "clear",
            ", ".join(inconclusive) or "none",
        ),
        (
            "no_go",
            "ZNE validity controls incomplete",
            "triggered" if not_assessed else "clear",
            f"{not_assessed} of {len(zne_cells)} proposed cells lack single-method ZNE assessment",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["criterion_type", "criterion", "status", "evidence"])
    frame["status_basis"] = frame.criterion.map(
        lambda name: "computed_from_artifacts" if name in COMPUTED_CRITERIA else "review_judgment"
    )
    frame["decision_basis"] = "review_judgment"
    blocked = frame.loc[frame.criterion_type == "go", "status"].ne("met").any()
    triggered = frame.loc[frame.criterion_type == "no_go", "status"].eq("triggered").any()
    frame["decision"] = "no_go" if blocked or triggered else "go"
    return frame


def project_summary(manifest: pd.DataFrame, criteria: pd.DataFrame) -> pd.DataFrame:
    """Return the headline value of each completed track with its evidence label."""

    values = paper_macros()
    resolved, certified = qaoa_selection_counts()
    selector = pd.read_csv(PROCESSED / "selector_v2_summary.csv").set_index("criterion")
    selector_passed = bool(selector.loc["verdict_all_criteria", "passed"] in (True, "True"))
    rows = [
        (
            "expectation_accuracy",
            "operating_cells",
            values["OperatingCells"],
            "regime_labels_v2.csv",
        ),
        ("expectation_accuracy", "raw_winners", values["VTwoRawWinners"], "regime_labels_v2.csv"),
        (
            "expectation_accuracy",
            "practical_ties",
            values["VTwoPracticalTies"],
            "regime_labels_v2.csv",
        ),
        (
            "expectation_accuracy",
            "readout_winners",
            values["VTwoReadoutWinners"],
            "regime_labels_v2.csv",
        ),
        ("expectation_accuracy", "zne_winners", values["VTwoZNEWinners"], "regime_labels_v2.csv"),
        (
            "expectation_accuracy",
            "negative_gain_summaries",
            values["NegativeGains"],
            "benchmark_summary_v1.csv",
        ),
        ("zne_validity", "credible_cells", values["ZNECredibleCells"], "zne_validity_v1.csv"),
        ("zne_validity", "assessed_cells", values["ZNEValidityCells"], "zne_validity_v1.csv"),
        (
            "zne_validity",
            "hardware_efficient_credible_cells",
            values["ZNEHardwareCredibleCells"],
            "zne_validity_v1.csv",
        ),
        (
            "combined_mitigation",
            "lowest_rmse_conditions",
            values["CombinedExpectationWins"],
            "combined_qem_summary_v1.csv",
        ),
        (
            "combined_mitigation",
            "validity_rejected_conditions",
            values["CombinedValidityRejected"],
            "combined_qem_validity_v1.csv",
        ),
        (
            "decision_reliability",
            "near_tie_error_min",
            values["CombinedNearTieMinimum"],
            "combined_qem_summary_v1.csv",
        ),
        (
            "decision_reliability",
            "near_tie_error_max",
            values["CombinedNearTieMaximum"],
            "combined_qem_summary_v1.csv",
        ),
        (
            "readout_scalability",
            "mismatch_error_min",
            values["MismatchDecisionMinimum"],
            "readout_mismatch_summary_v1.csv",
        ),
        (
            "readout_scalability",
            "mismatch_error_max",
            values["MismatchDecisionMaximum"],
            "readout_mismatch_summary_v1.csv",
        ),
        (
            "application_case_study",
            "completed_selection_groups",
            resolved,
            "qaoa_selection_summary_v1.csv",
        ),
        (
            "application_case_study",
            "certified_selection_groups",
            certified,
            "qaoa_selection_summary_v1.csv",
        ),
        (
            "calibration_drift",
            "never_recalibrate_abrupt_error",
            values["NeverAbrupt"],
            "calibration_lifetime_v1.csv",
        ),
        (
            "calibration_drift",
            "never_recalibrate_control_error",
            values["NeverRandom"],
            "calibration_lifetime_v1.csv",
        ),
        (
            "calibration_drift",
            "trigger_shot_saving",
            values["TriggerShotSaving"],
            "calibration_lifetime_v1.csv",
        ),
        (
            "sequential_certification",
            "false_certification_corrected_max",
            values["NullWilsonMax"],
            "sequential_decision_v1.csv",
        ),
        (
            "sequential_certification",
            "false_certification_naive_max",
            values["NullNaiveMax"],
            "sequential_decision_v1.csv",
        ),
        (
            "selector",
            "holdout_decision_compatibility",
            values["HoldoutCompatibilityVTwo"],
            "selector_evaluation_v2.csv",
        ),
        ("selector", "selector_harmful_cells", values["SelectorHarm"], "selector_audit_v1.csv"),
        ("selector", "always_raw_harmful_cells", values["AlwaysRawHarm"], "selector_audit_v1.csv"),
        (
            "selector",
            "runtime_selector_all_criteria_passed",
            selector_passed,
            "selector_v2_summary.csv",
        ),
        ("hardware_readiness", "proposed_cells", len(manifest), "hardware_manifest_v1.csv"),
        (
            "hardware_readiness",
            "required_total_shots",
            int(manifest.required_total_shots.sum()),
            "hardware_manifest_v1.csv",
        ),
        (
            "hardware_readiness",
            "decision",
            str(criteria.decision.iloc[0]),
            "hardware_readiness_criteria_v1.csv",
        ),
        (
            "hardware_readiness",
            "jobs_submitted",
            int(HARDWARE_CONFIG["jobs_submitted"]),
            "hardware_readiness_v1.yaml",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["track", "metric", "value", "source"])
    frame["evidence_label"] = "noisy_simulated"
    frame["claim_type"] = "computed"
    frame.loc[frame.metric == "decision", "claim_type"] = "review_judgment"
    review = frame.track == "hardware_readiness"
    frame.loc[review, "evidence_label"] = "review_of_simulated_evidence"
    return frame


def evidence_matrix(criteria: pd.DataFrame) -> pd.DataFrame:
    """Return the final evidence matrix; every row states that no hardware data exists."""

    v = paper_macros()
    resolved, certified = qaoa_selection_counts()
    decision = str(criteria.decision.iloc[0]).replace("_", "-")
    rows = [
        (
            "Expectation accuracy",
            "GHZ and Bell references",
            "statevector references",
            f"{v['OperatingCells']} cells at 50 repeats",
            f"No universal winner: {v['VTwoRawWinners']} raw, {v['VTwoPracticalTies']} ties, "
            f"{v['VTwoReadoutWinners']} readout, {v['VTwoZNEWinners']} ZNE.",
            "Which method wins when the signal is strong and the circuit is new?",
            "Raw wins are mostly near-zero references; one circuit per width and depth.",
        ),
        (
            "Decision reliability",
            "worked counterexample",
            "exact margins",
            "controlled two-candidate decisions",
            f"Lower RMSE does not ensure a better decision; near-tie error is "
            f"{v['CombinedNearTieMinimum']}-{v['CombinedNearTieMaximum']} for combined QEM.",
            "How does this extend beyond two candidates?",
            "Controlled margins on small circuits.",
        ),
        (
            "ZNE validity",
            "observable floors",
            "folding equivalence",
            f"{v['ZNEValidityCells']} gated cells with garbage folding",
            f"Apparent gain is not validated gain: {v['ZNECredibleCells']} of "
            f"{v['ZNEValidityCells']} cells pass, none hardware-efficient.",
            "Does the gate hold for other circuits and fits?",
            "Two circuits, linear fit, and folds 1, 3, and 5.",
        ),
        (
            "Combined mitigation",
            "none",
            "reference values",
            f"{v['CombinedExpectationCells']} conditions plus controls",
            f"Lowest RMSE in {v['CombinedExpectationWins']} of {v['CombinedExpectationCells']} "
            f"conditions, yet all {v['CombinedValidityCells']} validity checks fail.",
            "Are readout and gate errors separable on a device?",
            "Six cells, two budgets, two allocations, higher overhead.",
        ),
        (
            "Readout scalability",
            "resource scaling",
            "small-width references",
            "widths 2 to 12, with mismatch",
            f"Lower memory is not robustness: mismatch error reaches "
            f"{v['MismatchDecisionMinimum']}-{v['MismatchDecisionMaximum']}.",
            "How large is real correlated readout error?",
            "Synthetic channels; correction amplifies sampling spread.",
        ),
        (
            "Application case study",
            "exact MaxCut optimum",
            "ideal candidate energies",
            "4,500 candidate rows",
            f"Lower energy RMSE does not imply better selection; {certified} of {resolved} "
            "completed selections certify.",
            "Does this hold for other graphs and depths?",
            "Three graphs, five frozen candidates, and p=1.",
        ),
        (
            "Calibration drift",
            "none",
            "none",
            f"{v['LifetimeRows']} one-qubit decisions",
            f"Triggered recalibration saves about {v['TriggerShotSaving']} of shots with similar "
            f"error; never recalibrating reaches {v['NeverAbrupt']} under abrupt drift.",
            "What drift does a real device show?",
            "Directed synthetic stress; not a physical lifetime.",
        ),
        (
            "Sequential certification",
            "union bound over looks",
            "exact margins",
            f"{v['SequentialRows']} policy rows",
            f"Corrected looks keep false certification at or below {v['NullWilsonMax']}, versus "
            f"up to {v['NullNaiveMax']} for peeking; bias is not detected.",
            "How can bias be detected before certifying?",
            "Two-qubit circuits, one observable, conservative bounds.",
        ),
        (
            "Hardware readiness",
            "none",
            "none",
            "frozen predictions for six cells",
            f"Review decision: {decision}. No job was submitted.",
            "Can a backend-specific, powered, controlled protocol be justified?",
            "Predictions depend on simulator noise; no backend data.",
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "track",
            "analytic_evidence",
            "noiseless_simulation",
            "noisy_simulation",
            "supported_finding",
            "unresolved_question",
            "limitation",
        ],
    )
    frame.insert(4, "hardware_evidence", "none")
    frame["text_source"] = "manual_interpretation"
    frame["numbers_source"] = "paper_macros"
    frame.loc[frame.track == "Hardware readiness", "numbers_source"] = "review_judgment"
    return frame


def _latex(text: object) -> str:
    return str(text).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def write_paper_rows(matrix: pd.DataFrame, manifest: pd.DataFrame, criteria: pd.DataFrame) -> None:
    """Write paper tables from the same frames that the CSV files hold."""

    def mark(value: str) -> str:
        return "--" if value == "none" else "Y"

    lines = []
    for row in matrix.itertuples():
        marks = " & ".join(
            mark(value)
            for value in (
                row.analytic_evidence,
                row.noiseless_simulation,
                row.noisy_simulation,
                row.hardware_evidence,
            )
        )
        lines.append(
            f"{_latex(row.track)} & {marks} & {_latex(row.supported_finding)} & "
            f"{_latex(row.unresolved_question)} & {_latex(row.limitation)} \\\\"
        )
    EVIDENCE_ROWS_PATH.write_text(
        "\n".join([*lines, r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n"
    )
    cell_lines = []
    for row in manifest.itertuples():

        def value(number: float) -> str:
            return "--" if pd.isna(number) else f"{number:.3f}"

        shots = row.required_total_shots / 1_000_000
        validity = str(row.expected_zne_validity)
        validity = validity.replace("not_applicable", "--").replace("not_assessed", "none")
        cell_lines.append(
            f"{CELL_NAMES[row.cell_id]} & {value(row.predicted_raw)} & "
            f"{value(row.predicted_readout)} & {value(row.predicted_zne)} & "
            f"{value(row.predicted_combined)} & {validity} & {shots:.1f} \\\\"
        )
    HARDWARE_ROWS_PATH.write_text(
        "\n".join([*cell_lines, r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n"
    )
    go = criteria.loc[criteria.criterion_type == "go"]
    triggers = criteria.loc[criteria.criterion_type == "no_go"]
    cells = manifest.set_index("cell_id")
    near_tie = int(cells.loc["decision_near_tie", "required_shots_per_repeat"])
    macros = {
        "HardwareGoMet": int((go.status == "met").sum()),
        "HardwareGoTotal": len(go),
        "HardwareTriggers": int((triggers.status == "triggered").sum()),
        "HardwareTriggerTotal": len(triggers),
        "HardwareTotalShots": f"{manifest.required_total_shots.sum() / 1_000_000:.1f}",
        "HardwareNearTieShots": f"{near_tie:,}".replace(",", "{,}"),
    }
    FINAL_MACROS_PATH.write_text(
        "\n".join(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=True)
    criteria = readiness_criteria(manifest)
    summary = project_summary(manifest, criteria)
    matrix = evidence_matrix(criteria)
    criteria.to_csv(CRITERIA_PATH, index=False, lineterminator="\n")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n")
    matrix.to_csv(MATRIX_PATH, index=False, lineterminator="\n")
    write_paper_rows(matrix, manifest, criteria)


if __name__ == "__main__":
    main()
