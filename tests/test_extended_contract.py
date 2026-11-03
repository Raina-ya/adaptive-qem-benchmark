import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/extended_benchmark_v1.yaml"


def _config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_extended_contract_configuration_is_frozen_and_parseable() -> None:
    config = _config()
    assert config["version"] == "extended_benchmark_v1"
    assert config["root_seed"] == 20261102
    assert config["decision_margins"] == [0.02, 0.05, 0.10, 0.20]
    assert config["total_shot_budgets"] == [1024, 4096, 16384]
    assert config["confidence_level"] == 0.95
    assert config["reference_tolerance"] == 1e-12


def test_evidence_labels_and_statuses_are_explicit() -> None:
    config = _config()
    assert config["evidence_labels"] == [
        "analytic",
        "noiseless_simulated",
        "noisy_simulated",
        "hardware_derived",
        "inferred_hypothesis",
    ]
    assert set(config["row_statuses"]) == {
        "pending",
        "complete",
        "failed",
        "unresolved",
        "abstained",
        "not_applicable",
    }


def test_resource_and_statistical_safeguards_are_enabled() -> None:
    config = _config()
    required_true = {
        "fixed_budget_required",
        "preserve_repeat_rows",
        "wilson_interval_required",
        "factor_level_zne_records_required",
        "zne_validity_assessment_required",
        "combined_overhead_includes_calibration_and_folding",
        "holdout_rules_frozen_before_evaluation",
        "selector_oracle_features_forbidden",
        "hardware_claim_requires_hardware_derived_rows",
    }
    assert all(config[key] is True for key in required_true)
    assert config["hardware_execution_authorized"] is False
    assert config["machine_learning_selector_authorized"] is False


def test_contract_documents_cover_all_falsifiable_research_tracks() -> None:
    questions = (ROOT / "docs/extended_research_questions.md").read_text(encoding="utf-8")
    contract = (ROOT / "docs/extended_experiment_contract.md").read_text(encoding="utf-8")
    normalized_contract = " ".join(contract.split()).lower()
    for term in (
        "Decision reliability",
        "ZNE validity",
        "Combined mitigation",
        "Readout scalability",
        "Hardware readiness",
        "Falsifier",
    ):
        assert term in questions
    for term in (
        "Fixed total shot budget",
        "Wilson intervals",
        "oracle isolation",
        "matched-cost negative-control",
        "hardware_execution_authorized",
    ):
        assert term.lower() in normalized_contract


def test_extended_definitions_include_primary_metrics_and_boundaries() -> None:
    text = (ROOT / "docs/extended_definitions.md").read_text(encoding="utf-8")
    for term in (
        "Exact decision margin",
        "Estimated decision margin",
        "Decision error",
        "Simple regret",
        "Unresolved decision",
        "Signal retention",
        "Observable floor",
        "Extrapolation artefact",
        "Garbage-folding control",
        "Calibration amortization",
        "Decision compatibility",
        "Cost-aware utility",
        "Hardware readiness",
    ):
        assert term in text
