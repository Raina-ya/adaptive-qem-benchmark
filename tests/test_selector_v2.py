import inspect
import math

import numpy as np
import pandas as pd
import pytest

from src.selector_v2 import (
    CONFIG,
    NO_SAFE_OUTPUTS,
    PilotEvidence,
    pauli_standard_error,
    pilot_budget,
    select_v2,
)


def _evidence(**overrides: float) -> PilotEvidence:
    values = {"e1": 0.8, "se1": 0.02, "e5": 0.5, "se5": 0.02, "condition_number": 1.2}
    values.update(overrides)
    return PilotEvidence(**values)


def test_config_is_preregistered_and_has_no_noise_names() -> None:
    assert CONFIG["status"] == "preregistered_before_any_holdout_v2_outcome"
    assert CONFIG["no_noise_names"] is True
    assert CONFIG["simulator_only"] is True
    assert {item["output"] for item in CONFIG["rules_in_order"]} >= NO_SAFE_OUTPUTS


def test_selector_inputs_exclude_references_and_noise_names() -> None:
    fields = set(inspect.signature(PilotEvidence).parameters)
    assert fields == {"e1", "se1", "e5", "se5", "condition_number"}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "readout"),
        ({"e1": 0.03}, "request_more_shots"),
        ({"condition_number": 8.0}, "recalibrate"),
        ({"condition_number": math.inf}, "recalibrate"),
        ({"e5": -0.5}, "invalid_zne_scaling"),
        ({"e5": 0.95}, "invalid_zne_scaling"),
        ({"condition_number": 3.0}, "abstain"),
        ({"e1": 0.3, "e5": 0.2}, "unresolved"),
    ],
)
def test_each_rule_returns_its_frozen_output(overrides: dict[str, float], expected: str) -> None:
    assert select_v2(_evidence(**overrides)).decision == expected


def test_rule_order_checks_signal_before_calibration() -> None:
    assert select_v2(_evidence(e1=0.01, condition_number=50.0)).decision == "request_more_shots"


def test_small_negative_signal_uses_magnitude() -> None:
    assert select_v2(_evidence(e1=-0.8, e5=-0.5)).decision == "readout"


def test_nonsignificant_changes_do_not_flag_invalid_scaling() -> None:
    assert select_v2(_evidence(e5=0.82)).decision == "readout"
    assert select_v2(_evidence(e5=-0.01)).decision == "readout"


def test_pilot_budget_is_charged_inside_the_total() -> None:
    for total in (2048, 8192):
        pilot, calibration, first, fifth = pilot_budget(total)
        assert pilot == total // 4
        assert calibration + first + fifth == pilot
        assert first >= fifth > 0


def test_standard_error_never_collapses_at_extreme_estimates() -> None:
    assert pauli_standard_error(1.0, 100) > 0.0
    assert pauli_standard_error(0.0, 100) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        pauli_standard_error(0.5, 0)


def _real_generator():
    pytest.importorskip("qiskit_aer")
    import src.generate_selector_holdout_v2 as generator

    return generator


def test_manifest_is_fresh_complete_and_unique() -> None:
    generator = _real_generator()
    manifest = generator.build_manifest(repeats=2)
    assert len(manifest) == 10 * 6 * 2 * 4 * 2
    assert manifest.seed.is_unique
    assert set(manifest.shots) == {2048, 8192}
    old_widths = {2, 4, 6}
    assert not set(manifest.width) & old_widths


def test_committed_manifest_matches_the_frozen_grid() -> None:
    generator = _real_generator()
    committed = pd.read_csv(generator.MANIFEST_PATH)
    pd.testing.assert_frame_equal(committed, generator.build_manifest(), check_dtype=False)


def test_replicates_draw_different_frozen_circuits() -> None:
    _real_generator()
    from src.hardware_efficient_circuits import (
        hardware_efficient_circuit,
        parameter_signature,
    )

    first = parameter_signature(hardware_efficient_circuit(3, 2, replicate=0))
    second = parameter_signature(hardware_efficient_circuit(3, 2, replicate=1))
    original = parameter_signature(hardware_efficient_circuit(3, 2))
    assert first != second
    assert first != original
    assert first == parameter_signature(hardware_efficient_circuit(3, 2, replicate=0))


def test_selector_row_charges_exactly_the_declared_budget() -> None:
    generator = _real_generator()
    manifest = generator.build_manifest(repeats=1)
    row = manifest.loc[
        (manifest.circuit_id == "ghz_w3")
        & (manifest.noise == "low")
        & (manifest.shots == 2048)
        & (manifest.method == "selector_v2")
    ].iloc[0]
    circuits = {item[1]: item[5] for item in generator.circuit_grid()}
    output = generator.execute_row(
        row.to_dict(), circuits["ghz_w3"], generator.noise_table()["low"]
    )
    assert output["status"] == "complete"
    assert output["pilot_shots"] == 512
    assert output["total_shots_used"] in {512, 2048}
    if output["decision"] == "readout":
        assert output["total_shots_used"] == 2048
    else:
        assert output["decision"] in NO_SAFE_OUTPUTS
        assert output["total_shots_used"] == 512
    assert not np.isnan(output["pilot_e1"])


def _scored_raw(
    *, cells: int, readout_error: float, raw_error: float, decision: str
) -> pd.DataFrame:
    rows = []
    for index in range(cells):
        base = {
            "family": "ghz",
            "circuit_id": f"c{index}",
            "width": 3,
            "depth": 1,
            "replicate": 0,
            "noise": "low",
            "shots": 2048,
        }
        for repeat in range(4):
            sign = 1.0 if repeat % 2 else -1.0
            for method, error in (("raw", raw_error), ("readout", readout_error), ("zne", 0.5)):
                rows.append(
                    {
                        **base,
                        "method": method,
                        "repeat": repeat,
                        "status": "complete",
                        "decision": None,
                        "estimate": 1.0 + sign * error,
                        "total_shots_used": 2048,
                    }
                )
            rows.append(
                {
                    **base,
                    "method": "selector_v2",
                    "repeat": repeat,
                    "status": "complete",
                    "decision": decision,
                    "estimate": 1.0 + sign * readout_error if decision == "readout" else np.nan,
                    "total_shots_used": 2048 if decision == "readout" else 512,
                }
            )
    return pd.DataFrame(rows)


def test_scoring_marks_reliable_decisions_and_passes_h1_h3_logic() -> None:
    generator = _real_generator()
    raw = _scored_raw(cells=12, readout_error=0.02, raw_error=0.2, decision="readout")
    references = {f"c{index}": 1.0 for index in range(12)}
    cells = generator.score_cells(raw, references)
    assert set(cells.cell_type) == {"decided"}
    assert cells.selector_reliable.all()
    assert cells.raw_catastrophic.all()
    summary = generator.evaluate_criteria(cells).set_index("criterion")
    assert bool(summary.loc["H1_decided_cells_reliable", "passed"]) is True
    assert bool(summary.loc["H2_no_safe_outputs_informative", "passed"]) is False
    assert bool(summary.loc["verdict_all_criteria", "passed"]) is False


def test_scoring_separates_informative_from_unnecessary_no_safe_cells() -> None:
    generator = _real_generator()
    references = {f"c{index}": 1.0 for index in range(12)}
    useful = generator.score_cells(
        _scored_raw(cells=12, readout_error=0.02, raw_error=0.2, decision="unresolved"), references
    )
    assert useful.no_safe_avoided_failure.all()
    wasted = generator.score_cells(
        _scored_raw(cells=12, readout_error=0.2, raw_error=0.02, decision="abstain"), references
    )
    assert not wasted.no_safe_avoided_failure.any()
    summary = generator.evaluate_criteria(useful).set_index("criterion")
    assert bool(summary.loc["H2_no_safe_outputs_informative", "passed"]) is True
    assert bool(summary.loc["H1_decided_cells_reliable", "passed"]) is False
