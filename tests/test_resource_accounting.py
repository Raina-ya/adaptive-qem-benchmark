import pytest

from src.resource_accounting import readout_mitigation_ledger, split_shots, unmitigated_ledger


def test_unmitigated_budget_is_exact() -> None:
    ledger = unmitigated_ledger(1024)
    assert ledger.target_shots == ledger.total_shot_budget == 1024
    assert ledger.calibration_shots == 0


def test_readout_budget_counts_calibration() -> None:
    ledger = readout_mitigation_ledger(4096, width=2)
    assert ledger.calibration_shots == 819
    assert ledger.target_shots == 3277
    assert ledger.target_shots + ledger.calibration_shots == 4096
    assert ledger.calibration_circuit_executions == 4


def test_split_shots_is_exact_and_deterministic() -> None:
    assert split_shots(10, 4) == (3, 3, 2, 2)
    with pytest.raises(ValueError):
        split_shots(3, 4)
