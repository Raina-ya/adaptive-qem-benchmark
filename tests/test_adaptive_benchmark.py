import pandas as pd

from src.adaptive_benchmark import (
    build_hardware_efficient_manifest,
    build_noise_shift_manifest,
    execute_hardware_efficient_row,
    noise_shift_specs,
)
from src.benchmark_runner import MANIFEST_COLUMNS, METHODS
from src.config import DEPTHS, NOISE_LEVELS, SHOT_BUDGETS, WIDTHS
from src.noise_models import build_noise_model


def test_hardware_efficient_manifest_is_complete_and_all_applicable() -> None:
    manifest = build_hardware_efficient_manifest(repeats=2)
    expected = len(WIDTHS) * len(DEPTHS) * len(NOISE_LEVELS) * len(SHOT_BUDGETS) * len(METHODS) * 2
    assert len(manifest) == expected
    assert tuple(manifest.columns) == MANIFEST_COLUMNS
    assert manifest.status.eq("pending").all()
    assert manifest.seed.is_unique


def test_noise_shift_manifest_uses_new_conditions_and_seeds() -> None:
    manifest = build_noise_shift_manifest(repeats=2)
    assert set(manifest.noise) == set(noise_shift_specs())
    assert set(manifest.shots) == {4096}
    assert manifest.seed.is_unique
    for spec in noise_shift_specs().values():
        assert build_noise_model(spec) is not None


def test_hardware_efficient_row_keeps_reference_and_resources() -> None:
    row = pd.Series(
        {
            "width": 2,
            "depth": 1,
            "noise": "low",
            "shots": 1024,
            "method": "raw",
            "repeat": 0,
            "status": "pending",
            "seed": 1234,
        }
    )
    result = execute_hardware_efficient_row(row)
    assert result["status"] == "complete"
    assert result["circuit_id"] == "he_w2_d1"
    assert result["family"] == "hardware_efficient"
    assert result["total_circuit_executions"] == 1
    assert -1.0 <= result["noiseless_statevector_reference"] <= 1.0
