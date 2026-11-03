from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative)


def _assert_manifest_accounting(prefix: str, raw_name: str) -> None:
    manifest = _read(f"experiments/manifests/{prefix}_manifest.csv")
    raw = _read(f"results/raw/{raw_name}")
    coverage = _read(f"results/processed/{prefix}_coverage.csv")

    key = ["width", "depth", "noise", "shots", "method", "repeat", "seed"]
    assert len(raw) == len(manifest)
    assert not manifest.duplicated(key).any()
    assert not raw.duplicated(key).any()
    assert raw["seed"].is_unique
    assert set(raw["status"]) <= {"complete", "failed", "not_applicable"}
    assert int(coverage["row_count"].sum()) == len(raw)

    expected = manifest[key].sort_values(key).reset_index(drop=True)
    actual = raw[key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_hardware_efficient_manifest_and_coverage_account_for_every_row() -> None:
    _assert_manifest_accounting(
        "hardware_efficient_v1", "hardware_efficient_v1_repeats-50.csv"
    )


def test_noise_shift_manifest_and_coverage_account_for_every_row() -> None:
    _assert_manifest_accounting(
        "noise_shift_holdout_v1", "noise_shift_holdout_v1_repeats-50.csv"
    )


def test_committed_summary_metrics_recompute_from_raw_rows() -> None:
    raw = _read("results/raw/hardware_efficient_v1_repeats-50.csv")
    summary = _read("results/processed/benchmark_summary_v1.csv")
    summary = summary[summary["family"] == "hardware_efficient"]
    complete = raw[raw["status"] == "complete"].copy()
    complete["error"] = complete["estimate"] - complete["noiseless_statevector_reference"]

    group = ["family", "circuit_id", "width", "depth", "noise", "shots", "method"]
    recomputed = (
        complete.groupby(group, sort=True)["error"]
        .agg(
            complete_repeats="size",
            bias="mean",
            variance=lambda values: values.var(ddof=1),
            rmse=lambda values: np.sqrt(np.mean(np.square(values))),
        )
        .reset_index()
    )
    merged = summary.merge(recomputed, on=group, suffixes=("_saved", "_new"))
    assert len(merged) == len(summary)
    for column in ["bias", "variance", "rmse"]:
        np.testing.assert_allclose(
            merged[f"{column}_saved"], merged[f"{column}_new"], rtol=1e-9, atol=1e-12
        )
    assert (merged["complete_repeats_saved"] == merged["complete_repeats_new"]).all()


def test_selector_evaluation_does_not_mix_rule_design_and_holdout() -> None:
    design = _read("results/processed/selector_rule_design_predictions.csv")
    holdout = _read("results/processed/selector_holdout_evaluation.csv")
    assert set(design["split"]) == {"rule_design"}
    assert set(holdout["split"]) == {"holdout"}
    assert len(holdout) >= 20
