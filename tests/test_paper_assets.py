from pathlib import Path

import pandas as pd

from src.generate_paper_assets import load_core_metrics, write_metrics

ROOT = Path(__file__).resolve().parents[1]


def test_paper_metrics_match_adaptive_benchmark_evidence() -> None:
    metrics = load_core_metrics()
    regimes = pd.read_csv(ROOT / "results/processed/regime_labels_v1.csv")
    summaries = pd.read_csv(ROOT / "results/processed/benchmark_summary_v1.csv")
    hardware = pd.read_csv(
        ROOT / "results/raw/hardware_efficient_v1_repeats-50.csv"
    )

    assert metrics["operating_cells"] == 144
    assert regimes["oracle_label"].value_counts().to_dict() == {
        "practical_tie": 103,
        "readout": 18,
        "zne": 14,
        "inconclusive": 9,
    }
    assert int((hardware["status"] == "failed").sum()) == 99
    assert int((summaries["rmse_gain_vs_raw"] < 0).sum()) == 179


def test_generated_paper_macros_contain_current_selector_results() -> None:
    write_metrics()
    text = (ROOT / "paper/generated/adaptive_metrics.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\HoldoutCompatibility}{41.4\%}" in text
    assert r"\newcommand{\ShiftCompatibility}{50.0\%}" in text
    assert r"\newcommand{\HardwareFailures}{99}" in text
