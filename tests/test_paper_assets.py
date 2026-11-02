from src.generate_paper_assets import load_results, method_summary, regime_winners


def test_paper_metrics_match_frozen_benchmark_evidence() -> None:
    frame = load_results()
    summary = method_summary(frame).set_index("method")
    _, wins = regime_winners(frame)

    assert frame.status.value_counts().to_dict() == {"complete": 5372, "failed": 28}
    assert summary.loc["readout", "rmse"] < summary.loc["zne", "rmse"]
    assert summary.loc["zne", "rmse"] < summary.loc["raw", "rmse"]
    assert wins == {"raw": 9, "readout": 21, "zne": 4}
