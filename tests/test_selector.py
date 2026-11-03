import pandas as pd

from src.regime_map import assign_holdout, build_regime_labels
from src.selector import select_qem_method


def test_selector_returns_explainable_abstention() -> None:
    selection = select_qem_method(
        noise="high",
        shots=4096,
        width=4,
        depth=3,
        gate_count=40,
        condition_number=9.0,
        fit_residual=0.001,
    )
    assert selection.method == "abstain"
    assert "condition" in selection.trace


def test_selector_signature_has_no_exact_reference() -> None:
    import inspect

    assert "reference" not in inspect.signature(select_qem_method).parameters


def test_holdout_and_labels_allow_nonwinner_outcomes() -> None:
    rows = []
    for width in (2, 4, 6, 8, 10):
        for method, rmse, overhead in (("raw", 0.10, 1), ("readout", 0.099, 5), ("zne", 0.20, 3)):
            rows.append(
                {
                    "family": "hardware_efficient",
                    "circuit_id": f"he_w{width}_d1",
                    "width": width,
                    "depth": 1,
                    "noise": "low",
                    "shots": 4096,
                    "method": method,
                    "complete_repeats": 50,
                    "rmse": rmse,
                    "execution_overhead": overhead,
                    "rmse_gain_ci_low": -0.01,
                    "rmse_gain_ci_high": 0.01,
                }
            )
    summary = pd.DataFrame(rows)
    split = assign_holdout(summary)
    labels = build_regime_labels(summary)
    assert "holdout" in set(split.split)
    assert len(labels) == 5
    assert "equivalent_methods_json" in labels
    assert set(labels.oracle_label) <= {
        "raw",
        "readout",
        "zne",
        "practical_tie",
        "inconclusive",
    }
