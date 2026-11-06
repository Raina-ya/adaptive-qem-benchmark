import numpy as np
import pandas as pd
import pytest

from src import qaoa_mechanism as mechanism


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    return mechanism.load_rows()


@pytest.fixture(scope="module")
def cells(rows: pd.DataFrame) -> pd.DataFrame:
    return mechanism.cell_table(rows)


def _toy() -> pd.DataFrame:
    return mechanism.decompose(
        pd.DataFrame(
            {
                "graph_id": "g",
                "total_selection_budget": 100,
                "method": "m",
                "repeat": [0, 0, 0, 1, 1, 1],
                "candidate_id": ["a", "b", "c"] * 2,
                "candidate_role": ["x", "y", "z"] * 2,
                "ideal": [3.0, 2.0, 1.0] * 2,
                "estimate": [2.5, 2.4, 1.2, 3.1, 1.8, 0.9],
            }
        )
    )


def test_energy_error_splits_exactly_into_common_and_contrast(cells: pd.DataFrame) -> None:
    left = cells.energy_rmse**2
    right = cells.common_bias_rms**2 + cells.contrast_rmse**2
    assert left.to_numpy() == pytest.approx(right.to_numpy(), abs=1e-12)


def test_contrast_splits_into_systematic_and_sampling(cells: pd.DataFrame) -> None:
    left = cells.contrast_rmse**2
    right = cells.contrast_systematic_rms**2 + cells.contrast_sampling_rms**2
    assert left.to_numpy() == pytest.approx(right.to_numpy(), abs=1e-12)


def test_a_common_shift_changes_rmse_but_never_the_selection() -> None:
    toy = _toy()
    shifted = mechanism.decompose(toy.assign(estimate=toy.estimate + 7.0))
    first, second = mechanism.decisions(toy), mechanism.decisions(shifted)
    assert list(first.selected) == list(second.selected)
    assert list(first.regret) == list(second.regret)
    assert first.contrast_mse.to_numpy() == pytest.approx(second.contrast_mse.to_numpy())
    assert (second.mse > first.mse).all()


def test_cells_match_the_committed_selection_summary(cells: pd.DataFrame) -> None:
    summary = pd.read_csv(mechanism.PROCESSED / "qaoa_selection_summary_v1.csv")
    overall = summary.loc[summary.candidate_gap_regime == "overall"]
    keys = ["graph_id", "total_selection_budget", "method"]
    merged = cells.merge(overall, on=keys, suffixes=("", "_saved"), validate="one_to_one")
    assert len(merged) == 17
    pairs = (
        ("top1_error", "decision_error_probability"),
        ("mean_simple_regret", "mean_simple_regret_saved"),
        ("energy_rmse", "energy_rmse_saved"),
    )
    for mine, saved in pairs:
        assert merged[mine].to_numpy() == pytest.approx(merged[saved].to_numpy(), abs=1e-9)


def test_mcnemar_exact_values() -> None:
    assert mechanism.mcnemar_exact(0, 0) == 1.0
    assert mechanism.mcnemar_exact(0, 5) == pytest.approx(2 * 0.5**5)
    assert mechanism.mcnemar_exact(22, 6) == mechanism.mcnemar_exact(6, 22)
    assert mechanism.mcnemar_exact(22, 6) == pytest.approx(0.00371, abs=1e-5)


def test_holm_is_monotone_and_capped() -> None:
    assert mechanism.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert mechanism.holm([0.9, 0.9]) == [1.0, 1.0]


def test_zne_fit_variance_factor_is_the_analytic_value() -> None:
    weights = np.array([13.0 / 12.0, 1.0 / 3.0, -5.0 / 12.0])
    assert mechanism.zne_variance_factor() == pytest.approx(3.0 * np.sum(weights**2))
    assert mechanism.zne_variance_factor() == pytest.approx(4.375)


def test_zne_lowers_contrast_rmse_in_all_six_cells() -> None:
    table = pd.read_csv(mechanism.COMPARISONS_PATH)
    zne = table.loc[table.method == "zne"]
    assert len(zne) == 6
    assert (zne.contrast_rmse_ci_high < 0.0).all()
    assert (zne.energy_rmse_ci_high < 0.0).all()


def test_mitigation_never_raises_top1_accuracy_here_and_only_one_survives_holm() -> None:
    table = pd.read_csv(mechanism.COMPARISONS_PATH)
    assert (table.correct_rate_difference <= 0.0).all()
    assert (table.mcnemar_holm_p < 0.05).sum() == 1
    resolved = table.loc[table.mcnemar_holm_p < 0.05].iloc[0]
    assert (resolved.graph_id, resolved.method, resolved.total_selection_budget) == (
        "medium_dense8",
        "zne",
        16384,
    )


def test_gap_spread_grows_near_the_analytic_factor() -> None:
    table = pd.read_csv(mechanism.COUNTERFACTUAL_PATH)
    assert ((table.observed_sd_ratio > 1.9) & (table.observed_sd_ratio < 2.6)).all()
    assert table.theoretical_sd_ratio.iloc[0] == pytest.approx(4.375**0.5)


def test_results_are_labeled_exploratory() -> None:
    for path in (mechanism.CELLS_PATH, mechanism.COMPARISONS_PATH, mechanism.COUNTERFACTUAL_PATH):
        assert set(pd.read_csv(path).analysis_role) == {"exploratory_post_hoc"}
