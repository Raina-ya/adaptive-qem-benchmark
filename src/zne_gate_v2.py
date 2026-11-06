"""Choose a runtime-only ZNE gate on training and validation circuits, then score it on holdout.

No v2 check reads the exact reference. The exact reference only defines "really improved", through
independent draws in the truth table. Holdout rows reach neither selection function.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import derive_seed
from src.generate_signal_v2 import RAW_PATH, TRUTH_PATH, fit_weights, split_evenly
from src.observable_floors import ObservableFloor
from src.signal_manifest_v2 import CONFIG, MANIFEST_PATH
from src.zne_validity import assess_zne_validity
from src.zne_validity_gate import evaluate_zne_gate

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
CONDITIONS_PATH = PROCESSED / "zne_gate_v2_conditions.csv"
SELECTION_PATH = PROCESSED / "zne_gate_v2_selection.csv"
CHECKS_PATH = PROCESSED / "zne_gate_v2_checks.csv"
HOLDOUT_PATH = PROCESSED / "zne_gate_v2_holdout.csv"
REJECTIONS_PATH = PROCESSED / "zne_gate_v2_rejections.csv"
MODELS_PATH = PROCESSED / "zne_model_v2_selection.csv"
POLICIES_PATH = PROCESSED / "zne_model_v2_policies.csv"
Z_95 = 1.959963984540054
CHI2_1DF_95 = 3.841458820694124
PRACTICAL = float(CONFIG["practical_threshold"])
BOOTSTRAPS = 2000
TOP_ON_TRAIN = 5
TOP_MODELS_ON_TRAIN = 3
MAX_FA_INCREASE = 0.05
MAX_READOUT_AMPLIFICATION = 4.0
PHYSICAL = ("physical_strict", "physical_ci", "physical_tolerance", "physical_constrained")
TREND = (
    "trend_v1",
    "trend_monotonic",
    "trend_corrected_monotonic",
    "trend_leave_one_out",
    "trend_weighted_residual",
    "trend_quadratic",
    "trend_slope_sign",
    "trend_none",
)
# Each failing check names a mechanism, so no rejection is left unexplained.
MECHANISMS = {
    "physical_strict": "point_estimate_outside_physical_bounds",
    "physical_ci": "physical_bound_exceeded_beyond_sampling_interval",
    "physical_tolerance": "overshoot_larger_than_sampling_tolerance",
    "trend_v1": "v1_destroyed_or_unstable_rate_too_high",
    "trend_monotonic": "signal_rises_with_noise_factor",
    "trend_corrected_monotonic": "readout_corrected_signal_rises_with_noise_factor",
    "trend_leave_one_out": "extrapolation_depends_on_one_noise_factor",
    "trend_weighted_residual": "weighted_linear_fit_residual_too_large",
    "trend_quadratic": "quadratic_and_linear_extrapolations_disagree",
    "trend_slope_sign": "fitted_slope_sign_unstable_across_repeats",
    "garbage": "garbage_folding_not_distinguished_from_genuine",
    "retention": "signal_retention_too_low_or_baseline_unidentifiable",
    "estimated_gain": "estimated_gain_not_resolved_after_variance_cost",
}
# Only the floor and the bounds are read here, and neither depends on a reference value.
FLOOR = ObservableFloor(
    observable="traceless_pauli",
    physical_minimum=-1.0,
    physical_maximum=1.0,
    expected_noise_floor=0.0,
    reference_value=0.0,
    maximum_attainable_improvement=0.0,
    floor_model="zero_floor",
    evidence_label="manifest_declared",
)


@dataclass(frozen=True)
class Candidate:
    """One gate: a physical rule, a trend rule, and three optional checks."""

    physical: str
    trend: str
    garbage: bool
    retention: bool
    estimated_gain: bool

    @property
    def checks(self) -> tuple[str, ...]:
        items = [self.physical, self.trend]
        items.extend(
            name
            for name, used in (
                ("garbage", self.garbage),
                ("retention", self.retention),
                ("estimated_gain", self.estimated_gain),
            )
            if used
        )
        return tuple(item for item in items if item not in ("physical_constrained", "trend_none"))

    @property
    def name(self) -> str:
        return "+".join(
            (
                self.physical,
                self.trend,
                *(
                    name
                    for name, used in (
                        ("garbage", self.garbage),
                        ("retention", self.retention),
                        ("estimated_gain", self.estimated_gain),
                    )
                    if used
                ),
            )
        )

    @property
    def uses_clipped_estimate(self) -> bool:
        return self.physical == "physical_constrained"


def candidates() -> list[Candidate]:
    return [
        Candidate(*values)
        for values in itertools.product(
            PHYSICAL, TREND, (False, True), (False, True), (False, True)
        )
    ]


def _means_and_errors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def _no_significant_rise(values: np.ndarray) -> bool:
    means, errors = _means_and_errors(values)
    distance = np.abs(means)
    tolerance = Z_95 * np.sqrt(errors[:-1] ** 2 + errors[1:] ** 2)
    return bool(np.all(np.diff(distance) <= tolerance))


def runtime_checks(
    values: np.ndarray,
    errors: np.ndarray,
    garbage: np.ndarray,
    corrected: np.ndarray,
    raw: np.ndarray,
    raw_errors: np.ndarray,
    readout_amplification: float,
    factors: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    """Compute every check and metric from observations, without any reference."""

    weights = fit_weights(factors, 1)
    fit = values @ weights
    fit_errors = np.sqrt(errors**2 @ weights**2)
    scale = float(np.median(fit_errors))
    excess = np.abs(fit) - 1.0
    means, mean_errors = _means_and_errors(values)
    output: dict[str, object] = {
        "physical_strict": bool(excess.max() <= 1e-12),
        "physical_ci": bool(np.mean(np.abs(fit) - Z_95 * fit_errors > 1.0) <= 0.05),
        "physical_tolerance": bool(excess.max() <= 3.0 * scale),
        "physical_constrained": True,
    }
    output["trend_monotonic"] = _no_significant_rise(values)
    output["trend_corrected_monotonic"] = _no_significant_rise(corrected)
    lines = [
        means[list(pair)] @ fit_weights(tuple(factors[i] for i in pair), 1)
        for pair in itertools.combinations(range(len(factors)), 2)
    ]
    full = float(means @ weights)
    output["trend_leave_one_out"] = bool(max(abs(line - full) for line in lines) <= 2.0 * scale)
    # A weighted line through the factors gives a chi-square with one degree of freedom.
    design = np.vstack([np.ones(len(factors)), factors]).T
    chi_squares = []
    for row, sigma in zip(values, errors, strict=True):
        inverse = 1.0 / np.maximum(sigma, 1e-12) ** 2
        normal = design.T * inverse @ design
        solved = np.linalg.solve(normal, design.T * inverse @ row)
        chi_squares.append(float(np.sum(inverse * (row - design @ solved) ** 2)))
    output["trend_weighted_residual"] = bool(np.median(chi_squares) <= CHI2_1DF_95)
    quadratic_weights = fit_weights(factors, 2)
    difference = quadratic_weights - weights
    difference_scale = float(np.median(np.sqrt(errors**2 @ difference**2)))
    output["trend_quadratic"] = bool(abs(means @ difference) <= 2.0 * difference_scale)
    centered = np.asarray(factors, dtype=float) - np.mean(factors)
    slopes = values @ centered
    output["trend_slope_sign"] = bool(np.mean(np.sign(slopes) == np.sign(slopes.mean())) >= 0.9)
    output["trend_none"] = True
    # The garbage control must fall further than the genuine fold at the last factor.
    garbage_means, garbage_errors = _means_and_errors(garbage)
    advantage = abs(means[-1]) - abs(garbage_means[-1])
    output["garbage"] = bool(advantage - Z_95 * np.hypot(mean_errors[-1], garbage_errors[-1]) > 0.0)
    identifiable = abs(means[0]) > Z_95 * mean_errors[0]
    low = (abs(means[-1]) - Z_95 * mean_errors[-1]) / (abs(means[0]) + Z_95 * mean_errors[0])
    output["retention"] = bool(identifiable and low > 0.10)
    # If the fit were unbiased, E[(fit - raw)^2] - 2 var(fit) estimates raw MSE minus fit MSE.
    estimated = (fit - raw) ** 2 - 2.0 * fit_errors**2
    estimated_mean, estimated_error = _means_and_errors(estimated)
    output["estimated_gain"] = bool(estimated_mean - Z_95 * estimated_error > 0.0)
    output["conditioning_readout"] = bool(readout_amplification <= MAX_READOUT_AMPLIFICATION)
    output.update(
        readout_amplification=readout_amplification,
        extrapolation_leverage=float(np.sum(weights**2) * len(factors)),
        variance_ratio=float(np.mean(fit_errors**2 / np.maximum(raw_errors**2, 1e-15))),
        curvature_z=float(
            (means[0] - 2.0 * means[1] + means[2])
            / max(float(np.sqrt(np.sum(mean_errors**2 * np.array([1.0, 4.0, 1.0])))), 1e-15)
        ),
        estimated_gain_mean=float(estimated_mean),
        max_overshoot=float(max(excess.max(), 0.0)),
    )
    return output


def v1_decisions(
    values: np.ndarray,
    errors: np.ndarray,
    garbage: np.ndarray,
    raw: np.ndarray,
    reference: float,
) -> dict[str, object]:
    """Apply the frozen v1 gate, and its checks that need no reference."""

    weights = fit_weights((1, 3, 5), 1)
    fit, garbage_fit = values @ weights, garbage @ weights
    baseline = np.abs(raw - reference)
    assessed = [
        assess_zne_validity((1, 3, 5), tuple(row), tuple(sigma), FLOOR)
        for row, sigma in zip(values, errors, strict=True)
    ]
    labels = np.array([item.validity_classification for item in assessed])
    retention = np.array(
        [
            item.signal_retention_ratios[-1]
            if np.isfinite(item.signal_retention_ratios[-1])
            else 0.0
            for item in assessed
        ]
    )
    result = evaluate_zne_gate(
        baseline - np.abs(fit - reference),
        baseline - np.abs(garbage_fit - reference),
        retention,
        np.zeros(len(values)),
        np.maximum(np.abs(fit) - 1.0, 0.0),
        labels == "signal_destroyed",
        labels == "extrapolation_unstable",
    )
    output = {
        "v1_signal_retention": result.signal_retention_pass,
        "v1_physical_bounds": result.physical_bounds_pass,
        "v1_garbage_control": result.garbage_control_pass,
        "v1_gain_uncertainty": result.gain_uncertainty_pass,
        "trend_v1": result.trend_pass,
        "v1_destroyed_rate": result.destroyed_rate,
        "v1_unstable_rate": result.unstable_rate,
        "v1_full": result.validity_status == "credible",
    }
    output["v1_runtime"] = bool(
        result.signal_retention_pass and result.physical_bounds_pass and result.trend_pass
    )
    return output


def _plus_values(column: pd.Series, shots: list[int]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array([[int(item) for item in text.split(";")] for text in column], dtype=float)
    total = np.asarray(shots, dtype=float)
    estimates = 2.0 * counts / total - 1.0
    return estimates, np.sqrt(np.maximum(1.0 - estimates**2, 1.0 / total) / total)


def build_conditions(
    repeats: pd.DataFrame, truth: pd.DataFrame, manifest: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Compute one row of decisions per condition, with truth joined for scoring only."""

    target = config["gate_target"]
    target_truth = truth.loc[
        (truth.design == target["design"]) & (truth.model == target["model"])
    ].set_index(["circuit_id", "noise", "total_shot_budget"])
    info = manifest.set_index("circuit_id")
    rows = []
    groups = repeats.groupby(["circuit_id", "noise", "total_shot_budget"], sort=True)
    for key, group in groups:
        circuit, noise, budget = key
        group = group.sort_values("repeat")
        shots = split_evenly(int(budget), 3)
        values, errors = _plus_values(group[f"{target['design']}_plus"], shots)
        garbage, _ = _plus_values(group.garbage_135_plus, shots)
        corrected = np.array(
            [[float(item) for item in text.split(";")] for text in group.corrected_135]
        )
        raw = 2.0 * group.raw_plus.to_numpy(float) / budget - 1.0
        raw_errors = np.sqrt(np.maximum(1.0 - raw**2, 1.0 / budget) / budget)
        settings = config["noise"][noise]
        support = sum(symbol != "I" for symbol in info.loc[circuit, "observable_q0_first"])
        attenuation = 1.0 - settings.get("readout_p01", 0.0) - settings.get("readout_p10", 0.0)
        row_truth = target_truth.loc[key]
        row: dict[str, object] = {
            "circuit_id": circuit,
            "noise": noise,
            "total_shot_budget": int(budget),
            "split": info.loc[circuit, "split"],
            "signal_band": info.loc[circuit, "signal_band"],
            "width": int(info.loc[circuit, "width"]),
            "depth": int(info.loc[circuit, "depth"]),
            "raw_rmse": float(row_truth.raw_rmse),
            "fit_rmse": float(row_truth.fit_rmse),
            "fit_rmse_clipped": float(row_truth.fit_rmse_clipped),
        }
        row.update(
            runtime_checks(
                values, errors, garbage, corrected, raw, raw_errors, attenuation ** (-support)
            )
        )
        row.update(v1_decisions(values, errors, garbage, raw, float(row_truth.reference)))
        row["improved"] = row["raw_rmse"] - row["fit_rmse"] > PRACTICAL
        row["improved_clipped"] = row["raw_rmse"] - row["fit_rmse_clipped"] > PRACTICAL
        rows.append(row)
    return pd.DataFrame(rows)


def decide(candidate: Candidate, table: pd.DataFrame) -> pd.Series:
    if not candidate.checks:
        return pd.Series(True, index=table.index)
    return table[list(candidate.checks)].all(axis=1)


def confusion(accepted: pd.Series, truth: pd.Series) -> dict[str, float]:
    true_accept = int((accepted & truth).sum())
    false_accept = int((accepted & ~truth).sum())
    true_reject = int((~accepted & ~truth).sum())
    false_reject = int((~accepted & truth).sum())
    fa_rate = false_accept / (false_accept + true_reject) if false_accept + true_reject else np.nan
    fr_rate = false_reject / (false_reject + true_accept) if false_reject + true_accept else np.nan
    return {
        "true_accept": true_accept,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "false_reject": false_reject,
        "false_accept_rate": fa_rate,
        "false_reject_rate": fr_rate,
        "balanced_accuracy": 1.0 - (fa_rate + fr_rate) / 2.0,
    }


def _score(candidate: Candidate, table: pd.DataFrame) -> dict[str, float]:
    truth = table.improved_clipped if candidate.uses_clipped_estimate else table.improved
    return confusion(decide(candidate, table), truth)


def select_gate(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Candidate, pd.DataFrame]:
    """Keep the five best on training, then pick on validation; ties favour fewer checks."""

    rows = []
    for candidate in candidates():
        score = _score(candidate, train)
        rows.append(
            {
                "candidate": candidate,
                "name": candidate.name,
                "check_count": len(candidate.checks),
                "train_balanced_accuracy": score["balanced_accuracy"],
                "train_false_accept_rate": score["false_accept_rate"],
                "train_false_reject_rate": score["false_reject_rate"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["train_balanced_accuracy", "check_count", "name"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top = frame.head(TOP_ON_TRAIN).copy()
    scores = [_score(item, validation) for item in top.candidate]
    top["validation_balanced_accuracy"] = [score["balanced_accuracy"] for score in scores]
    top["validation_false_accept_rate"] = [score["false_accept_rate"] for score in scores]
    top["validation_false_reject_rate"] = [score["false_reject_rate"] for score in scores]
    top = top.sort_values(
        ["validation_balanced_accuracy", "check_count", "name"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    top["selected"] = [True, *[False] * (len(top) - 1)]
    return top.candidate.iloc[0], top.drop(columns="candidate")


def select_model(truth: pd.DataFrame) -> tuple[tuple[str, str], pd.DataFrame]:
    """Rank design and model pairs by mean RMSE gain, which already includes variance cost."""

    fit = truth.loc[truth.split.isin(("train", "validation"))].copy()
    fit["gain"] = fit.raw_rmse - fit.fit_rmse
    means = fit.groupby(["split", "design", "model"]).gain.mean().unstack("split").reset_index()
    means = means.sort_values(
        ["train", "design", "model"], ascending=[False, True, True], kind="mergesort"
    )
    top = means.head(TOP_MODELS_ON_TRAIN).sort_values(
        ["validation", "design", "model"], ascending=[False, True, True], kind="mergesort"
    )
    chosen = (str(top.design.iloc[0]), str(top.model.iloc[0]))
    means["selected"] = (means.design == chosen[0]) & (means.model == chosen[1])
    means["reached_validation"] = means.index.isin(top.index)
    return chosen, means.rename(columns={"train": "train_gain", "validation": "validation_gain"})


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    rate = successes / total
    denominator = 1.0 + Z_95**2 / total
    center = (rate + Z_95**2 / (2.0 * total)) / denominator
    spread = Z_95 * np.sqrt(rate * (1.0 - rate) / total + Z_95**2 / (4.0 * total**2)) / denominator
    return float(center - spread), float(center + spread)


def _cluster_intervals(
    table: pd.DataFrame, accepted: pd.Series, truth: pd.Series
) -> dict[str, float]:
    """Resample whole circuits, because conditions on one circuit are not independent."""

    frame = pd.DataFrame({"circuit": table.circuit_id, "accepted": accepted, "truth": truth})
    frame["fa"] = frame.accepted & ~frame.truth
    frame["fr"] = ~frame.accepted & frame.truth
    frame["neg"] = ~frame.truth
    frame["pos"] = frame.truth
    totals = frame.groupby("circuit")[["fa", "fr", "neg", "pos"]].sum().to_numpy(float)
    rng = np.random.default_rng(derive_seed("zne-gate-v2-cluster"))
    picks = rng.integers(0, len(totals), size=(BOOTSTRAPS, len(totals)))
    sums = totals[picks].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = sums[:, 0] / sums[:, 2]
        fr = sums[:, 1] / sums[:, 3]
    return {
        "fa_cluster_low": float(np.nanquantile(fa, 0.025)),
        "fa_cluster_high": float(np.nanquantile(fa, 0.975)),
        "fr_cluster_low": float(np.nanquantile(fr, 0.025)),
        "fr_cluster_high": float(np.nanquantile(fr, 0.975)),
    }


def holdout_table(selected: Candidate, holdout: pd.DataFrame) -> pd.DataFrame:
    """Score the frozen choice against both forms of v1, on circuits it has never seen."""

    truth = holdout.improved
    gates = {
        "v1_full_reads_reference": (holdout.v1_full, truth),
        "v1_runtime_checks": (holdout.v1_runtime, truth),
        f"v2_selected:{selected.name}": (
            decide(selected, holdout),
            holdout.improved_clipped if selected.uses_clipped_estimate else truth,
        ),
    }
    rows = []
    for name, (accepted, labels) in gates.items():
        score = confusion(accepted, labels)
        fa_low, fa_high = wilson(
            score["false_accept"], score["false_accept"] + score["true_reject"]
        )
        fr_low, fr_high = wilson(
            score["false_reject"], score["false_reject"] + score["true_accept"]
        )
        rows.append(
            {
                "gate": name,
                "conditions": len(holdout),
                "independent_circuits": int(holdout.circuit_id.nunique()),
                **score,
                "fa_wilson_low": fa_low,
                "fa_wilson_high": fa_high,
                "fr_wilson_low": fr_low,
                "fr_wilson_high": fr_high,
                **_cluster_intervals(holdout, accepted, labels),
            }
        )
    frame = pd.DataFrame(rows)
    v2 = frame.iloc[2]
    adopted = bool(
        v2.balanced_accuracy > frame.iloc[1].balanced_accuracy
        and v2.false_accept_rate <= frame.iloc[1].false_accept_rate + MAX_FA_INCREASE
    )
    frame["v2_adopted_over_v1_runtime"] = adopted
    return frame


def check_table(table: pd.DataFrame) -> pd.DataFrame:
    """Tabulate each check alone against the truth, by split."""

    names = [
        *PHYSICAL,
        *TREND,
        "garbage",
        "retention",
        "estimated_gain",
        "conditioning_readout",
        "v1_signal_retention",
        "v1_physical_bounds",
        "v1_garbage_control",
        "v1_gain_uncertainty",
        "v1_full",
        "v1_runtime",
    ]
    rows = []
    for split, group in table.groupby("split", sort=True):
        for name in names:
            rows.append(
                {
                    "split": split,
                    "check": name,
                    "reads_exact_reference": name
                    in ("v1_garbage_control", "v1_gain_uncertainty", "v1_full"),
                    **confusion(group[name].astype(bool), group.improved),
                }
            )
    return pd.DataFrame(rows)


def rejection_reasons(selected: Candidate, holdout: pd.DataFrame) -> pd.DataFrame:
    """Name every failed check behind every holdout rejection."""

    rows = []
    for _, row in holdout.iterrows():
        failed = [check for check in selected.checks if not row[check]]
        if failed:
            rows.append(
                {
                    "circuit_id": row.circuit_id,
                    "noise": row.noise,
                    "total_shot_budget": row.total_shot_budget,
                    "signal_band": row.signal_band,
                    "really_improved": bool(row.improved),
                    "failure_mechanisms": ";".join(MECHANISMS[check] for check in failed),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "circuit_id",
            "noise",
            "total_shot_budget",
            "signal_band",
            "really_improved",
            "failure_mechanisms",
        ],
    )


def policy_table(
    selected: Candidate, model: tuple[str, str], holdout: pd.DataFrame, truth: pd.DataFrame
) -> pd.DataFrame:
    """Compare holdout RMSE of each policy, with raw as the fallback after a rejection."""

    chosen = truth.loc[(truth.design == model[0]) & (truth.model == model[1])]
    keys = ["circuit_id", "noise", "total_shot_budget"]
    frame = holdout.merge(
        chosen[[*keys, "fit_rmse"]].rename(columns={"fit_rmse": "model_rmse"}), on=keys
    )
    best = truth.groupby(keys).fit_rmse.min().rename("best_fit_rmse").reset_index()
    frame = frame.merge(best, on=keys)
    gate = decide(selected, frame).to_numpy()
    fit = frame.fit_rmse_clipped if selected.uses_clipped_estimate else frame.fit_rmse
    v1 = frame.v1_runtime.to_numpy()
    policies = {
        "always_raw": frame.raw_rmse.to_numpy(),
        "always_global_135_linear": frame.fit_rmse.to_numpy(),
        f"always_selected_{model[0]}_{model[1]}": frame.model_rmse.to_numpy(),
        "v1_runtime_gate_else_raw": np.where(v1, frame.fit_rmse, frame.raw_rmse),
        "v2_gate_else_raw": np.where(gate, fit, frame.raw_rmse),
        "v2_gate_selected_model_else_raw": np.where(gate, frame.model_rmse, frame.raw_rmse),
        "oracle_best_of_raw_and_fits_upper_bound": np.minimum(
            frame.raw_rmse, frame.best_fit_rmse
        ).to_numpy(),
    }
    circuits = frame.circuit_id.to_numpy()
    unique = np.unique(circuits)
    rng = np.random.default_rng(derive_seed("zne-model-v2-cluster"))
    picks = rng.integers(0, len(unique), size=(BOOTSTRAPS, len(unique)))
    rows = []
    raw = policies["always_raw"]
    for name, values in policies.items():
        per = pd.DataFrame({"circuit": circuits, "gain": raw - values}).groupby("circuit").gain
        totals, counts = per.sum().to_numpy(), per.size().to_numpy()
        boot = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
        rows.append(
            {
                "policy": name,
                "conditions": len(frame),
                "independent_circuits": len(unique),
                "mean_rmse": float(values.mean()),
                "mean_gain_vs_raw": float((raw - values).mean()),
                "gain_cluster_low": float(np.quantile(boot, 0.025)),
                "gain_cluster_high": float(np.quantile(boot, 0.975)),
                "budget_scope": "per-run budget; gate repeats and garbage runs are not charged",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    repeats = pd.read_csv(RAW_PATH)
    truth = pd.read_csv(TRUTH_PATH).merge(manifest[["circuit_id", "split"]], on="circuit_id")
    table = build_conditions(repeats, truth, manifest, CONFIG)
    table.to_csv(CONDITIONS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    train = table.loc[table.split == "train"]
    validation = table.loc[table.split == "validation"]
    holdout = table.loc[table.split == "holdout"]
    selected, selection = select_gate(train, validation)
    selection.to_csv(SELECTION_PATH, index=False, lineterminator="\n", float_format="%.12g")
    model, models = select_model(truth)
    models.to_csv(MODELS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    check_table(table).to_csv(CHECKS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    holdout_table(selected, holdout).to_csv(
        HOLDOUT_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    rejection_reasons(selected, holdout).to_csv(REJECTIONS_PATH, index=False, lineterminator="\n")
    policy_table(selected, model, holdout, truth.loc[truth.split == "holdout"]).to_csv(
        POLICIES_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )


if __name__ == "__main__":
    main()
