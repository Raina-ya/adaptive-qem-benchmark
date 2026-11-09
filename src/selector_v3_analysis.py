"""Score the selector and the certificate on tasks kept apart from rule choice."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.decision_tasks_v3 import BUILDERS, CONFIG, MANIFEST_PATH
from src.density_sim import derive_seed
from src.generate_selector_v3 import ALARMS, RAW_PATH
from src.selector_v3 import Rule, apply_rule

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
RULE_PATH = PROCESSED / "selector_v3_rule_selection.csv"
HOLDOUT_PATH = PROCESSED / "selector_v3_holdout.csv"
GROUPS_PATH = PROCESSED / "selector_v3_groups.csv"
LEAVE_OUT_PATH = PROCESSED / "selector_v3_leave_group_out.csv"
ACCEPTANCE_PATH = PROCESSED / "selector_v3_acceptance.csv"
CERTIFICATE_PATH = PROCESSED / "certification_v2_summary.csv"
CERTIFICATE_GROUPS_PATH = PROCESSED / "certification_v2_groups.csv"
ALARMS_PATH = PROCESSED / "certification_v2_alarms.csv"
ARRAYS = (
    "base_raw",
    "base_readout",
    "base_zne",
    "base_combined",
    "pilot_raw",
    "pilot_raw_se",
    "pilot_readout",
    "pilot_fit",
    "pilot_fit_se",
    "branch_raw",
    "branch_raw_se",
    "branch_readout",
    "branch_readout_se",
    "branch_zne",
    "branch_zne_se",
    "branch_combined",
    "branch_combined_se",
)
BASELINES = {
    "always_raw": "base_raw",
    "always_readout": "base_readout",
    "always_zne": "base_zne",
    "always_combined": "base_combined",
}


def parse(frame: pd.DataFrame, candidates: int) -> dict[str, np.ndarray]:
    data = {}
    for name in ARRAYS:
        joined = ";".join(frame[name].tolist())
        data[name] = np.fromstring(joined, sep=";").reshape(len(frame), candidates)
    data["drift_statistic"] = frame.drift_statistic.to_numpy(float)
    return data


def task_table(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return per-task ideal values and two-qubit gate counts per candidate."""

    rows = []
    for task_id, group in manifest.groupby("task_id", sort=True):
        group = group.sort_values("candidate")
        counts = [
            sum(
                gate.name == "cx" for gate in BUILDERS[r.family](r.width, r.depth, r.parameter_seed)
            )
            for r in group.itertuples()
        ]
        rows.append(
            {
                "task_id": task_id,
                "family": group.family.iloc[0],
                "width": int(group.width.iloc[0]),
                "depth": int(group.depth.iloc[0]),
                "gap_class": group.gap_class.iloc[0],
                "split": group.split.iloc[0],
                "ideal": group.ideal_expectation.to_numpy(float),
                "gate_count": float(np.mean(counts)),
            }
        )
    return pd.DataFrame(rows)


def gate_shots(
    frame: pd.DataFrame, branch: np.ndarray | None, name: str, config: dict
) -> np.ndarray:
    """Return two-qubit gate executions per shot budget for one policy, including all folds."""

    budget = frame.budget.to_numpy(float)
    gates = frame.gate_count.to_numpy(float)
    mean_folds = float(np.mean(config["factors"]))
    if name == "always_raw":
        return budget * gates
    if name == "always_readout":
        return budget * (1.0 - config["baseline_calibration_fraction"]) * gates
    if name == "always_zne":
        return budget * mean_folds * gates
    if name == "always_combined":
        return budget * (1.0 - config["baseline_calibration_fraction"]) * mean_folds * gates
    pilot = budget * config["pilot_fold_fraction"] * mean_folds * gates
    remaining = 1.0 - config["pilot_calibration_fraction"] - config["pilot_fold_fraction"]
    folds = np.where(branch == 1, mean_folds, 1.0)
    return pilot + budget * remaining * folds * gates


def score(frame: pd.DataFrame, pick: np.ndarray, abstain: np.ndarray) -> dict[str, np.ndarray]:
    ideal = np.stack(frame.ideal.to_numpy())
    best = ideal.max(axis=1)
    index = np.arange(len(frame))
    regret = best - ideal[index, pick]
    error = regret > 1e-12
    return {"regret": regret, "error": error, "abstain": abstain}


def summarize(frame: pd.DataFrame, outcome: dict, overhead: np.ndarray, calibration: float) -> dict:
    covered = ~outcome["abstain"]
    return {
        "runs": len(frame),
        "tasks": int(frame.task_id.nunique()),
        "coverage": float(covered.mean()),
        "abstention_rate": float(outcome["abstain"].mean()),
        "forced_regret": float(outcome["regret"].mean()),
        "covered_regret": float(outcome["regret"][covered].mean()) if covered.any() else np.nan,
        "conditional_error": float(outcome["error"][covered].mean()) if covered.any() else np.nan,
        "unconditional_error": float((outcome["error"] & covered).mean()),
        "reliability_coverage": float((covered & ~outcome["error"]).mean()),
        "calibration_share": calibration,
        "mean_gate_shots": float(overhead.mean()),
    }


def evaluate_policies(frame, data, rule: Rule, config: dict) -> dict[str, dict]:
    """Return outcomes for every baseline and the selector, on the same runs."""

    policies = {}
    calibration = float(config["baseline_calibration_fraction"])
    for name, column in BASELINES.items():
        pick = np.argmax(data[column], axis=1)
        share = calibration if name in ("always_readout", "always_combined") else 0.0
        policies[name] = (
            score(frame, pick, np.zeros(len(frame), dtype=bool)),
            gate_shots(frame, None, name, config),
            share,
        )
    branch, pick, abstain, _ = apply_rule(data, rule)
    policies["selector"] = (
        score(frame, pick, abstain),
        gate_shots(frame, branch, "selector", config),
        float(config["pilot_calibration_fraction"]),
    )
    # The same abstention test on always-readout tells whether switching method adds anything.
    gap_sorted = np.sort(data["branch_readout"], axis=1)
    chosen = np.argmax(data["branch_readout"], axis=1)
    runner = np.argsort(data["branch_readout"], axis=1)[:, -2]
    index = np.arange(len(frame))
    joint = np.hypot(
        data["branch_readout_se"][index, chosen], data["branch_readout_se"][index, runner]
    )
    readout_abstain = (gap_sorted[:, -1] - gap_sorted[:, -2]) < rule.z_separate * joint
    policies["abstaining_readout"] = (
        score(frame, chosen, readout_abstain),
        gate_shots(frame, np.zeros(len(frame), dtype=int), "selector", config),
        float(config["pilot_calibration_fraction"]),
    )
    return policies


def oracle_row(frame, policies) -> dict:
    """Pick the best baseline per condition using scored regret, as an upper bound only."""

    keys = frame.task_id + "|" + frame.noise + "|" + frame.budget.astype(str)
    table = pd.DataFrame(
        {name: policies[name][0]["regret"] for name in BASELINES}, index=keys.values
    )
    best = table.groupby(level=0).mean().min(axis=1)
    return {"policy": "oracle_best_baseline", "forced_regret": float(best.mean()), "coverage": 1.0}


def choose_rule(frame_train, data_train, frame_val, data_val, config) -> tuple[Rule, pd.DataFrame]:
    rows = []
    grid = config["selector_grid"]
    for values in itertools.product(grid["z_separate"], grid["z_deviate"], grid["allow_raw"]):
        rule = Rule(*values)
        _, pick, abstain, _ = apply_rule(data_train, rule)
        outcome = score(frame_train, pick, abstain)
        covered = ~abstain
        rows.append(
            {
                "rule": rule,
                "name": rule.name,
                "z_deviate": rule.z_deviate,
                "z_separate": rule.z_separate,
                "allow_raw": rule.allow_raw,
                "train_coverage": float(covered.mean()),
                "train_covered_regret": float(outcome["regret"][covered].mean())
                if covered.any()
                else np.inf,
            }
        )
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame.train_coverage >= float(config["minimum_coverage"])]
    order = ["train_covered_regret", "z_deviate", "z_separate", "allow_raw"]
    top = (
        eligible.sort_values(order, ascending=[True, False, True, True], kind="mergesort")
        .head(5)
        .copy()
    )
    values, coverage = [], []
    for rule in top.rule:
        _, pick, abstain, _ = apply_rule(data_val, rule)
        outcome = score(frame_val, pick, abstain)
        covered = ~abstain
        values.append(float(outcome["regret"][covered].mean()) if covered.any() else np.inf)
        coverage.append(float(covered.mean()))
    top["validation_covered_regret"] = values
    top["validation_coverage"] = coverage
    top = top.sort_values(
        ["validation_covered_regret", "z_deviate", "z_separate", "allow_raw"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    top["selected"] = [True, *[False] * (len(top) - 1)]
    return top.rule.iloc[0], top.drop(columns="rule")


def cluster_contrast(
    frame, values: np.ndarray, seed_name: str, bootstraps: int
) -> tuple[float, float]:
    """Resample tasks, because runs of one task are not independent."""

    per = pd.DataFrame({"task": frame.task_id.values, "value": values}).groupby("task").value
    totals, counts = per.sum().to_numpy(), per.size().to_numpy()
    rng = np.random.default_rng(derive_seed(seed_name))
    picks = rng.integers(0, len(totals), size=(bootstraps, len(totals)))
    boots = totals[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def acceptance_table(frame, policies, config) -> pd.DataFrame:
    """Apply the three frozen acceptance routes against always-readout."""

    bootstraps = int(config["bootstraps"])
    selector, selector_cost, _ = policies["selector"]
    readout, readout_cost, _ = policies["always_readout"]
    abstaining, _, _ = policies["abstaining_readout"]
    gain = readout["regret"] - selector["regret"]
    low, high = cluster_contrast(frame, gain, "selector-v3-regret", bootstraps)
    cost_ratio = selector_cost.mean() / readout_cost.mean()
    similar_regret = float(config["similar_regret"])
    similar_cost = float(config["similar_overhead_fraction"])
    reliability = (~selector["abstain"] & ~selector["error"]).astype(float) - (
        ~abstaining["abstain"] & ~abstaining["error"]
    ).astype(float)
    rel_low, rel_high = cluster_contrast(frame, reliability, "selector-v3-reliability", bootstraps)
    return pd.DataFrame(
        [
            {
                "route": "lower_regret_than_always_readout",
                "estimate": float(gain.mean()),
                "cluster_low": low,
                "cluster_high": high,
                "passed": low > 0.0,
            },
            {
                "route": "similar_regret_lower_overhead",
                "estimate": float(cost_ratio),
                "cluster_low": low,
                "cluster_high": high,
                "passed": bool(low >= -similar_regret and cost_ratio < 1.0),
            },
            {
                "route": "similar_overhead_higher_reliable_coverage",
                "estimate": float(reliability.mean()),
                "cluster_low": rel_low,
                "cluster_high": rel_high,
                "passed": bool(abs(cost_ratio - 1.0) <= similar_cost and rel_low > 0.0),
            },
        ]
    )


def certificate_outcomes(frame: pd.DataFrame) -> dict[str, pd.Series]:
    ideal = np.stack(frame.ideal.to_numpy())
    best = np.argmax(ideal, axis=1)
    index = np.arange(len(frame))

    def wrong(column: str, passed: np.ndarray) -> np.ndarray:
        winner = frame[column].to_numpy(int)
        safe = np.where(winner >= 0, winner, 0)
        return passed & (ideal[index, safe] < ideal[index, best])

    joint = frame.certified_look.to_numpy() > 0
    layer = frame.l1_look.to_numpy() > 0
    naive = frame.naive_pass.to_numpy(bool)
    return {
        "both_layers": (joint, wrong("certified_winner", joint), frame.certified_look.to_numpy()),
        "layer_one_only": (layer, wrong("l1_winner", layer), frame.l1_look.to_numpy()),
        "naive_independent_z": (
            naive,
            wrong("naive_winner", naive),
            np.where(naive, 5, 0),
        ),
    }


def certificate_rows(frame: pd.DataFrame, identity: dict) -> list[dict]:
    rows = []
    batches = int(CONFIG["certification"]["batches"])
    for name, (passed, wrong, look) in certificate_outcomes(frame).items():
        rate = float(wrong.sum() / passed.sum()) if passed.sum() else np.nan
        looks = look[passed]
        rows.append(
            {
                **identity,
                "rule": name,
                "runs": len(frame),
                "tasks": int(frame.task_id.nunique()),
                "certified_rate": float(passed.mean()),
                "false_certification_per_run": float(wrong.mean()),
                "false_certification_given_certified": rate,
                "missed_or_unresolved_rate": float(1.0 - passed.mean()),
                "mean_shot_fraction_when_certified": float(looks.mean() / batches)
                if len(looks)
                else np.nan,
            }
        )
    return rows


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    tasks = task_table(manifest)
    raw = pd.read_csv(RAW_PATH).merge(tasks, on="task_id", validate="many_to_one")
    candidates = int(CONFIG["candidates_per_task"])
    parts = {
        split: raw.loc[raw.split == split].reset_index(drop=True)
        for split in ("train", "validation", "holdout")
    }
    arrays = {split: parse(frame, candidates) for split, frame in parts.items()}
    rule, selection = choose_rule(
        parts["train"], arrays["train"], parts["validation"], arrays["validation"], CONFIG
    )
    selection.to_csv(RULE_PATH, index=False, lineterminator="\n", float_format="%.12g")

    frame, data = parts["holdout"], arrays["holdout"]
    policies = evaluate_policies(frame, data, rule, CONFIG)
    rows = [
        {"policy": name, **summarize(frame, outcome, cost, share)}
        for name, (outcome, cost, share) in policies.items()
    ]
    rows.append(oracle_row(frame, policies))
    pd.DataFrame(rows).to_csv(HOLDOUT_PATH, index=False, lineterminator="\n", float_format="%.12g")
    acceptance_table(frame, policies, CONFIG).to_csv(
        ACCEPTANCE_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )

    group_rows = []
    for column in ("family", "width", "noise", "gap_class", "budget"):
        for value in sorted(frame[column].unique()):
            mask = (frame[column] == value).to_numpy()
            sub = frame.loc[mask].reset_index(drop=True)
            for name, (outcome, cost, share) in policies.items():
                part = {key: item[mask] for key, item in outcome.items()}
                group_rows.append(
                    {
                        "grouping": column,
                        "value": value,
                        "policy": name,
                        **summarize(sub, part, cost[mask], share),
                    }
                )
    pd.DataFrame(group_rows).to_csv(
        GROUPS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )

    leave_rows = []
    for column in ("family", "width", "noise"):
        for value in sorted(raw[column].unique()):
            keep_train = parts["train"][column] != value
            keep_val = parts["validation"][column] != value
            train = parts["train"].loc[keep_train].reset_index(drop=True)
            validation = parts["validation"].loc[keep_val].reset_index(drop=True)
            held = frame.loc[frame[column] == value].reset_index(drop=True)
            if held.empty:
                continue
            fold_rule, _ = choose_rule(
                train,
                parse(train, candidates),
                validation,
                parse(validation, candidates),
                CONFIG,
            )
            fold = evaluate_policies(held, parse(held, candidates), fold_rule, CONFIG)
            for name in ("selector", "always_readout", "abstaining_readout"):
                outcome, cost, share = fold[name]
                leave_rows.append(
                    {
                        "left_out": column,
                        "value": value,
                        "rule": fold_rule.name,
                        "policy": name,
                        **summarize(held, outcome, cost, share),
                    }
                )
    pd.DataFrame(leave_rows).to_csv(
        LEAVE_OUT_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )

    certificate_summary = certificate_rows(frame, {"grouping": "all", "value": "all"})
    pd.DataFrame(certificate_summary).to_csv(
        CERTIFICATE_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    certificate_groups = []
    for column in ("noise", "gap_class", "family", "width", "budget"):
        for value in sorted(frame[column].unique()):
            sub = frame.loc[frame[column] == value].reset_index(drop=True)
            certificate_groups.extend(certificate_rows(sub, {"grouping": column, "value": value}))
    pd.DataFrame(certificate_groups).to_csv(
        CERTIFICATE_GROUPS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )
    fired = frame.loc[frame.l1_look > 0]
    alarm_rows = [
        {
            "alarm": name,
            "runs_passing_layer_one": len(fired),
            "fire_rate": float(((fired.first_alarm_bits.to_numpy() >> bit) & 1).mean())
            if len(fired)
            else np.nan,
        }
        for bit, name in enumerate(ALARMS)
    ]
    pd.DataFrame(alarm_rows).to_csv(
        ALARMS_PATH, index=False, lineterminator="\n", float_format="%.12g"
    )


if __name__ == "__main__":
    main()
