"""Batched sequential certification of two-candidate decisions under a shot budget."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import derive_seed
from src.metrics import wilson_interval
from src.resource_accounting import split_shots

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/configs/sequential_decision_v1.yaml"
RAW_PATH = ROOT / "results/raw/sequential_decision_v1.csv"
SUMMARY_PATH = ROOT / "results/processed/sequential_decision_summary_v1.csv"
SHOTS_FIGURE = ROOT / "figures/shots_to_certification.png"
TRADEOFF_FIGURE = ROOT / "figures/certification_error_tradeoff.png"

SUMMARY_KEYS = ["task", "margin", "noise", "max_total_shots", "method", "policy"]
POLICY_LABELS = {
    "fixed_budget": "Fixed budget",
    "wilson_stopping": "Wilson",
    "gap_confidence_stopping": "Gap confidence",
    "variance_aware_stopping": "Variance aware",
    "naive_wald_peeking": "Naive peeking (control)",
}
POLICY_COLORS = {
    "fixed_budget": "#2a78d6",
    "wilson_stopping": "#eb6834",
    "gap_confidence_stopping": "#1baf7a",
    "variance_aware_stopping": "#6250d6",
    "naive_wald_peeking": "#6B6B6B",
}
METHOD_MARKERS = {"raw": "o", "readout": "s", "zne": "^"}
INK = "#2B2B2B"
MUTED = "#6B6B6B"
GRID = "#E4E4E4"
# Rounding keeps a probability of exactly one half on one side of NumPy's binomial branch.
PROBABILITY_DECIMALS = 12


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    """Load the JSON-compatible frozen configuration."""

    return json.loads(path.read_text(encoding="utf-8"))


def _ry(theta: float) -> np.ndarray:
    half = theta / 2.0
    single = np.array(
        [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]], dtype=float
    )
    return np.kron(np.eye(2), single)


# Basis index is q1 q0, so qubit 0 is the least significant bit and the CX control.
_CX = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
    dtype=float,
)
_Z0 = np.diag([1.0, -1.0, 1.0, -1.0])


def _depolarize_q0(rho: np.ndarray, probability: float) -> np.ndarray:
    if probability == 0.0:
        return rho
    q1_marginal = np.array(
        [
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]],
        ]
    )
    return (1.0 - probability) * rho + probability * np.kron(q1_marginal, np.eye(2) / 2.0)


def _depolarize_pair(rho: np.ndarray, probability: float) -> np.ndarray:
    if probability == 0.0:
        return rho
    return (1.0 - probability) * rho + probability * np.eye(4) / 4.0


def circuit_expectation(
    expectation: float, depth: int, factor: int, one_qubit: float, two_qubit: float
) -> float:
    """Return the exact noisy Z0 value of U(U†U)^n before readout error."""

    if factor < 1 or factor % 2 == 0:
        raise ValueError("Folding factor must be a positive odd integer")
    theta = math.acos(expectation)
    rho = np.zeros((4, 4), dtype=float)
    rho[0, 0] = 1.0

    def forward(state: np.ndarray) -> np.ndarray:
        gate = _ry(theta)
        state = _depolarize_q0(gate @ state @ gate.T, one_qubit)
        for _ in range(depth):
            state = _depolarize_pair(_CX @ state @ _CX.T, two_qubit)
        return state

    def backward(state: np.ndarray) -> np.ndarray:
        for _ in range(depth):
            state = _depolarize_pair(_CX @ state @ _CX.T, two_qubit)
        gate = _ry(-theta)
        return _depolarize_q0(gate @ state @ gate.T, one_qubit)

    rho = forward(rho)
    for _ in range((factor - 1) // 2):
        rho = forward(backward(rho))
    return float(np.trace(_Z0 @ rho))


def intercept_weights(factors: tuple[int, ...]) -> np.ndarray:
    """Return OLS weights that evaluate a fitted line at noise factor zero."""

    values = np.asarray(factors, dtype=float)
    centered = values - values.mean()
    return 1.0 / len(values) - values.mean() * centered / float(centered @ centered)


def batch_schedule(target_shots: int, batch_size: int) -> tuple[int, ...]:
    """Split target shots into full batches and one final partial batch."""

    if target_shots < 1 or batch_size < 1:
        raise ValueError("Target shots and batch size must be positive")
    full, remainder = divmod(target_shots, batch_size)
    return (batch_size,) * full + ((remainder,) if remainder else ())


def wilson_bounds(
    probability: np.ndarray, trials: np.ndarray, z: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return elementwise Wilson score bounds."""

    denominator = 1.0 + z**2 / trials
    center = (probability + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2))
        / denominator
    )
    return np.clip(center - radius, 0.0, 1.0), np.clip(center + radius, 0.0, 1.0)


def mover_interval(
    estimate: np.ndarray,
    gradient: np.ndarray,
    probability: np.ndarray,
    trials: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine component Wilson bounds into an interval for a linear contrast."""

    z = float(norm.ppf(1.0 - alpha / 2.0))
    lower, upper = wilson_bounds(probability, trials, z)
    toward_low = np.where(gradient >= 0.0, probability - lower, upper - probability)
    toward_high = np.where(gradient >= 0.0, upper - probability, probability - lower)
    down = np.sqrt(np.sum((gradient * toward_low) ** 2, axis=-1))
    up = np.sqrt(np.sum((gradient * toward_high) ** 2, axis=-1))
    return estimate - down, estimate + up


def hoeffding_halfwidth(gradient: np.ndarray, trials: np.ndarray, alpha: float) -> np.ndarray:
    """Return a variance-free two-sided Hoeffding half-width for a Bernoulli contrast."""

    return np.sqrt(math.log(2.0 / alpha) / 2.0 * np.sum(gradient**2 / trials, axis=-1))


def empirical_bernstein_halfwidth(
    gradient: np.ndarray, probability: np.ndarray, trials: np.ndarray, alpha: float
) -> np.ndarray:
    """Return a per-component empirical-Bernstein half-width, union-bounded and summed."""

    log_term = math.log(3.0 * gradient.shape[-1] / alpha)
    radius = (
        np.sqrt(2.0 * probability * (1.0 - probability) * log_term / trials)
        + 3.0 * log_term / trials
    )
    return np.sum(np.abs(gradient) * np.minimum(radius, 1.0), axis=-1)


def wald_halfwidth(
    gradient: np.ndarray, probability: np.ndarray, trials: np.ndarray, alpha: float
) -> np.ndarray:
    """Return a plug-in normal half-width with no repeated-look correction."""

    z = float(norm.ppf(1.0 - alpha / 2.0))
    variance = probability * (1.0 - probability) / trials
    return z * np.sqrt(np.sum(gradient**2 * variance, axis=-1))


def policy_intervals(
    policy: str,
    estimate: np.ndarray,
    gradient: np.ndarray,
    probability: np.ndarray,
    trials: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one interval per look, using only the policy's own error allowance."""

    looks = len(estimate)
    per_look = alpha / looks
    if policy == "fixed_budget":
        return mover_interval(estimate, gradient, probability, trials, alpha)
    if policy == "wilson_stopping":
        return mover_interval(estimate, gradient, probability, trials, per_look)
    if policy == "gap_confidence_stopping":
        half = hoeffding_halfwidth(gradient, trials, per_look)
    elif policy == "variance_aware_stopping":
        half = empirical_bernstein_halfwidth(gradient, probability, trials, per_look)
    elif policy == "naive_wald_peeking":
        half = wald_halfwidth(gradient, probability, trials, alpha)
    else:
        raise ValueError(f"Unknown stopping policy: {policy}")
    return estimate - half, estimate + half


def first_certified_look(policy: str, lower: np.ndarray, upper: np.ndarray) -> int | None:
    """Return the zero-based stopping look, or None if the interval never excludes zero."""

    certified = (lower > 0.0) | (upper < 0.0)
    if policy == "fixed_budget":
        return len(lower) - 1 if certified[-1] else None
    hits = np.flatnonzero(certified)
    return int(hits[0]) if hits.size else None


@dataclass(frozen=True)
class CellSpec:
    """One task, noise, budget, and method combination."""

    task: str
    margin: float
    noise: str
    max_total_shots: int
    method: str


@dataclass(frozen=True)
class Trajectory:
    """Cumulative look-level evidence shared by every policy for one repeat."""

    estimate: np.ndarray
    gradient: np.ndarray
    probability: np.ndarray
    trials: np.ndarray
    target_shots: np.ndarray
    executions: np.ndarray
    gate_shots: np.ndarray
    calibration_shots: int
    status: str


def _candidate_noise(spec: CellSpec, config: dict[str, object]) -> dict[str, dict[str, float]]:
    task = config["tasks"][spec.task]
    strength = config["noise_strengths"][spec.noise]
    fraction = float(config["readout_p10_fraction_of_p01"])
    multiplier = float(task["readout_multiplier_a"])
    p01 = float(strength["readout_p01"])
    return {
        "a": {"p01": multiplier * p01, "p10": multiplier * fraction * p01},
        "b": {"p01": p01, "p10": fraction * p01},
    }


def observed_probabilities(spec: CellSpec, config: dict[str, object]) -> dict[str, np.ndarray]:
    """Return P(Z=+1 observed) per candidate and sampled folding factor."""

    task = config["tasks"][spec.task]
    strength = config["noise_strengths"][spec.noise]
    factors = tuple(config["zne_factors"]) if spec.method == "zne" else (1,)
    readout = _candidate_noise(spec, config)
    base = float(task["base_expectation"])
    output: dict[str, np.ndarray] = {}
    for label, sign, depth in (
        ("a", 1.0, int(task["depth_a"])),
        ("b", -1.0, int(task["depth_b"])),
    ):
        p01 = readout[label]["p01"]
        p10 = readout[label]["p10"]
        values = []
        for factor in factors:
            value = circuit_expectation(
                base + sign * spec.margin / 2.0,
                depth,
                factor,
                float(strength["one_qubit"]),
                float(strength["two_qubit"]),
            )
            observed = (1.0 - p01 - p10) * value + (p10 - p01)
            values.append((1.0 + observed) / 2.0)
        output[label] = np.array(values, dtype=float)
    return output


def sampling_probability(probability: float) -> float:
    """Round before sampling so last-bit differences cannot switch the binomial branch."""

    return round(float(probability), PROBABILITY_DECIMALS)


def asymptotic_gap(spec: CellSpec, config: dict[str, object]) -> float:
    """Return the gap this estimator converges to with unlimited shots."""

    probabilities = observed_probabilities(spec, config)
    readout = _candidate_noise(spec, config)
    gaps = {}
    for label in ("a", "b"):
        observed = 2.0 * probabilities[label] - 1.0
        if spec.method == "raw":
            gaps[label] = float(observed[0])
        elif spec.method == "readout":
            p01 = readout[label]["p01"]
            p10 = readout[label]["p10"]
            gaps[label] = float((observed[0] - (p10 - p01)) / (1.0 - p01 - p10))
        else:
            weights = intercept_weights(tuple(config["zne_factors"]))
            gaps[label] = float(weights @ observed)
    return gaps["a"] - gaps["b"]


def simulate_trajectory(spec: CellSpec, seed: int, config: dict[str, object]) -> Trajectory:
    """Sample one full-budget trajectory; policies later read its looks in order."""

    task = config["tasks"][spec.task]
    rng = np.random.default_rng(seed)
    probabilities = observed_probabilities(spec, config)
    readout = _candidate_noise(spec, config)
    factors = tuple(config["zne_factors"]) if spec.method == "zne" else (1,)
    calibration_shots = (
        int(spec.max_total_shots * float(config["readout_calibration_fraction"]))
        if spec.method == "readout"
        else 0
    )
    schedule = batch_schedule(
        spec.max_total_shots - calibration_shots, int(config["batch_target_shots"])
    )
    count = len(factors)
    successes = np.zeros((len(schedule), 2, count), dtype=float)
    trials = np.zeros((len(schedule), 2, count), dtype=float)
    gate_shots = np.zeros(len(schedule), dtype=float)
    depths = (int(task["depth_a"]), int(task["depth_b"]))
    for look, batch in enumerate(schedule):
        for candidate, (label, shots) in enumerate(
            zip(("a", "b"), split_shots(batch, 2), strict=True)
        ):
            for index, factor_shots in enumerate(split_shots(shots, count)):
                successes[look, candidate, index] = rng.binomial(
                    factor_shots, sampling_probability(probabilities[label][index])
                )
                trials[look, candidate, index] = factor_shots
                gate_shots[look] += factor_shots * factors[index] * depths[candidate]
    successes = successes.cumsum(axis=0)
    trials = trials.cumsum(axis=0)
    looks = len(schedule)
    p_hat = (successes / trials).reshape(looks, 2 * count)
    n_hat = trials.reshape(looks, 2 * count)
    target_shots = np.cumsum(schedule, dtype=float)
    executions = (np.arange(looks, dtype=float) + 1.0) * 2.0 * count
    status = "complete"

    if spec.method == "raw":
        gradient = np.tile([2.0, -2.0], (looks, 1))
        estimate = np.sum(gradient * p_hat, axis=1)
    elif spec.method == "zne":
        weights = intercept_weights(factors)
        row = np.concatenate([2.0 * weights, -2.0 * weights])
        gradient = np.tile(row, (looks, 1))
        # The OLS intercept weights sum to one, so the constant terms of A and B cancel.
        estimate = np.sum(gradient * p_hat, axis=1)
    else:
        shared = task["calibration"] == "shared"
        pairs = 1 if shared else 2
        state_shots = split_shots(calibration_shots, 2 * pairs)
        cal_p: list[float] = []
        cal_n: list[float] = []
        hats: list[tuple[float, float]] = []
        cursor = 0
        for label in ("a", "b")[:pairs]:
            n0, n1 = state_shots[cursor], state_shots[cursor + 1]
            cursor += 2
            a_hat = rng.binomial(n0, readout[label]["p01"]) / n0
            b_hat = rng.binomial(n1, readout[label]["p10"]) / n1
            hats.append((a_hat, b_hat))
            cal_p.extend([a_hat, b_hat])
            cal_n.extend([float(n0), float(n1)])
        if shared:
            hats.append(hats[0])
        scale_a = 1.0 - sum(hats[0])
        scale_b = 1.0 - sum(hats[1])
        if scale_a <= 0.0 or scale_b <= 0.0:
            status = "calibration_invalid"
            scale_a = scale_b = 1.0
        obs_a = 2.0 * p_hat[:, 0] - 1.0
        obs_b = 2.0 * p_hat[:, 1] - 1.0
        corrected_a = (obs_a - (hats[0][1] - hats[0][0])) / scale_a
        corrected_b = (obs_b - (hats[1][1] - hats[1][0])) / scale_b
        estimate = corrected_a - corrected_b
        target_grad = np.column_stack(
            [np.full(looks, 2.0 / scale_a), np.full(looks, -2.0 / scale_b)]
        )
        if shared:
            # The offset cancels; only the common scale moves the shared-calibration gap.
            cal_grad = np.column_stack([estimate / scale_a, estimate / scale_a])
        else:
            cal_grad = np.column_stack(
                [
                    (1.0 + corrected_a) / scale_a,
                    (corrected_a - 1.0) / scale_a,
                    -(1.0 + corrected_b) / scale_b,
                    -(corrected_b - 1.0) / scale_b,
                ]
            )
        gradient = np.hstack([target_grad, cal_grad])
        p_hat = np.hstack([p_hat, np.tile(cal_p, (looks, 1))])
        n_hat = np.hstack([n_hat, np.tile(cal_n, (looks, 1))])
        executions = executions + 2.0 * pairs

    return Trajectory(
        estimate=estimate,
        gradient=gradient,
        probability=p_hat,
        trials=n_hat,
        target_shots=target_shots,
        executions=executions,
        gate_shots=gate_shots.cumsum(),
        calibration_shots=calibration_shots,
        status=status,
    )


def score_policy(
    policy: str, trajectory: Trajectory, exact_gap: float, alpha: float
) -> dict[str, object]:
    """Apply one stopping rule and score it against the exact gap afterward."""

    lower, upper = policy_intervals(
        policy,
        trajectory.estimate,
        trajectory.gradient,
        trajectory.probability,
        trajectory.trials,
        alpha,
    )
    looks = len(lower)
    if trajectory.status != "complete":
        stop = looks - 1
        status, certified_choice = "failed", ""
    else:
        hit = first_certified_look(policy, lower, upper)
        if hit is None:
            stop, status, certified_choice = looks - 1, "unresolved", ""
        else:
            stop, status = hit, "certified"
            certified_choice = "a" if lower[hit] > 0.0 else "b"
    estimate = float(trajectory.estimate[stop])
    if certified_choice:
        forced_choice = certified_choice
    else:
        forced_choice = "a" if estimate > 0.0 else "b" if estimate < 0.0 else "tie"
    has_winner = exact_gap != 0.0
    correct = "a" if exact_gap > 0.0 else "b"
    forced_error = forced_choice != correct if has_winner else None
    wrong_certified = bool(certified_choice) and (certified_choice != correct or not has_winner)
    regret = abs(exact_gap)
    target = float(trajectory.target_shots[stop])
    return {
        "status": status,
        "looks_available": looks,
        "stop_look": stop + 1,
        "target_shots_used": int(target),
        "calibration_shots": trajectory.calibration_shots,
        "total_shots_used": int(target) + trajectory.calibration_shots,
        "circuit_executions": int(trajectory.executions[stop]),
        "two_qubit_gate_shots": int(trajectory.gate_shots[stop]),
        "gap_estimate": estimate,
        "interval_low": float(lower[stop]),
        "interval_high": float(upper[stop]),
        "certified_choice": certified_choice,
        "forced_choice": forced_choice,
        "wrong_certified": wrong_certified,
        "wrong_early_stop": wrong_certified and stop + 1 < looks,
        "forced_error": forced_error,
        "forced_regret": regret if forced_error else 0.0,
        "abstain_regret": regret if wrong_certified and has_winner else 0.0,
    }


def _grid(config: dict[str, object]) -> list[CellSpec]:
    cells = []
    for task, definition in config["tasks"].items():
        for margin in definition["margins"]:
            for noise in definition["noise"]:
                for budget in config["max_total_shots"]:
                    for method in config["methods"]:
                        cells.append(CellSpec(task, float(margin), noise, int(budget), method))
    return cells


def cell_seed(spec: CellSpec, repeat: int, config: dict[str, object]) -> int:
    """Derive one trajectory seed; every policy in a repeat reads the same data."""

    return derive_seed(
        "sequential-decision-v1",
        list(config["tasks"]).index(spec.task),
        round(spec.margin * 1000),
        list(config["noise_strengths"]).index(spec.noise),
        spec.max_total_shots,
        list(config["methods"]).index(spec.method),
        repeat,
    )


def generate_rows(config: dict[str, object], repeats: int | None = None) -> pd.DataFrame:
    """Run the complete frozen grid and keep every repeat-level policy outcome."""

    alpha = 1.0 - float(config["confidence_level"])
    policies = [*config["policies"], *config["negative_control_policies"]]
    total = int(config["repeats"]) if repeats is None else repeats
    rows: list[dict[str, object]] = []
    for spec in _grid(config):
        limit = asymptotic_gap(spec, config)
        for repeat in range(total):
            seed = cell_seed(spec, repeat, config)
            trajectory = simulate_trajectory(spec, seed, config)
            for policy in policies:
                rows.append(
                    {
                        "task": spec.task,
                        "margin": spec.margin,
                        "noise": spec.noise,
                        "max_total_shots": spec.max_total_shots,
                        "method": spec.method,
                        "policy": policy,
                        "repeat": repeat,
                        "seed": seed,
                        "asymptotic_gap": limit,
                        **score_policy(policy, trajectory, spec.margin, alpha),
                        "result_label": "noisy_simulated",
                    }
                )
    return pd.DataFrame(rows)


def _rate(count: int, trials: int) -> tuple[float, float, float]:
    low, high = wilson_interval(count, trials)
    return count / trials, low, high


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeat rows without removing unresolved or wrong outcomes."""

    rows: list[dict[str, object]] = []
    for keys, group in raw.groupby(SUMMARY_KEYS, sort=True):
        trials = len(group)
        row: dict[str, object] = dict(zip(SUMMARY_KEYS, keys, strict=True))
        unresolved = float((group.status == "unresolved").mean())
        wrong, wrong_low, wrong_high = _rate(int(group.wrong_certified.sum()), trials)
        early, early_low, early_high = _rate(int(group.wrong_early_stop.sum()), trials)
        forced_regret = float(group.forced_regret.mean())
        abstain_regret = float(group.abstain_regret.mean())
        row.update(
            repeats=trials,
            asymptotic_gap=float(group.asymptotic_gap.iloc[0]),
            mean_total_shots=float(group.total_shots_used.mean()),
            median_total_shots=float(group.total_shots_used.median()),
            mean_circuit_executions=float(group.circuit_executions.mean()),
            mean_two_qubit_gate_shots=float(group.two_qubit_gate_shots.mean()),
            certified_rate=float((group.status == "certified").mean()),
            unresolved_rate=unresolved,
            failed_rate=float((group.status == "failed").mean()),
            wrong_certified_rate=wrong,
            wrong_certified_wilson_low=wrong_low,
            wrong_certified_wilson_high=wrong_high,
            wrong_early_stop_rate=early,
            wrong_early_stop_wilson_low=early_low,
            wrong_early_stop_wilson_high=early_high,
            forced_regret=forced_regret,
            abstain_regret=abstain_regret,
            abstention_break_even_penalty=(
                (forced_regret - abstain_regret) / unresolved if unresolved > 0.0 else np.nan
            ),
        )
        if float(keys[1]) == 0.0:
            row.update(
                forced_decision_error=np.nan,
                forced_error_wilson_low=np.nan,
                forced_error_wilson_high=np.nan,
            )
        else:
            forced = group.forced_error.astype(bool)
            error, error_low, error_high = _rate(int(forced.sum()), trials)
            row.update(
                forced_decision_error=error,
                forced_error_wilson_low=error_low,
                forced_error_wilson_high=error_high,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _style_axis(axis: plt.Axes) -> None:
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def plot_shots_to_certification(raw: pd.DataFrame, path: Path = SHOTS_FIGURE) -> None:
    """Plot how the budget share spent depends on margin, method, and stopping rule."""

    budget = 16384
    subset = raw.loc[
        (raw.task == "symmetric_gap")
        & (raw.noise == "medium")
        & (raw.max_total_shots == budget)
        & (raw.policy != "naive_wald_peeking")
    ]
    mean = (
        subset.groupby(["method", "policy", "margin"], sort=True)
        .total_shots_used.mean()
        .reset_index()
    )
    titles = {"raw": "Raw", "readout": "Readout mitigation", "zne": "Linear ZNE"}
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
    for axis, method in zip(axes, ("raw", "readout", "zne"), strict=True):
        for policy in POLICY_LABELS:
            line = mean.loc[(mean.method == method) & (mean.policy == policy)]
            if line.empty:
                continue
            axis.plot(
                line.margin,
                line.total_shots_used,
                color=POLICY_COLORS[policy],
                marker="o",
                markersize=5,
                linewidth=2,
                label=POLICY_LABELS[policy],
            )
        axis.set_title(titles[method], color=INK, fontsize=11, loc="left")
        axis.set_xlabel("Exact decision margin", color=MUTED, fontsize=9)
        axis.set_xticks(sorted(mean.margin.unique()))
        axis.set_ylim(0, budget * 1.06)
        _style_axis(axis)
    axes[0].set_ylabel("Mean total shots, including calibration", color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=4, frameon=False, fontsize=9)
    figure.suptitle(
        "Clear decisions stop early; near ties use most of the 16,384-shot budget "
        "(symmetric task, medium noise)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_certification_tradeoff(raw: pd.DataFrame, path: Path = TRADEOFF_FIGURE) -> None:
    """Plot wrong certification against unresolved rate for each task."""

    budget = 16384
    subset = raw.loc[raw.max_total_shots == budget].copy()
    subset["group"] = subset.task
    subset.loc[subset.margin == 0.0, "group"] = "null_gap"
    subset["unresolved"] = subset.status == "unresolved"
    mean = (
        subset.groupby(["group", "method", "policy"], sort=True)[
            ["unresolved", "wrong_certified"]
        ]
        .mean()
        .reset_index()
    )
    panels = (
        ("null_gap", "No true winner (margin 0)"),
        ("symmetric_gap", "Symmetric task"),
        ("gate_imbalanced", "Gate-noise imbalance"),
        ("readout_imbalanced", "Readout imbalance"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharex=True, sharey=True)
    for axis, (group, title) in zip(axes, panels, strict=True):
        frame = mean.loc[mean.group == group]
        for row in frame.itertuples():
            axis.scatter(
                row.unresolved,
                row.wrong_certified,
                s=70,
                marker=METHOD_MARKERS[row.method],
                color=POLICY_COLORS[row.policy],
                edgecolors="white",
                linewidths=1.2,
                zorder=3,
            )
        axis.axhline(0.05, color=MUTED, linestyle="--", linewidth=1)
        axis.set_title(title, color=INK, fontsize=11, loc="left")
        axis.set_xlabel("Unresolved rate", color=MUTED, fontsize=9)
        axis.set_xlim(-0.04, 1.04)
        axis.set_ylim(-0.04, 1.04)
        _style_axis(axis)
    axes[0].set_ylabel("Wrong-certification rate", color=MUTED, fontsize=9)
    axes[0].annotate("5% nominal", (-0.02, 0.07), ha="left", color=MUTED, fontsize=8)
    policy_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=8,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        for policy in POLICY_LABELS
    ]
    method_names = {"raw": "Raw", "readout": "Readout", "zne": "ZNE"}
    method_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="",
            marker=marker,
            markersize=8,
            color=MUTED,
            label=method_names[method],
        )
        for method, marker in METHOD_MARKERS.items()
    ]
    figure.legend(
        handles=policy_handles + method_handles,
        loc="lower center",
        ncols=8,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Look correction controls false certification but cannot remove estimator bias "
        "(16,384 shots, pooled over margins and noise)",
        color=INK,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    config = load_config()
    raw = generate_rows(config)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH, index=False, lineterminator="\n", float_format="%.12g")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summarize(raw).to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.12g")
    plot_shots_to_certification(raw)
    plot_certification_tradeoff(raw)


if __name__ == "__main__":
    main()
