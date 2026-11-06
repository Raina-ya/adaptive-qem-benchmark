"""Build the condition-level ZNE credibility gate from C4 raw evidence."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.zne_validity_gate import evaluate_zne_gate

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "results/raw/zne_falsification_v1.csv"
OUTPUT_PATH = ROOT / "results/processed/zne_validity_v1.csv"
FIGURE_PATH = ROOT / "figures/zne_signal_retention_map.png"
KEYS = ["circuit_id", "noise", "total_shot_budget"]
NOISE_ORDER = ("zero", "low", "medium", "high")


def _final_retention(payload: str) -> float:
    values = json.loads(payload)
    return float(values[-1])


def build_validity_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Pair genuine and garbage repeats and apply the frozen validity gate."""

    if not frame.status.eq("complete").all():
        raise ValueError("Validity gate requires explicit complete input rows")
    genuine = frame.loc[frame.method == "genuine_folding"].copy()
    garbage = frame.loc[frame.method == "garbage_folding"].copy()
    rows: list[dict[str, object]] = []
    for keys, group in genuine.groupby(KEYS, sort=True):
        paired = group.merge(
            garbage,
            on=[*KEYS, "repeat"],
            suffixes=("_genuine", "_garbage"),
            validate="one_to_one",
        ).sort_values("repeat")
        reference = float(paired.reference_genuine.iloc[0])
        lower, upper = (-1.0, 1.0)
        overshoots = np.maximum.reduce(
            (
                np.full(len(paired), lower) - paired.estimate_genuine.to_numpy(),
                paired.estimate_genuine.to_numpy() - np.full(len(paired), upper),
                np.zeros(len(paired)),
            )
        )
        retentions = np.array(
            [_final_retention(payload) for payload in paired.signal_retention_json_genuine]
        )
        result = evaluate_zne_gate(
            paired.true_gain_genuine.to_numpy(),
            paired.true_gain_garbage.to_numpy(),
            retentions,
            paired.negative_mass_genuine.to_numpy(),
            overshoots,
            paired.validity_classification_genuine.eq("signal_destroyed").to_numpy(),
            paired.validity_classification_genuine.eq("extrapolation_unstable").to_numpy(),
        )
        row = dict(zip(KEYS, keys, strict=True))
        row.update(
            reference=reference,
            apparent_gain_mean=float(paired.apparent_gain_genuine.mean()),
            validated_gain=(
                result.true_gain_mean if result.validity_status == "credible" else np.nan
            ),
            **result.__dict__,
            failure_reasons_json=json.dumps(result.failure_reasons),
            result_label="noisy_simulated",
            status="complete",
        )
        row.pop("failure_reasons")
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(frame: pd.DataFrame) -> None:
    circuits = sorted(frame.circuit_id.unique())
    row_labels = [f"{circuit}\n{noise}" for circuit in circuits for noise in NOISE_ORDER]
    budgets = sorted(frame.total_shot_budget.unique())
    matrix = np.zeros((len(row_labels), len(budgets)))
    annotations: list[list[str]] = [["" for _ in budgets] for _ in row_labels]
    for row_index, (circuit, noise) in enumerate(
        (circuit, noise) for circuit in circuits for noise in NOISE_ORDER
    ):
        for column_index, budget in enumerate(budgets):
            row = frame[
                (frame.circuit_id == circuit)
                & (frame.noise == noise)
                & (frame.total_shot_budget == budget)
            ].iloc[0]
            matrix[row_index, column_index] = row.validity_status == "credible"
            annotations[row_index][column_index] = (
                f"{row.validity_status}\nR={row.final_retention_mean:.2f}"
            )
    figure, axis = plt.subplots(figsize=(7.5, 6.2))
    image = axis.imshow(
        matrix, aspect="auto", vmin=0, vmax=1, cmap=ListedColormap(["#e45756", "#54a24b"])
    )
    del image
    axis.set_xticks(range(len(budgets)), budgets)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set(xlabel="Total shot budget", ylabel="Circuit and noise")
    axis.set_title("ZNE credibility requires every declared validity check")
    for row_index, row_annotations in enumerate(annotations):
        for column_index, text in enumerate(row_annotations):
            axis.text(column_index, row_index, text, ha="center", va="center", fontsize=8)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=180)
    plt.close(figure)


def main() -> None:
    frame = pd.read_csv(INPUT_PATH)
    validity = build_validity_table(frame)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    validity.to_csv(OUTPUT_PATH, index=False)
    _plot(validity)


if __name__ == "__main__":
    main()
