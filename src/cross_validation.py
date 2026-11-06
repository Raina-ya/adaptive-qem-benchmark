"""Compare the independent reference with committed first-path results and mutated versions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import independent_reference as ref

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
PROCESSED = ROOT / "results/processed"
OUTPUT_PATH = PROCESSED / "cross_validation_v1.csv"
MUTATION_PATH = PROCESSED / "cross_validation_mutations_v1.csv"
EXACT_TOLERANCE = 1e-12
# Readout correction is solved by different NNLS algorithms, each with its own stopping rule.
SOLVER_TOLERANCE = 1e-9
# Full-matrix correction at width 8 uses the first path's iterative solver.
ITERATIVE_LABEL = "readout, width 8 (iterative first-path solver)"
TOP_K = 2
SELECTION_KEYS = ["graph_id", "method", "total_selection_budget", "repeat"]
WIDTHS = (2, 4, 6)
DEPTHS = (1, 3, 6)
READOUT_CELL = {"circuit_id": "he_w2_d1", "noise": "asymmetric_readout", "method": "readout"}


def _bucket(difference: float) -> str:
    """Report a difference by order of magnitude so results stay stable across platforms."""

    for limit in (1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-6, 1e-3):
        if difference <= limit:
            return f"<={limit:g}"
    return ">1e-3"


def _row(
    group: str,
    check: str,
    compared: str,
    count: int,
    largest: float,
    tolerance: float,
    *,
    known_limitation: str = "",
) -> dict[str, object]:
    return {
        "known_limitation": known_limitation,
        "group": group,
        "check": check,
        "first_path": compared,
        "values_compared": count,
        "largest_difference": _bucket(largest),
        "tolerance": tolerance,
        "passed": bool(largest <= tolerance),
    }


def exact_checks() -> list[dict[str, object]]:
    """Compare enumerated states with analytic values and stored statevector references."""

    rows = []
    bell = ref.ghz_state(2)
    bell_gates = ref.ghz_state_from_gates(2)
    rows.append(
        _row(
            "exact",
            "bell amplitudes, direct versus gates",
            "analytic",
            bell.size,
            float(np.abs(bell - bell_gates).max()),
            EXACT_TOLERANCE,
        )
    )
    differences = [abs(ref.pauli_expectation(bell, pauli) - 1.0) for pauli in ("XX", "ZZ")]
    differences.append(abs(ref.pauli_expectation(bell, "YY") + 1.0))
    rows.append(_row("exact", "bell XX, ZZ, YY", "analytic", 3, max(differences), EXACT_TOLERANCE))
    ghz_differences = []
    for width in WIDTHS:
        state = ref.ghz_state_from_gates(width)
        ghz_differences.append(abs(ref.pauli_expectation(state, "X" * width) - 1.0))
        for left in range(width - 1):
            pauli = "".join("Z" if q in (left, left + 1) else "I" for q in range(width))
            ghz_differences.append(abs(ref.pauli_expectation(state, pauli) - 1.0))
    rows.append(
        _row(
            "exact",
            "ghz stabilizers widths 2, 4, 6",
            "analytic",
            len(ghz_differences),
            max(ghz_differences),
            EXACT_TOLERANCE,
        )
    )
    stored = _efficient_references()
    differences = [
        abs(
            ref.pauli_expectation(
                ref.hardware_efficient_state(width, depth),
                ref.observable_for_hardware_efficient(width),
            )
            - stored[f"he_w{width}_d{depth}"]
        )
        for width in WIDTHS
        for depth in DEPTHS
    ]
    rows.append(
        _row(
            "exact",
            "hardware-efficient references, nine circuits",
            "hardware_efficient_v1 stored reference",
            len(differences),
            max(differences),
            EXACT_TOLERANCE,
        )
    )
    return rows


def _efficient_references() -> dict[str, float]:
    frame = pd.read_csv(
        RAW / "hardware_efficient_v1_repeats-50.csv",
        usecols=["circuit_id", "noiseless_statevector_reference"],
    )
    return frame.groupby("circuit_id").noiseless_statevector_reference.first().to_dict()


def _readout_rows() -> pd.DataFrame:
    frame = pd.read_csv(RAW / "noise_shift_holdout_v1_repeats-50.csv", low_memory=False)
    selection = frame
    for column, value in READOUT_CELL.items():
        selection = selection.loc[selection[column] == value]
    return selection.loc[selection.status == "complete"].sort_values("repeat")


def readout_checks() -> list[dict[str, object]]:
    """Rebuild one asymmetric-readout cell from its fixed counts, repeat by repeat."""

    rows = _readout_rows()
    pauli = str(rows.observable_q0_first.iloc[0])
    width = int(rows.width.iloc[0])
    raw_gap, matrix_gap, condition_gap, corrected_gap, estimate_gap = [], [], [], [], []
    shot_gap = 0
    for row in rows.itertuples():
        payload = json.loads(row.raw_payload_json)
        counts = json.loads(payload["raw_counts_json"])
        records = json.loads(payload["calibration_counts_json"])
        raw_gap.append(abs(ref.expectation_from_counts(counts, pauli) - payload["raw_estimate"]))
        matrix = ref.assignment_matrix(records, width)
        matrix_gap.append(
            float(np.abs(matrix - np.array(json.loads(payload["assignment_matrix_json"]))).max())
        )
        condition_gap.append(
            abs(ref.infinity_condition_number(matrix) - payload["condition_number"])
        )
        observed = ref.probabilities_from_counts(counts, width)
        corrected = ref.correct_readout(matrix, observed)
        stored = np.array(json.loads(payload["corrected_probabilities_json"]))
        corrected_gap.append(float(np.abs(corrected - stored).max()))
        estimate_gap.append(
            abs(ref.expectation_from_probabilities(corrected, pauli) - row.estimate)
        )
        target = sum(counts.values())
        calibration = sum(sum(item["counts"].values()) for item in records)
        expected = ref.split_evenly(int(payload["calibration_shots"]), 2**width)
        shots_used = tuple(sum(item["counts"].values()) for item in records)
        shot_gap += int(
            target != payload["target_shots"]
            or calibration != payload["calibration_shots"]
            or target + calibration != payload["total_shot_budget"]
            or shots_used != expected
            or row.total_circuit_executions != 1 + 2**width
        )
    cell = "he_w2_d1 asymmetric readout, 50 repeats"
    source = "noise_shift_holdout_v1 repeat rows"
    count = len(rows)
    return [
        _row(
            "readout",
            f"{cell}: raw parity from counts",
            source,
            count,
            max(raw_gap),
            EXACT_TOLERANCE,
        ),
        _row(
            "readout",
            f"{cell}: assignment matrix",
            source,
            count * 16,
            max(matrix_gap),
            EXACT_TOLERANCE,
        ),
        _row(
            "readout",
            f"{cell}: condition number",
            source,
            count,
            max(condition_gap),
            SOLVER_TOLERANCE,
        ),
        _row(
            "readout",
            f"{cell}: corrected probabilities",
            source,
            count * 4,
            max(corrected_gap),
            SOLVER_TOLERANCE,
        ),
        _row(
            "readout",
            f"{cell}: corrected expectation",
            source,
            count,
            max(estimate_gap),
            SOLVER_TOLERANCE,
        ),
        _row(
            "readout", f"{cell}: shot and execution accounting", source, count, float(shot_gap), 0.0
        ),
    ]


def _qaoa_rows() -> pd.DataFrame:
    return pd.read_csv(RAW / "qaoa_selection_v1.csv", low_memory=False, keep_default_na=False)


def _number(value: object) -> float:
    return float(value) if value != "" else float("nan")


def _estimate_from_row(row, edges, width, matrix) -> float:
    if row.method == "raw":
        return ref.energy_from_counts(json.loads(row.raw_counts), edges)
    if row.method == "zne":
        factors = json.loads(row.factor_level_counts)
        estimates = [ref.energy_from_counts(factors[str(f)], edges) for f in (1, 3, 5)]
        return ref.linear_intercept((1.0, 3.0, 5.0), estimates)
    observed = ref.probabilities_from_counts(json.loads(row.raw_counts), width)
    corrected = ref.correct_readout(matrix, observed)
    return float(corrected @ ref.cut_table(width, edges))


def qaoa_checks() -> list[dict[str, object]]:
    """Rebuild candidate energies, rankings, and selection metrics from fixed counts."""

    frame = _qaoa_rows()
    graphs = ref.benchmark_graphs()
    rows: list[dict[str, object]] = []
    ideal_gap, cut_gap, edge_gap = [], [], 0
    seen = frame.drop_duplicates(["graph_id", "candidate_id"])
    for row in seen.itertuples():
        width, edges = graphs[row.graph_id]
        edge_gap += int(len(edges) != int(row.edge_count))
        energy = ref.ideal_energy(
            width, edges, json.loads(row.beta_parameters)[0], json.loads(row.gamma_parameters)[0]
        )
        ideal_gap.append(abs(energy - float(row.exact_ideal_energy)))
        cut_gap.append(abs(ref.exact_maxcut(width, edges)[0] - float(row.exact_classical_maxcut)))
    rows.append(
        _row(
            "qaoa",
            "ideal energies, 15 candidates",
            "qaoa_selection_v1",
            len(ideal_gap),
            max(ideal_gap),
            EXACT_TOLERANCE,
        )
    )
    rows.append(
        _row(
            "qaoa",
            "exact MaxCut optimum and edge counts",
            "qaoa_selection_v1",
            len(cut_gap),
            max(cut_gap) + edge_gap,
            0.0,
        )
    )

    complete = frame.loc[frame.status == "complete"]
    # Width 8 uses an iterative solver with a looser stopping rule, so it is reported separately.
    groups = {
        "raw and ZNE, all graphs": complete.loc[complete.method.isin(("raw", "zne"))],
        "readout, widths 4 and 6": complete.loc[
            (complete.method == "readout") & (complete.width < 8)
        ],
        "readout, width 8 (iterative first-path solver)": complete.loc[
            (complete.method == "readout") & (complete.width == 8)
        ],
    }
    calibration = {}
    for row in frame.loc[frame.calibration_data != ""].itertuples():
        calibration[(row.graph_id, row.total_selection_budget, row.repeat)] = row.calibration_data
    for label, group in groups.items():
        rows.extend(_selection_group(label, group, graphs, calibration))
    return rows


def _selection_group(label, group, graphs, calibration) -> list[dict[str, object]]:
    energy_gap, raw_energy_gap, rmse_gap, regret_gap = [], [], [], []
    selection_flip, ranking_flip, recall_gap, shot_gap = 0, 0, [], 0
    decisions = 0
    for keys, selection in group.groupby(SELECTION_KEYS, sort=True):
        if len(selection) != 5:
            continue
        graph_id, method, budget, repeat = keys
        width, edges = graphs[graph_id]
        matrix = None
        if method == "readout":
            records = json.loads(calibration[(graph_id, budget, repeat)])
            matrix = ref.assignment_matrix(records, width)
        estimates, stored = [], []
        for row in selection.itertuples():
            estimates.append(_estimate_from_row(row, edges, width, matrix))
            stored.append(float(row.corrected_or_extrapolated_estimate))
            raw_counts = json.loads(row.raw_counts)
            raw_energy_gap.append(
                abs(ref.energy_from_counts(raw_counts, edges) - float(row.raw_energy))
            )
        energy_gap.append(float(np.abs(np.array(estimates) - np.array(stored)).max()))
        ids = selection.candidate_id.tolist()
        exact = selection.exact_ideal_energy.astype(float).tolist()
        mine = ref.selection_metrics(ids, exact, estimates, top_k=TOP_K)
        theirs = ref.selection_metrics(ids, exact, stored, top_k=TOP_K)
        selection_flip += int(mine["selected"] != theirs["selected"])
        ranking_flip += int(mine["ranking"] != theirs["ranking"])
        rmse_gap.append(abs(mine["energy_rmse"] - theirs["energy_rmse"]))
        regret_gap.append(abs(mine["simple_regret"] - theirs["simple_regret"]))
        recall_gap.append(abs(mine["top_k_recall"] - theirs["top_k_recall"]))
        shot_gap += _shot_mismatches(selection, method, budget, width)
        decisions += 1
    tolerance = SOLVER_TOLERANCE if "readout" in label else EXACT_TOLERANCE
    source = "qaoa_selection_v1 repeat rows"
    limitation = (
        "first-path iterative solver stops at a relative step of 1e-9, so energies are approximate"
        if label == ITERATIVE_LABEL
        else ""
    )
    return [
        _row(
            "qaoa",
            f"{label}: energies from fixed counts",
            source,
            decisions * 5,
            max(energy_gap),
            tolerance,
            known_limitation=limitation,
        ),
        _row(
            "qaoa",
            f"{label}: raw energies",
            source,
            decisions * 5,
            max(raw_energy_gap),
            EXACT_TOLERANCE,
        ),
        _row("qaoa", f"{label}: selected candidate", source, decisions, float(selection_flip), 0.0),
        _row(
            "qaoa", f"{label}: full candidate ranking", source, decisions, float(ranking_flip), 0.0
        ),
        _row(
            "qaoa", f"{label}: simple regret", source, decisions, max(regret_gap), EXACT_TOLERANCE
        ),
        _row("qaoa", f"{label}: top-k recall", source, decisions, max(recall_gap), 0.0),
        _row(
            "qaoa",
            f"{label}: shot and execution accounting",
            source,
            decisions,
            float(shot_gap),
            0.0,
        ),
    ]


def _shot_mismatches(selection: pd.DataFrame, method: str, budget: int, width: int) -> int:
    selection_shots = budget // 2
    calibration = selection_shots // 5 if method == "readout" else 0
    expected = ref.split_evenly(selection_shots - calibration, 5)
    used = []
    for row in selection.itertuples():
        if method == "zne":
            factors = json.loads(row.factor_level_counts)
            used.append(sum(sum(factors[str(f)].values()) for f in (1, 3, 5)))
        else:
            used.append(sum(json.loads(row.raw_counts).values()))
    executions = {"raw": 5, "readout": 5 + 2**width, "zne": 15}[method]
    wrong = tuple(used) != expected
    wrong = wrong or (selection.selection_shots.astype(int) != selection_shots).any()
    wrong = wrong or (selection.calibration_shots.astype(int) != calibration).any()
    wrong = wrong or (selection.selection_execution_count.astype(int) != executions).any()
    return int(wrong)


def summary_check() -> list[dict[str, object]]:
    """Rebuild the committed per-cell selection summary from independent decisions."""

    frame = _qaoa_rows()
    graphs = ref.benchmark_graphs()
    complete = frame.loc[(frame.status == "complete") & (frame.width < 8)]
    complete = pd.concat(
        [
            complete,
            frame.loc[
                (frame.status == "complete")
                & frame.method.isin(("raw", "zne"))
                & (frame.width == 8)
            ],
        ]
    )
    calibration = {
        (row.graph_id, row.total_selection_budget, row.repeat): row.calibration_data
        for row in frame.loc[frame.calibration_data != ""].itertuples()
    }
    decisions = []
    for keys, selection in complete.groupby(SELECTION_KEYS, sort=True):
        graph_id, method, budget, repeat = keys
        width, edges = graphs[graph_id]
        matrix = None
        if method == "readout":
            matrix = ref.assignment_matrix(
                json.loads(calibration[(graph_id, budget, repeat)]), width
            )
        estimates = [
            _estimate_from_row(row, edges, width, matrix) for row in selection.itertuples()
        ]
        metrics = ref.selection_metrics(
            selection.candidate_id.tolist(),
            selection.exact_ideal_energy.astype(float).tolist(),
            estimates,
            top_k=TOP_K,
        )
        decisions.append(
            {
                "graph_id": graph_id,
                "method": method,
                "total_selection_budget": budget,
                "decision_error": metrics["decision_error"],
                "simple_regret": metrics["simple_regret"],
                "top_k_recall": metrics["top_k_recall"],
            }
        )
    keys = ["graph_id", "method", "total_selection_budget"]
    mine = pd.DataFrame(decisions).groupby(keys).mean().sort_index()
    saved = pd.read_csv(PROCESSED / "qaoa_selection_summary_v1.csv")
    saved = saved.loc[saved.candidate_gap_regime == "overall"].set_index(keys)
    saved = saved.loc[mine.index]
    gaps = [
        (mine.decision_error - saved.decision_error_probability).abs().max(),
        (mine.simple_regret - saved.mean_simple_regret).abs().max(),
        (mine.top_k_recall - saved.mean_top_k_recall).abs().max(),
    ]
    return [
        _row(
            "qaoa",
            "per-cell decision error, regret, and recall",
            "qaoa_selection_summary_v1",
            len(mine) * 3,
            float(max(gaps)),
            SOLVER_TOLERANCE,
        )
    ]


def mutation_table() -> pd.DataFrame:
    """Break one convention at a time and record whether the comparison notices."""

    rows = []

    def add(name: str, target: str, difference: float, tolerance: float, sensitive: bool) -> None:
        rows.append(
            {
                "mutation": name,
                "target": target,
                "convention_sensitive": sensitive,
                "detected": bool(difference > tolerance),
                "largest_difference": _bucket(difference),
            }
        )

    state = ref.hardware_efficient_state(2, 1)
    stored = _efficient_references()["he_w2_d1"]
    add(
        "reverse bit order",
        "hardware-efficient XZ exact expectation",
        abs(ref.pauli_expectation(state, "XZ", reverse_bits=True) - stored),
        EXACT_TOLERANCE,
        True,
    )
    rows_readout = _readout_rows()
    pauli = str(rows_readout.observable_q0_first.iloc[0])
    width = int(rows_readout.width.iloc[0])
    payload = json.loads(rows_readout.raw_payload_json.iloc[0])
    counts = json.loads(payload["raw_counts_json"])
    records = json.loads(payload["calibration_counts_json"])
    stored_matrix = np.array(json.loads(payload["assignment_matrix_json"]))
    add(
        "reverse bit order",
        "XZ parity from fixed counts (full-support Pauli)",
        abs(
            ref.expectation_from_counts(counts, pauli, reverse_bits=True) - payload["raw_estimate"]
        ),
        EXACT_TOLERANCE,
        False,
    )
    add(
        "reverse bit order",
        "ZI parity from fixed counts",
        abs(
            ref.expectation_from_counts(counts, "ZI", reverse_bits=True)
            - ref.expectation_from_counts(counts, "ZI")
        ),
        EXACT_TOLERANCE,
        True,
    )
    add(
        "reverse bit order",
        "assignment matrix",
        float(
            np.abs(ref.assignment_matrix(records, width, reverse_bits=True) - stored_matrix).max()
        ),
        EXACT_TOLERANCE,
        True,
    )
    add(
        "transpose assignment matrix",
        "assignment matrix",
        float(np.abs(ref.assignment_matrix(records, width, transpose=True) - stored_matrix).max()),
        EXACT_TOLERANCE,
        True,
    )
    observed = ref.probabilities_from_counts(counts, width)
    transposed = ref.correct_readout(
        ref.assignment_matrix(records, width, transpose=True), observed
    )
    add(
        "transpose assignment matrix",
        "corrected expectation",
        abs(
            ref.expectation_from_probabilities(transposed, pauli)
            - float(rows_readout.estimate.iloc[0])
        ),
        SOLVER_TOLERANCE,
        True,
    )
    frame = _qaoa_rows()
    graphs = ref.benchmark_graphs()
    for graph_id, sensitive in (("medium_dense8", True), ("sparse6", False), ("ring4", False)):
        width_q, edges = graphs[graph_id]
        selection = frame.loc[
            (frame.graph_id == graph_id) & (frame.method == "raw") & (frame.status == "complete")
        ]
        largest = max(
            abs(
                ref.energy_from_counts(json.loads(row.raw_counts), edges, reverse_bits=True)
                - float(row.raw_energy)
            )
            for row in selection.itertuples()
        )
        add(
            "reverse bit order",
            f"{graph_id} energies from fixed counts",
            largest,
            EXACT_TOLERANCE,
            sensitive,
        )
    width_q, edges = graphs["medium_dense8"]
    row = frame.loc[frame.graph_id == "medium_dense8"].iloc[0]
    add(
        "flip QAOA gamma sign",
        "medium_dense8 ideal energy",
        abs(
            ref.ideal_energy(
                width_q,
                edges,
                json.loads(row.beta_parameters)[0],
                json.loads(row.gamma_parameters)[0],
                gamma_sign=-1.0,
            )
            - float(row.exact_ideal_energy)
        ),
        EXACT_TOLERANCE,
        True,
    )
    bell = ref.ghz_state(2)
    add(
        "reverse bit order",
        "bell XX expectation",
        abs(ref.pauli_expectation(bell, "XX", reverse_bits=True) - 1.0),
        EXACT_TOLERANCE,
        False,
    )
    return pd.DataFrame(rows)


def build_report() -> pd.DataFrame:
    return pd.DataFrame([*exact_checks(), *readout_checks(), *qaoa_checks(), *summary_check()])


def main() -> None:
    report = build_report()
    mutations = mutation_table()
    report.to_csv(OUTPUT_PATH, index=False, lineterminator="\n")
    mutations.to_csv(MUTATION_PATH, index=False, lineterminator="\n")
    failed = report.loc[~report.passed & (report.known_limitation.fillna("") == "")]
    missed = mutations.loc[mutations.convention_sensitive != mutations.detected]
    if len(failed) or len(missed):
        raise SystemExit(f"Cross-validation failed:\n{failed.to_string()}\n{missed.to_string()}")


if __name__ == "__main__":
    main()
