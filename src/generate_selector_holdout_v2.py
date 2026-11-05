"""Run the preregistered fresh grid for selector v2 and score it afterward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit

from src.adaptive_benchmark import noise_shift_specs
from src.circuits import ghz_circuit
from src.config import derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.hardware_efficient_circuits import (
    hardware_efficient_circuit,
    hardware_efficient_observable,
)
from src.metrics import wilson_interval
from src.mitigation.folding import fold_global
from src.mitigation.readout import (
    _inverse_and_condition_inf,
    estimate_assignment_matrix,
    run_readout_mitigation,
)
from src.mitigation.zne import run_linear_zne
from src.noise_models import NoiseSpec, noise_spec
from src.observables import exact_expectation
from src.selector_v2 import (
    CONFIG,
    PilotEvidence,
    pauli_standard_error,
    pilot_budget,
    select_v2,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/manifests/selector_v2_holdout_manifest.csv"
RAW_PATH = ROOT / "results/raw/selector_holdout_v2.csv"
CELLS_PATH = ROOT / "results/processed/selector_v2_cells.csv"
SUMMARY_PATH = ROOT / "results/processed/selector_v2_summary.csv"
HOLDOUT = CONFIG["holdout"]
METHODS = ("raw", "readout", "zne", "selector_v2")
CELL_KEYS = ["family", "circuit_id", "width", "depth", "replicate", "noise", "shots"]
NEGLIGIBLE = float(CONFIG["negligible_regret_max"])
CATASTROPHIC = float(CONFIG["catastrophic_regret_absolute"])
RELATIVE = float(CONFIG["catastrophic_regret_relative_to_absolute_reference"])


def circuit_grid() -> list[tuple[str, str, int, int, int, QuantumCircuit, str]]:
    """Return every preregistered circuit and its q0-first observable."""

    grid = []
    for width in HOLDOUT["families"]["ghz"]["widths"]:
        grid.append(("ghz", f"ghz_w{width}", width, 1, 0, ghz_circuit(width), "X" * width))
    family = HOLDOUT["families"]["hardware_efficient"]
    for width in family["widths"]:
        for depth in family["depths"]:
            for replicate in family["replicates"]:
                grid.append(
                    (
                        "hardware_efficient",
                        f"he_w{width}_d{depth}_r{replicate}",
                        width,
                        depth,
                        replicate,
                        hardware_efficient_circuit(width, depth, replicate=replicate),
                        hardware_efficient_observable(width),
                    )
                )
    return grid


def noise_table() -> dict[str, NoiseSpec]:
    """Return mixed levels and shifted channels by their declared names."""

    table = {level: noise_spec("mixed", level) for level in HOLDOUT["noise_levels"]}
    shifts = noise_shift_specs()
    table.update({name: shifts[name] for name in HOLDOUT["noise_shifts"]})
    return table


def build_manifest(repeats: int | None = None) -> pd.DataFrame:
    """Build the complete frozen run list without executing a circuit."""

    total = int(HOLDOUT["repeats"]) if repeats is None else repeats
    rows = []
    for index, (family, circuit_id, width, depth, replicate, _, observable) in enumerate(
        circuit_grid()
    ):
        for noise_index, noise in enumerate(noise_table()):
            for shots in HOLDOUT["total_shots"]:
                for method_index, method in enumerate(METHODS):
                    for repeat in range(total):
                        rows.append(
                            {
                                "family": family,
                                "circuit_id": circuit_id,
                                "width": width,
                                "depth": depth,
                                "replicate": replicate,
                                "observable": observable,
                                "noise": noise,
                                "shots": shots,
                                "method": method,
                                "repeat": repeat,
                                "seed": derive_seed(
                                    HOLDOUT["seed_namespace"],
                                    index,
                                    noise_index,
                                    shots,
                                    method_index,
                                    repeat,
                                ),
                                "status": "pending",
                            }
                        )
    return pd.DataFrame(rows)


def run_pilot(
    circuit: QuantumCircuit, observable: str, total_shots: int, spec: NoiseSpec, seed: int
) -> tuple[PilotEvidence, int]:
    """Spend the pilot budget and return runtime-visible evidence."""

    pilot, calibration, first, fifth = pilot_budget(total_shots)
    assignment, _, _ = estimate_assignment_matrix(
        circuit.num_qubits, calibration, spec, seed=derive_seed("selector-v2-calibration", seed)
    )
    _, condition = _inverse_and_condition_inf(assignment)
    low = estimate_pauli(
        circuit,
        observable,
        shots=first,
        seed=derive_seed("selector-v2-factor", seed, 1),
        noise_spec=spec,
    )
    high = estimate_pauli(
        fold_global(circuit, 5),
        observable,
        shots=fifth,
        seed=derive_seed("selector-v2-factor", seed, 5),
        noise_spec=spec,
    )
    evidence = PilotEvidence(
        e1=low.estimate,
        se1=pauli_standard_error(low.estimate, first),
        e5=high.estimate,
        se5=pauli_standard_error(high.estimate, fifth),
        condition_number=condition,
    )
    return evidence, pilot


def _run_readout(
    circuit: QuantumCircuit, observable: str, shots: int, spec: NoiseSpec, seed: int
) -> tuple[float, dict[str, object]]:
    result = run_readout_mitigation(
        circuit, observable, total_shot_budget=shots, noise_spec=spec, seed=seed
    )
    if result.status != "ok" or result.corrected_estimate is None:
        raise RuntimeError(f"readout mitigation status: {result.status}")
    return float(result.corrected_estimate), {
        "condition_number": result.condition_number,
        "target_counts_json": result.raw_counts_json,
        "calibration_counts_json": result.calibration_counts_json,
    }


def execute_row(
    row: dict[str, object], circuit: QuantumCircuit, spec: NoiseSpec
) -> dict[str, object]:
    """Execute one manifest row and retain failures as explicit rows."""

    output = {
        **row,
        "decision": None,
        "rule_trace": None,
        "pilot_shots": 0,
        "pilot_e1": np.nan,
        "pilot_se1": np.nan,
        "pilot_e5": np.nan,
        "pilot_se5": np.nan,
        "pilot_condition_number": np.nan,
        "estimate": np.nan,
        "total_shots_used": 0,
        "payload_json": None,
        "error": None,
        "result_label": "noisy_simulated",
    }
    method, shots, seed = str(row["method"]), int(row["shots"]), int(row["seed"])
    observable = str(row["observable"])
    try:
        if method == "raw":
            result = estimate_pauli(
                circuit, observable, shots=shots, seed=seed, noise_spec=spec
            )
            output.update(
                estimate=result.estimate,
                total_shots_used=shots,
                payload_json=json.dumps({"counts_json": result.raw_counts_json}),
            )
        elif method == "readout":
            estimate, payload = _run_readout(circuit, observable, shots, spec, seed)
            output.update(
                estimate=estimate, total_shots_used=shots, payload_json=json.dumps(payload)
            )
        elif method == "zne":
            result = run_linear_zne(
                circuit, observable, total_shot_budget=shots, noise_spec=spec, seed=seed
            )
            output.update(
                estimate=result.extrapolated_estimate,
                total_shots_used=shots,
                payload_json=json.dumps(
                    {
                        "factor_estimates": result.factor_estimates,
                        "shots_by_factor": result.shots_by_factor,
                        "residual_sum_squares": result.residual_sum_squares,
                    }
                ),
            )
        else:
            evidence, pilot = run_pilot(circuit, observable, shots, spec, seed)
            selection = select_v2(evidence)
            output.update(
                decision=selection.decision,
                rule_trace=selection.trace,
                pilot_shots=pilot,
                pilot_e1=evidence.e1,
                pilot_se1=evidence.se1,
                pilot_e5=evidence.e5,
                pilot_se5=evidence.se5,
                pilot_condition_number=evidence.condition_number,
                total_shots_used=pilot,
            )
            if selection.decision == "readout":
                estimate, payload = _run_readout(
                    circuit,
                    observable,
                    shots - pilot,
                    spec,
                    derive_seed("selector-v2-final", seed),
                )
                output.update(
                    estimate=estimate, total_shots_used=shots, payload_json=json.dumps(payload)
                )
        output["status"] = "complete"
    except Exception as error:  # row-level audit must retain failures
        output["status"] = "failed"
        output["error"] = f"{type(error).__name__}: {error}"
    return output


def generate(manifest: pd.DataFrame, path: Path = RAW_PATH) -> pd.DataFrame:
    """Run every manifest row, writing a checkpoint that a final rename replaces."""

    circuits = {item[1]: item[5] for item in circuit_grid()}
    noises = noise_table()
    checkpoint = path.with_name(f".{path.name}.checkpoint")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        rows.append(execute_row(row, circuits[row["circuit_id"]], noises[row["noise"]]))
        if index % 500 == 0:
            pd.DataFrame(rows).to_csv(checkpoint, index=False, lineterminator="\n")
            print(f"rows={index}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")
    checkpoint.unlink(missing_ok=True)
    return frame


def score_cells(raw: pd.DataFrame, references: dict[str, float]) -> pd.DataFrame:
    """Score one cell at a time after every decision has already been recorded."""

    rows: list[dict[str, object]] = []
    frame = raw.copy()
    frame["error_value"] = frame.estimate - frame.circuit_id.map(references)
    for keys, group in frame.groupby(CELL_KEYS, sort=True):
        row: dict[str, object] = dict(zip(CELL_KEYS, keys, strict=True))
        reference = references[str(row["circuit_id"])]
        rmse: dict[str, float] = {}
        complete: dict[str, int] = {}
        for method in ("raw", "readout", "zne"):
            rows_method = group.loc[(group.method == method) & (group.status == "complete")]
            complete[method] = len(rows_method)
            rmse[method] = float(np.sqrt(np.mean(rows_method.error_value**2)))
        best = float(np.nanmin(list(rmse.values())))
        policy = group.loc[group.method == "selector_v2"]
        decided = policy.loc[(policy.decision == "readout") & (policy.status == "complete")]
        share = len(decided) / len(policy)
        selector_regret = (
            float(np.sqrt(np.mean(decided.error_value**2)) - best) if len(decided) else np.nan
        )
        counted = share >= float(CONFIG["decided_cell_min_share"])
        raw_regret = rmse["raw"] - best
        row.update(
            reference=reference,
            repeats=len(policy),
            decided_share=share,
            selector_mode=_mode(policy.decision),
            cell_type="decided" if counted else "no_safe",
            mean_shots_used=float(policy.total_shots_used.mean()),
            best_full_budget_rmse=best,
            raw_rmse=rmse["raw"],
            readout_rmse=rmse["readout"],
            zne_rmse=rmse["zne"],
            readout_complete_repeats=complete["readout"],
            selector_regret=selector_regret,
            raw_regret=raw_regret,
            readout_regret=rmse["readout"] - best,
            selector_reliable=bool(counted and selector_regret <= NEGLIGIBLE),
            selector_catastrophic=bool(counted and _catastrophic(selector_regret, reference)),
            raw_catastrophic=_catastrophic(raw_regret, reference),
            readout_catastrophic=_catastrophic(rmse["readout"] - best, reference),
            no_safe_avoided_failure=bool(not counted and raw_regret > NEGLIGIBLE),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mode(values: pd.Series) -> str:
    counts = values.dropna().value_counts()
    return str(sorted(counts[counts == counts.max()].index)[0]) if len(counts) else "failed"


def _catastrophic(regret: float, reference: float) -> bool:
    if np.isnan(regret):
        return False
    return regret >= CATASTROPHIC or regret >= abs(reference) * RELATIVE


def evaluate_criteria(cells: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen criteria and report every result, including failures."""

    decided = cells.loc[cells.cell_type == "decided"]
    no_safe = cells.loc[cells.cell_type == "no_safe"]
    rows: list[dict[str, object]] = []

    def add(
        name: str, value: float | None, low: float | None, high: float | None, passed: bool
    ) -> None:
        rows.append(
            {
                "criterion": name,
                "value": value,
                "wilson_low": low,
                "wilson_high": high,
                "passed": passed,
            }
        )

    if len(decided):
        count = int(decided.selector_reliable.sum())
        low, high = wilson_interval(count, len(decided))
        rate = count / len(decided)
    else:
        count, low, high, rate = 0, None, None, None
    h1 = len(decided) >= 10 and rate is not None and rate >= 0.80 and low >= 0.50
    add("H1_decided_cells_reliable", rate, low, high, h1)
    if len(no_safe):
        avoided = int(no_safe.no_safe_avoided_failure.sum())
        low2, high2 = wilson_interval(avoided, len(no_safe))
        share = avoided / len(no_safe)
    else:
        low2, high2, share = None, None, None
    h2 = len(no_safe) >= 10 and low2 is not None and low2 > 0.50
    add("H2_no_safe_outputs_informative", share, low2, high2, h2)
    if len(decided):
        selector = float(decided.selector_regret.mean())
        raw_regret = float(decided.raw_regret.mean())
        readout = float(decided.readout_regret.mean())
        catastrophic = int(decided.selector_catastrophic.sum())
        h3 = (
            selector < raw_regret
            and selector < readout
            and catastrophic <= int(decided.raw_catastrophic.sum())
            and catastrophic <= int(decided.readout_catastrophic.sum())
        )
    else:
        selector, raw_regret, readout, h3 = None, None, None, False
    add("H3_better_than_fixed_policies", selector, None, None, h3)
    add("verdict_all_criteria", None, None, None, bool(h1 and h2 and h3))
    rows.extend(
        [
            {"criterion": "cells_total", "value": float(len(cells))},
            {"criterion": "cells_decided", "value": float(len(decided))},
            {"criterion": "cells_no_safe", "value": float(len(no_safe))},
            {"criterion": "decided_mean_selector_regret", "value": selector},
            {"criterion": "decided_mean_raw_regret", "value": raw_regret},
            {"criterion": "decided_mean_readout_regret", "value": readout},
        ]
    )
    return pd.DataFrame(rows)


def analyze(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a finished run using exact references only after decisions exist."""

    references = {
        circuit_id: exact_expectation(circuit, observable)
        for _, circuit_id, _, _, _, circuit, observable in circuit_grid()
    }
    cells = score_cells(raw, references)
    return cells, evaluate_criteria(cells)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the frozen manifest without running any circuit",
    )
    args = parser.parse_args(argv)
    manifest = build_manifest()
    if args.manifest_only:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST_PATH, index=False, lineterminator="\n")
        return
    committed = pd.read_csv(MANIFEST_PATH)
    pd.testing.assert_frame_equal(committed, manifest, check_dtype=False)
    raw = generate(manifest)
    cells, summary = analyze(raw)
    cells.to_csv(CELLS_PATH, index=False, lineterminator="\n", float_format="%.12g")
    summary.to_csv(SUMMARY_PATH, index=False, lineterminator="\n", float_format="%.12g")



if __name__ == "__main__":
    main()
