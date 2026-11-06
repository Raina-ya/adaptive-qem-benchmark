"""Check manifests, recomputed summaries, figures, and the compiled paper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/raw"
PROCESSED = ROOT / "results/processed"
MANIFESTS = ROOT / "experiments/manifests"
REPORT_PATH = PROCESSED / "final_verification_v1.csv"
PDF_PATH = ROOT / "paper/main.pdf"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
TOLERANCE = 1e-9
CORRELATED_MODELS = ("pair_correlated", "calibration_mismatch")
MISMATCH_MAX_WIDTH = 6
PLANNED_NOT_EXECUTED = 960
QAOA_CANDIDATES = 5
REQUIRED_PDF_TEXT = (
    "Abstract",
    "Hardware readiness",
    "no-go",
    "References",
    "Reproducibility",
    "not an independently timestamped preregistration",
    "publication order only",
    "not independently verified",
    "src.claim_ledger",
)


def _row(group: str, check: str, expected: object, observed: object, passed: bool) -> dict:
    return {
        "group": group,
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
    }


def _rows(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns, low_memory=False)


def manifest_checks() -> list[dict]:
    """Confirm that every planned row or repeat has one raw record."""

    rows = []
    for manifest, raw, keep in (
        ("volumetric_v1_manifest", "day12_ghz_volumetric_v1_repeats-50", "applicable"),
        ("hardware_efficient_v1_manifest", "hardware_efficient_v1_repeats-50", "all"),
        ("noise_shift_holdout_v1_manifest", "noise_shift_holdout_v1_repeats-50", "all"),
        ("decision_benchmark_v1_manifest", "decision_margin_benchmark_v1", "all"),
        ("combined_qem_v1_manifest", "combined_qem_v1", "primary"),
        ("qaoa_selection_v1_manifest", "qaoa_selection_v1", "all"),
        ("readout_scalability_v1_manifest", "readout_scalability_v1", "repeats"),
        ("selector_v2_holdout_manifest", "selector_holdout_v2", "all"),
    ):
        planned = _rows(MANIFESTS / f"{manifest}.csv")
        recorded = _rows(RAW / f"{raw}.csv", ["status"] if keep != "primary" else ["record_kind"])
        expected = len(planned)
        observed = len(recorded)
        if keep == "applicable":
            expected = int((planned.status != "not_applicable").sum())
            observed = int((recorded.status != "not_applicable").sum())
        elif keep == "primary":
            observed = int((recorded.record_kind == "primary").sum())
        elif keep == "repeats":
            direct = planned.loc[planned.readout_model.isin(("independent", "asymmetric"))]
            expected = int(direct.planned_repeats.sum())
        rows.append(_row("manifest", manifest, expected, observed, expected == observed))
    rows.extend(_correlated_readout_checks())
    return rows


def _correlated_readout_checks() -> list[dict]:
    """Account for correlated readout rows, which ran outside the scalability grid."""

    planned = _rows(MANIFESTS / "readout_scalability_v1_manifest.csv")
    planned = planned.loc[planned.readout_model.isin(CORRELATED_MODELS)]
    small = int(planned.loc[planned.width <= MISMATCH_MAX_WIDTH, "planned_repeats"].sum())
    large = int(planned.loc[planned.width > MISMATCH_MAX_WIDTH, "planned_repeats"].sum())
    recorded = _rows(RAW / "readout_mismatch_v1.csv", ["width", "target_model"])
    recorded = recorded.loc[recorded.target_model.isin(CORRELATED_MODELS)]
    executed_large = int((recorded.width > MISMATCH_MAX_WIDTH).sum())
    return [
        _row(
            "manifest",
            "readout correlated widths 2-6, run in readout_mismatch_v1",
            small,
            len(recorded),
            small == len(recorded),
        ),
        _row(
            "manifest",
            f"readout correlated widths 8-12, planned_not_executed ({large} repeats disclosed)",
            0,
            executed_large,
            executed_large == 0 and large == PLANNED_NOT_EXECUTED,
        ),
    ]


def _compare(check: str, saved: pd.Series, computed: pd.Series) -> dict:
    joined = pd.concat([saved.rename("saved"), computed.rename("computed")], axis=1, join="inner")
    difference = (joined.saved - joined.computed).abs()
    both_missing = joined.saved.isna() & joined.computed.isna()
    matches = (difference <= TOLERANCE) | both_missing
    valid = len(joined) == len(saved) and bool(matches.all())
    largest = float(difference.max()) if difference.notna().any() else 0.0
    observed = f"{len(joined)} values within {TOLERANCE:g}" if valid else f"max {largest:.3g}"
    return _row("summary", check, f"{len(saved)} values within {TOLERANCE:g}", observed, valid)


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def _absolute_rmse_check(check: str, summary: str, raw: str, keys: list[str]) -> dict:
    saved = pd.read_csv(PROCESSED / f"{summary}.csv").set_index(keys).rmse
    frame = _rows(RAW / f"{raw}.csv", [*keys, "status", "absolute_error"])
    complete = frame.loc[frame.status == "complete"]
    computed = complete.groupby(keys).absolute_error.apply(_rmse)
    return _compare(check, saved.dropna(), computed)


def summary_checks() -> list[dict]:
    """Recompute representative metrics of each direct summary from repeat rows."""

    rows = []
    hardware = _rows(
        RAW / "hardware_efficient_v1_repeats-50.csv",
        [
            "family",
            "circuit_id",
            "width",
            "depth",
            "noise",
            "shots",
            "method",
            "status",
            "estimate",
            "noiseless_statevector_reference",
        ],
    )
    hardware = hardware.loc[hardware.status == "complete"]
    errors = hardware.estimate - hardware.noiseless_statevector_reference
    keys = ["circuit_id", "noise", "shots", "method"]
    computed = errors.groupby([hardware[key] for key in keys]).apply(_rmse)
    benchmark = pd.read_csv(PROCESSED / "benchmark_summary_v1.csv")
    saved = benchmark.loc[benchmark.family == "hardware_efficient"].set_index(keys).rmse
    rows.append(_compare("benchmark_summary_v1 rmse", saved, computed))

    ghz = _rows(
        RAW / "day12_ghz_volumetric_v1_repeats-50.csv",
        ["width", "noise", "shots", "method", "status", "estimate"],
    )
    ghz = ghz.loc[ghz.status == "complete"].copy()
    ghz["circuit_id"] = ghz.width.map(lambda width: f"ghz_w{int(width)}")
    computed = (ghz.estimate - 1.0).groupby([ghz[key] for key in keys]).apply(_rmse)
    saved = benchmark.loc[benchmark.family == "ghz"].set_index(keys).rmse
    rows.append(_compare("benchmark_summary_v1 GHZ rmse", saved, computed))

    keys = ["cell_id", "total_shot_budget", "allocation_policy", "method"]
    combined = _rows(RAW / "combined_qem_v1.csv", [*keys, "status", "record_kind", "signed_error"])
    combined = combined.loc[(combined.record_kind == "primary") & (combined.status == "complete")]
    rows.append(
        _compare(
            "combined_qem_summary_v1 rmse",
            pd.read_csv(PROCESSED / "combined_qem_summary_v1.csv").set_index(keys).rmse,
            combined.groupby(keys).signed_error.apply(_rmse),
        )
    )

    keys = ["margin", "noise", "shots", "method", "record_mode"]
    decision = _rows(RAW / "decision_margin_benchmark_v1.csv", [*keys, "status", "decision_error"])
    decision = decision.loc[decision.status == "complete"]
    rows.append(
        _compare(
            "decision_margin_summary_v1 error",
            pd.read_csv(PROCESSED / "decision_margin_summary_v1.csv")
            .set_index(keys)
            .decision_error_probability,
            decision.groupby(keys).decision_error.apply(lambda item: item.astype(bool).mean()),
        )
    )

    keys = ["task", "margin", "noise", "max_total_shots", "method", "policy"]
    sequential = _rows(RAW / "sequential_decision_v1.csv", [*keys, "wrong_certified"])
    rows.append(
        _compare(
            "sequential_decision_summary_v1 wrong certification",
            pd.read_csv(PROCESSED / "sequential_decision_summary_v1.csv")
            .set_index(keys)
            .wrong_certified_rate,
            sequential.groupby(keys).wrong_certified.mean(),
        )
    )

    readout_keys = ["width", "distribution", "readout_model", "total_shot_budget", "method"]
    for summary, raw in (
        ("readout_tensor_validation_summary_v1", "readout_tensor_validation_v1"),
        ("readout_subspace_validation_summary_v1", "readout_subspace_validation_v1"),
        ("readout_matrix_free_validation_summary_v1", "readout_matrix_free_validation_v1"),
        ("readout_scalability_summary_v1", "readout_scalability_v1"),
    ):
        rows.append(_absolute_rmse_check(f"{summary} rmse", summary, raw, readout_keys))
    rows.append(
        _absolute_rmse_check(
            "readout_mismatch_summary_v1 rmse",
            "readout_mismatch_summary_v1",
            "readout_mismatch_v1",
            [
                "width",
                "distribution",
                "target_model",
                "calibration_model",
                "total_shot_budget",
                "method",
            ],
        )
    )

    qaoa = _rows(RAW / "qaoa_selection_v1.csv", ["status"])
    summary = pd.read_csv(PROCESSED / "qaoa_selection_summary_v1.csv")
    overall = summary.loc[summary.candidate_gap_regime == "overall"]
    resolved = int(overall.resolved_repeats.sum())
    complete = int((qaoa.status == "complete").sum()) // QAOA_CANDIDATES
    rows.append(
        _row("summary", "qaoa_selection_summary_v1 resolved groups", complete, resolved,
             complete == resolved)
    )

    cells = pd.read_csv(PROCESSED / "selector_v2_cells.csv")
    references = cells.set_index("circuit_id").reference.groupby(level=0).first()
    raw_v2 = _rows(
        RAW / "selector_holdout_v2.csv",
        ["circuit_id", "noise", "shots", "method", "status", "estimate"],
    )
    baseline = raw_v2.loc[(raw_v2.method == "raw") & (raw_v2.status == "complete")]
    keys = ["circuit_id", "noise", "shots"]
    error = baseline.estimate - baseline.circuit_id.map(references)
    rows.append(
        _compare(
            "selector_v2_cells raw rmse",
            cells.set_index(keys).raw_rmse,
            error.groupby([baseline[key] for key in keys]).apply(_rmse),
        )
    )
    return rows


def figure_checks() -> list[dict]:
    """Require every figure to be a nonempty PNG."""

    rows = []
    for directory in (ROOT / "figures", ROOT / "paper/figures"):
        paths = sorted(directory.glob("*.png"))
        invalid = [
            path.name
            for path in paths
            if path.stat().st_size == 0 or path.read_bytes()[:8] != PNG_SIGNATURE
        ]
        rows.append(
            _row(
                "figure",
                directory.relative_to(ROOT).as_posix(),
                f"{len(paths)} valid PNG files",
                f"{len(paths) - len(invalid)} valid PNG files",
                len(paths) > 0 and not invalid,
            )
        )
    return rows


def pdf_checks(path: Path = PDF_PATH) -> list[dict]:
    """Check that the paper parses, has pages, and contains required sections."""

    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    normalized = " ".join(text.split())
    rows = [_row("pdf", "page count", "at least 1", "parsed", len(reader.pages) >= 1)]
    for item in REQUIRED_PDF_TEXT:
        rows.append(_row("pdf", f"contains {item}", True, item in normalized, item in normalized))
    return rows


def build_report() -> pd.DataFrame:
    """Run every check and keep failures visible."""

    return pd.DataFrame(
        [*manifest_checks(), *summary_checks(), *figure_checks(), *pdf_checks()]
    )


def main() -> None:
    report = build_report()
    report.to_csv(REPORT_PATH, index=False, lineterminator="\n")
    failed = report.loc[~report.passed]
    if len(failed):
        raise SystemExit(f"{len(failed)} final verification checks failed:\n{failed.to_string()}")


if __name__ == "__main__":
    main()
