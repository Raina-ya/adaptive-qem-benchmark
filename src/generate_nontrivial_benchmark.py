"""Generate nontrivial-family and distribution-shift repeat-level evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.adaptive_benchmark import (
    build_hardware_efficient_manifest,
    build_noise_shift_manifest,
    execute_hardware_efficient_row,
    hardware_efficient_coverage,
    noise_shift_specs,
)
from src.generate_days08_12 import _write_canonical_csv

RESULTS_RAW = Path("results/raw")
RESULTS_PROCESSED = Path("results/processed")
FIGURES = Path("figures")
MANIFESTS = Path("experiments/manifests")


def _execute_with_checkpoint(
    manifest: pd.DataFrame,
    *,
    checkpoint_path: Path,
    shifted_noise: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    shifts = noise_shift_specs() if shifted_noise else {}
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        rows.append(
            execute_hardware_efficient_row(
                row,
                noise_override=shifts[str(row.noise)] if shifted_noise else None,
            )
        )
        if index % 250 == 0:
            _write_canonical_csv(pd.DataFrame(rows), checkpoint_path)
    result = pd.DataFrame(rows)
    _write_canonical_csv(result, checkpoint_path)
    return result


def _error_vs_depth(results: pd.DataFrame, path: Path) -> None:
    complete = results.loc[results.status == "complete"].copy()
    complete["absolute_error"] = (
        complete.estimate - complete.noiseless_statevector_reference
    ).abs()
    summary = (
        complete.groupby(["depth", "method"], sort=True).absolute_error.mean().reset_index()
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for method, group in summary.groupby("method", sort=True):
        axis.plot(group.depth, group.absolute_error, marker="o", label=method)
    axis.set(
        xlabel="Hardware-efficient depth parameter",
        ylabel="Mean absolute error",
        xticks=sorted(summary.depth.unique()),
    )
    axis.set_title("Circuit depth changes mitigation accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    for directory in (RESULTS_RAW, RESULTS_PROCESSED, FIGURES, MANIFESTS):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = build_hardware_efficient_manifest(repeats=50)
    _write_canonical_csv(manifest, MANIFESTS / "hardware_efficient_v1_manifest.csv")
    checkpoint = RESULTS_RAW / ".hardware_efficient_v1_checkpoint.csv"
    results = _execute_with_checkpoint(
        manifest, checkpoint_path=checkpoint, shifted_noise=False
    )
    output = RESULTS_RAW / "hardware_efficient_v1_repeats-50.csv"
    checkpoint.replace(output)
    _write_canonical_csv(
        hardware_efficient_coverage(results),
        RESULTS_PROCESSED / "hardware_efficient_v1_coverage.csv",
    )
    _error_vs_depth(results, FIGURES / "hardware_efficient_error_vs_depth.png")

    holdout_manifest = build_noise_shift_manifest(repeats=50)
    _write_canonical_csv(holdout_manifest, MANIFESTS / "noise_shift_holdout_v1_manifest.csv")
    holdout_checkpoint = RESULTS_RAW / ".noise_shift_holdout_v1_checkpoint.csv"
    holdout = _execute_with_checkpoint(
        holdout_manifest,
        checkpoint_path=holdout_checkpoint,
        shifted_noise=True,
    )
    holdout_output = RESULTS_RAW / "noise_shift_holdout_v1_repeats-50.csv"
    holdout_checkpoint.replace(holdout_output)
    _write_canonical_csv(
        hardware_efficient_coverage(holdout),
        RESULTS_PROCESSED / "noise_shift_holdout_v1_coverage.csv",
    )


if __name__ == "__main__":
    main()
