"""Report the frozen v2 benchmark by signal band, attenuation, noise, and folding method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.density_sim import derive_seed
from src.generate_signal_v2 import TRUTH_PATH
from src.signal_manifest_v2 import CONFIG, MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "results/processed"
BANDS_PATH = PROCESSED / "signal_v2_bands.csv"
CIRCUITS_PATH = PROCESSED / "signal_v2_circuits.csv"
ATTENUATION_PATH = PROCESSED / "signal_v2_attenuation.csv"
NOISE_PATH = PROCESSED / "signal_v2_noise.csv"
FOLDING_PATH = PROCESSED / "signal_v2_folding.csv"
CONTRAST_PATH = PROCESSED / "signal_v2_band_contrast.csv"
PRACTICAL = float(CONFIG["practical_threshold"])
ATTENUATION_EDGES = (0.01, 0.05, 0.2)
ATTENUATIONS = ("below_0.01", "0.01_to_0.05", "0.05_to_0.2", "at_least_0.2")
BOOTSTRAPS = 2000


def prepare(truth: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Add outcome labels, excluding the noise-free control from noisy summaries."""

    frame = truth.merge(
        manifest[["circuit_id", "width", "depth", "signal_band", "split", "ideal_expectation"]],
        on="circuit_id",
    )
    frame["gain"] = frame.raw_rmse - frame.fit_rmse
    frame["outcome"] = np.select(
        [frame.gain > PRACTICAL, frame.gain < -PRACTICAL], ["fit_better", "raw_better"], "tie"
    )
    frame["attenuation"] = (frame.raw_mean - frame.reference).abs()
    frame["attenuation_band"] = [
        ATTENUATIONS[int(np.searchsorted(ATTENUATION_EDGES, value, side="right"))]
        for value in frame.attenuation
    ]
    near_zero = frame.signal_band == "near_zero"
    frame["raw_normalized_rmse"] = (frame.raw_rmse / frame.reference.abs()).where(~near_zero)
    frame["fit_normalized_rmse"] = (frame.fit_rmse / frame.reference.abs()).where(~near_zero)
    return frame


def _summary(group: pd.DataFrame) -> dict[str, float]:
    return {
        "conditions": len(group),
        "independent_circuits": int(group.circuit_id.nunique()),
        "median_raw_rmse": float(group.raw_rmse.median()),
        "median_fit_rmse": float(group.fit_rmse.median()),
        "median_raw_normalized_rmse": float(group.raw_normalized_rmse.median()),
        "median_fit_normalized_rmse": float(group.fit_normalized_rmse.median()),
        "mean_gain": float(group.gain.mean()),
        "fit_better_share": float((group.outcome == "fit_better").mean()),
        "raw_better_share": float((group.outcome == "raw_better").mean()),
        "tie_share": float((group.outcome == "tie").mean()),
    }


def _table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = [
        {**dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)), **_summary(group)}
        for key, group in frame.groupby(by, sort=True)
    ]
    return pd.DataFrame(rows)


def circuit_variation(frame: pd.DataFrame) -> pd.DataFrame:
    """Report the spread between circuits inside each band."""

    rows = []
    for (band, design, model), group in frame.groupby(
        ["signal_band", "design", "model"], sort=True
    ):
        per = group.groupby("circuit_id").agg(gain=("gain", "mean"), raw=("raw_rmse", "mean"))
        rows.append(
            {
                "signal_band": band,
                "design": design,
                "model": model,
                "independent_circuits": len(per),
                "mean_circuit_gain": float(per.gain.mean()),
                "sd_circuit_gain": float(per.gain.std(ddof=1)),
                "q10_circuit_gain": float(per.gain.quantile(0.1)),
                "q90_circuit_gain": float(per.gain.quantile(0.9)),
                "min_circuit_gain": float(per.gain.min()),
                "max_circuit_gain": float(per.gain.max()),
                "sd_circuit_raw_rmse": float(per.raw.std(ddof=1)),
                "circuits_with_positive_gain": int((per.gain > 0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def band_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare near-zero and boundary bands, resampling circuits. Protocol hypothesis H1."""

    rows = []
    rng = np.random.default_rng(derive_seed("signal-v2-contrast"))
    target = frame.loc[(frame.design == "global_135") & (frame.model == "linear")]
    per = target.groupby(["signal_band", "circuit_id"]).outcome.apply(
        lambda values: (values == "fit_better").mean()
    )
    near, boundary = per.loc["near_zero"].to_numpy(), per.loc["boundary_adjacent"].to_numpy()
    draws = rng.choice(boundary, (BOOTSTRAPS, len(boundary))).mean(axis=1) - rng.choice(
        near, (BOOTSTRAPS, len(near))
    ).mean(axis=1)
    rows.append(
        {
            "contrast": "boundary_minus_near_zero_fit_better_share",
            "near_zero_circuits": len(near),
            "boundary_circuits": len(boundary),
            "estimate": float(boundary.mean() - near.mean()),
            "cluster_low": float(np.quantile(draws, 0.025)),
            "cluster_high": float(np.quantile(draws, 0.975)),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    frame = prepare(pd.read_csv(TRUTH_PATH), pd.read_csv(MANIFEST_PATH))
    noisy = frame.loc[frame.noise != "none"]
    target = noisy.loc[(noisy.design == "global_135") & (noisy.model == "linear")]
    tables = {
        BANDS_PATH: _table(target, ["signal_band", "total_shot_budget"]),
        CIRCUITS_PATH: circuit_variation(noisy),
        ATTENUATION_PATH: _table(target, ["attenuation_band", "signal_band"]),
        NOISE_PATH: _table(frame, ["noise", "design", "model"]),
        FOLDING_PATH: _table(noisy, ["folding", "design", "model", "noise"]),
        CONTRAST_PATH: band_contrast(noisy),
    }
    for path, table in tables.items():
        table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


if __name__ == "__main__":
    main()
