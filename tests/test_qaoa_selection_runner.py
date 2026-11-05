import json

import pandas as pd

import src.applications.qaoa_selection as qaoa
from src.noise_models import noise_spec


def _group(method: str) -> list[dict[str, object]]:
    graph = qaoa.benchmark_graphs()[0]
    candidates = qaoa.select_benchmark_candidates(graph)
    return qaoa.run_candidate_selection_group(
        graph,
        candidates,
        method=method,
        total_selection_budget=4096,
        repeat=0,
        manifest_seeds=tuple(range(100, 105)),
        noise=noise_spec("mixed", "medium"),
    )


def test_raw_rows_preserve_candidates_and_reconstruct_energy(monkeypatch) -> None:
    monkeypatch.setattr(
        qaoa,
        "_sample_complete_bitstrings",
        lambda circuit, *, shots, seed, noise: {"0101": shots - 1, "0000": 1},
    )

    rows = _group("raw")
    graph = qaoa.benchmark_graphs()[0]

    assert len(rows) == 5
    assert len({row["candidate_id"] for row in rows}) == 5
    assert all(row["status"] == "complete" for row in rows)
    for row in rows:
        counts = json.loads(row["raw_counts"])
        assert qaoa.expected_cut_from_counts(graph, counts) == row["raw_energy"]
    assert sum(row["candidate_target_shots"] for row in rows) == 2048
    assert rows[0]["selection_execution_count"] == 5


def test_readout_and_zne_overhead_are_fully_charged(monkeypatch) -> None:
    monkeypatch.setattr(
        qaoa,
        "_sample_complete_bitstrings",
        lambda circuit, *, shots, seed, noise: {"0101": shots},
    )

    readout = _group("readout")
    zne = _group("zne")

    assert sum(row["candidate_target_shots"] for row in readout) + readout[0][
        "calibration_shots"
    ] == 2048
    assert sum(bool(row["calibration_data"]) for row in readout) == 1
    assert readout[0]["selection_execution_count"] == 5 + 2**4
    assert sum(sum(json.loads(row["factor_level_shots"])) for row in zne) == 2048
    assert all(row["candidate_execution_count"] == 3 for row in zne)
    assert zne[0]["selection_execution_count"] == 15


def test_failure_is_explicit_and_does_not_drop_candidate(monkeypatch) -> None:
    calls = 0

    def fail_once(circuit, *, shots, seed, noise):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return {"0101": shots}

    monkeypatch.setattr(qaoa, "_sample_complete_bitstrings", fail_once)
    rows = _group("raw")

    assert len(rows) == 5
    assert [row["status"] for row in rows].count("failed") == 1
    assert "injected failure" in next(
        row["failure_reason"] for row in rows if row["status"] == "failed"
    )


def test_fixed_seeds_regenerate_identical_rows(monkeypatch) -> None:
    def seeded_counts(circuit, *, shots, seed, noise):
        first = seed % (shots + 1)
        return {"0101": first, "0000": shots - first}

    monkeypatch.setattr(qaoa, "_sample_complete_bitstrings", seeded_counts)

    pd.testing.assert_frame_equal(pd.DataFrame(_group("raw")), pd.DataFrame(_group("raw")))
