from src.generate_decision_benchmark import build_manifest


def test_decision_manifest_is_complete_and_deterministic() -> None:
    first = build_manifest(repeats=2)
    second = build_manifest(repeats=2)
    assert first.equals(second)
    assert len(first) == 4 * 3 * 3 * 2 * 4 * 2
    assert first.seed.is_unique
    assert set(first.status) == {"pending"}


def test_every_manifest_cell_uses_declared_grid() -> None:
    manifest = build_manifest(repeats=2)
    assert set(manifest.margin) == {0.02, 0.05, 0.10, 0.20}
    assert set(manifest.shots) == {1024, 4096, 16384}
    assert set(manifest.method) == {"raw", "readout", "zne"}
    assert set(manifest.record_mode) == {"independent_records", "shared_records"}
    assert set(manifest.noise) == {"zero", "low", "medium", "high"}
