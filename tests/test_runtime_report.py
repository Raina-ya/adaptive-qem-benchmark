from src.generate_runtime_report import load_runtime


def test_runtime_rows_classify_each_duration() -> None:
    frame = load_runtime()

    assert set(frame.timing_kind) == {"measured", "estimated"}
    assert frame.timing_kind.notna().all()
    assert (frame.loc[frame.timing_kind == "estimated", "notes"].str.contains("estimated")).all()
