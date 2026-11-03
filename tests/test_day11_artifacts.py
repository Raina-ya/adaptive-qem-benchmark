import pandas as pd

from src.generate_days08_12 import _write_canonical_csv


def test_canonical_csv_size_is_platform_independent(tmp_path) -> None:
    path = tmp_path / "pilot.csv"
    frame = pd.DataFrame({"value": [1, 2]})

    size = _write_canonical_csv(frame, path)
    payload = path.read_bytes()

    assert b"\r\n" not in payload
    assert payload.count(b"\n") == 3
    assert size == len(payload)
