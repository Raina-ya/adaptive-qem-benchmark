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


def test_canonical_csv_normalizes_float_noise_in_json(tmp_path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    common = {"estimate": [0.12345678901234], "count": [1024]}
    frame_left = pd.DataFrame(
        {**common, "raw_payload_json": ['{"value":0.12345678901234,"count":1024}']}
    )
    frame_right = pd.DataFrame(
        {**common, "raw_payload_json": ['{"value":0.12345678901235,"count":1024}']}
    )

    _write_canonical_csv(frame_left, left)
    _write_canonical_csv(frame_right, right)

    assert left.read_bytes() == right.read_bytes()
