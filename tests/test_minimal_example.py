import json

import numpy as np

from src.minimal_example import run_minimal_example
from src.observables import parity_eigenvalue


def test_bell_xx_matches_analytic_reference() -> None:
    row = run_minimal_example(shots=1024, repeat=0)
    np.testing.assert_allclose(row["noiseless_statevector_reference"], 1.0, atol=1e-12)
    assert row["estimate"] == 1.0
    assert sum(json.loads(row["raw_counts_json"]).values()) == 1024


def test_minimal_example_is_seed_reproducible() -> None:
    assert run_minimal_example(repeat=3) == run_minimal_example(repeat=3)


def test_q0_first_bit_ordering() -> None:
    assert parity_eigenvalue("01", "ZI") == -1
    assert parity_eigenvalue("01", "IZ") == 1
