import numpy as np
import pytest

from src.circuits import ghz_circuit, ghz_observables
from src.config import WIDTHS
from src.observables import exact_expectation
from src.references import ghz_reference_table


@pytest.mark.parametrize("width", WIDTHS)
def test_ghz_analytic_stabilizers(width: int) -> None:
    circuit = ghz_circuit(width)
    for observable in ghz_observables(width):
        np.testing.assert_allclose(exact_expectation(circuit, observable), 1.0, atol=1e-12)


def test_ghz_reference_schema_and_metadata() -> None:
    table = ghz_reference_table()
    assert set(table["width"]) == set(WIDTHS)
    assert (table["logical_two_qubit_gate_count"] == table["width"] - 1).all()
    np.testing.assert_allclose(
        table["analytic_reference"], table["noiseless_statevector_reference"], atol=1e-12
    )
