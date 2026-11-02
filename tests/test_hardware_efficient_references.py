from src.config import DEPTHS, WIDTHS
from src.hardware_efficient_circuits import hardware_efficient_circuit, parameter_signature
from src.references import hardware_efficient_reference_table


def test_hardware_efficient_parameters_regenerate_identically() -> None:
    first = hardware_efficient_circuit(4, 3)
    second = hardware_efficient_circuit(4, 3)
    assert parameter_signature(first) == parameter_signature(second)


def test_hardware_efficient_grid_and_nontrivial_references() -> None:
    table = hardware_efficient_reference_table()
    expected = {(width, depth) for width in WIDTHS for depth in DEPTHS}
    observed = set(zip(table["width"], table["depth_parameter"], strict=True))
    assert observed == expected
    assert (table["noiseless_statevector_reference"].abs() < 0.95).any()
