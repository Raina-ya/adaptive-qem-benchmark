import math

import numpy as np
import pytest

from src.density_sim import (
    Gate,
    Noise,
    assignment_matrix,
    fold_global,
    fold_local,
    garbage_fold,
    hardware_efficient_gates,
    ideal_expectation,
    outcome_probabilities,
    parity_signs,
)
from src.independent_reference import hardware_efficient_state, pauli_expectation

CLEAN = Noise("clean")
PLUS = [Gate("h", (0,))]


def expectation(gates, pauli, noise=CLEAN):
    return float(outcome_probabilities(gates, pauli, noise) @ parity_signs(pauli))


def test_density_path_matches_independent_statevector() -> None:
    from src.independent_reference import derive_seed

    gates = hardware_efficient_gates(4, 3, derive_seed("hardware-efficient", 4, 3))
    state = hardware_efficient_state(4, 3)
    for pauli in ("XZXZ", "ZZII", "ZIII"):
        assert expectation(gates, pauli) == pytest.approx(
            pauli_expectation(state, pauli), abs=1e-12
        )
        assert ideal_expectation(gates, pauli) == pytest.approx(
            pauli_expectation(state, pauli), abs=1e-12
        )


def test_depolarizing_scales_a_one_qubit_signal_by_one_minus_p() -> None:
    assert expectation(PLUS, "X", Noise("d", depolarizing_1q=0.1)) == pytest.approx(0.9**2)


def test_two_qubit_depolarizing_scales_a_bell_correlation() -> None:
    bell = [Gate("h", (0,)), Gate("cx", (0, 1))]
    # Measuring XX adds two Hadamards, which carry no noise here.
    assert expectation(bell, "XX", Noise("d", depolarizing_2q=0.2)) == pytest.approx(0.8)


def test_amplitude_damping_matches_the_analytic_coherence_decay() -> None:
    gamma = 0.19
    # The Hadamard pair is noisy: coherence shrinks by sqrt(1 - gamma), twice with measurement.
    value = expectation(PLUS, "X", Noise("a", damping_1q=gamma))
    assert 0.0 < value < 1.0
    one = Gate("ry", (0,), math.pi)
    assert expectation([one], "Z", Noise("a", damping_1q=gamma)) == pytest.approx(
        -(1.0 - 2.0 * gamma)
    )


def test_dephasing_leaves_populations_and_shrinks_coherence() -> None:
    p = 0.2
    assert expectation(
        [Gate("ry", (0,), math.pi)], "Z", Noise("p", dephasing_1q=p)
    ) == pytest.approx(-1.0)
    assert expectation(
        [Gate("ry", (0,), math.pi / 2.0)], "X", Noise("p", dephasing_1q=p)
    ) == pytest.approx(1.0 - p)


def test_readout_matrix_applies_each_direction_with_qubit_zero_low() -> None:
    probabilities = outcome_probabilities([], "ZI", Noise("r", readout_p01=0.1, readout_p10=0.2))
    assert probabilities == pytest.approx([0.81, 0.09, 0.09, 0.01])
    matrix = assignment_matrix(1, 0.1, 0.2)
    assert matrix.sum(axis=0) == pytest.approx([1.0, 1.0])


@pytest.mark.parametrize("fold", [fold_global, fold_local])
@pytest.mark.parametrize("factor", [3, 5, 7])
def test_folding_preserves_the_noiseless_value_and_scales_gate_count(fold, factor) -> None:
    gates = hardware_efficient_gates(3, 2, 7)
    assert len(fold(gates, factor)) == factor * len(gates)
    assert expectation(fold(gates, factor), "XZX") == pytest.approx(
        expectation(gates, "XZX"), abs=1e-12
    )


def test_even_factors_are_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        fold_global([], 2)


def test_folding_cancels_overrotation_but_accumulates_detuning() -> None:
    gate = [Gate("ry", (0,), 0.7)]
    over = Noise("o", overrotation=0.1)
    assert expectation(fold_global(gate, 3), "Z", over) == pytest.approx(
        expectation(gate, "Z", over)
    )
    detuned = Noise("d", detuning=0.1)
    assert expectation(fold_global(gate, 3), "Z", detuned) == pytest.approx(math.cos(0.7 + 0.3))


def test_garbage_fold_changes_the_noiseless_value() -> None:
    gates = hardware_efficient_gates(2, 2, 3)
    assert expectation(garbage_fold(gates, 3, 11), "ZZ") != pytest.approx(
        expectation(gates, "ZZ"), abs=1e-3
    )


def test_density_matrix_matches_aer_for_every_channel() -> None:
    aer = pytest.importorskip("qiskit_aer")
    if not hasattr(pytest.importorskip("qiskit_aer.noise"), "amplitude_damping_error"):
        pytest.skip("The real qiskit_aer package is not installed")
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import (
        NoiseModel,
        amplitude_damping_error,
        depolarizing_error,
        pauli_error,
    )

    p1, p2, g1, g2, d1, d2, p01, p10 = 0.01, 0.04, 0.02, 0.03, 0.015, 0.025, 0.02, 0.016
    noise = Noise(
        "all",
        depolarizing_1q=p1,
        depolarizing_2q=p2,
        damping_1q=g1,
        damping_2q=g2,
        dephasing_1q=d1,
        dephasing_2q=d2,
        readout_p01=p01,
        readout_p10=p10,
    )
    one = (
        depolarizing_error(p1, 1)
        .compose(amplitude_damping_error(g1))
        .compose(pauli_error([("Z", d1 / 2.0), ("I", 1.0 - d1 / 2.0)]))
    )
    single = amplitude_damping_error(g2).compose(
        pauli_error([("Z", d2 / 2.0), ("I", 1.0 - d2 / 2.0)])
    )
    two = depolarizing_error(p2, 2).compose(single.tensor(single))
    model = NoiseModel()
    model.add_all_qubit_quantum_error(one, ["ry", "rz", "h"])
    model.add_all_qubit_quantum_error(two, ["cx"])
    gates = hardware_efficient_gates(3, 2, 99)
    circuit = QuantumCircuit(3)
    for gate in gates:
        if gate.name == "cx":
            circuit.cx(*gate.qubits)
        else:
            getattr(circuit, gate.name)(gate.angle, gate.qubits[0])
    circuit.save_density_matrix(label="rho")
    backend = aer.AerSimulator(method="density_matrix", noise_model=model)
    rho = np.asarray(backend.run(circuit).result().data()["rho"])
    from src.density_sim import diagonal, run

    ours = run(
        gates,
        3,
        Noise(
            "no-readout",
            **{
                k: getattr(noise, k)
                for k in (
                    "depolarizing_1q",
                    "depolarizing_2q",
                    "damping_1q",
                    "damping_2q",
                    "dephasing_1q",
                    "dephasing_2q",
                )
            },
        ),
    )
    assert ours.reshape(8, 8) == pytest.approx(rho, abs=1e-10)
    assert diagonal(ours, 3).sum() == pytest.approx(1.0)
    assert assignment_matrix(3, p01, p10) @ diagonal(ours, 3) == pytest.approx(
        outcome_probabilities(gates, "III", noise), abs=1e-12
    )
