import json

import numpy as np
import pytest

from src.circuits import bell_circuit
from src.mitigation.readout_matrix_free import (
    TensorSubspaceOperator,
    run_matrix_free_readout_mitigation,
    solve_matrix_free_nnls,
)
from src.noise_models import NoiseSpec


def _materialize_for_test(operator: TensorSubspaceOperator) -> np.ndarray:
    return np.column_stack(
        [operator.matvec(np.eye(operator.shape[1])[:, index]) for index in range(operator.shape[1])]
    )


def test_operator_matches_explicit_tensor_subspace_matrix() -> None:
    q0 = np.array([[0.93, 0.04], [0.07, 0.96]])
    q1 = np.array([[0.89, 0.02], [0.11, 0.98]])
    support = (0, 1, 3)
    operator = TensorSubspaceOperator((q0, q1), support)
    expected = np.kron(q1, q0)[np.ix_(support, support)]

    np.testing.assert_allclose(_materialize_for_test(operator), expected, atol=1e-15)
    vector = np.array([0.2, 0.3, 0.5])
    np.testing.assert_allclose(operator.rmatvec(vector), expected.T @ vector, atol=1e-15)


def test_matrix_free_solver_matches_explicit_nonnegative_solution() -> None:
    local = np.array([[0.96, 0.05], [0.04, 0.95]])
    operator = TensorSubspaceOperator((local, local), (0, 1, 2, 3))
    exact = np.array([0.45, 0.05, 0.15, 0.35])
    observed = operator.matvec(exact)

    result = solve_matrix_free_nnls(operator, observed, tolerance=1e-12)

    assert result.status == "ok"
    assert result.converged
    assert result.solution is not None
    np.testing.assert_allclose(result.solution, exact, rtol=0.0, atol=2e-10)


def test_solver_does_not_allocate_support_square_matrix(monkeypatch) -> None:
    local = np.array([[0.97, 0.04], [0.03, 0.96]])
    operator = TensorSubspaceOperator((local,) * 3, tuple(range(8)))
    observed = operator.matvec(np.full(8, 1 / 8))
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if shape == operator.shape:
            raise AssertionError("dense support matrix was allocated")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", guarded_zeros)
    result = solve_matrix_free_nnls(operator, observed)

    assert result.status == "ok"


def test_nonconvergence_is_explicit() -> None:
    local = np.array([[0.8, 0.2], [0.2, 0.8]])
    operator = TensorSubspaceOperator((local, local), (0, 1, 2, 3))
    result = solve_matrix_free_nnls(
        operator,
        np.array([0.8, 0.1, 0.05, 0.05]),
        max_iterations=1,
        tolerance=1e-16,
    )

    assert result.solution is None
    assert not result.converged
    assert result.status == "not_converged"


def test_matrix_free_integration_preserves_budget_and_diagnostics() -> None:
    result = run_matrix_free_readout_mitigation(
        bell_circuit(),
        "ZZ",
        total_shot_budget=4096,
        noise_spec=NoiseSpec("independent", readout_p01=0.03, readout_p10=0.03),
        seed=20261104,
    )

    assert result.status == "ok"
    assert result.corrected_estimate is not None
    assert result.target_shots + result.calibration_shots == 4096
    assert result.calibration_circuit_executions == 4
    assert result.materialized_matrix_entries == 0
    assert result.converged
    assert result.iterations > 0
    assert json.loads(result.support_states_json)


def test_operator_rejects_invalid_support() -> None:
    with pytest.raises(ValueError, match="unique"):
        TensorSubspaceOperator((np.eye(2),), (0, 0))
