import pytest

from src.circuits import bell_circuit
from src.config import NOISE_LEVELS, derive_seed
from src.estimators.unmitigated import estimate_pauli
from src.noise_models import build_noise_model, noise_spec


@pytest.mark.parametrize("kind", ["readout", "one_qubit", "two_qubit", "mixed"])
@pytest.mark.parametrize("level", NOISE_LEVELS)
def test_noise_model_builds_for_declared_grid(kind: str, level: str) -> None:
    model = build_noise_model(noise_spec(kind, level))
    assert model is not None


def test_zero_noise_returns_noiseless_bell_reference() -> None:
    result = estimate_pauli(
        bell_circuit(),
        "XX",
        shots=1024,
        seed=derive_seed("zero-noise-test"),
        noise_spec=noise_spec("mixed", "zero"),
    )
    assert result.estimate == 1.0


def test_noise_is_attached_to_intended_instruction_classes() -> None:
    model = build_noise_model(noise_spec("mixed", "medium"))
    assert "measure" in model.noise_instructions
    assert "cx" in model.noise_instructions
    assert any(gate in model.noise_instructions for gate in ("h", "sx", "ry"))
