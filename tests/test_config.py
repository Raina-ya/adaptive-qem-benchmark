from src.config import ROOT_SEED, derive_seed


def test_seed_derivation_is_stable_and_namespaced() -> None:
    assert ROOT_SEED == 20261102
    assert derive_seed("example", 1, 2) == derive_seed("example", 1, 2)
    assert derive_seed("example", 1, 2) != derive_seed("other", 1, 2)
