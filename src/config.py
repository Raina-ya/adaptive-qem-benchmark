"""Shared deterministic configuration."""

from __future__ import annotations

import hashlib

ROOT_SEED = 20261102
DEFAULT_SHOTS = 1024
WIDTHS = (2, 4, 6)
DEPTHS = (1, 3, 6)
NOISE_LEVELS = ("zero", "low", "medium", "high")
SHOT_BUDGETS = (1024, 4096, 16384)
READOUT_CALIBRATION_FRACTION = 0.20


def derive_seed(namespace: str, *indices: int) -> int:
    """Derive a stable 32-bit child seed without Python's randomized hash()."""

    payload = ":".join([str(ROOT_SEED), namespace, *(str(index) for index in indices)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)
