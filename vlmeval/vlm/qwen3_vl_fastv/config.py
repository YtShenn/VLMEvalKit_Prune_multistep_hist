"""Configuration for the isolated, training-free FastV backend."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FastVConfig:
    enabled: bool = True
    k: int = 2
    r: float = 0.5
    debug: bool = False

    @classmethod
    def from_env(cls) -> "FastVConfig":
        try:
            k = int(os.getenv("FASTV_K", "2"))
            r = float(os.getenv("FASTV_R", "0.5"))
        except ValueError as exc:
            raise ValueError("FASTV_K must be an integer and FASTV_R a float") from exc
        if k < 0:
            raise ValueError("FASTV_K must be non-negative")
        if not 0.0 <= r < 1.0:
            raise ValueError("FASTV_R must be in [0, 1)")
        return cls(_flag("FASTV_ENABLED", "1"), k, r, _flag("FASTV_DEBUG", "0"))
