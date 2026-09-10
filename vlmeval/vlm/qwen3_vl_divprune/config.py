"""Configuration for the isolated, training-free DivPrune backend."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DivPruneConfig:
    """``keep_ratio`` is a retention ratio; pruning is ``1 - keep_ratio``."""

    enabled: bool = True
    keep_ratio: float = 0.098
    scope: str = "global_all_visual"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "DivPruneConfig":
        try:
            keep_ratio = float(os.getenv("DIVPRUNE_KEEP_RATIO", "0.098"))
        except ValueError as exc:
            raise ValueError("DIVPRUNE_KEEP_RATIO must be a float") from exc
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("DIVPRUNE_KEEP_RATIO must be in (0, 1]")
        scope = os.getenv("DIVPRUNE_SCOPE", "global_all_visual").strip().lower()
        if scope not in {"global_all_visual", "history_only"}:
            raise ValueError("DIVPRUNE_SCOPE must be global_all_visual or history_only")
        return cls(_flag("DIVPRUNE_ENABLED", "1"), keep_ratio, scope, _flag("DIVPRUNE_DEBUG", "0"))
