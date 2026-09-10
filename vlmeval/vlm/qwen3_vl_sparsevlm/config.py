"""Configuration for the standalone inference-only SparseVLM baseline."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SparseVLMConfig:
    enabled: bool = True
    layers: tuple[int, ...] = (3, 6, 15)
    retain_ratio: float = 0.5
    version: str = "1_0"
    scope: str = "all"
    max_history_frames: int = 4
    debug: bool = False

    @classmethod
    def from_env(cls) -> "SparseVLMConfig":
        raw_layers = os.getenv("SPARSEVLM_LAYERS", "3,6,15")
        try:
            layers = tuple(sorted({int(x.strip()) for x in raw_layers.split(",") if x.strip()}))
            ratio = float(os.getenv("SPARSEVLM_RETAIN_RATIO", os.getenv("RETAIN_RATIO", "0.5")))
            max_history = int(os.getenv("SPARSEVLM_MAX_HISTORY_FRAMES", "4"))
        except ValueError as exc:
            raise ValueError("SPARSEVLM_LAYERS, SPARSEVLM_RETAIN_RATIO and SPARSEVLM_MAX_HISTORY_FRAMES are invalid") from exc
        scope = os.getenv("SPARSEVLM_SCOPE", "all").strip().lower()
        if not layers or min(layers) < 0:
            raise ValueError("SPARSEVLM_LAYERS must contain non-negative decoder layer indices")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("SPARSEVLM_RETAIN_RATIO must be in (0, 1]")
        if scope not in {"all", "history", "current"}:
            raise ValueError("SPARSEVLM_SCOPE must be all, history, or current")
        if not 0 <= max_history <= 4:
            raise ValueError("SPARSEVLM_MAX_HISTORY_FRAMES must be in [0, 4]")
        return cls(_flag("SPARSEVLM_ENABLED", "1"), layers, ratio,
                   os.getenv("SPARSEVLM_VERSION", os.getenv("USE_VERSION", "1_0")),
                   scope, max_history, _flag("SPARSEVLM_DEBUG", "0"))
