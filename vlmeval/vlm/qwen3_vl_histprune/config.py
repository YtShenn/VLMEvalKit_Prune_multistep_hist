"""Configuration for HistPrune-GUI reproduction/adaptation for Qwen3-VL."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class HistPruneConfig:
    """All ratios are history-only; the current screenshot is never pruned."""
    mode: str = "random"
    history_keep_ratio: float = 0.40
    temporal_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    drop_layer: int = 4
    random_seed: int = 0
    sobel_edge_threshold: float = 50.0
    sobel_ratio_threshold: float = 0.01
    visual_cache: bool = False
    max_history_frames: int = 4

    @classmethod
    def from_env(cls) -> "HistPruneConfig":
        mode = os.getenv("HISTPRUNE_MODE", "random").strip().lower()
        if mode not in {"random", "uniform", "sobel_foreground", "sobel_background"}:
            raise ValueError(f"Unsupported HISTPRUNE_MODE={mode!r}")
        raw = os.getenv("HISTPRUNE_TEMPORAL_WEIGHTS", "0.4,0.3,0.2,0.1")
        try:
            weights = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
        except ValueError as exc:
            raise ValueError("HISTPRUNE_TEMPORAL_WEIGHTS must be comma-separated floats") from exc
        if not weights or any(x < 0 for x in weights) or sum(weights) <= 0:
            raise ValueError("HISTPRUNE_TEMPORAL_WEIGHTS must contain non-negative values with positive sum")
        ratio = _float("HISTPRUNE_HISTORY_KEEP_RATIO", .40)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("HISTPRUNE_HISTORY_KEEP_RATIO must be in [0, 1]")
        return cls(mode=mode, history_keep_ratio=ratio, temporal_weights=weights,
                   drop_layer=max(0, _int("HISTPRUNE_DROP_LAYER", 4)),
                   random_seed=_int("HISTPRUNE_RANDOM_SEED", 0),
                   sobel_edge_threshold=_float("HISTPRUNE_SOBEL_EDGE_THRESHOLD", 50),
                   sobel_ratio_threshold=_float("HISTPRUNE_SOBEL_RATIO_THRESHOLD", .01),
                   visual_cache=os.getenv("HISTPRUNE_VISUAL_CACHE", "0").strip() in {"1", "true", "yes"})
