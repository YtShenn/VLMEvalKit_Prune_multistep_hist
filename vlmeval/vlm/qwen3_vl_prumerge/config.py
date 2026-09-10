"""Runtime configuration for the isolated Qwen3-VL PruMerge baseline."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PruMergeConfig:
    # Disabled by default so selecting the wrapper alone cannot silently alter
    # a comparison run. The provided runner explicitly enables it.
    enabled: bool = False
    variant: str = "prumerge"
    min_keep_tokens: int = 4
    max_keep_tokens: int = 128
    iqr_multiplier: float = 1.5
    similarity_temperature: float = 1.0
    max_history_steps: int = 4
    scope: str = "all"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "PruMergeConfig":
        try:
            min_keep = int(os.getenv("QWEN3VL_PRUMERGE_MIN_KEEP_TOKENS", "4"))
            max_keep = int(os.getenv("QWEN3VL_PRUMERGE_MAX_KEEP_TOKENS", "128"))
            iqr = float(os.getenv("QWEN3VL_PRUMERGE_IQR_MULTIPLIER", "1.5"))
            temperature = float(os.getenv("QWEN3VL_PRUMERGE_SIMILARITY_TEMPERATURE", "1.0"))
            history = int(os.getenv("QWEN3VL_PRUMERGE_MAX_HISTORY_STEPS", "4"))
        except ValueError as exc:
            raise ValueError("QWEN3VL_PRUMERGE numeric options are invalid") from exc
        variant = os.getenv("QWEN3VL_PRUMERGE_VARIANT", "prumerge").strip().lower()
        scope = os.getenv("QWEN3VL_PRUMERGE_SCOPE", "all").strip().lower()
        if variant not in {"prumerge", "prumerge_plus"}:
            raise ValueError("QWEN3VL_PRUMERGE_VARIANT must be prumerge or prumerge_plus")
        if scope not in {"all", "history", "current"}:
            raise ValueError("QWEN3VL_PRUMERGE_SCOPE must be all, history, or current")
        if min_keep < 1 or max_keep < min_keep:
            raise ValueError("QWEN3VL_PRUMERGE keep bounds must satisfy 1 <= min <= max")
        if iqr < 0.0 or temperature <= 0.0:
            raise ValueError("QWEN3VL_PRUMERGE_IQR_MULTIPLIER must be non-negative and temperature positive")
        if not 0 <= history <= 4:
            raise ValueError("QWEN3VL_PRUMERGE_MAX_HISTORY_STEPS must be in [0, 4]")
        return cls(
            enabled=_flag("QWEN3VL_PRUMERGE_ENABLED", "0"), variant=variant,
            min_keep_tokens=min_keep, max_keep_tokens=max_keep,
            iqr_multiplier=iqr, similarity_temperature=temperature,
            max_history_steps=history, scope=scope,
            debug=_flag("QWEN3VL_PRUMERGE_DEBUG", "0"),
        )
