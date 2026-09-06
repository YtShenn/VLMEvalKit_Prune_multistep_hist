from __future__ import annotations

import os
from dataclasses import dataclass


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class STLiteConfig:
    history_steps: int = 4
    keep_ratio: float = 0.20
    window_size: int = 8
    alpha: float = 0.1
    css_kernel_size: int = 3
    use_tsg: bool = True
    use_css: bool = True
    min_tokens: int = 64
    max_capacity_prompt: int | None = None
    pooling: str = "avgpool"
    tsg_redundancy_threshold: float = 0.95

    @classmethod
    def from_env(cls, **overrides) -> "STLiteConfig":
        cfg = cls(
            history_steps=env_int("ST_LITE_HISTORY_STEPS", 4),
            keep_ratio=env_float("ST_LITE_KEEP_RATIO", 0.20),
            window_size=env_int("ST_LITE_WINDOW_SIZE", 8),
            alpha=env_float("ST_LITE_ALPHA", 0.1),
            css_kernel_size=env_int("ST_LITE_CSS_KERNEL_SIZE", 3),
            use_tsg=env_flag("ST_LITE_USE_TSG", "1"),
            use_css=env_flag("ST_LITE_USE_CSS", "1"),
            min_tokens=env_int("ST_LITE_MIN_TOKENS", 64),
            pooling=os.getenv("ST_LITE_POOLING", "avgpool"),
            tsg_redundancy_threshold=env_float("ST_LITE_TSG_THRESHOLD", 0.95),
        )
        max_capacity = os.getenv("ST_LITE_MAX_CAPACITY", "").strip()
        if max_capacity:
            cfg.max_capacity_prompt = max(1, int(max_capacity))
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.history_steps = max(0, int(cfg.history_steps))
        cfg.keep_ratio = max(0.0, min(1.0, float(cfg.keep_ratio)))
        cfg.window_size = max(1, int(cfg.window_size))
        cfg.css_kernel_size = max(1, int(cfg.css_kernel_size))
        if cfg.css_kernel_size % 2 == 0:
            cfg.css_kernel_size += 1
        cfg.min_tokens = max(1, int(cfg.min_tokens))
        if cfg.max_capacity_prompt is not None:
            cfg.max_capacity_prompt = max(1, int(cfg.max_capacity_prompt))
        return cfg
