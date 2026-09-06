from __future__ import annotations

import os

from ..qwen3_vl.model import Qwen3VLChat
from .attention_patch import init_gui_kv, update_gui_kv_config
from .gui_kv_utils import GUIKVConfig
from .history import configure_guikv_history_defaults


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class Qwen3VLGUIKVChat(Qwen3VLChat):
    """Qwen3-VL backend with isolated GUI-KV cache compression."""

    def __init__(
        self,
        *args,
        max_capacity_prompt: int | None = None,
        kv_cache_budget: int | None = None,
        total_keep_ratio: float | None = None,
        window_size: int | None = None,
        alpha: float | None = None,
        temperature: float | None = None,
        pooling: str | None = None,
        kernel_size: int | None = None,
        history_steps: int | None = None,
        **kwargs,
    ) -> None:
        history_steps = _env_int("GUIKV_HISTORY_STEPS", 4) if history_steps is None else int(history_steps)
        configure_guikv_history_defaults(history_steps=history_steps)
        kwargs["use_vllm"] = False
        kwargs.setdefault("attn_implementation", os.getenv("GUIKV_ATTENTION_IMPLEMENTATION", "sdpa"))
        super().__init__(*args, **kwargs)

        if kv_cache_budget is not None and max_capacity_prompt is None:
            max_capacity_prompt = kv_cache_budget
        if max_capacity_prompt is None and os.getenv("GUIKV_MAX_CAPACITY_PROMPT", "").strip():
            max_capacity_prompt = _env_int("GUIKV_MAX_CAPACITY_PROMPT", 0)
        config = GUIKVConfig(
            max_capacity_prompt=int(max_capacity_prompt) if max_capacity_prompt is not None and int(max_capacity_prompt) > 0 else None,
            total_keep_ratio=float(
                total_keep_ratio if total_keep_ratio is not None else _env_float("GUIKV_TOTAL_KEEP_RATIO", 0.40)
            ),
            window_size=int(window_size or _env_int("GUIKV_WINDOW_SIZE", 8)),
            kernel_size=int(kernel_size or _env_int("GUIKV_KERNEL_SIZE", 5)),
            pooling=str(pooling or os.getenv("GUIKV_POOLING", "avgpool")),
            alpha=float(alpha if alpha is not None else _env_float("GUIKV_ALPHA", 2.0)),
            temperature=float(temperature if temperature is not None else _env_float("GUIKV_TEMPERATURE", 3.5)),
        )
        self.guikv_config = config
        self.guikv_history_steps = history_steps
        if not hasattr(self, "model"):
            raise RuntimeError("GUI-KV is implemented for the transformers Qwen3-VL backend, not vLLM.")
        init_gui_kv(self.model, config)

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        update_gui_kv_config(
            self.model,
            max_capacity_prompt=kwargs.get("max_capacity_prompt", kwargs.get("kv_cache_budget", None)),
            total_keep_ratio=kwargs.get("total_keep_ratio", None),
            window_size=kwargs.get("window_size", None),
            alpha=kwargs.get("alpha", None),
            temperature=kwargs.get("temperature", None),
        )
        return super().generate_inner_transformers(message, dataset=dataset, **kwargs)
