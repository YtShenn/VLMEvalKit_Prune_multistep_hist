from __future__ import annotations

import os
import time

from ..qwen3_vl.model import Qwen3VLChat
from .attention_patch import get_stlite_stats, init_stlite, update_stlite_config
from .config import STLiteConfig
from .history import configure_stlite_history_defaults


class Qwen3VLSTLiteChat(Qwen3VLChat):
    """Qwen3-VL backend with isolated ST-Lite KV cache compression."""

    def __init__(
        self,
        *args,
        st_lite_history_steps: int | None = None,
        st_lite_keep_ratio: float | None = None,
        st_lite_window_size: int | None = None,
        st_lite_alpha: float | None = None,
        st_lite_css_kernel_size: int | None = None,
        st_lite_use_tsg: bool | None = None,
        st_lite_use_css: bool | None = None,
        st_lite_min_tokens: int | None = None,
        st_lite_max_capacity: int | None = None,
        **kwargs,
    ) -> None:
        cfg = STLiteConfig.from_env(
            history_steps=st_lite_history_steps,
            keep_ratio=st_lite_keep_ratio,
            window_size=st_lite_window_size,
            alpha=st_lite_alpha,
            css_kernel_size=st_lite_css_kernel_size,
            use_tsg=st_lite_use_tsg,
            use_css=st_lite_use_css,
            min_tokens=st_lite_min_tokens,
            max_capacity_prompt=st_lite_max_capacity,
        )
        configure_stlite_history_defaults(history_steps=cfg.history_steps)
        kwargs["use_vllm"] = False
        kwargs.setdefault("attn_implementation", os.getenv("ST_LITE_ATTENTION_IMPLEMENTATION", "sdpa"))
        super().__init__(*args, **kwargs)
        self.stlite_config = cfg
        self.stlite_history_steps = int(cfg.history_steps)
        if not hasattr(self, "model"):
            raise RuntimeError("ST-Lite is implemented for the transformers Qwen3-VL backend, not vLLM.")
        init_stlite(self.model, cfg)

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        self._stlite_last_sample_stats = None
        update_stlite_config(
            self.model,
            keep_ratio=kwargs.get("st_lite_keep_ratio", kwargs.get("keep_ratio", None)),
            window_size=kwargs.get("st_lite_window_size", kwargs.get("window_size", None)),
            alpha=kwargs.get("st_lite_alpha", None),
            css_kernel_size=kwargs.get("st_lite_css_kernel_size", None),
            min_tokens=kwargs.get("st_lite_min_tokens", None),
            max_capacity_prompt=kwargs.get("st_lite_max_capacity", kwargs.get("max_capacity_prompt", None)),
        )
        total_start = time.perf_counter()
        response = super().generate_inner_transformers(message, dataset=dataset, **kwargs)
        stats = get_stlite_stats(self.model)
        if stats:
            # Keep FLOPs visible under ST-Lite-specific keys while reusing the
            # existing Qwen3-VL runtime estimator.
            flops_stats = dict(getattr(self.model, "_vlmeval_last_sample_flops", {}) or {})
            if flops_stats:
                stats.update(
                    {
                        "ST_LITE_vision_flops": float(flops_stats.get("vision_flops", 0.0) or 0.0),
                        "ST_LITE_llm_flops": float(flops_stats.get("llm_flops", 0.0) or 0.0),
                        "ST_LITE_lm_head_flops": float(flops_stats.get("lm_head_flops", 0.0) or 0.0),
                        "ST_LITE_e2e_flops": float(flops_stats.get("e2e_flops", 0.0) or 0.0),
                        "ST_LITE_flops_forward_steps": int(flops_stats.get("forward_steps", 0) or 0),
                    }
                )
            per_layer = stats.get("ST_LITE_per_layer", {}) or {}
            layer_records = [value for value in per_layer.values() if isinstance(value, dict)]
            if layer_records:
                before_values = [float(v.get("history_visual_tokens", 0) or 0) for v in layer_records]
                after_values = [float(v.get("history_visual_tokens_after", 0) or 0) for v in layer_records]
                rate_values = [
                    after / before
                    for before, after in zip(before_values, after_values)
                    if before > 0
                ]
                if rate_values:
                    stats["ST_LITE_history_visual_tokens"] = float(sum(before_values) / len(before_values))
                    stats["ST_LITE_history_visual_tokens_after"] = float(sum(after_values) / len(after_values))
                    stats["ST_LITE_history_visual_retention_rate"] = float(sum(rate_values) / len(rate_values))
            stats["ST_LITE_end_to_end_latency_s"] = float(time.perf_counter() - total_start)
            decode_s = stats["ST_LITE_end_to_end_latency_s"] - float(stats.get("ST_LITE_prefill_latency_s", 0.0))
            stats["ST_LITE_decode_latency_s"] = max(0.0, float(decode_s))
            try:
                self.model.config.text_config._stlite_last_stats = stats
            except Exception:
                pass
            # The inference loop consumes this per-sample snapshot when it
            # builds summary.json. Keep it separate from existing timing data.
            self._stlite_last_sample_stats = dict(stats)
            if os.getenv("ST_LITE_DEBUG_STATS", "0") == "1":
                print(f"[ST_LITE] {stats}", flush=True)
        return response
