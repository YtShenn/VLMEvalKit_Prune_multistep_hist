"""Independent public wrapper for SparseVLM; no existing backend is reconfigured."""
from __future__ import annotations

from ..qwen3_vl.model import Qwen3VLChat
from .config import SparseVLMConfig


def _history_and_current(message: list[dict]) -> tuple[list[int], int | None]:
    images = [i for i, x in enumerate(message) if isinstance(x, dict) and x.get("type") == "image"]
    if not images:
        return [], None
    current = next((i for i in reversed(images) if message[i].get("is_current") or str(message[i].get("frame_role", "")).lower() == "current"), images[-1])
    return [i for i in images if i != current], current


class Qwen3VLSparseVLMChat(Qwen3VLChat):
    """SparseVLM baseline using the isolated custom Qwen3 decoder path."""
    def __init__(self, *args, **kwargs) -> None:
        self.sparsevlm_config = SparseVLMConfig.from_env()
        kwargs["use_vllm"] = False
        # This merely selects the custom decoder. SparseVLM itself is gated by
        # config attributes, not QWEN3VL_ENABLE_ATTN_PRUNE.
        kwargs["use_attn_prune"] = True
        super().__init__(*args, **kwargs)
        # SparseVLM trims prefill KV states itself.  Do not inherit the legacy
        # cache-off default of the unrelated AttnPrune backend.
        self.model.config.use_cache = True
        self.model.generation_config.use_cache = True
        self._publish()

    def _publish(self) -> None:
        # HF creates separate config copies for the outer generation model,
        # Qwen3VLModel and Qwen3VLTextModel.  The token metadata builder reads
        # the middle copy, while the decoder reads the innermost copy.
        core = getattr(self.model, "model", None)
        configs = [
            getattr(self.model, "config", None),
            getattr(self.model.config, "text_config", None),
            getattr(core, "config", None),
            getattr(getattr(core, "config", None), "text_config", None),
        ]
        for cfg in configs:
            if cfg is not None:
                cfg._sparsevlm_enabled = bool(self.sparsevlm_config.enabled)
                cfg._sparsevlm_config = self.sparsevlm_config
        inner = getattr(core, "language_model", None)
        if inner is not None:
            inner.config._sparsevlm_enabled = bool(self.sparsevlm_config.enabled)
            inner.config._sparsevlm_config = self.sparsevlm_config

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        history, _ = _history_and_current(message)
        if len(history) > self.sparsevlm_config.max_history_frames:
            raise ValueError("SparseVLM supports at most four history screenshots")
        self._publish()
        self._sparsevlm_last_sample_stats = None
        try:
            return super().generate_inner_transformers(message, dataset=dataset, **kwargs)
        finally:
            stats = getattr(self.model.config.text_config, "_sparsevlm_last_stats", None)
            if isinstance(stats, dict):
                self._sparsevlm_last_sample_stats = dict(stats)
