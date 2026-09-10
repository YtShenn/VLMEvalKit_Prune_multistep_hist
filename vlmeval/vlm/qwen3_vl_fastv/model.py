"""Independent Qwen3-VL wrapper enabling FastV only."""
from __future__ import annotations

from ..qwen3_vl.model import Qwen3VLChat
from .config import FastVConfig


class Qwen3VLFastVChat(Qwen3VLChat):
    """Faithful FastV scoring with Qwen3-VL's dynamic multi-image token map."""

    def __init__(self, *args, **kwargs) -> None:
        self.fastv_config = FastVConfig.from_env()
        kwargs["use_vllm"] = False
        kwargs["fastv"] = True
        super().__init__(*args, **kwargs)
        if not hasattr(self, "model"):
            raise RuntimeError("FastV requires the transformers Qwen3-VL backend.")
        # Qwen3VLTextModel is constructed with `_from_config`, so its config
        # is a distinct object from `model.config.text_config`. Publish to the
        # outer config (for visual metadata construction) and inner decoder
        # config (for the layer-K FastV operation).
        core = getattr(self.model, "model", None)
        inner = getattr(core, "language_model", None)
        configs = [
            getattr(self.model.config, "text_config", None),
            getattr(core, "config", None),
            getattr(getattr(core, "config", None), "text_config", None),
            getattr(inner, "config", None),
        ]
        for cfg in configs:
            if cfg is not None:
                cfg._fastv_enabled = self.fastv_config.enabled
                cfg._fastv_config = self.fastv_config
