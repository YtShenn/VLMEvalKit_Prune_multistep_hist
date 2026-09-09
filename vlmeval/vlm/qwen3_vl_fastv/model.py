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
        cfg = self.model.config.text_config
        cfg._fastv_enabled = self.fastv_config.enabled
        cfg._fastv_config = self.fastv_config
