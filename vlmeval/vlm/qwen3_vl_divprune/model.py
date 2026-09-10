"""Independent Qwen3-VL wrapper enabling only Layer-0 DivPrune."""
from __future__ import annotations

from ..qwen3_vl.model import Qwen3VLChat
from .config import DivPruneConfig


class Qwen3VLDivPruneChat(Qwen3VLChat):
    def __init__(self, *args, **kwargs) -> None:
        self.divprune_config = DivPruneConfig.from_env()
        kwargs["use_vllm"] = False
        kwargs["divprune"] = True
        super().__init__(*args, **kwargs)
        if not hasattr(self, "model"):
            raise RuntimeError("DivPrune requires the transformers Qwen3-VL backend.")
        core = getattr(self.model, "model", None)
        inner = getattr(core, "language_model", None)
        for cfg in (getattr(self.model.config, "text_config", None), getattr(core, "config", None),
                    getattr(getattr(core, "config", None), "text_config", None), getattr(inner, "config", None)):
            if cfg is not None:
                cfg._divprune_enabled = self.divprune_config.enabled
                cfg._divprune_config = self.divprune_config
