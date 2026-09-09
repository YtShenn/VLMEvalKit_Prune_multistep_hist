"""Independent HistPrune-GUI Qwen3-VL wrapper (history frames only)."""
from __future__ import annotations

import os
from copy import copy

from PIL import Image

from ..qwen3_vl.model import Qwen3VLChat
from .config import HistPruneConfig


def _path(item: dict) -> str | None:
    value = str(item.get("value", ""))
    value = value[7:] if value.startswith("file://") else value
    return value if value and os.path.isfile(value) else None


def history_and_current_indices(message: list[dict]) -> tuple[list[int], int | None]:
    images = [i for i, item in enumerate(message) if isinstance(item, dict) and item.get("type") == "image"]
    if not images:
        return [], None
    explicit = [i for i in images if bool(message[i].get("is_current", False)) or str(message[i].get("frame_role", "")).lower() == "current"]
    current = explicit[-1] if explicit else images[-1]
    return [i for i in images if i != current], current


class Qwen3VLHistPruneChat(Qwen3VLChat):
    """HistPrune-GUI reproduction/adaptation for Qwen3-VL.

    This class only enables the isolated custom Qwen3 model.  Its default
    parent path, GUIKV, STLite, GUIPruner and attention pruning remain intact.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.histprune_config = HistPruneConfig.from_env()
        kwargs["use_vllm"] = False
        # Reuse the repository's custom Qwen3 model only as a Qwen3-compatible
        # decoder implementation; its attention-pruning switch remains off.
        kwargs["histprune"] = True
        super().__init__(*args, **kwargs)
        if not hasattr(self, "model"):
            raise RuntimeError("HistPrune requires the transformers Qwen3-VL backend.")
        self._publish_config([], [])

    def _publish_config(self, images, keys) -> None:
        cfg = self.model.config.text_config
        cfg._histprune_enabled = True
        cfg._histprune_config = self.histprune_config
        cfg._histprune_history_images = images
        # Consumed by the model-side episode cache when enabled.
        self.model.config._histprune_visual_cache_enabled = self.histprune_config.visual_cache
        self.model.config._histprune_visual_cache_keys = keys

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        history_indices, current_index = history_and_current_indices(message)
        if len(history_indices) > self.histprune_config.max_history_frames:
            raise ValueError("HistPrune accepts at most four history screenshots; truncate the trajectory before inference.")
        history_images = []
        history_keys = []
        for index in history_indices:
            path = _path(message[index])
            if path is None and self.histprune_config.mode.startswith("sobel"):
                raise RuntimeError("Sobel HistPrune requires readable local paths for history screenshots.")
            if path is not None:
                with Image.open(path) as image:
                    history_images.append(image.convert("RGB"))
                history_keys.append(os.path.abspath(path))
            else:
                history_images.append(None)
                history_keys.append("")
        # Image order passed to Qwen remains old -> new -> current.  The model
        # turns it into the prescribed recent-first rank map.
        current_key = _path(message[current_index]) if current_index is not None else None
        self._publish_config(history_images, history_keys + ([os.path.abspath(current_key)] if current_key else [""]))
        try:
            return super().generate_inner_transformers(message, dataset=dataset, **kwargs)
        finally:
            # PIL references are per request; cache itself is explicitly episode-managed.
            self.model.config.text_config._histprune_history_images = []

