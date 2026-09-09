"""Standalone Qwen3-VL wrapper for GUIPruner-reproduction."""

from __future__ import annotations

import os
import tempfile
from copy import copy

from PIL import Image

from ..qwen3_vl.model import Qwen3VLChat
from .attention_patch import init_guipruner
from .config import GUIPrunerConfig
from .history import configure_guipruner_history_defaults
from .tar import apply_tar


def _image_path(item: dict) -> str | None:
    value = str(item.get("value", ""))
    if value.startswith("file://"):
        value = value[7:]
    return value if value and os.path.isfile(value) else None


def _history_and_current_indices(message: list[dict]) -> tuple[list[int], int | None, str]:
    """Prefer explicit current/history metadata and labels; document order fallback."""
    image_indices: list[int] = []
    current_candidates: list[int] = []
    saw_current_label = False
    for i, item in enumerate(message):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = str(item.get("value", "")).lower()
            saw_current_label = saw_current_label or "current screenshot" in text or "current image" in text or "[current_image]" in text
            continue
        if item.get("type") != "image":
            continue
        image_indices.append(i)
        if bool(item.get("is_current", False)) or str(item.get("frame_role", "")).lower() == "current" or saw_current_label:
            current_candidates.append(i)
        # A prompt label describes the immediately following screenshot, not
        # every image that happens to occur later in the message.
        saw_current_label = False
    if not image_indices:
        return [], None, "no_image"
    if current_candidates:
        current = current_candidates[-1]
        return [i for i in image_indices if i != current], current, "metadata_or_prompt_label"
    # Existing builders conventionally append the current screenshot.  This is
    # an explicit, tested fallback rather than an unrecorded assumption.
    return image_indices[:-1], image_indices[-1], "last_image_fallback"


class Qwen3VLGUIPrunerChat(Qwen3VLChat):
    """GUIPruner-reproduction (unofficial reproduction), Qwen3-VL only."""

    def __init__(
        self,
        *args,
        history_steps: int | None = None,
        history_keep_ratio: float | None = None,
        temporal_decay: float | None = None,
        current_keep_ratio: float | None = None,
        background_saliency: float | None = None,
        overall_keep_ratio: float | None = None,
        prune_layer: int | None = None,
        **kwargs,
    ) -> None:
        def env(name: str, value, cast):
            return cast(value) if value is not None else cast(os.getenv(name, default_env[name]))

        default_env = {
            "GUI_PRUNER_HISTORY_STEPS": "4", "GUI_PRUNER_HISTORY_KEEP_RATIO": "0.40",
            "GUI_PRUNER_TEMPORAL_DECAY": "0.20", "GUI_PRUNER_CURRENT_KEEP_RATIO": "0.75",
            "GUI_PRUNER_BACKGROUND_SALIENCY": "0.30", "GUI_PRUNER_PRUNE_LAYER": "2",
        }
        self.guipruner_config = GUIPrunerConfig(
            history_steps=env("GUI_PRUNER_HISTORY_STEPS", history_steps, int),
            history_keep_ratio=env("GUI_PRUNER_HISTORY_KEEP_RATIO", history_keep_ratio, float),
            temporal_decay=env("GUI_PRUNER_TEMPORAL_DECAY", temporal_decay, float),
            current_keep_ratio=env("GUI_PRUNER_CURRENT_KEEP_RATIO", current_keep_ratio, float),
            background_saliency=env("GUI_PRUNER_BACKGROUND_SALIENCY", background_saliency, float),
            overall_keep_ratio=(
                float(overall_keep_ratio) if overall_keep_ratio is not None
                else (float(os.environ["GUI_PRUNER_OVERALL_KEEP_RATIO"])
                      if os.getenv("GUI_PRUNER_OVERALL_KEEP_RATIO", "").strip() else None)
            ),
            prune_layer=env("GUI_PRUNER_PRUNE_LAYER", prune_layer, int),
        )
        configure_guipruner_history_defaults(self.guipruner_config.history_steps)
        kwargs["use_vllm"] = False
        # SSP needs exact shallow attention; init_guipruner verifies eager support.
        kwargs["attn_implementation"] = "eager"
        super().__init__(*args, **kwargs)
        if not hasattr(self, "model"):
            raise RuntimeError("GUIPruner-reproduction supports the transformers Qwen3-VL backend only.")
        init_guipruner(self.model, self.guipruner_config)

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        history_idx, current_idx, identification = _history_and_current_indices(message)
        if len(history_idx) > self.guipruner_config.history_steps:
            raise ValueError("Input contains more history screenshots than GUIPruner history_steps.")
        if not history_idx:
            self.model.config.text_config._guipruner_history_original_visual_tokens = 0
            return super().generate_inner_transformers(message, dataset=dataset, **kwargs)
        loaded: list[Image.Image] = []
        for index in history_idx:
            path = _image_path(message[index])
            if path is None:
                raise RuntimeError("GUIPruner TAR requires local, readable history image paths.")
            with Image.open(path) as image:
                loaded.append(image.convert("RGB"))
        vision = self.model.config.vision_config
        resized, audit = apply_tar(
            loaded, keep_ratio=self.guipruner_config.history_keep_ratio,
            decay=self.guipruner_config.temporal_decay, patch_size=int(vision.patch_size),
            spatial_merge_size=int(vision.spatial_merge_size),
        )
        patched = [copy(item) if isinstance(item, dict) else item for item in message]
        history_original_tokens = sum(
            (image.width // (int(vision.patch_size) * int(vision.spatial_merge_size)))
            * (image.height // (int(vision.patch_size) * int(vision.spatial_merge_size)))
            for image in loaded
        )
        # SSP runs inside ``super``. Publish this pre-TAR token count before
        # entering it so a requested global budget can be resolved there.
        self.model.config.text_config._guipruner_history_original_visual_tokens = history_original_tokens
        paths: list[str] = []
        try:
            for index, image in zip(history_idx, resized):
                handle = tempfile.NamedTemporaryFile(prefix="guipruner_tar_", suffix=".png", delete=False)
                handle.close()
                image.save(handle.name)
                paths.append(handle.name)
                patched[index]["value"] = handle.name
            result = super().generate_inner_transformers(patched, dataset=dataset, **kwargs)
            cfg = self.model.config.text_config
            cfg._guipruner_tar_audit = {
                "implementation": "GUIPruner-reproduction (unofficial reproduction)",
                "history_identification": identification,
                "current_message_index": current_idx,
                "history_original_visual_tokens": history_original_tokens,
                "overall_keep_ratio": self.guipruner_config.overall_keep_ratio,
                "records": [record.__dict__ for record in audit],
            }
            return result
        finally:
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
