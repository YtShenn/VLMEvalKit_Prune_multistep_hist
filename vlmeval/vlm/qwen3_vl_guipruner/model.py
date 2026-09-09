"""Standalone Qwen3-VL wrapper for GUIPruner-reproduction."""

from __future__ import annotations

import os
import tempfile
from copy import copy

from PIL import Image

from ..qwen3_vl.model import Qwen3VLChat
from .attention_patch import init_guipruner
from .config import GUIPrunerConfig
from .flops import correct_split_prefill_flops
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


def _processor_visual_tokens(processor, images: list[Image.Image], spatial_merge_size: int) -> int:
    """Count Qwen3-VL visual tokens with its own image processor, not geometry.

    This is used only to establish the unpruned logical prompt denominator for
    an end-to-end retention target; no model forward is performed.
    """
    try:
        encoded = processor.image_processor(images=images, return_tensors="pt")
        grid = encoded.get("image_grid_thw") if hasattr(encoded, "get") else encoded.image_grid_thw
        if grid is None:
            raise RuntimeError("image_grid_thw missing")
        rows = grid.detach().cpu().tolist() if hasattr(grid, "detach") else grid
        return sum(int(t) * int(h) * int(w) // (spatial_merge_size * spatial_merge_size) for t, h, w in rows)
    except Exception as exc:
        raise RuntimeError(
            "GUIPruner cannot establish the exact unpruned Qwen3-VL prompt-token denominator."
        ) from exc


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
        global_budget_policy: str | None = None,
        prune_layer: int | None = None,
        **kwargs,
    ) -> None:
        conflicting_backends = [
            name for name in ("QWEN3VL_ENABLE_ATTN_PRUNE", "QWEN3VL_USE_ATTN_PRUNE_MODEL")
            if os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}
        ]
        if conflicting_backends:
            raise RuntimeError(
                "GUIPruner-reproduction must use the unmodified transformers Qwen3-VL "
                "backend; disable " + ", ".join(conflicting_backends) + "."
            )
        def env(name: str, value, cast):
            return cast(value) if value is not None else cast(os.getenv(name, default_env[name]))

        default_env = {
            "GUI_PRUNER_HISTORY_STEPS": "4", "GUI_PRUNER_HISTORY_KEEP_RATIO": "0.40",
            "GUI_PRUNER_TEMPORAL_DECAY": "0.20", "GUI_PRUNER_CURRENT_KEEP_RATIO": "0.75",
            "GUI_PRUNER_BACKGROUND_SALIENCY": "0.30", "GUI_PRUNER_PRUNE_LAYER": "2",
            "GUI_PRUNER_GLOBAL_BUDGET_POLICY": "clip",
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
            global_budget_policy=(
                str(global_budget_policy) if global_budget_policy is not None
                else str(os.getenv("GUI_PRUNER_GLOBAL_BUDGET_POLICY", default_env["GUI_PRUNER_GLOBAL_BUDGET_POLICY"]))
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

    def _capture_sample_stats(self, history_before: int, history_after: int) -> None:
        """Expose actual, alignment-aware retention for inference summary.json."""
        stats = dict(getattr(self.model.config.text_config, "_guipruner_last_stats", {}) or {})
        current_before = int(stats.get("current_visual_tokens_before", 0) or 0)
        current_after = int(stats.get("current_visual_tokens_after", 0) or 0)
        baseline_prompt_before = int(stats.get("baseline_prompt_tokens", 0) or 0)
        prompt_before_ssp = int(stats.get("prompt_tokens_before_ssp", 0) or 0)
        prompt_after_ssp = int(stats.get("prompt_tokens_after_ssp", 0) or 0)
        if current_before <= 0:
            raise RuntimeError(
                "GUIPruner-reproduction SSP did not execute on the current image; "
                "refusing to emit an unpruned result without GUIPruner audit metrics."
            )
        # ``stage`` matches GUIPruner's optional eta and the repository's
        # state-packet accounting: TAR output plus current image.  ``raw``
        # remains available as the end-to-end ratio against original history.
        stage_before = int(history_after) + current_before
        stage_after = int(history_after) + current_after
        raw_before = int(history_before) + current_before
        raw_after = stage_after
        stats.update(
            {
                "GUIPruner_history_visual_tokens_before": int(history_before),
                "GUIPruner_history_visual_tokens_after": int(history_after),
                "GUIPruner_current_visual_tokens_before": current_before,
                "GUIPruner_current_visual_tokens_after": current_after,
                "GUIPruner_visual_tokens_before": stage_before,
                "GUIPruner_visual_tokens_after": stage_after,
                "GUIPruner_raw_visual_tokens_before": raw_before,
                "GUIPruner_raw_visual_tokens_after": raw_after,
                "GUIPruner_actual_history_retention_rate": (
                    float(history_after / history_before) if history_before > 0 else None
                ),
                "GUIPruner_actual_global_retention_rate": float(stage_after / stage_before),
                "GUIPruner_actual_global_pruning_rate": float(1.0 - stage_after / stage_before),
                "GUIPruner_actual_end_to_end_retention_rate": float(raw_after / raw_before),
                "GUIPruner_actual_end_to_end_pruning_rate": float(1.0 - raw_after / raw_before),
                "GUIPruner_prompt_tokens_before_baseline": baseline_prompt_before or None,
                "GUIPruner_prompt_tokens_before_ssp": prompt_before_ssp,
                "GUIPruner_prompt_tokens_after_ssp": prompt_after_ssp,
                "GUIPruner_actual_prompt_retention_rate": (
                    float(prompt_after_ssp / baseline_prompt_before)
                    if baseline_prompt_before > 0 else None
                ),
                "GUIPruner_actual_post_tar_prompt_retention_rate": (
                    float(prompt_after_ssp / prompt_before_ssp)
                    if prompt_before_ssp > 0 else None
                ),
            }
        )
        self._guipruner_last_sample_stats = stats

    def _correct_split_flops(self) -> None:
        """Replace the common tracker's full-prefill assumption after SSP ran."""
        stats = dict(getattr(self.model.config.text_config, "_guipruner_last_stats", {}) or {})
        before = int(stats.get("prompt_tokens_before_ssp", 0) or 0)
        after = int(stats.get("prompt_tokens_after_ssp", 0) or 0)
        if before <= 0 or after <= 0:
            raise RuntimeError("GUIPruner FLOPs correction requires successful SSP token audit.")
        generic = dict(getattr(self.model, "_vlmeval_last_sample_flops", {}) or {})
        if not generic or float(generic.get("e2e_flops", 0.0) or 0.0) <= 0:
            return
        corrected = correct_split_prefill_flops(
            self.model,
            generic,
            prompt_tokens_before_ssp=before,
            prompt_tokens_after_ssp=after,
            prune_layer_one_based=self.guipruner_config.prune_layer,
        )
        self.model._vlmeval_last_sample_flops = corrected
        records = getattr(self, "_vlmeval_generate_timing_records", None) or []
        if records and isinstance(records[-1], dict):
            records[-1].update(corrected)

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        self._guipruner_last_sample_stats = None
        history_idx, current_idx, identification = _history_and_current_indices(message)
        if len(history_idx) > self.guipruner_config.history_steps:
            raise ValueError("Input contains more history screenshots than GUIPruner history_steps.")
        if not history_idx:
            self.model.config.text_config._guipruner_history_original_visual_tokens = 0
            self.model.config.text_config._guipruner_history_original_processor_visual_tokens = 0
            self.model.config.text_config._guipruner_history_visual_tokens_after_tar = 0
            result = super().generate_inner_transformers(message, dataset=dataset, **kwargs)
            self._correct_split_flops()
            self._capture_sample_stats(history_before=0, history_after=0)
            return result
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
        history_original_processor_tokens = _processor_visual_tokens(
            self.processor, loaded, int(vision.spatial_merge_size)
        )
        # SSP runs inside ``super``. Publish this pre-TAR token count before
        # entering it so a requested global budget can be resolved there.
        self.model.config.text_config._guipruner_history_original_visual_tokens = history_original_tokens
        self.model.config.text_config._guipruner_history_original_processor_visual_tokens = (
            history_original_processor_tokens
        )
        self.model.config.text_config._guipruner_history_visual_tokens_after_tar = sum(
            record.actual_visual_tokens for record in audit
        )
        paths: list[str] = []
        try:
            for index, image in zip(history_idx, resized):
                handle = tempfile.NamedTemporaryFile(prefix="guipruner_tar_", suffix=".png", delete=False)
                handle.close()
                image.save(handle.name)
                paths.append(handle.name)
                patched[index]["value"] = handle.name
            result = super().generate_inner_transformers(patched, dataset=dataset, **kwargs)
            self._correct_split_flops()
            cfg = self.model.config.text_config
            cfg._guipruner_tar_audit = {
                "implementation": "GUIPruner-reproduction (unofficial reproduction)",
                "history_identification": identification,
                "current_message_index": current_idx,
                "history_original_visual_tokens": history_original_tokens,
                "overall_keep_ratio": self.guipruner_config.overall_keep_ratio,
                "records": [record.__dict__ for record in audit],
            }
            self._capture_sample_stats(
                history_before=history_original_tokens,
                history_after=sum(record.actual_visual_tokens for record in audit),
            )
            return result
        finally:
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
