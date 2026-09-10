"""Independent Qwen3-VL wrapper for the PruMerge-inspired baseline."""
from __future__ import annotations

from copy import deepcopy

from ..qwen3_vl.model import Qwen3VLChat
from .config import PruMergeConfig


def _is_current(item: dict) -> bool:
    return bool(item.get("is_current", False)) or str(item.get("frame_role", "")).lower() == "current"


def truncate_history_steps(message: list[dict], max_history_steps: int) -> list[dict]:
    """Keep the current visual step and at most the last N prior visual steps.

    The common VLMEvalKit representation is a flat content list.  If it does
    not carry an explicit step id, an image/video plus the text immediately
    preceding it is treated as a step.  This deliberately keeps associated
    text and media together instead of dropping only old screenshots.
    """
    if max_history_steps < 0:
        raise ValueError("max_history_steps must be non-negative")
    visual = [i for i, item in enumerate(message) if isinstance(item, dict) and item.get("type") in {"image", "video"}]
    if len(visual) <= 1:
        return list(message)
    current = next((i for i in reversed(visual) if _is_current(message[i])), visual[-1])
    history = [i for i in visual if i != current and i < current]
    if len(history) <= max_history_steps:
        return list(message)
    keep_visual = set(history[-max_history_steps:] + [current])
    # Assign each item to the visual item that terminates its flat-content
    # segment.  Text leading a retained image follows that image as context.
    starts = {v: (visual[idx - 1] + 1 if idx else 0) for idx, v in enumerate(visual)}
    keep_indices: set[int] = set()
    for v in keep_visual:
        start = starts[v]
        keep_indices.update(range(start, v + 1))
    # Preserve text after the current visual item (usually the user question).
    keep_indices.update(range(current + 1, len(message)))
    # System-like prefix before any image contains global instruction and is safe.
    if visual:
        keep_indices.update(range(0, starts[visual[0]]))
    return [item for i, item in enumerate(message) if i in keep_indices]


class Qwen3VLPruMergeChat(Qwen3VLChat):
    """Training-free Qwen3-VL adaptation of PruMerge selection and merging."""

    def __init__(self, *args, **kwargs) -> None:
        self.prumerge_config = PruMergeConfig.from_env()
        kwargs["use_vllm"] = False
        # Select the repository's cache/DeepStack-safe custom Qwen3 decoder.
        # PruMerge itself remains gated by the isolated config below.
        kwargs["prumerge"] = True
        kwargs["use_attn_prune"] = True
        super().__init__(*args, **kwargs)
        if not hasattr(self, "model"):
            raise RuntimeError("Qwen3VL-PruMerge requires the transformers Qwen3-VL backend")
        # PruMerge physically rebuilds the prefill sequence before layer 0;
        # normal generation cache remains enabled, matching FastV/SparseVLM.
        self.model.config.use_cache = True
        self.model.generation_config.use_cache = True
        self._publish()

    def _publish(self) -> None:
        outer = getattr(self.model.config, "text_config", None)
        inner = getattr(getattr(self.model, "model", None), "language_model", None)
        for cfg in (outer, getattr(inner, "config", None)):
            if cfg is not None:
                cfg._prumerge_enabled = bool(self.prumerge_config.enabled)
                cfg._prumerge_config = self.prumerge_config

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        limited_message = truncate_history_steps(list(message), self.prumerge_config.max_history_steps)
        self._publish()
        self._prumerge_last_sample_stats = None
        try:
            return super().generate_inner_transformers(limited_message, dataset=dataset, **kwargs)
        finally:
            cfg = getattr(self.model.config, "text_config", None)
            stats = getattr(cfg, "_prumerge_last_stats", None) if cfg is not None else None
            if isinstance(stats, dict):
                self._prumerge_last_sample_stats = deepcopy(stats)
