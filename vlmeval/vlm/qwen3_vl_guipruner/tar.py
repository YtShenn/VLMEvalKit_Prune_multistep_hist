"""Temporal-Adaptive Resolution (TAR), Algorithm 1 of GUIPruner."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class TARRecord:
    history_index: int
    temporal_lag: int
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    theoretical_token_quota: float
    actual_visual_tokens: int


def temporal_token_quotas(num_history: int, orig_tokens: int, keep_ratio: float, decay: float) -> list[float]:
    """Return quotas ordered from oldest to newest history image.

    The formula's k=1 is the newest frame.  Prompt history is normally oldest
    to newest, so this function deliberately returns that order.
    """
    if num_history < 0 or orig_tokens < 0:
        raise ValueError("num_history and orig_tokens must be non-negative.")
    if num_history == 0:
        return []
    budget = math.floor(num_history * orig_tokens * keep_ratio)
    if num_history == 1:
        return [float(budget)]
    newest_weights = [decay + (1.0 - decay) * (num_history - k) / (num_history - 1) for k in range(1, num_history + 1)]
    newest_quotas = [budget * weight / sum(newest_weights) for weight in newest_weights]
    return list(reversed(newest_quotas))


def _aligned_size(width: int, height: int, scale: float, align: int) -> tuple[int, int]:
    # Nearest lower aligned size makes the actual cost never exceed a quota.
    new_w = max(align, int(math.floor(width * scale / align)) * align)
    new_h = max(align, int(math.floor(height * scale / align)) * align)
    return new_w, new_h


def apply_tar(
    history_images: list[Image.Image],
    *,
    keep_ratio: float,
    decay: float,
    patch_size: int,
    spatial_merge_size: int,
) -> tuple[list[Image.Image], list[TARRecord]]:
    """Resize only history images; caller owns the unchanged current image."""
    if len(history_images) > 4:
        raise ValueError("GUIPruner-reproduction accepts at most four history images.")
    if not history_images:
        return [], []
    align = int(patch_size) * int(spatial_merge_size)
    if align <= 0:
        raise ValueError("patch_size * spatial_merge_size must be positive.")
    # The paper assumes a common original resolution.  We use each image's
    # actual token count so mixed-resolution prompts remain budget-auditable.
    first_w, first_h = history_images[-1].size  # newest image = N_orig reference
    orig_tokens = (first_w // align) * (first_h // align)
    quotas = temporal_token_quotas(len(history_images), orig_tokens, keep_ratio, decay)
    output, records = [], []
    for idx, (image, quota) in enumerate(zip(history_images, quotas)):
        width, height = image.size
        base_tokens = max(1, (width // align) * (height // align))
        scale = math.sqrt(max(0.0, quota) / base_tokens)
        out_size = _aligned_size(width, height, scale, align)
        resized = image.resize(out_size, Image.Resampling.BILINEAR)
        actual = (out_size[0] // align) * (out_size[1] // align)
        # idx is oldest-first, therefore lag descends toward the current state.
        output.append(resized)
        records.append(TARRecord(idx, len(history_images) - idx, (width, height), out_size, quota, actual))
    return output, records
