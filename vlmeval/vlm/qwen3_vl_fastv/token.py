"""Small, model-independent FastV token selection primitives."""
from __future__ import annotations

import torch


def build_token_meta(visual_pos_masks: torch.Tensor | None) -> dict | None:
    """Return all injected visual-token positions, including every history image.

    Qwen3-VL has already expanded every image placeholder by this point, so the
    mask is robust to heterogeneous image grids and text between image blocks.
    """
    if visual_pos_masks is None or visual_pos_masks.ndim != 2 or visual_pos_masks.shape[0] != 1:
        return None
    positions = torch.where(visual_pos_masks[0].bool())[0]
    if positions.numel() == 0:
        return None
    return {"visual_indices": positions}


def select_keep_indices(seq_len: int, visual_indices: torch.Tensor, scores: torch.Tensor, drop_ratio: float) -> torch.Tensor:
    """Official FastV global top-k: preserve non-visual tokens and sorted order."""
    if not 0.0 <= float(drop_ratio) < 1.0:
        raise ValueError("drop_ratio must be in [0, 1)")
    visual_indices = visual_indices.to(dtype=torch.long)
    if visual_indices.numel() != scores.numel():
        raise ValueError("FastV scores must align with visual_indices")
    if int(visual_indices.numel()) == 0 or float(drop_ratio) == 0.0:
        return torch.arange(seq_len, device=visual_indices.device)
    keep_count = int(round(int(visual_indices.numel()) * (1.0 - float(drop_ratio))))
    # The official topk expression requires k > 0. This is its safe equivalent
    # for a valid R<1 configuration on very small visual blocks.
    keep_count = max(1, min(int(visual_indices.numel()), keep_count))
    chosen = torch.topk(scores.float(), k=keep_count, largest=True).indices
    keep_mask = torch.ones(seq_len, dtype=torch.bool, device=visual_indices.device)
    keep_mask[visual_indices] = False
    keep_mask[visual_indices.index_select(0, chosen)] = True
    return torch.where(keep_mask)[0]


def select_kv_sequence(cache_tensor: torch.Tensor, keep_indices: torch.Tensor) -> torch.Tensor:
    """Select the sequence axis of a standard [B, H, S, D] KV tensor.

    The custom decoder applies the same indices to its cache object; retaining
    this tiny primitive makes that contract unit-testable without a model.
    """
    if cache_tensor.ndim != 4:
        raise ValueError("expected KV tensor shaped [batch, heads, sequence, dim]")
    return cache_tensor.index_select(2, keep_indices.to(cache_tensor.device, dtype=torch.long))
