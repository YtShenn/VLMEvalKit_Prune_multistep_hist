"""Pure, testable SparseVLM token bookkeeping primitives."""
from __future__ import annotations

import math
import torch


def image_blocks(visual_mask: torch.Tensor | None, image_grid_thw: torch.Tensor | None, merge: int) -> list[torch.Tensor] | None:
    """Return per-image sequence positions; supports text between image blocks."""
    if visual_mask is None or image_grid_thw is None or visual_mask.ndim != 2 or visual_mask.shape[0] != 1:
        return None
    positions = torch.where(visual_mask[0].bool())[0]
    counts = (image_grid_thw.prod(-1) // int(merge) ** 2).detach().cpu().tolist()
    if sum(int(n) for n in counts) != int(positions.numel()):
        return None
    offset, blocks = 0, []
    for n in counts:
        n = int(n)
        blocks.append(positions[offset:offset + n])
        offset += n
    return blocks


def select_scope(blocks: list[torch.Tensor], scope: str) -> torch.Tensor:
    if not blocks:
        return torch.empty(0, dtype=torch.long)
    selected = blocks if scope == "all" else (blocks[-1:] if scope == "current" else blocks[:-1])
    return torch.cat(selected) if selected else blocks[0].new_empty((0,), dtype=torch.long)


def select_raters(hidden: torch.Tensor, visual_indices: torch.Tensor, text_indices: torch.Tensor) -> torch.Tensor:
    """SparseVLM §3.2 mean-above-mean text raters, with a safe empty result."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or not visual_indices.numel() or not text_indices.numel():
        return text_indices.new_empty((0,), dtype=torch.long)
    affinity = hidden[:, visual_indices, :] @ hidden[:, text_indices, :].transpose(1, 2)
    scores = affinity.softmax(-1).mean(1)[0]
    chosen = text_indices[scores > scores.mean()]
    return chosen if chosen.numel() else text_indices[scores.argmax()].reshape(1)


def retain_indices(scores: torch.Tensor, retain_ratio: float) -> torch.Tensor:
    if scores.ndim != 1:
        scores = scores.flatten()
    if not scores.numel():
        return scores.new_empty((0,), dtype=torch.long)
    k = min(scores.numel(), max(1, int(math.ceil(scores.numel() * float(retain_ratio)))))
    return torch.topk(scores.float(), k=k).indices.sort().values


def physical_keep_indices(seq_len: int, visual_indices: torch.Tensor, kept_relative: torch.Tensor) -> torch.Tensor:
    mask = torch.ones(seq_len, dtype=torch.bool, device=visual_indices.device)
    mask[visual_indices] = False
    mask[visual_indices.index_select(0, kept_relative)] = True
    return torch.where(mask)[0]
