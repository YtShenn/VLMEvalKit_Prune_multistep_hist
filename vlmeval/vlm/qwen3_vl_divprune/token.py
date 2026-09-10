"""Pure, GPU-friendly Layer-0 DivPrune token-selection primitives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def image_blocks(visual_mask: torch.Tensor | None, image_grid_thw: torch.Tensor | None, merge: int) -> list[torch.Tensor] | None:
    """Map expanded visual positions to image blocks without assuming contiguity."""
    if visual_mask is None or image_grid_thw is None or visual_mask.ndim != 2 or visual_mask.shape[0] != 1:
        return None
    positions = torch.where(visual_mask[0].bool())[0]
    counts = (image_grid_thw.prod(-1) // int(merge) ** 2).detach().cpu().tolist()
    if sum(int(n) for n in counts) != int(positions.numel()):
        return None
    offset, blocks = 0, []
    for count in counts:
        count = int(count)
        blocks.append(positions[offset: offset + count])
        offset += count
    return blocks


def max_min_keep_relative(features: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Greedy max-min cosine-distance selector from DivPrune Algorithm 1.

    Return sorted relative positions.  The deterministic ``argmax`` tie break
    is the first index, matching normal PyTorch semantics.
    """
    if features.ndim != 2:
        raise ValueError("DivPrune features must have shape [tokens, hidden]")
    count = int(features.shape[0])
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=features.device)
    if not 0.0 < float(keep_ratio) <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    keep = min(count, max(1, int(round(count * float(keep_ratio)))))
    if keep == count:
        return torch.arange(count, dtype=torch.long, device=features.device)
    values = features.float()
    if not torch.isfinite(values).all():
        raise ValueError("DivPrune received non-finite visual features")
    normalized = F.normalize(values, p=2, dim=-1, eps=1e-12)
    distance = 1.0 - normalized @ normalized.transpose(0, 1)
    # First selection: candidate with the most distant nearest neighbour.
    distance.fill_diagonal_(float("inf"))
    nearest = distance.min(dim=1).values
    first = nearest.argmax()
    selected = torch.empty(keep, dtype=torch.long, device=features.device)
    selected[0] = first
    available = torch.ones(count, dtype=torch.bool, device=features.device)
    available[first] = False
    min_to_selected = distance[:, first].clone()
    for index in range(1, keep):
        scores = min_to_selected.masked_fill(~available, -float("inf"))
        chosen = scores.argmax()
        selected[index] = chosen
        available[chosen] = False
        min_to_selected = torch.minimum(min_to_selected, distance[:, chosen])
    return selected.sort().values


def build_token_meta(
    inputs_embeds: torch.Tensor,
    visual_mask: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    merge: int,
    keep_ratio: float,
    scope: str,
) -> dict | None:
    """Build one prefill-only plan; all positions remain in original order."""
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        return None
    blocks = image_blocks(visual_mask, image_grid_thw, merge)
    if not blocks:
        return None
    all_visual = torch.cat(blocks)
    selected_visual = all_visual if scope == "global_all_visual" else torch.cat(blocks[:-1]) if len(blocks) > 1 else all_visual.new_empty((0,))
    if selected_visual.numel() == 0:
        return {"blocks": blocks, "selected_visual": selected_visual, "kept_visual": all_visual,
                "per_image_before": [int(b.numel()) for b in blocks], "per_image_after": [int(b.numel()) for b in blocks],
                "reason": "history_scope_has_no_history"}
    features = inputs_embeds[0].index_select(0, selected_visual)
    kept_relative = max_min_keep_relative(features, keep_ratio)
    kept_selected = selected_visual.index_select(0, kept_relative)
    # The scope not selected for pruning must be retained in full (notably the
    # current image for the explicitly non-faithful history_only option).
    retained = torch.cat((kept_selected, all_visual[~torch.isin(all_visual, selected_visual)])).sort().values
    after = [int(torch.isin(block, retained).sum().item()) for block in blocks]
    return {
        "blocks": blocks, "selected_visual": selected_visual, "kept_visual": retained,
        "per_image_before": [int(block.numel()) for block in blocks], "per_image_after": after,
        "selection_count": int(selected_visual.numel()), "reason": "ready",
    }
