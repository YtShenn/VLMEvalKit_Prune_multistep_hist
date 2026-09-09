"""Pure token-budget and selection primitives for HistPrune-GUI.

HistPrune-GUI reproduction/adaptation for Qwen3-VL.  Ranks are recent-first.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch


def allocate_history_budget(counts_recent_first: Sequence[int], keep_ratio: float,
                            weights_recent_first: Sequence[float]) -> list[int]:
    """Allocate one fixed history budget with capped weighted largest remainder."""
    counts = [max(0, int(x)) for x in counts_recent_first]
    total = sum(counts)
    budget = int(round(float(keep_ratio) * total))
    budget = max(0, min(total, budget))
    if not counts or budget == 0:
        return [0] * len(counts)
    if budget == total:
        return counts
    weights = [float(weights_recent_first[min(i, len(weights_recent_first) - 1)]) for i in range(len(counts))]
    assigned = [0] * len(counts)
    remaining = budget
    # Repeated water filling handles an image whose desired share exceeds capacity.
    available = {i for i, n in enumerate(counts) if n > 0}
    while remaining and available:
        denom = sum(weights[i] for i in available)
        if denom <= 0:
            ordered = sorted(available)
            proposals = {i: remaining / len(ordered) for i in ordered}
        else:
            proposals = {i: remaining * weights[i] / denom for i in available}
        floors = {i: min(counts[i] - assigned[i], int(math.floor(proposals[i]))) for i in available}
        added = sum(floors.values())
        for i, value in floors.items():
            assigned[i] += value
        remaining -= added
        available = {i for i in available if assigned[i] < counts[i]}
        if not remaining or not available:
            break
        # deterministic largest remainder; rank 0 wins exact ties.
        order = sorted(available, key=lambda i: (-(proposals.get(i, 0.0) - math.floor(proposals.get(i, 0.0))), i))
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if assigned[i] < counts[i]:
                assigned[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    assert sum(assigned) == budget, (assigned, budget)
    return assigned


def uniform_indices(n: int, k: int, device=None) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= n:
        return torch.arange(n, device=device)
    return torch.div(torch.arange(k, device=device) * n, k, rounding_mode="floor").long()


def select_indices(n: int, k: int, mode: str, seed: int, rank: int,
                   edge_mask: torch.Tensor | None = None, device=None) -> torch.Tensor:
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if k >= n:
        return torch.arange(n, device=device)
    if mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1000003 * int(rank))
        return torch.randperm(n, generator=gen)[:k].to(device=device).sort().values
    if mode == "uniform":
        return uniform_indices(n, k, device=device)
    if edge_mask is None or edge_mask.numel() != n:
        raise ValueError(f"{mode} needs an aligned Sobel edge mask of length {n}")
    preferred = edge_mask.bool() if mode == "sobel_foreground" else ~edge_mask.bool()
    primary = torch.where(preferred)[0]
    secondary = torch.where(~preferred)[0]
    keep_primary = uniform_indices(int(primary.numel()), min(k, int(primary.numel())), device=primary.device)
    selected = primary.index_select(0, keep_primary)
    remainder = k - int(selected.numel())
    if remainder:
        selected = torch.cat((selected, secondary.index_select(0, uniform_indices(int(secondary.numel()), remainder, secondary.device))))
    return selected.to(device=device).sort().values


def select_from_rank_map(rank_map: torch.Tensor, config, edge_masks: Sequence[torch.Tensor | None] | None = None):
    """Return a sequence keep mask and auditable per-rank statistics."""
    if rank_map.ndim != 2 or rank_map.shape[0] != 1:
        raise ValueError("HistPrune only supports batch size 1 rank maps")
    ranks = [r for r in range(int(config.max_history_frames)) if int((rank_map[0] == r).sum())]
    counts = [int((rank_map[0] == r).sum()) for r in ranks]
    budgets = allocate_history_budget(counts, config.history_keep_ratio, config.temporal_weights)
    keep = torch.ones_like(rank_map, dtype=torch.bool)
    actual = []
    for local, (rank, n, k) in enumerate(zip(ranks, counts, budgets)):
        positions = torch.where(rank_map[0] == rank)[0]
        edge = edge_masks[rank] if edge_masks is not None and rank < len(edge_masks) else None
        chosen = select_indices(n, k, config.mode, config.random_seed, rank, edge, positions.device)
        keep[0, positions] = False
        keep[0, positions.index_select(0, chosen)] = True
        actual.append(int(chosen.numel()))
    return keep, {"ranks": ranks, "counts": counts, "budgets": budgets, "actual": actual,
                  "history_tokens": sum(counts), "history_keep_budget": sum(budgets)}


def build_token_meta(input_ids: torch.Tensor | None, image_mask: torch.Tensor | None,
                     image_grid_thw: torch.Tensor | None, spatial_merge_size: int,
                     config, history_images) -> dict | None:
    """Map old->new image placeholder blocks into a recent-first history rank map."""
    if input_ids is None or image_mask is None or image_grid_thw is None:
        return None
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or image_mask.ndim != 2 or image_mask.shape[0] != 1:
        return None
    counts_prompt = (image_grid_thw.prod(-1) // int(spatial_merge_size) ** 2).tolist()
    if len(counts_prompt) < 2:  # current-only input
        return None
    history_count = len(counts_prompt) - 1
    if history_count > int(config.max_history_frames) or len(history_images) != history_count:
        raise ValueError("HistPrune image order/count differs from the declared <=4 history frames")
    positions = torch.where(image_mask[0].bool())[0]
    if int(sum(int(x) for x in counts_prompt)) != int(positions.numel()):
        raise ValueError("HistPrune cannot safely align image_grid_thw to image placeholder positions")
    rank_map = torch.full_like(input_ids, -1, dtype=torch.long)
    offset = 0
    edge_masks = [None] * int(config.max_history_frames)
    from .sobel import sobel_edge_mask
    for prompt_index, raw_count in enumerate(counts_prompt[:-1]):
        n = int(raw_count)
        segment = positions[offset:offset + n]
        offset += n
        rank = history_count - 1 - prompt_index
        rank_map[0, segment] = rank
        if config.mode.startswith("sobel"):
            image = history_images[prompt_index]
            if image is None:
                raise ValueError("Sobel HistPrune requires local readable history images")
            _, h, w = (int(x) for x in image_grid_thw[prompt_index].detach().cpu().tolist())
            gh, gw = h // int(spatial_merge_size), w // int(spatial_merge_size)
            edge = sobel_edge_mask(image, gh, gw, config.sobel_edge_threshold, config.sobel_ratio_threshold)
            temporal = max(1, n // max(1, edge.numel()))
            edge_masks[rank] = edge.repeat(temporal)[:n]
    return {"rank_map": rank_map, "edge_masks": edge_masks, "history_frame_count": history_count}
