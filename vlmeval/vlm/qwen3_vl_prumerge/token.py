"""Pure tensor primitives for Qwen3-VL PruMerge-inspired compression.

Qwen3-VL does not expose a CLIP CLS token.  Saliency therefore uses the
cosine similarity to a per-block global visual query (the normalized mean of
the merged visual features), rather than claiming the original CLS-IQR score.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MergePlan:
    """A per-block partition: every input token maps to one kept centre."""
    keep: torch.Tensor
    assignment: torch.Tensor
    weights: torch.Tensor
    scores: torch.Tensor


def contiguous_visual_blocks(visual_mask: torch.Tensor | None) -> list[torch.Tensor] | None:
    """Return ordered physical sequence positions for each contiguous visual block."""
    if visual_mask is None or visual_mask.ndim != 2 or visual_mask.shape[0] != 1:
        return None
    pos = torch.where(visual_mask[0].bool())[0]
    if not pos.numel():
        return []
    split = torch.where(pos[1:] != pos[:-1] + 1)[0] + 1
    return list(torch.tensor_split(pos, split))


def scope_block_indices(num_blocks: int, scope: str) -> set[int]:
    if num_blocks <= 0:
        return set()
    if scope == "current":
        return {num_blocks - 1}
    if scope == "history":
        return set(range(max(0, num_blocks - 1)))
    return set(range(num_blocks))


def visual_saliency(features: torch.Tensor) -> torch.Tensor:
    """Qwen-adapted global-query saliency for one visual block."""
    if features.ndim != 2 or not features.shape[0]:
        return features.new_empty((0,))
    unit = F.normalize(features.float(), dim=-1, eps=1e-6)
    query = F.normalize(unit.mean(0, keepdim=True), dim=-1, eps=1e-6)
    return (unit @ query.transpose(0, 1)).squeeze(-1)


def _bounded_indices(scores: torch.Tensor, candidate: torch.Tensor, min_keep: int, max_keep: int) -> torch.Tensor:
    n = int(scores.numel())
    if not n:
        return scores.new_empty((0,), dtype=torch.long)
    budget_lo = min(n, max(1, int(min_keep)))
    budget_hi = min(n, max(budget_lo, int(max_keep)))
    candidate = candidate.unique(sorted=True)
    if candidate.numel() > budget_hi:
        candidate = candidate[torch.topk(scores.index_select(0, candidate), budget_hi).indices]
    if candidate.numel() < budget_lo:
        chosen = torch.zeros(n, dtype=torch.bool, device=scores.device)
        chosen[candidate] = True
        extras = torch.argsort(scores, descending=True)
        extras = extras[~chosen[extras]][: budget_lo - candidate.numel()]
        candidate = torch.cat((candidate, extras))
    return candidate.sort().values


def select_iqr(scores: torch.Tensor, min_keep: int, max_keep: int, multiplier: float) -> torch.Tensor:
    """Select high IQR outliers, with deterministic bounded top-score fallback."""
    if scores.ndim != 1:
        scores = scores.flatten()
    if not scores.numel():
        return scores.new_empty((0,), dtype=torch.long)
    q1, q3 = torch.quantile(scores.float(), torch.tensor([0.25, 0.75], device=scores.device))
    upper = q3 + float(multiplier) * (q3 - q1)
    return _bounded_indices(scores, torch.where(scores > upper)[0], min_keep, max_keep)


def uniform_extra_indices(length: int, selected: torch.Tensor, count: int) -> torch.Tensor:
    """Evenly spread supplemental indices, excluding already selected tokens."""
    if length <= 0 or count <= 0:
        return selected.new_empty((0,), dtype=torch.long)
    all_idx = torch.arange(length, device=selected.device)
    free = all_idx[~torch.isin(all_idx, selected)]
    if not free.numel():
        return free
    count = min(int(count), int(free.numel()))
    pick = torch.linspace(0, free.numel() - 1, count, device=free.device).round().long()
    return free[pick].unique(sorted=True)


def make_merge_plan(features: torch.Tensor, *, min_keep: int, max_keep: int, iqr_multiplier: float,
                    variant: str, similarity_temperature: float = 1.0) -> MergePlan:
    """Build a selection and similarity-based aggregation partition for one block."""
    if features.ndim != 2:
        raise ValueError("features must have shape [tokens, hidden]")
    n = int(features.shape[0])
    scores = visual_saliency(features)
    keep = select_iqr(scores, min_keep, max_keep, iqr_multiplier)
    if variant == "prumerge_plus" and keep.numel() < min(n, max_keep):
        extra_budget = min(int(keep.numel()), min(n, max_keep) - int(keep.numel()))
        keep = torch.cat((keep, uniform_extra_indices(n, keep, extra_budget))).unique(sorted=True)
    if not keep.numel():
        return MergePlan(keep, keep, scores, scores)
    unit = F.normalize(features.float(), dim=-1, eps=1e-6)
    centres = unit.index_select(0, keep)
    assignment = (unit @ centres.transpose(0, 1) / float(similarity_temperature)).argmax(-1)
    # Saliency is shifted positive so every token contributes to its selected centre.
    weights = (scores - scores.min()).clamp_min(0) + 1e-6
    return MergePlan(keep=keep, assignment=assignment, weights=weights, scores=scores)


def aggregate_features(features: torch.Tensor, plan: MergePlan) -> torch.Tensor:
    """Weighted aggregation for the selected centres; output order follows plan.keep."""
    if not plan.keep.numel():
        return features.new_empty((0, features.shape[-1]))
    out = torch.zeros((plan.keep.numel(), features.shape[-1]), dtype=features.dtype, device=features.device)
    weights = plan.weights.to(features.device, dtype=features.dtype)
    assignment = plan.assignment.to(features.device)
    out.index_add_(0, assignment, features * weights.unsqueeze(-1))
    denom = torch.zeros((plan.keep.numel(),), dtype=features.dtype, device=features.device)
    denom.index_add_(0, assignment, weights)
    return out / denom.clamp_min(torch.finfo(features.dtype).eps).unsqueeze(-1)


def aggregate_visual_blocks(features: torch.Tensor, lengths: list[int], plans: list[MergePlan | None]) -> torch.Tensor:
    """Apply plans to visual blocks without mixing images, videos, or history steps."""
    if features.ndim != 2 or len(lengths) != len(plans) or sum(lengths) != int(features.shape[0]):
        raise ValueError("visual feature blocks and plans are inconsistent")
    out, start = [], 0
    for length, plan in zip(lengths, plans):
        block = features[start:start + int(length)]
        out.append(aggregate_features(block, plan) if plan is not None else block)
        start += int(length)
    return torch.cat(out, dim=0) if out else features.new_empty((0, features.shape[-1]))


def plans_metadata(plans: list[MergePlan | None], lengths: list[int]) -> list[dict]:
    """Compact CPU-safe per-block summary suitable for result JSON."""
    records = []
    for index, (plan, length) in enumerate(zip(plans, lengths)):
        kept = int(plan.keep.numel()) if plan is not None else int(length)
        records.append({"block_index": index, "tokens_before": int(length), "tokens_after": kept,
                        "compressed": bool(plan is not None)})
    return records
