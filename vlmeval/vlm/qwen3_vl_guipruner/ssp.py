"""Stratified Structure-aware Pruning (SSP), Algorithm 2 of GUIPruner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SSPSelection:
    foreground: torch.Tensor
    background: torch.Tensor
    uniform: torch.Tensor
    final: torch.Tensor
    budget: int


def foreground_mask(image: Image.Image, grid_h: int, grid_w: int) -> torch.Tensor:
    """Paper Appendix A.4.3 edge pipeline, mapped by max-pooling to token grid."""
    try:
        import cv2
    except ImportError as exc:  # Never silently substitute a different detector.
        raise RuntimeError("GUIPruner SSP requires opencv-python for its paper edge pipeline.") from exc
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.createCLAHE().apply(gray)
    edges = cv2.bitwise_or(cv2.Canny(gray, 50, 150), cv2.Canny(gray, 100, 200))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = np.zeros_like(edges)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if area >= 4 and 1 / 50 <= aspect <= 50:
            cv2.drawContours(valid, [contour], -1, 255, thickness=1)
    # exact max pooling into the LLM visual-token grid, including non-divisible images.
    tensor = torch.from_numpy(valid > 0).float()[None, None]
    return torch.nn.functional.adaptive_max_pool2d(tensor, (grid_h, grid_w)).bool().flatten()


def _uniform_from_remaining(remaining: torch.Tensor, count: int) -> torch.Tensor:
    if count <= 0 or remaining.numel() == 0:
        return remaining.new_empty((0,), dtype=torch.long)
    if count >= remaining.numel():
        return remaining
    # Deterministic UGS: evenly spaced representatives of the remaining grid order.
    slots = torch.linspace(0, remaining.numel() - 1, steps=count, device=remaining.device).round().long()
    return remaining[torch.unique_consecutive(slots)]


def select_stratified(scores: torch.Tensor, foreground: torch.Tensor, keep_ratio: float, rho: float) -> SSPSelection:
    """Faithful three-budget selection. Returned indices are sorted original order."""
    scores = scores.flatten()
    foreground = foreground.flatten().bool().to(scores.device)
    if scores.numel() != foreground.numel():
        raise ValueError("SSP score and foreground-mask lengths must match.")
    total = int(scores.numel())
    budget = int(total * keep_ratio)
    fg_pool = torch.where(foreground)[0]
    bg_pool = torch.where(~foreground)[0]
    fg_n = min(int(fg_pool.numel() * keep_ratio), budget)
    bg_n = min(int(bg_pool.numel() * keep_ratio * rho), max(0, budget - fg_n))
    fg = fg_pool[torch.topk(scores[fg_pool], fg_n, sorted=False).indices] if fg_n else fg_pool[:0]
    bg = bg_pool[torch.topk(scores[bg_pool], bg_n, sorted=False).indices] if bg_n else bg_pool[:0]
    chosen = torch.cat((fg, bg))
    remain_mask = torch.ones(total, dtype=torch.bool, device=scores.device)
    remain_mask[chosen] = False
    uniform = _uniform_from_remaining(torch.where(remain_mask)[0], budget - chosen.numel())
    final = torch.sort(torch.cat((fg, bg, uniform))).values
    if final.numel() != budget or torch.unique(final).numel() != final.numel():
        raise RuntimeError("GUIPruner SSP budget invariant failed.")
    return SSPSelection(torch.sort(fg).values, torch.sort(bg).values, torch.sort(uniform).values, final, budget)
