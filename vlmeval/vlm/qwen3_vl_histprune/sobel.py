"""Sobel patch masks aligned to the post-merge Qwen3-VL visual grid."""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image


def sobel_edge_mask(image: Image.Image, grid_h: int, grid_w: int, edge_threshold: float, ratio_threshold: float) -> torch.Tensor:
    """Classify each LLM visual cell; resizing makes the mask grid-aligned."""
    gray = np.asarray(image.convert("L").resize((grid_w, grid_h), Image.Resampling.BILINEAR), dtype=np.float32)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    # ratio threshold is meaningful at patch level; one grid cell is one merged patch.
    magnitude = np.hypot(gx, gy)
    threshold = max(float(edge_threshold), 255.0 * float(ratio_threshold))
    return torch.from_numpy((magnitude >= threshold).reshape(-1))
