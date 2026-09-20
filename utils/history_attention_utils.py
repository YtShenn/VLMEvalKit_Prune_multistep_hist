"""Pure helpers for the history-visual-attention diagnostic.

This module deliberately contains no model patches.  Its grid accounting is based on
the processor supplied ``image_grid_thw`` and the image-placeholder positions in the
*actual* LLM sequence, never on the source image dimensions.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np


FORMAT_TOKEN_RE = re.compile(r"^[\s\[\]{}(),:\"']+$")


def grid_rows(image_grid_thw, spatial_merge_size=1):
    """Return final LLM grids (T, H/merge, W/merge), validating divisibility."""
    if hasattr(image_grid_thw, "detach"):
        image_grid_thw = image_grid_thw.detach().cpu().tolist()
    if image_grid_thw is None:
        return []
    if image_grid_thw and not isinstance(image_grid_thw[0], (tuple, list)):
        image_grid_thw = [image_grid_thw]
    m = max(1, int(spatial_merge_size))
    out = []
    for row in image_grid_thw:
        t, h, w = map(int, row[:3])
        if h % m or w % m:
            raise ValueError(f"image_grid_thw={row} is not divisible by merge={m}")
        out.append((t, h // m, w // m))
    return out


def locate_visual_segments(input_ids, image_token_id, grids, image_records):
    """Map each processor image, in order, to exact placeholder positions.

    Qwen VL expands one image placeholder into one final LLM visual token per
    merged-grid cell.  We require exact counts rather than silently proportionally
    splitting visual positions.
    """
    ids = input_ids.detach().cpu().tolist() if hasattr(input_ids, "detach") else input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    pos = [i for i, x in enumerate(ids) if int(x) == int(image_token_id)]
    expected = [t * h * w for t, h, w in grids]
    if len(expected) != len(image_records):
        raise ValueError(f"processor supplied {len(expected)} grids for {len(image_records)} prompt images")
    if len(pos) != sum(expected):
        raise ValueError(f"visual placeholder count {len(pos)} != grid count {sum(expected)}")
    out, cursor = [], 0
    for record, (t, h, w), count in zip(image_records, grids, expected):
        segment = pos[cursor:cursor + count]
        if segment != list(range(segment[0], segment[0] + count)):
            raise ValueError("visual placeholders for an image are non-contiguous")
        x = dict(record)
        x.update(sequence_start=segment[0], sequence_end=segment[-1] + 1,
                 grid_t=t, grid_height=h, grid_width=w, visual_tokens=count)
        out.append(x)
        cursor += count
    return out


def semantic_token_indices(token_texts):
    """Conservative configurable default: omit only unmistakable JSON punctuation."""
    return [i for i, text in enumerate(token_texts) if text.strip() and not FORMAT_TOKEN_RE.fullmatch(text)]


def normalized_importance(attentions, query_positions, hist_positions, layers="last_third", eps=1e-12):
    """Compute I(k), S_hist, using [layer, batch, head, query, key] tensors."""
    if not attentions:
        raise ValueError("Model returned no attentions; use eager attention backend.")
    start = len(attentions) * 2 // 3 if layers == "last_third" else 0
    q = np.asarray(query_positions, dtype=np.int64)
    k = np.asarray(hist_positions, dtype=np.int64)
    cond, masses = [], []
    for attn in attentions[start:]:
        a = attn[0, :, q, :][:, :, k].float().cpu().numpy()  # H,Q,K
        denom = a.sum(axis=-1, keepdims=True)
        cond.append(a / (denom + eps))
        masses.append(denom[..., 0])
    cond = np.concatenate(cond, axis=0)  # (L*H), Q, K
    return cond.mean(axis=(0, 1)), float(np.concatenate(masses, axis=0).mean())


def entropy(prob, eps=1e-12):
    prob = np.asarray(prob, dtype=float)
    prob = prob / max(float(prob.sum()), eps)
    return float(-(prob * np.log(prob + eps)).sum())


def roi_mask_xyxy(box, grid_h, grid_w, image_size):
    """Cell-center ROI rasterization; image_size is only for coordinate conversion."""
    x1, y1, x2, y2 = map(float, box)
    iw, ih = image_size
    xs = (np.arange(grid_w) + .5) * iw / grid_w
    ys = (np.arange(grid_h) + .5) * ih / grid_h
    return ((ys[:, None] >= y1) & (ys[:, None] <= y2) &
            (xs[None, :] >= x1) & (xs[None, :] <= x2))


def load_compatible_roi_json(path):
    """Load repository ROI JSON: keys are screenshot names, values are xyxy boxes.

    Existing ScreenSpot-Pro recall code stores candidate coordinates at 1/4 of pixel
    scale; its caller explicitly applies x4.  This loader keeps that convention for
    ``roi_results`` style files, while accepting explicit ``coordinate_scale``.
    """
    raw = json.loads(Path(path).read_text())
    scale = float(raw.get("coordinate_scale", 4.0)) if isinstance(raw, dict) else 4.0
    mapping = raw.get("rois", raw) if isinstance(raw, dict) else raw
    return mapping, scale


def roi_boxes_for_path(mapping, path, scale=4.0):
    stem, name = Path(path).stem, Path(path).name
    boxes = mapping.get(path, mapping.get(name, mapping.get(stem, [])))
    return [[float(v) * scale for v in b] for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]


def bootstrap_episode(records, field, n=2000, seed=42):
    """Episode bootstrap, never patch bootstrap."""
    groups = {}
    for x in records:
        if x.get(field) is not None:
            groups.setdefault(str(x.get("episode_id", x.get("sample_id"))), []).append(float(x[field]))
    vals = np.asarray([np.mean(v) for v in groups.values()])
    if not len(vals): return {"mean": None, "ci95": [None, None], "episodes": 0}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return {"mean": float(vals.mean()), "ci95": [float(np.quantile(boots,.025)), float(np.quantile(boots,.975))], "episodes": len(vals)}
