#!/usr/bin/env python3
"""Merge ROI boxes and optionally visualize merged boxes on existing images.

Input JSON format:
{
  "image_key.png": [[x1, y1, x2, y2], ...]
}

Output JSON keeps the same keys and stores merged boxes in the same list format.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw


# =========================
# Global Config
# =========================
INPUT_JSON = "qwen3vl_1-15_layer_0.25scale/roi_results.json"
OUTPUT_JSON = "qwen3vl_1-15_layer_0.25scale/roi_results_merged.json"

VISUALIZE = True
VIS_INPUT_DIR = "roi_cropped_images_scale_onimg_0.25"
VIS_OUTPUT_DIR = "roi_cropped_images_scale_onimg_0.25_merged"
LINE_COLOR = (0, 0, 255)
LINE_WIDTH = 3

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _to_box(box: List[float]) -> Tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"Invalid box length: {box}")
    x1, y1, x2, y2 = map(float, box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _rect_distance(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return math.hypot(dx, dy)


def _overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _box_size(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(x2 - x1, y2 - y1)


def _merge_two(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def merge_boxes_recursive(boxes: List[List[float]]) -> List[List[int]]:
    originals = [_to_box(b) for b in boxes]
    n = len(originals)
    used = [False] * n
    merged: List[Tuple[float, float, float, float]] = []

    for i in range(n):
        if used[i]:
            continue

        used[i] = True
        current = originals[i]

        changed = True
        while changed:
            changed = False
            for j in range(n):
                if used[j]:
                    continue

                cand = originals[j]
                dist = _rect_distance(current, cand)
                should_merge = _overlap(current, cand) #or (dist < _box_size(cand))
                if should_merge:
                    current = _merge_two(current, cand)
                    used[j] = True
                    changed = True

        merged.append(current)

    merged.sort(key=lambda b: (b[0], b[1], b[2], b[3]))
    out: List[List[int]] = []
    for x1, y1, x2, y2 in merged:
        out.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
    return out


def _extract_idx(name: str) -> Optional[str]:
    m = re.search(r"idx(\d+)", name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def _extract_timestamp(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})[\s_-](\d{2})[-_](\d{2})[-_](\d{2})", name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}_{h}-{mi}-{s}"


def _make_match_id(name: str) -> Optional[Tuple[str, str]]:
    idx = _extract_idx(name)
    ts = _extract_timestamp(name)
    if idx is None or ts is None:
        return None
    return idx, ts


def build_merged_results(input_json: Path, output_json: Path) -> Dict[str, List[List[int]]]:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    merged: Dict[str, List[List[int]]] = {}
    for key, boxes in data.items():
        if not isinstance(boxes, list):
            merged[key] = []
            continue
        merged[key] = merge_boxes_recursive(boxes)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    return merged


def visualize_merged_boxes(
    merged_results: Dict[str, List[List[int]]],
    vis_input_dir: Path,
    vis_output_dir: Path,
) -> None:
    key_to_boxes: Dict[Tuple[str, str], List[List[int]]] = {}
    duplicate_count = 0

    for key, boxes in merged_results.items():
        match_id = _make_match_id(key)
        if match_id is None:
            continue
        if match_id in key_to_boxes:
            duplicate_count += 1
        key_to_boxes[match_id] = boxes

    total_images = 0
    matched_images = 0

    for img_path in vis_input_dir.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        total_images += 1
        match_id = _make_match_id(img_path.name)
        if match_id is None:
            continue

        boxes = key_to_boxes.get(match_id)
        if boxes is None:
            continue

        matched_images += 1
        rel = img_path.relative_to(vis_input_dir)
        out_path = vis_output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in boxes:
            draw.rectangle([x1*4, y1*4, x2*4, y2*4], outline=LINE_COLOR, width=LINE_WIDTH)
        img.save(out_path)

    print(f"[VIS] input images: {total_images}")
    print(f"[VIS] matched images: {matched_images}")
    print(f"[VIS] output dir: {vis_output_dir}")
    if duplicate_count > 0:
        print(f"[VIS] warning: duplicate (idx, timestamp) keys in json: {duplicate_count}")


def main() -> None:
    input_json = Path(INPUT_JSON)
    output_json = Path(OUTPUT_JSON)
    vis_input_dir = Path(VIS_INPUT_DIR)
    vis_output_dir = Path(VIS_OUTPUT_DIR)

    if not input_json.exists():
        raise FileNotFoundError(f"Input json not found: {input_json}")

    merged_results = build_merged_results(input_json, output_json)
    print(f"[JSON] merged results saved to: {output_json}")
    print(f"[JSON] total keys: {len(merged_results)}")

    if VISUALIZE:
        if not vis_input_dir.exists():
            raise FileNotFoundError(f"Visualization input dir not found: {vis_input_dir}")
        visualize_merged_boxes(merged_results, vis_input_dir, vis_output_dir)


if __name__ == "__main__":
    main()
