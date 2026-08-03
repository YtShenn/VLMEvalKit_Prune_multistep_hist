#!/usr/bin/env python3
"""Generate ScreenSpot_Pro GT-centered ROI boxes and visualization images.

For each ScreenSpot_Pro sample, this script:
1. Builds one ROI box centered at GT bbox center.
2. Sets ROI area ratio by ``--area-ratio`` (default 0.25).
3. Saves ROI map json using keys like: idx{n}_{question_slug}_{image_stem}_attn.png
4. Draws ROI in red on original image and saves visualization image.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw


# =========================
# Global Config
# =========================
AREA_RATIO = 0.0625  # 0.25
IMAGES_ROOT = None  # None -> auto detect via _default_images_root()
VIZ_OUTPUT_DIR = "roi_cropped_images_scale_onimg_gtcenter_1" #"roi_cropped_images_scale_onimg_gtcenter"#
OUTPUT_JSON = "qwen3vl_1-15_layer_0.5scale_name_switcnxy/roi_results_gt_center_1.json" #"qwen3vl_1-15_layer_0.5scale_name_switcnxy/roi_results_gt_center.json"#
SAVE_VIS_IMAGES = False


DEFAULT_DATASET_SOURCES = {
    "ScreenSpot_Pro_Development": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_Development.tsv",
    "ScreenSpot_Pro_Creative": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_Creative.tsv",
    "ScreenSpot_Pro_CAD": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_CAD.tsv",
    "ScreenSpot_Pro_Scientific": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_Scientific.tsv",
    "ScreenSpot_Pro_Office": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_Office.tsv",
    "ScreenSpot_Pro_OS": "http://opencompass.openxlab.space/utils/benchmarks/GUI/ScreenSpot_Pro/ScreenSpot_Pro_OS.tsv",
}


@dataclass
class RowInfo:
    global_index: str
    split_name: str
    question: str
    image_path: str
    gt_bbox_xyxy: list[float]


def _safe_value(value: float) -> float:
    if value == -1:
        return 0.0
    return float(value)


def _parse_bbox_to_xyxy(bbox: Any) -> list[float]:
    if isinstance(bbox, str):
        bbox = ast.literal_eval(bbox)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"Unsupported bbox format: {bbox}")

    x1, y1, x2, y2 = [_safe_value(v) for v in bbox]
    # Fallback for datasets that may store xywh.
    if x2 < x1 or y2 < y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    return [x1, y1, x2, y2]


def _sanitize_for_filename(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,;:!@#%+=()-]", "", text)
    text = text.strip().replace(" ", "_")
    if len(text) > max_len:
        text = text[:max_len]
    return text or "na"


def _build_result_key(sample_index: str, question: str, image_path: str) -> str:
    base = Path(image_path).stem
    q_slug = _sanitize_for_filename(question, max_len=80)
    return f"idx{sample_index}_{q_slug}_{base}_attn.png"


def _load_tsv(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            return pd.read_csv(path_or_url, sep="\t")
        except Exception:
            file_name = Path(path_or_url).name
            local_candidates = []
            env_root = os.environ.get("LMU_DATA_ROOT", "").strip()
            if env_root:
                local_candidates.append(Path(env_root) / file_name)
            local_candidates.extend(
                [
                    Path("/mnt/storage2/users/ytshen_data/LMUData") / file_name,
                    Path("/home/ytshen/LMUData") / file_name,
                ]
            )
            for candidate in local_candidates:
                if candidate.exists():
                    return pd.read_csv(candidate, sep="\t")
            raise
    return pd.read_csv(path_or_url, sep="\t")


def _pick_question(item: pd.Series) -> str:
    for key in ["question", "instruction", "query", "prompt", "description"]:
        if key in item and not pd.isna(item[key]):
            return str(item[key])
    return "na"


def _default_images_root() -> Path:
    if "LMUData" in os.environ and Path(os.environ["LMUData"]).exists():
        return Path(os.environ["LMUData"]) / "images"
    if Path("/mnt/storage/users/ytshen_data/LMUData").exists():
        return Path("/mnt/storage/users/ytshen_data/LMUData/images")
    return Path.home() / "LMUData" / "images"


def _resolve_image_path(images_root: Path, split_name: str, image_path: str) -> Path | None:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    c1 = images_root / split_name / image_path
    if c1.exists():
        return c1

    c2 = images_root / image_path
    if c2.exists():
        return c2

    # Fallback: basename search (slow, only when needed).
    for c in (images_root / split_name, images_root):
        if c.exists():
            matches = list(c.rglob(Path(image_path).name))
            if matches:
                return matches[0]

    return None


def load_rows_from_sources(dataset_sources: dict[str, str]) -> list[RowInfo]:
    rows: list[RowInfo] = []
    for split_name, path_or_url in dataset_sources.items():
        df = _load_tsv(path_or_url)
        for split_index, (_, item) in enumerate(df.iterrows(), start=1):
            rows.append(
                RowInfo(
                    global_index=str(split_index),
                    split_name=split_name,
                    question=_pick_question(item),
                    image_path=str(item.get("image_path", "")),
                    gt_bbox_xyxy=_parse_bbox_to_xyxy(item["bbox"]),
                )
            )
    return rows


def _fit_window(center: float, side_len: float, min_v: float, max_v: float) -> tuple[float, float]:
    side_len = max(1.0, min(side_len, max_v - min_v))
    start = center - side_len / 2
    end = center + side_len / 2

    if start < min_v:
        end += min_v - start
        start = min_v
    if end > max_v:
        start -= end - max_v
        end = max_v

    start = max(min_v, start)
    end = min(max_v, end)
    return start, end


def _build_center_roi(gt_bbox_xyxy: list[float], img_w: int, img_h: int, area_ratio: float) -> list[int]:
    x1, y1, x2, y2 = gt_bbox_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    side_scale = math.sqrt(area_ratio)
    roi_w = img_w * side_scale
    roi_h = img_h * side_scale

    rx1, rx2 = _fit_window(center_x, roi_w, 0.0, float(img_w))
    ry1, ry2 = _fit_window(center_y, roi_h, 0.0, float(img_h))

    ix1, iy1 = int(round(rx1)), int(round(ry1))
    ix2, iy2 = int(round(rx2)), int(round(ry2))

    if ix2 <= ix1:
        ix2 = min(img_w, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(img_h, iy1 + 1)

    return [ix1/2, iy1/2, ix2/2, iy2/2]


def main() -> None:
    if not (0 < AREA_RATIO <= 1.0):
        raise ValueError("AREA_RATIO must be in (0, 1].")

    images_root = Path(IMAGES_ROOT) if IMAGES_ROOT else _default_images_root()
    viz_output_dir = Path(VIZ_OUTPUT_DIR)
    output_json = Path(OUTPUT_JSON)
    if SAVE_VIS_IMAGES:
        viz_output_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows_from_sources(DEFAULT_DATASET_SOURCES)

    result: dict[str, list[list[int]]] = {}
    missing_images = 0

    for row in rows:
        image_full_path = _resolve_image_path(images_root, row.split_name, row.image_path)
        if image_full_path is None:
            missing_images += 1
            continue

        with Image.open(image_full_path).convert("RGB") as image:
            img_w, img_h = image.size
            roi_box = _build_center_roi(row.gt_bbox_xyxy, img_w, img_h, AREA_RATIO)

            key = _build_result_key(row.global_index, row.question, row.image_path)
            result[key] = [roi_box]

            if SAVE_VIS_IMAGES:
                draw = ImageDraw.Draw(image)
                line_w = max(2, min(img_w, img_h) // 300)
                draw.rectangle(roi_box, outline=(255, 0, 0), width=line_w)

                split_out = viz_output_dir / row.split_name
                split_out.mkdir(parents=True, exist_ok=True)
                image.save(split_out / key)

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"[GT-ROI] total_rows={len(rows)} saved_entries={len(result)} missing_images={missing_images}")
    print(f"[GT-ROI] roi_json={output_json}")
    if SAVE_VIS_IMAGES:
        print(f"[GT-ROI] viz_dir={viz_output_dir}")


if __name__ == "__main__":
    main()
