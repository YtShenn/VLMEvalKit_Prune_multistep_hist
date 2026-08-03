import argparse
import ast
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


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
    category: str
    ui_type: str
    screenshot_token: str | None
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
    if x2 < x1 or y2 < y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    return [x1, y1, x2, y2]


def _boxes_intersect(box1_xyxy: list[float], box2_xyxy: list[float]) -> bool:
    x1a, y1a, x2a, y2a = box1_xyxy
    x1b, y1b, x2b, y2b = box2_xyxy
    return not (x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a)


def _extract_screenshot_token(text: str | None) -> str | None:
    if text is None:
        return None
    text = str(text)
    match = re.search(r"(screenshot_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", text)
    if match:
        return match.group(1)
    
    match_from = re.search(r"(Screenshot from \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})", text)
    if match_from:
        return match_from.group(1)

    base = Path(text).name
    stem = Path(base).stem
    if stem.endswith("_attn"):
        stem = stem[:-5]
    return stem if stem else None


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


def load_rows_from_sources(dataset_sources: dict[str, str]) -> list[RowInfo]:
    rows: list[RowInfo] = []
    global_counter = 0
    for split_name, path_or_url in dataset_sources.items():
        df = _load_tsv(path_or_url)
        for _, item in df.iterrows():
            global_counter += 1
            bbox_xyxy = _parse_bbox_to_xyxy(item["bbox"])

            category = str(item.get("category", "Unknown"))
            ui_type = str(item.get("ui_type", "Unknown"))
            screenshot_token = _extract_screenshot_token(item.get("image_path", ""))

            rows.append(
                RowInfo(
                    global_index=str(global_counter),
                    split_name=split_name,
                    category=category,
                    ui_type=ui_type,
                    screenshot_token=screenshot_token,
                    gt_bbox_xyxy=bbox_xyxy,
                )
            )
    return rows


def load_rows_from_single_tsv(tsv_path: str, split_name: str = "ScreenSpot_Pro") -> list[RowInfo]:
    df = _load_tsv(tsv_path)
    rows: list[RowInfo] = []
    for index, item in enumerate(df.itertuples(index=False), start=1):
        bbox_xyxy = _parse_bbox_to_xyxy(getattr(item, "bbox"))
        category = str(getattr(item, "category", "Unknown"))
        ui_type = str(getattr(item, "ui_type", "Unknown"))
        screenshot_token = _extract_screenshot_token(getattr(item, "image_path", ""))
        rows.append(
            RowInfo(
                global_index=str(index),
                split_name=split_name,
                category=category,
                ui_type=ui_type,
                screenshot_token=screenshot_token,
                gt_bbox_xyxy=bbox_xyxy,
            )
        )
    return rows


def compute_recall(rows: list[RowInfo], roi_map: dict[str, Any]) -> dict[str, Any]:
    roi_by_token: dict[str, list[list[float]]] = defaultdict(list)
    roi_key_count_by_token: dict[str, int] = defaultdict(int)
    invalid_roi_keys = 0
    for key, candidates in roi_map.items():
        token = _extract_screenshot_token(key)
        # print("Processing ROI key: ", key, " extracted token: ", token)
        if token is None:
            invalid_roi_keys += 1
            continue
        roi_key_count_by_token[token] += 1
        if isinstance(candidates, list):
            # print("has candidates")
            for box in candidates:
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    roi_by_token[token].append([float(box[0])*4, float(box[1])*4, float(box[2])*4, float(box[3])*4])

    total = 0
    hit_total = 0
    by_split = defaultdict(lambda: {"total": 0, "hit": 0})
    by_category = defaultdict(lambda: {"total": 0, "hit": 0})
    by_ui_type = defaultdict(lambda: {"total": 0, "hit": 0})

    missing_roi_rows = []
    empty_candidates_rows = []
    missing_screenshot_token_rows = []

    for row in rows:
        total += 1
        split_stats = by_split[row.split_name]
        category_stats = by_category[row.category]
        ui_type_stats = by_ui_type[row.ui_type]
        split_stats["total"] += 1
        category_stats["total"] += 1
        ui_type_stats["total"] += 1

        token = row.screenshot_token
        # print("token: ", token)
        if not token:
            missing_screenshot_token_rows.append(row.global_index)
            continue

        candidates = roi_by_token.get(token)
        # print("candidates: ", candidates)
        if candidates is None:
            missing_roi_rows.append(row.global_index)
            continue
        if len(candidates) == 0:
            empty_candidates_rows.append(row.global_index)
            continue
        
        # print("gt_bbox_xyxy: ", row.gt_bbox_xyxy)
        hit = any(_boxes_intersect(row.gt_bbox_xyxy, box) for box in candidates)
        # print("hit: ", hit)
        if hit:
            hit_total += 1
            split_stats["hit"] += 1
            category_stats["hit"] += 1
            ui_type_stats["hit"] += 1

    def finalize(stats_dict: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
        output = {}
        for key, stats in sorted(stats_dict.items(), key=lambda kv: kv[0]):
            t = stats["total"]
            h = stats["hit"]
            output[key] = {
                "total": t,
                "hit": h,
                "recall": (h / t) if t > 0 else math.nan,
            }
        return output

    return {
        "overall": {
            "total": total,
            "hit": hit_total,
            "recall": (hit_total / total) if total > 0 else math.nan,
        },
        "by_split": finalize(by_split),
        "by_category": finalize(by_category),
        "by_ui_type": finalize(by_ui_type),
        "diagnostics": {
            "roi_entries": len(roi_map),
            "matched_screenshot_tokens": len(roi_by_token),
            "duplicate_roi_tokens_count": sum(1 for _, c in roi_key_count_by_token.items() if c > 1),
            "invalid_roi_keys_count": invalid_roi_keys,
            "missing_screenshot_token_rows_count": len(missing_screenshot_token_rows),
            "missing_roi_rows_count": len(missing_roi_rows),
            "empty_candidates_rows_count": len(empty_candidates_rows),
            "missing_screenshot_token_rows_preview": missing_screenshot_token_rows[:20],
            "missing_roi_rows_preview": missing_roi_rows[:20],
            "empty_candidates_rows_preview": empty_candidates_rows[:20],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROI candidate recall against ScreenSpot_Pro GT.")
    parser.add_argument(
        "--roi-json",
        type=str,
        default="qwen3vl_1-15_layer_0.25scale_num=2/roi_results.json",
        help="Path to roi_results.json",
    )
    parser.add_argument(
        "--single-tsv",
        type=str,
        default=None,
        help="Use one TSV file instead of the default 6 ScreenSpot_Pro splits.",
    )
    parser.add_argument(
        "--single-split-name",
        type=str,
        default="ScreenSpot_Pro",
        help="Split name when --single-tsv is used.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save detailed recall report as json.",
    )
    args = parser.parse_args()

    roi_path = Path(args.roi_json)
    with roi_path.open("r", encoding="utf-8") as file:
        roi_map = json.load(file)

    if args.single_tsv:
        rows = load_rows_from_single_tsv(args.single_tsv, args.single_split_name)
    else:
        rows = load_rows_from_sources(DEFAULT_DATASET_SOURCES)

    report = compute_recall(rows, roi_map)

    overall = report["overall"]
    print(f"[ROIRecall] total={overall['total']} hit={overall['hit']} recall={overall['recall']:.6f}")

    print("[ROIRecall] By split:")
    for split_name, split_stats in report["by_split"].items():
        print(f"  - {split_name}: total={split_stats['total']} hit={split_stats['hit']} recall={split_stats['recall']:.6f}")

    print("[ROIRecall] By category:")
    for category, category_stats in report["by_category"].items():
        print(f"  - {category}: total={category_stats['total']} hit={category_stats['hit']} recall={category_stats['recall']:.6f}")

    print("[ROIRecall] By ui_type:")
    for ui_type, ui_stats in report["by_ui_type"].items():
        print(f"  - {ui_type}: total={ui_stats['total']} hit={ui_stats['hit']} recall={ui_stats['recall']:.6f}")

    diagnostics = report["diagnostics"]
    print(
        "[ROIRecall] Diagnostics: "
        f"roi_entries={diagnostics['roi_entries']} "
        f"matched_screenshot_tokens={diagnostics['matched_screenshot_tokens']} "
        f"duplicate_roi_tokens_count={diagnostics['duplicate_roi_tokens_count']} "
        f"missing_roi_rows_count={diagnostics['missing_roi_rows_count']} "
        f"empty_candidates_rows_count={diagnostics['empty_candidates_rows_count']}"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"[ROIRecall] Saved report: {output_path}")


if __name__ == "__main__":
    main()
