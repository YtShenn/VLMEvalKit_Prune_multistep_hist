#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlmeval.dataset import AndroidControlCurated


def infer_dataset_name_from_file(path: str) -> str:
    full_path = str(path)
    name = os.path.basename(full_path)
    candidates = [
        "AndroidControl_Curated_Low_Point",
        "AndroidControl_Curated_High_Point",
        "AndroidControl_Curated_Low_BBox",
        "AndroidControl_Curated_High_BBox",
        "AndroidControl_Curated_High_Task_Improved",
    ]
    for c in candidates:
        if c in name or c in full_path:
            return c
    raise ValueError(
        "Cannot infer AndroidControl dataset name from the xlsx path. "
        "This helper only supports AndroidControl xlsx files. "
        "Please pass --dataset explicitly, for example "
        "--dataset AndroidControl_Curated_High_BBox."
    )


def _task_key(item: dict) -> str:
    for k in ["task_filename", "task_id", "episode", "revised_task", "instruction"]:
        v = item.get(k, None)
        if v is not None:
            s = str(v).strip()
            if s != "":
                return f"{k}:{s}"
    return f"index:{item.get('index', '')}"


def _ensure_task_sr(metrics: dict, xlsx_path: str) -> dict:
    if "Task_Success_Rate" in metrics:
        return metrics

    detail_file = metrics.get("Detail_File", None)
    if not detail_file:
        detail_file = str(xlsx_path).replace(".xlsx", "_android_control_detail.json")
    if not os.path.exists(detail_file):
        return metrics

    try:
        with open(detail_file, "r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        return metrics
    if not isinstance(rows, list):
        return metrics

    task_stat = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        tk = _task_key(row)
        if tk not in task_stat:
            task_stat[tk] = {"all_correct": True}
        task_stat[tk]["all_correct"] = task_stat[tk]["all_correct"] and bool(row.get("type_bbox_flag", False))

    total_tasks = len(task_stat)
    success_tasks = sum(1 for v in task_stat.values() if v["all_correct"])
    metrics["Task_Total"] = total_tasks
    metrics["Task_Success_Count"] = success_tasks
    metrics["Task_Success_Rate"] = round(success_tasks / max(total_tasks, 1) * 100, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute AndroidControl metrics from an existing xlsx prediction file."
    )
    parser.add_argument("--xlsx", help="Path to AndroidControl prediction xlsx file.", \
                        default="OUTPUT/outputs_qwen3vl_android_control_curated_4B_template_prefill_timing_history/AndroidControl_Curated_High_Task_Improved/Qwen3-VL-4B-Instruct/T20260729_GHEAD/Qwen3-VL-4B-Instruct_AndroidControl_Curated_High_Task_Improved.xlsx")
    parser.add_argument(
        "--dataset",
        default=None,
        help="AndroidControl dataset name. If omitted, infer from xlsx filename.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        help="Optional path to save metrics json. Default: <xlsx>_metrics.json",
    )
    args = parser.parse_args()

    dataset_name = args.dataset or infer_dataset_name_from_file(args.xlsx)
    ds = AndroidControlCurated(dataset_name)
    metrics = ds.evaluate(args.xlsx)
    metrics = _ensure_task_sr(metrics, args.xlsx)

    save_path = args.save_json or args.xlsx.replace(".xlsx", "_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved metrics to: {save_path}")


if __name__ == "__main__":
    main()
