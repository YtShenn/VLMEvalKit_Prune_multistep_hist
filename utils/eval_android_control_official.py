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
from vlmeval.dataset.AndroidControl_Curated.official_eval import (
    OFFICIAL_GROUPS,
    aggregate_android_control_group_metrics,
    evaluate_android_control_eval_file_official,
    infer_android_control_dataset_name,
)


def _load_metrics_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AndroidControl-Curated outputs with official-compatible metrics and group summaries."
    )
    parser.add_argument("--xlsx", nargs="*", default=[], help="One or more AndroidControl prediction xlsx files.")
    parser.add_argument("--metrics-json", nargs="*", default=[], help="Optional existing official metrics json files to include in group aggregation.")
    parser.add_argument("--dataset", nargs="*", default=[], help="Dataset names matching --xlsx order. If omitted, infer from each xlsx path.")
    parser.add_argument("--save-dir", default=None, help="Directory to save summary files. Default: alongside the first input file.")
    args = parser.parse_args()

    if not args.xlsx and not args.metrics_json:
        raise ValueError("Please provide at least one --xlsx or --metrics-json input.")
    if args.dataset and len(args.dataset) != len(args.xlsx):
        raise ValueError("--dataset must have the same number of items as --xlsx.")

    metrics_by_dataset = {}
    per_file_outputs = {}

    for idx, xlsx_path in enumerate(args.xlsx):
        dataset_name = args.dataset[idx] if idx < len(args.dataset) else infer_android_control_dataset_name(xlsx_path)
        ds = AndroidControlCurated(dataset_name, skeleton=True)
        metrics = evaluate_android_control_eval_file_official(
            xlsx_path,
            dataset_name=dataset_name,
            image_resolver=ds._resolve_image_path,
        )
        metrics["Dataset"] = dataset_name
        metrics["Source_XLSX"] = xlsx_path
        metrics["Official_Group"] = next(
            (group_name for group_name, members in OFFICIAL_GROUPS.items() if dataset_name in members),
            None,
        )
        metrics_by_dataset[dataset_name] = metrics
        out_path = xlsx_path.replace(".xlsx", "_official_metrics.json")
        _save_json(out_path, metrics)
        per_file_outputs[dataset_name] = out_path

    for metrics_json in args.metrics_json:
        metrics = _load_metrics_json(metrics_json)
        dataset_name = metrics.get("Dataset", None) or infer_android_control_dataset_name(metrics_json)
        metrics_by_dataset[dataset_name] = metrics

    input_roots = args.xlsx or args.metrics_json
    base_dir = args.save_dir or str(Path(input_roots[0]).resolve().parent)
    os.makedirs(base_dir, exist_ok=True)

    group_summary = {}
    for group_name in ["AndroidControl-Curated-Easy", "AndroidControl-Curated-Hard"]:
        members = OFFICIAL_GROUPS[group_name]
        if all(member in metrics_by_dataset for member in members):
            group_summary[group_name] = aggregate_android_control_group_metrics(metrics_by_dataset, group_name)

    box_hard_group = "AndroidControl-Curated-Box-Hard"
    if all(member in metrics_by_dataset for member in OFFICIAL_GROUPS[box_hard_group]):
        group_summary[box_hard_group] = aggregate_android_control_group_metrics(metrics_by_dataset, box_hard_group)

    summary_payload = {
        "per_dataset": metrics_by_dataset,
        "groups": group_summary,
        "per_file_outputs": per_file_outputs,
    }
    summary_path = os.path.join(base_dir, "android_control_curated_official_summary.json")
    _save_json(summary_path, summary_payload)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"\nSaved summary to: {summary_path}")


if __name__ == "__main__":
    main()
