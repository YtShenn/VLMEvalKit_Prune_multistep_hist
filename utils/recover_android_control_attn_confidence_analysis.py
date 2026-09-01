#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import build_dataset
from vlmeval.inference import _is_structured_record
from vlmeval.smp import dump, load


def _rank_pkl_path(run_dir: str, rank: int, world_size: int, dataset_name: str) -> str:
    return os.path.join(run_dir, f"{rank}{world_size}_{dataset_name}.pkl")


def _find_run_dir(dataset_work_dir: str, model_name: str) -> str:
    model_dir = os.path.join(dataset_work_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model output dir not found: {model_dir}")
    candidates = [
        os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.startswith("T") and os.path.isdir(os.path.join(model_dir, name))
    ]
    if not candidates:
        raise FileNotFoundError(f"no T* run directory under: {model_dir}")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-work-dir", required=True)
    parser.add_argument("--dataset", default="AndroidControl_Curated_High_Task_Improved")
    parser.add_argument("--model", default="Qwen3-VL-4B-Instruct-AttnPrune")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--confidence-jsonl", default="")
    parser.add_argument("--analysis-out-dir", default="")
    parser.add_argument("--metrics", default="confidence,uncertainty,top10_mass,top20_mass,gini,max_mean_ratio")
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.strip() or _find_run_dir(args.dataset_work_dir, args.model)
    data_all = {}
    for rank in range(args.world_size):
        path = _rank_pkl_path(run_dir, rank, args.world_size, args.dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing rank pkl: {path}")
        data_all.update(load(path))

    dataset = build_dataset(args.dataset)
    data = dataset.data.copy()
    missing = [x for x in data["index"] if x not in data_all]
    if missing:
        raise RuntimeError(f"missing {len(missing)} predictions, first missing index: {missing[0]}")

    if all(_is_structured_record(data_all[x]) for x in data["index"]):
        data["prediction"] = [data_all[x]["prediction"] for x in data["index"]]
        data["extra_records"] = [data_all[x]["extra_records"] for x in data["index"]]
    else:
        data["prediction"] = [str(data_all[x]) for x in data["index"]]
    if "image" in data:
        data.pop("image")

    result_file = os.path.join(run_dir, f"{args.model}_{args.dataset}.xlsx")
    dump(data, result_file)
    print(f"[Recover] wrote result_file={result_file}")

    eval_results = dataset.evaluate(result_file)
    print(f"[Recover] eval_results={eval_results}")
    detail_json = str(eval_results.get("Detail_File", "")) if isinstance(eval_results, dict) else ""
    if not detail_json or not os.path.exists(detail_json):
        raise FileNotFoundError(f"detail json not found from eval results: {detail_json}")

    confidence_jsonl = args.confidence_jsonl.strip() or os.path.join(
        args.dataset_work_dir,
        "attn_confidence",
        "attn_confidence_records.jsonl",
    )
    analysis_out_dir = args.analysis_out_dir.strip() or os.path.join(
        args.dataset_work_dir,
        "attn_confidence_analysis",
    )
    cmd = [
        sys.executable,
        "utils/analyze_attn_confidence_success.py",
        "--confidence-jsonl",
        confidence_jsonl,
        "--detail-json",
        detail_json,
        "--work-dir",
        args.dataset_work_dir,
        "--out-dir",
        analysis_out_dir,
        "--metrics",
        args.metrics,
        "--bins",
        str(args.bins),
    ]
    print("[Recover] running " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
