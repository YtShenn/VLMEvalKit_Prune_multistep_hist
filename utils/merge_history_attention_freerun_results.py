#!/usr/bin/env python3
"""Merge isolated multi-GPU free-running attention worker outputs.

Worker heatmaps intentionally remain under ``workers/worker_XX`` to avoid file
collisions. This script writes one global JSONL, CSV, summary, and aggregate
figures at the parent run directory without loading all visual maps into memory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.history_attention_freerun_diagnostic import (ALL_CATEGORIES, CAUSAL_NOTE,
    bootstrap_episode, json_safe, make_figures, setup_dirs, write_csv)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers-dir", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--num-workers", type=int, default=None,
                   help="Expected worker count; rejects stale/missing worker directories.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    a = args_parser(); root = setup_dirs(a.work_dir); workers = sorted(Path(a.workers_dir).glob("worker_*"))
    if not workers: raise ValueError(f"No worker_* directories in {a.workers_dir}")
    if a.num_workers is not None:
        expected = [Path(a.workers_dir) / f"worker_{i:02d}" for i in range(a.num_workers)]
        missing = [str(x) for x in expected if not x.is_dir()]
        extras = [str(x) for x in workers if x not in expected]
        if missing or extras:
            raise ValueError(f"Worker directory mismatch; missing={missing}, stale_extra={extras}")
        workers = expected
    merged = root / "per_sample.jsonl"
    flat, outcomes = [], Counter(); samples_ok = structured_valid = 0; action_labels = []; overall_labels = []
    worker_meta, worker_summaries = [], []
    with merged.open("w", encoding="utf-8") as dst:
        for worker in workers:
            meta_path, rows_path = worker / "metadata.json", worker / "per_sample.jsonl"
            if not rows_path.exists(): raise FileNotFoundError(rows_path)
            worker_meta.append(json.loads(meta_path.read_text()) if meta_path.exists() else {"worker":worker.name})
            summary_path = worker / "summary.json"
            if not summary_path.exists(): raise FileNotFoundError(summary_path)
            worker_summaries.append(json.loads(summary_path.read_text()))
            for line in rows_path.open(encoding="utf-8"):
                dst.write(line)
                sample = json.loads(line); correctness = sample.get("correctness", {})
                outcomes[correctness.get("status", "skipped")] += 1
                if sample.get("status") != "ok": continue
                samples_ok += 1; structured_valid += sample.get("parsed_answer") is not None
                if correctness.get("action_type_correct") is not None: action_labels.append(correctness["action_type_correct"])
                if correctness.get("overall_success") is not None: overall_labels.append(correctness["overall_success"])
                for cat, payload in sample.get("categories", {}).items():
                    for frame in payload.get("frames", []):
                        flat.append({"sample_id":sample["sample_id"], "episode_id":sample["episode_id"],
                                     "outcome":correctness.get("status", "skipped"), "category":cat,
                                     "query_count":payload["query_count"], "S_hist":payload["mass"]["history"], **frame})
    write_csv(root / "frame_statistics.csv", flat); make_figures(root, flat)
    summaries, warnings = {}, []
    for cat in ALL_CATEGORIES:
        for outcome in ("all", "correct", "incorrect"):
            for step in sorted({x["relative_step"] for x in flat}):
                rows = [x for x in flat if x["category"] == cat and x["relative_step"] == step and (outcome == "all" or x["outcome"] == outcome)]
                if rows:
                    summaries[f"{cat}/{outcome}/t{step}"] = {k:bootstrap_episode(rows, k, seed=a.seed) for k in ("S_hist","F_i","A_i","entropy","effective_tokens","top_1pct_mass")}
        rows = [x for x in flat if x["category"] == cat]
        if rows:
            edge = np.mean([min(x["peak_relative_xy"][0], 1-x["peak_relative_xy"][0], x["peak_relative_xy"][1], 1-x["peak_relative_xy"][1]) < .08 for x in rows])
            common, count = Counter(tuple(x["peak_grid_row_col"]) for x in rows).most_common(1)[0]
            if edge > .5 or count / len(rows) > .4:
                warnings.append(f"{cat}: possible positional artifact (edge_peak_fraction={edge:.3f}, most_common_peak={common}, fraction={count/len(rows):.3f})")
    first = worker_summaries[0]
    summary = {"merge_complete":True, "workers":len(workers), "samples_requested":first.get("samples_requested"),
               "eligible_complete_history_rows":first.get("eligible_complete_history_rows"),
               "samples_selected":sum(x.get("samples_selected", 0) for x in worker_summaries),
               "samples_ok":samples_ok, "outcomes":dict(outcomes),
               "structured_valid":structured_valid, "action_type_accuracy":float(np.mean(action_labels)) if action_labels else None,
               "overall_success_rate":float(np.mean(overall_labels)) if overall_labels else None,
               "per_category_step":summaries, "artifact_warnings":warnings, "interpretation_boundary":CAUSAL_NOTE,
               "heatmap_location":"Per-sample heatmaps remain in workers/worker_XX/{heatmaps_absolute,heatmaps_conditional}.",
               "worker_metadata":worker_meta}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_safe))
    (root / "MERGED_FROM_WORKERS.txt").write_text("Global statistics/figures are here. Per-sample heatmaps remain in workers/worker_XX to avoid concurrent-write collisions.\n")


if __name__ == "__main__": main()
