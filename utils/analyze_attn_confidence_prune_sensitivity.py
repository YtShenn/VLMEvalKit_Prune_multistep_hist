#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_index(row: dict[str, Any]) -> str:
    for key in ("sample_index", "index", "idx"):
        if key in row and row.get(key) is not None:
            return str(row.get(key))
    return ""


def _truthy(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _correct_flag(row: dict[str, Any]) -> bool | None:
    for key in ("type_bbox_flag", "correct", "is_correct", "both_flag"):
        if key in row:
            flag = _truthy(row.get(key))
            if flag is not None:
                return flag
    return None


def _load_correctness(path: str) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, dict):
        rows = []
        for key, value in data.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("index", key)
                rows.append(row)
    else:
        rows = list(data)
    out = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = _sample_index(row)
        flag = _correct_flag(row)
        if idx and flag is not None:
            out[idx] = dict(row, success=bool(flag))
    return out


def _confidence_by_sample_layer(path: str, forward_idx: int, prefer: str) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for row in _load_jsonl(path):
        if int(row.get("forward_idx", 0) or 0) != int(forward_idx):
            continue
        idx = _sample_index(row)
        if not idx:
            continue
        try:
            layer = int(row.get("layer_idx"))
        except Exception:
            continue
        key = (idx, layer)
        if key not in out or prefer == "last":
            out[key] = row
    return out


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _auc(xs: list[float], labels: list[int]) -> float | None:
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0 or neg == 0:
        return None
    ranks = _ranks(xs)
    rank_sum_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _quantile_bins(rows: list[dict[str, Any]], metric: str, label_key: str, bins: int) -> list[dict[str, Any]]:
    pairs = []
    for row in rows:
        value = _safe_float(row.get(metric))
        label = row.get(label_key)
        if value is None or label is None:
            continue
        pairs.append((value, int(label), row))
    if not pairs:
        return []
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    out = []
    for b in range(bins):
        lo = int(round(b * n / bins))
        hi = int(round((b + 1) * n / bins))
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        vals = [x for x, _, _ in chunk]
        labels = [y for _, y, _ in chunk]
        baseline_correct = [int(r["baseline_success"]) for _, _, r in chunk]
        pruned_correct = [int(r["pruned_success"]) for _, _, r in chunk]
        baseline_correct_rows = [r for _, _, r in chunk if int(r["baseline_success"]) == 1]
        harm_labels = [int(r["prune_harmed"]) for r in baseline_correct_rows]
        out.append(
            {
                "bin": b + 1,
                "count": len(chunk),
                "score_min": min(vals),
                "score_max": max(vals),
                "score_mean": _mean(vals),
                f"{label_key}_rate": _mean(labels),
                "baseline_success_rate": _mean(baseline_correct),
                "pruned_success_rate": _mean(pruned_correct),
                "success_delta_pruned_minus_baseline": _mean(pruned_correct) - _mean(baseline_correct),
                "baseline_correct_count": len(baseline_correct_rows),
                "prune_harm_rate_among_baseline_correct": _mean(harm_labels) if harm_labels else None,
                "prune_harm_count": sum(harm_labels) if harm_labels else 0,
            }
        )
    return out


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-confidence-jsonl", required=True)
    parser.add_argument("--baseline-detail-json", required=True)
    parser.add_argument("--pruned-confidence-jsonl", required=True)
    parser.add_argument("--pruned-detail-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics", default="confidence,uncertainty,top10_mass,top20_mass,gini,max_mean_ratio")
    parser.add_argument("--forward-idx", type=int, default=0)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--score-source", choices=["baseline", "pruned"], default="baseline")
    parser.add_argument("--duplicate-policy", choices=["first", "last"], default="first")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    baseline_correct = _load_correctness(args.baseline_detail_json)
    pruned_correct = _load_correctness(args.pruned_detail_json)
    baseline_conf = _confidence_by_sample_layer(args.baseline_confidence_jsonl, args.forward_idx, args.duplicate_policy)
    pruned_conf = _confidence_by_sample_layer(args.pruned_confidence_jsonl, args.forward_idx, args.duplicate_policy)
    score_conf = baseline_conf if args.score_source == "baseline" else pruned_conf

    common_samples = sorted(set(baseline_correct) & set(pruned_correct), key=lambda x: int(x) if x.isdigit() else x)
    layers = sorted({layer for idx, layer in score_conf if idx in common_samples})
    joined_rows = []
    for idx in common_samples:
        b_success = bool(baseline_correct[idx]["success"])
        p_success = bool(pruned_correct[idx]["success"])
        for layer in layers:
            conf = score_conf.get((idx, layer))
            if not conf:
                continue
            row = dict(conf)
            row.update(
                {
                    "sample_index": idx,
                    "layer_idx": layer,
                    "baseline_success": int(b_success),
                    "pruned_success": int(p_success),
                    "prune_harmed": int(b_success and not p_success),
                    "prune_preserved": int(b_success and p_success),
                    "prune_helped": int((not b_success) and p_success),
                    "both_failed": int((not b_success) and (not p_success)),
                    "score_source": args.score_source,
                }
            )
            joined_rows.append(row)

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    summaries = []
    bins_payload = {}
    for layer in layers:
        layer_rows = [r for r in joined_rows if int(r["layer_idx"]) == int(layer)]
        base_correct_rows = [r for r in layer_rows if int(r["baseline_success"]) == 1]
        for metric in metrics:
            for label_key, subset in (
                ("prune_harmed", base_correct_rows),
                ("pruned_success", layer_rows),
                ("success_drop", layer_rows),
            ):
                vals, labels = [], []
                for row in subset:
                    value = _safe_float(row.get(metric))
                    if value is None:
                        continue
                    if label_key == "success_drop":
                        label = int(row["baseline_success"]) - int(row["pruned_success"])
                        label = 1 if label > 0 else 0
                    else:
                        label = int(row[label_key])
                    vals.append(value)
                    labels.append(label)
                if not vals:
                    continue
                pos_vals = [x for x, y in zip(vals, labels) if y == 1]
                neg_vals = [x for x, y in zip(vals, labels) if y == 0]
                auc_high = _auc(vals, labels)
                summary = {
                    "layer_idx": layer,
                    "metric": metric,
                    "label": label_key,
                    "score_source": args.score_source,
                    "n": len(vals),
                    "positive_count": sum(labels),
                    "negative_count": len(labels) - sum(labels),
                    "positive_rate": _mean(labels),
                    "mean_all": _mean(vals),
                    "std_all": _std(vals),
                    "mean_positive": _mean(pos_vals),
                    "mean_negative": _mean(neg_vals),
                    "mean_diff_positive_minus_negative": (
                        _mean(pos_vals) - _mean(neg_vals) if pos_vals and neg_vals else None
                    ),
                    "pearson_with_label": _pearson(vals, labels),
                    "spearman_with_label": _spearman(vals, labels),
                    "auc_higher_score_predicts_positive": auc_high,
                    "auc_lower_score_predicts_positive": (1.0 - auc_high if auc_high is not None else None),
                }
                summaries.append(summary)
                if label_key == "prune_harmed":
                    bins_payload[f"layer{layer}_{metric}_{label_key}"] = _quantile_bins(
                        base_correct_rows,
                        metric=metric,
                        label_key="prune_harmed",
                        bins=max(1, args.bins),
                    )

    transition_counts = {
        "baseline_correct_pruned_correct": sum(
            1 for idx in common_samples if baseline_correct[idx]["success"] and pruned_correct[idx]["success"]
        ),
        "baseline_correct_pruned_wrong": sum(
            1 for idx in common_samples if baseline_correct[idx]["success"] and not pruned_correct[idx]["success"]
        ),
        "baseline_wrong_pruned_correct": sum(
            1 for idx in common_samples if not baseline_correct[idx]["success"] and pruned_correct[idx]["success"]
        ),
        "baseline_wrong_pruned_wrong": sum(
            1 for idx in common_samples if not baseline_correct[idx]["success"] and not pruned_correct[idx]["success"]
        ),
    }
    transition_counts["common_samples"] = len(common_samples)
    transition_counts["baseline_success_rate"] = _mean([int(baseline_correct[idx]["success"]) for idx in common_samples])
    transition_counts["pruned_success_rate"] = _mean([int(pruned_correct[idx]["success"]) for idx in common_samples])
    transition_counts["prune_harm_rate_among_baseline_correct"] = (
        transition_counts["baseline_correct_pruned_wrong"]
        / max(1, transition_counts["baseline_correct_pruned_correct"] + transition_counts["baseline_correct_pruned_wrong"])
    )

    analysis = {
        "baseline_confidence_jsonl": args.baseline_confidence_jsonl,
        "baseline_detail_json": args.baseline_detail_json,
        "pruned_confidence_jsonl": args.pruned_confidence_jsonl,
        "pruned_detail_json": args.pruned_detail_json,
        "score_source": args.score_source,
        "forward_idx": int(args.forward_idx),
        "metrics": metrics,
        "transition_counts": transition_counts,
        "num_joined_rows": len(joined_rows),
        "layers": layers,
        "summaries": summaries,
    }
    with open(os.path.join(args.out_dir, "attn_confidence_prune_sensitivity_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, "attn_confidence_prune_sensitivity_bins.json"), "w", encoding="utf-8") as f:
        json.dump(bins_payload, f, ensure_ascii=False, indent=2)
    _write_csv(os.path.join(args.out_dir, "attn_confidence_prune_sensitivity_summary.csv"), summaries)
    _write_csv(os.path.join(args.out_dir, "attn_confidence_prune_sensitivity_joined.csv"), joined_rows)

    print("[PruneSensitivity] " + json.dumps(transition_counts, ensure_ascii=False))
    for row in summaries:
        if row["label"] == "prune_harmed" and row["metric"] in {"confidence", "uncertainty", "top10_mass", "gini"}:
            print(
                "[PruneSensitivity] "
                f"layer={row['layer_idx']} metric={row['metric']} n={row['n']} "
                f"harm_rate={row['positive_rate']:.6f} "
                f"mean_harmed={row['mean_positive']} mean_not_harmed={row['mean_negative']} "
                f"pearson={row['pearson_with_label']} spearman={row['spearman_with_label']} "
                f"auc_high={row['auc_higher_score_predicts_positive']} "
                f"auc_low={row['auc_lower_score_predicts_positive']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
