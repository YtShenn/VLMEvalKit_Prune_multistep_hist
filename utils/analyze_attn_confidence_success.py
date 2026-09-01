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
            if not line:
                continue
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


def _load_correctness(detail_json: str) -> dict[str, dict[str, Any]]:
    data = _load_json(detail_json)
    if isinstance(data, dict):
        rows = list(data.values())
        for key, value in data.items():
            if isinstance(value, dict) and "index" not in value and "sample_index" not in value:
                value["index"] = key
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


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> float | None:
    if not xs:
        return None
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


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


def _quantile_bins(pairs: list[tuple[float, int]], bins: int = 4) -> list[dict[str, Any]]:
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: x[0])
    n = len(pairs)
    out = []
    for b in range(bins):
        lo = int(round(b * n / bins))
        hi = int(round((b + 1) * n / bins))
        chunk = pairs[lo:hi]
        if not chunk:
            continue
        vals = [x for x, _ in chunk]
        labels = [y for _, y in chunk]
        out.append(
            {
                "bin": b + 1,
                "count": len(chunk),
                "score_min": min(vals),
                "score_max": max(vals),
                "score_mean": _mean(vals),
                "success_rate": _mean(labels),
                "success_count": sum(labels),
            }
        )
    return out


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _find_detail_json(work_dir: str) -> str | None:
    candidates = []
    for root, _, files in os.walk(work_dir):
        for name in files:
            if "_android_control_detail" in name and name.endswith(".json"):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


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
    parser.add_argument("--confidence-jsonl", required=True)
    parser.add_argument("--detail-json", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics", default="confidence,uncertainty,top10_mass,top20_mass,gini,max_mean_ratio")
    parser.add_argument("--forward-idx", type=int, default=0)
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()

    detail_json = args.detail_json.strip()
    if not detail_json and args.work_dir:
        detail_json = _find_detail_json(args.work_dir) or ""
    if not detail_json:
        raise FileNotFoundError("detail json not provided and no *_android_control_detail.json found under work-dir")

    records = _load_jsonl(args.confidence_jsonl)
    correctness = _load_correctness(detail_json)
    os.makedirs(args.out_dir, exist_ok=True)

    by_layer: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    joined_rows = []
    for rec in records:
        if int(rec.get("forward_idx", 0) or 0) != int(args.forward_idx):
            continue
        idx = _sample_index(rec)
        if not idx or idx not in correctness:
            continue
        layer = int(rec.get("layer_idx"))
        key = f"{idx}:{layer}"
        if key in by_layer[layer]:
            continue
        label = 1 if correctness[idx]["success"] else 0
        row = dict(rec)
        row["success"] = label
        row["prediction"] = correctness[idx].get("prediction", "")
        by_layer[layer][key] = row
        joined_rows.append(row)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    summaries = []
    bins_payload = {}
    for layer in sorted(by_layer):
        layer_rows = list(by_layer[layer].values())
        labels = [int(r["success"]) for r in layer_rows]
        for metric in metrics:
            vals = []
            ys = []
            for row, label in zip(layer_rows, labels):
                value = _safe_float(row.get(metric))
                if value is None:
                    continue
                vals.append(value)
                ys.append(label)
            if not vals:
                continue
            succ_vals = [x for x, y in zip(vals, ys) if y == 1]
            fail_vals = [x for x, y in zip(vals, ys) if y == 0]
            bin_rows = _quantile_bins(list(zip(vals, ys)), bins=max(1, args.bins))
            bins_payload[f"layer{layer}_{metric}"] = bin_rows
            summaries.append(
                {
                    "layer_idx": layer,
                    "metric": metric,
                    "n": len(vals),
                    "success_count": sum(ys),
                    "failure_count": len(ys) - sum(ys),
                    "success_rate": _mean(ys),
                    "mean_all": _mean(vals),
                    "std_all": _std(vals),
                    "mean_success": _mean(succ_vals),
                    "mean_failure": _mean(fail_vals),
                    "mean_diff_success_minus_failure": (
                        _mean(succ_vals) - _mean(fail_vals) if succ_vals and fail_vals else None
                    ),
                    "pearson_with_success": _pearson(vals, ys),
                    "spearman_with_success": _spearman(vals, ys),
                    "auc_success_higher_score": _auc(vals, ys),
                }
            )

    summary_path = os.path.join(args.out_dir, "attn_confidence_success_analysis.json")
    joined_path = os.path.join(args.out_dir, "attn_confidence_success_joined.csv")
    summary_csv = os.path.join(args.out_dir, "attn_confidence_success_summary.csv")
    bins_path = os.path.join(args.out_dir, "attn_confidence_success_bins.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "confidence_jsonl": args.confidence_jsonl,
                "detail_json": detail_json,
                "forward_idx": int(args.forward_idx),
                "num_joined_rows": len(joined_rows),
                "num_matched_samples": len({str(r.get("sample_index")) for r in joined_rows}),
                "summaries": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(bins_path, "w", encoding="utf-8") as f:
        json.dump(bins_payload, f, ensure_ascii=False, indent=2)
    _write_csv(joined_path, joined_rows)
    _write_csv(summary_csv, summaries)

    print(f"[AttnConfidenceAnalysis] detail_json={detail_json}")
    print(f"[AttnConfidenceAnalysis] joined_rows={len(joined_rows)} summary={summary_path}")
    for row in summaries:
        if row["metric"] != "confidence":
            continue
        print(
            "[AttnConfidenceAnalysis] "
            f"layer={row['layer_idx']} n={row['n']} "
            f"success_rate={row['success_rate']:.4f} "
            f"mean_success={row['mean_success']} mean_failure={row['mean_failure']} "
            f"pearson={row['pearson_with_success']} spearman={row['spearman_with_success']} "
            f"auc={row['auc_success_higher_score']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
