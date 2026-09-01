#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any


DEFAULT_METRICS = (
    "confidence",
    "uncertainty",
    "normalized_entropy",
    "top1_mass",
    "top5_mass",
    "top10_mass",
    "top20_mass",
    "gini",
    "max_mean_ratio",
    "mean",
    "std",
)


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = max(0.0, min(1.0, q)) * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1.0 - frac) + sorted_xs[hi] * frac


def _histogram(xs: list[float], bins: int) -> dict[str, Any]:
    if not xs:
        return {"bins": [], "counts": []}
    lo = min(xs)
    hi = max(xs)
    if hi <= lo:
        return {"bins": [[lo, hi]], "counts": [len(xs)]}
    width = (hi - lo) / bins
    counts = [0 for _ in range(bins)]
    for x in xs:
        idx = min(bins - 1, max(0, int((x - lo) / width)))
        counts[idx] += 1
    edges = [[lo + i * width, lo + (i + 1) * width] for i in range(bins)]
    return {"bins": edges, "counts": counts}


def _summary(layer: int | str, metric: str, xs: list[float], bins: int) -> dict[str, Any]:
    xs_sorted = sorted(xs)
    mean_v = _mean(xs)
    std_v = _std(xs)
    q25 = _quantile(xs_sorted, 0.25)
    q75 = _quantile(xs_sorted, 0.75)
    min_v = xs_sorted[0] if xs_sorted else float("nan")
    max_v = xs_sorted[-1] if xs_sorted else float("nan")
    return {
        "layer_idx": layer,
        "metric": metric,
        "n": len(xs),
        "mean": mean_v,
        "std": std_v,
        "min": min_v,
        "p01": _quantile(xs_sorted, 0.01),
        "p05": _quantile(xs_sorted, 0.05),
        "p10": _quantile(xs_sorted, 0.10),
        "p25": q25,
        "p50": _quantile(xs_sorted, 0.50),
        "p75": q75,
        "p90": _quantile(xs_sorted, 0.90),
        "p95": _quantile(xs_sorted, 0.95),
        "p99": _quantile(xs_sorted, 0.99),
        "max": max_v,
        "range": max_v - min_v if xs else float("nan"),
        "iqr": q75 - q25 if xs else float("nan"),
        "cv": std_v / abs(mean_v) if xs and abs(mean_v) > 1e-12 else None,
        "histogram": _histogram(xs, bins=bins),
    }


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        "layer_idx",
        "metric",
        "n",
        "mean",
        "std",
        "min",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
        "range",
        "iqr",
        "cv",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--forward-idx", type=int, default=0)
    args = parser.parse_args()

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    rows = _read_jsonl(args.jsonl)
    values_by_layer_metric: dict[tuple[int, str], list[float]] = defaultdict(list)
    values_by_metric_all: dict[str, list[float]] = defaultdict(list)
    samples_by_layer: dict[int, set[str]] = defaultdict(set)

    used_rows = 0
    for row in rows:
        if int(row.get("forward_idx", 0) or 0) != int(args.forward_idx):
            continue
        try:
            layer = int(row.get("layer_idx"))
        except Exception:
            continue
        sample_index = str(row.get("sample_index", "") or "")
        if sample_index:
            samples_by_layer[layer].add(sample_index)
        used_rows += 1
        for metric in metrics:
            value = _safe_float(row.get(metric))
            if value is None:
                continue
            values_by_layer_metric[(layer, metric)].append(value)
            values_by_metric_all[metric].append(value)

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for layer, metric in sorted(values_by_layer_metric):
        summaries.append(_summary(layer, metric, values_by_layer_metric[(layer, metric)], bins=max(1, args.bins)))
    for metric in metrics:
        if values_by_metric_all.get(metric):
            summaries.append(_summary("all_layers", metric, values_by_metric_all[metric], bins=max(1, args.bins)))

    payload = {
        "jsonl": args.jsonl,
        "forward_idx": int(args.forward_idx),
        "total_input_rows": len(rows),
        "used_rows": used_rows,
        "layers": {
            str(layer): {
                "num_samples": len(samples),
                "sample_min": min(samples) if samples else None,
                "sample_max": max(samples) if samples else None,
            }
            for layer, samples in sorted(samples_by_layer.items())
        },
        "metrics": metrics,
        "summaries": summaries,
    }
    json_path = os.path.join(args.out_dir, "attn_confidence_distribution.json")
    csv_path = os.path.join(args.out_dir, "attn_confidence_distribution_summary.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_csv(csv_path, summaries)

    print(f"[AttnConfidenceDistribution] rows={used_rows} json={json_path}")
    for row in summaries:
        if row["metric"] != "confidence" or row["layer_idx"] == "all_layers":
            continue
        print(
            "[AttnConfidenceDistribution] "
            f"layer={row['layer_idx']} n={row['n']} "
            f"mean={row['mean']:.6f} std={row['std']:.6f} "
            f"p05={row['p05']:.6f} p50={row['p50']:.6f} p95={row['p95']:.6f} "
            f"range={row['range']:.6f} iqr={row['iqr']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
