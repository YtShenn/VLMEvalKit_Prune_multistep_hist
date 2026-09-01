#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any


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
            if line:
                rows.append(json.loads(line))
    return rows


def _values_by_layer(rows: list[dict[str, Any]], metric: str, forward_idx: int) -> dict[int, list[float]]:
    out: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row.get("forward_idx", 0) or 0) != int(forward_idx):
            continue
        try:
            layer = int(row.get("layer_idx"))
        except Exception:
            continue
        value = _safe_float(row.get(metric))
        if value is not None:
            out[layer].append(value)
    return dict(sorted(out.items()))


def _quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def _plot_confidence_hist(rows: list[dict[str, Any]], out_dir: str, forward_idx: int) -> None:
    import matplotlib.pyplot as plt

    by_layer = _values_by_layer(rows, "confidence", forward_idx)
    plt.figure(figsize=(9, 5.5))
    for layer, values in by_layer.items():
        plt.hist(values, bins=45, alpha=0.42, density=True, label=f"layer {layer}")
    plt.xlabel("confidence = 1 - normalized entropy")
    plt.ylabel("density")
    plt.title("Attention Confidence Distribution by Early Layer")
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_hist_by_layer.png"), dpi=220)
    plt.close()


def _plot_confidence_box(rows: list[dict[str, Any]], out_dir: str, forward_idx: int) -> None:
    import matplotlib.pyplot as plt

    by_layer = _values_by_layer(rows, "confidence", forward_idx)
    layers = list(by_layer)
    values = [by_layer[layer] for layer in layers]
    plt.figure(figsize=(8, 5.2))
    box = plt.boxplot(values, labels=[str(x) for x in layers], patch_artist=True, showfliers=False)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.48)
    plt.xlabel("layer index")
    plt.ylabel("confidence")
    plt.title("Confidence Spread Across Samples")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_boxplot_by_layer.png"), dpi=220)
    plt.close()


def _plot_confidence_ecdf(rows: list[dict[str, Any]], out_dir: str, forward_idx: int) -> None:
    import matplotlib.pyplot as plt

    by_layer = _values_by_layer(rows, "confidence", forward_idx)
    plt.figure(figsize=(8.5, 5.2))
    for layer, values in by_layer.items():
        xs = sorted(values)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        plt.plot(xs, ys, linewidth=2.0, label=f"layer {layer}")
    plt.xlabel("confidence")
    plt.ylabel("fraction of samples <= x")
    plt.title("Confidence ECDF by Layer")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confidence_ecdf_by_layer.png"), dpi=220)
    plt.close()


def _plot_quantile_band(rows: list[dict[str, Any]], out_dir: str, forward_idx: int) -> None:
    import matplotlib.pyplot as plt

    metrics = ["confidence", "top10_mass", "top20_mass", "gini"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        by_layer = _values_by_layer(rows, metric, forward_idx)
        layers = list(by_layer)
        p05 = [_quantile(by_layer[layer], 0.05) for layer in layers]
        p25 = [_quantile(by_layer[layer], 0.25) for layer in layers]
        p50 = [_quantile(by_layer[layer], 0.50) for layer in layers]
        p75 = [_quantile(by_layer[layer], 0.75) for layer in layers]
        p95 = [_quantile(by_layer[layer], 0.95) for layer in layers]
        ax.fill_between(layers, p05, p95, alpha=0.18, label="p05-p95")
        ax.fill_between(layers, p25, p75, alpha=0.28, label="p25-p75")
        ax.plot(layers, p50, marker="o", linewidth=2.0, label="median")
        ax.set_title(metric)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.22)
        ax.set_xticks(layers)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Metric Quantile Bands Across Samples", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_quantile_bands.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_box_grid(rows: list[dict[str, Any]], out_dir: str, forward_idx: int) -> None:
    import matplotlib.pyplot as plt

    metrics = ["confidence", "top1_mass", "top5_mass", "top10_mass", "top20_mass", "gini", "max_mean_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for ax, metric in zip(axes, metrics):
        by_layer = _values_by_layer(rows, metric, forward_idx)
        layers = list(by_layer)
        values = [by_layer[layer] for layer in layers]
        box = ax.boxplot(values, labels=[str(x) for x in layers], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        ax.set_title(metric)
        ax.set_xlabel("layer")
        ax.grid(axis="y", alpha=0.22)
    axes[-1].axis("off")
    fig.suptitle("Distribution of Confidence-Related Metrics", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metric_boxplots_by_layer.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--forward-idx", type=int, default=0)
    args = parser.parse_args()

    rows = _read_jsonl(args.jsonl)
    os.makedirs(args.out_dir, exist_ok=True)
    _plot_confidence_hist(rows, args.out_dir, args.forward_idx)
    _plot_confidence_box(rows, args.out_dir, args.forward_idx)
    _plot_confidence_ecdf(rows, args.out_dir, args.forward_idx)
    _plot_quantile_band(rows, args.out_dir, args.forward_idx)
    _plot_metric_box_grid(rows, args.out_dir, args.forward_idx)
    print(f"[AttnConfidencePlot] wrote plots to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
