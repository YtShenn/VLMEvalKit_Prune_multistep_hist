#!/usr/bin/env python3
"""Visualize temporal novelty in uncompressed GUI-agent history.

This is an offline observation tool.  It forces full-resolution historical
screenshots only in its own process, extracts Qwen3-VL post-merge visual-token
features, and visualizes ``1 - cosine`` novelty between adjacent history frames.
It never modifies VLMEval inference, prompts, state packets, or model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    from history_attention_diagnostic import (
        device_inputs,
        image_items,
        make_inputs,
        sampled_row_indices,
        complete_history_candidates,
    )
    from history_attention_utils import grid_rows, locate_visual_segments
except ModuleNotFoundError:  # pragma: no cover - module invocation support
    from .history_attention_diagnostic import (
        device_inputs,
        image_items,
        make_inputs,
        sampled_row_indices,
        complete_history_candidates,
    )
    from .history_attention_utils import grid_rows, locate_visual_segments


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='VLMEval dataset name.')
    p.add_argument('--model', required=True, help='Local/HuggingFace Qwen3-VL model path.')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--num-samples', type=int, default=6,
                   help='Small task-balanced pilot by default.')
    p.add_argument('--sample-by', choices=['task', 'sequential'], default='task')
    p.add_argument('--samples-per-task', type=int, default=1)
    p.add_argument('--sample-ids', default=None,
                   help='Comma-separated stable dataset indices; overrides task sampling.')
    p.add_argument('--history-steps', type=int, default=4)
    p.add_argument('--max-pairs-per-sample', type=int, default=0,
                   help='0 keeps all adjacent history pairs; positive keeps the most recent pairs.')
    p.add_argument('--seed', type=int, default=42)
    return p


def explicit_sample_indices(data, sample_ids: str) -> list[int]:
    requested = [x.strip() for x in str(sample_ids).split(',') if x.strip()]
    by_id = {str(row.get('index', i)): i for i, (_, row) in enumerate(data.iterrows())}
    missing = [x for x in requested if x not in by_id]
    if missing:
        raise ValueError(f'Unknown --sample-ids: {missing}')
    return [by_id[x] for x in requested]


def visual_feature_grids(model, full, segments) -> list[np.ndarray]:
    """Return one H x W x D post-merge feature grid per prompt image."""
    batch = device_inputs(full, model)
    with torch.no_grad():
        output = model.get_image_features(batch['pixel_values'], batch['image_grid_thw'])
    pooled = output.pooler_output
    if isinstance(pooled, (tuple, list)):
        pooled = torch.cat(list(pooled), dim=0)
    pooled = pooled.detach().float().cpu().numpy()
    grids, cursor = [], 0
    for seg in segments:
        count = int(seg['visual_tokens'])
        values = pooled[cursor:cursor + count]
        expected = int(seg['grid_t']) * int(seg['grid_height']) * int(seg['grid_width'])
        if len(values) != expected:
            raise ValueError(f'Feature count {len(values)} != expected grid size {expected}')
        grids.append(values.reshape(seg['grid_t'], seg['grid_height'], seg['grid_width'], -1).mean(0))
        cursor += count
    if cursor != len(pooled):
        raise ValueError(f'Unmapped visual features: consumed {cursor}, received {len(pooled)}')
    return grids


def novelty_map(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Novelty in current-grid coordinates after normalized spatial alignment."""
    prev = torch.from_numpy(previous).permute(2, 0, 1).unsqueeze(0)
    prev = F.interpolate(prev, size=current.shape[:2], mode='bilinear', align_corners=False)
    prev = prev[0].permute(1, 2, 0).numpy()
    numerator = (prev * current).sum(axis=-1)
    denominator = np.linalg.norm(prev, axis=-1) * np.linalg.norm(current, axis=-1)
    cosine = numerator / np.maximum(denominator, 1e-12)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def plot_sample(root: Path, sample_id: str, pairs: list[dict]) -> str:
    """Save one compact time-by-space panel for all adjacent pairs of a sample."""
    nrows = len(pairs)
    if not nrows:
        raise ValueError('No adjacent history pairs to plot.')
    vmax = max(0.05, float(np.quantile(np.concatenate([x['novelty'].ravel() for x in pairs]), 0.98)))
    fig, axes = plt.subplots(nrows, 3, figsize=(9.0, 2.65 * nrows), squeeze=False,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.16})
    im = None
    for row, pair in enumerate(pairs):
        previous = np.asarray(Image.open(pair['previous_path']).convert('RGB'))
        current = np.asarray(Image.open(pair['current_path']).convert('RGB'))
        ax_prev, ax_curr, ax_map = axes[row]
        ax_prev.imshow(previous)
        ax_curr.imshow(current)
        im = ax_map.imshow(pair['novelty'], cmap='coolwarm', vmin=0.0, vmax=vmax,
                           interpolation='nearest')
        ax_prev.set_ylabel(pair['label'], fontsize=9, fontweight='bold')
        for ax in (ax_prev, ax_curr, ax_map):
            ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax_prev.set_title('Earlier history frame', fontsize=10, fontweight='bold')
            ax_curr.set_title('Later history frame', fontsize=10, fontweight='bold')
            ax_map.set_title('Post-merge token novelty', fontsize=10, fontweight='bold')
        ax_map.text(0.01, 0.01, f"mean={pair['mean_novelty']:.3f}", transform=ax_map.transAxes,
                    fontsize=8, color='white', va='bottom', ha='left',
                    bbox={'facecolor': '#1f2937', 'alpha': .7, 'pad': 2, 'edgecolor': 'none'})
    if im is not None:
        cbar = fig.colorbar(im, ax=axes[:, 2].tolist(), fraction=.035, pad=.02)
        cbar.set_label(r'Novelty $1-\cos(f_t, f_{t-1})$', fontsize=9)
    fig.suptitle(f'Full-history temporal novelty — sample {sample_id}', fontsize=12, fontweight='bold', y=.995)
    out = root / 'figures' / f'{sample_id}_novelty.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    return str(out)


def main() -> None:
    args = args_parser().parse_args()
    root = Path(args.work_dir)
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Isolate the observation protocol from all normal evaluation switches.
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE'] = '0'
    os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES'] = str(args.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE'] = '0'
    os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS'] = '1'
    os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES'] = str(args.history_steps)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(args.data)
    data = getattr(dataset, 'data', None)
    if data is None:
        raise RuntimeError(f'Cannot obtain dataframe for dataset {args.data}')
    candidates = complete_history_candidates(data, args.history_steps)
    if args.sample_ids:
        selected = [i for i in explicit_sample_indices(data, args.sample_ids) if i in candidates]
    else:
        selected = sampled_row_indices(data, args.num_samples, args.sample_by,
                                       args.samples_per_task, args.seed, candidates)
    if not selected:
        raise RuntimeError('No selected samples have the requested complete history.')

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype='auto', device_map='auto', attn_implementation='sdpa'
    ).eval()
    image_id = getattr(model.config, 'image_token_id', None)
    if image_id is None:
        raise RuntimeError('Model config lacks image_token_id.')
    merge = getattr(getattr(model.config, 'vision_config', None), 'spatial_merge_size', 1)
    metadata = vars(args) | {
        'full_history_only': True,
        'state_packet_disabled_in_process': True,
        'feature_definition': 'Qwen3-VL post-merge visual-token feature, averaged over temporal grid dimension',
        'novelty_definition': '1 - cosine(feature(current token), bilinearly aligned feature(previous token))',
        'selected_rows': len(selected),
    }
    (root / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    records = []
    for idx in selected:
        row = data.iloc[idx]
        sample_id = str(row.get('index', idx)).replace('/', '_')
        result = {'sample_id': str(row.get('index', idx)), 'row_index': int(idx), 'status': 'skipped'}
        try:
            built = dataset.build_prompt(row)
            messages = built[0] if isinstance(built, tuple) else built
            prompt, full, _ = make_inputs(processor, messages, '')
            records_images = image_items(messages)
            segments = locate_visual_segments(
                full['input_ids'][0], image_id, grid_rows(full.get('image_grid_thw'), merge), records_images
            )
            histories = [x for x in segments if x['kind'] == 'history']
            if len(histories) < 2:
                raise ValueError(f'Need at least two history frames, got {len(histories)}')
            grids_all = visual_feature_grids(model, full, segments)
            grid_by_path = {str(seg['path']): grids_all[i] for i, seg in enumerate(segments)}
            pairs = []
            for position in range(1, len(histories)):
                prev, cur = histories[position - 1], histories[position]
                previous_grid, current_grid = grid_by_path[str(prev['path'])], grid_by_path[str(cur['path'])]
                novelty = novelty_map(previous_grid, current_grid)
                pairs.append({
                    'previous_path': str(prev['path']), 'current_path': str(cur['path']),
                    'novelty': novelty,
                    'label': f'History {position - len(histories)} → {position + 1 - len(histories)}',
                    'mean_novelty': float(novelty.mean()), 'median_novelty': float(np.median(novelty)),
                    'p90_novelty': float(np.quantile(novelty, .90)),
                    'previous_visual_tokens': int(prev['visual_tokens']),
                    'current_visual_tokens': int(cur['visual_tokens']),
                })
            if args.max_pairs_per_sample > 0:
                pairs = pairs[-args.max_pairs_per_sample:]
            figure = plot_sample(root, sample_id, pairs)
            np.savez_compressed(root / 'figures' / f'{sample_id}_novelty.npz',
                                **{f'pair_{i}': x['novelty'] for i, x in enumerate(pairs)})
            result.update(status='ok', figure=figure, pairs=[{
                key: value for key, value in pair.items()
                if key not in {'novelty'}
            } for pair in pairs])
        except Exception as exc:
            result['reason'] = f'{type(exc).__name__}: {exc}'
        records.append(result)
        print(f"[history-novelty] sample={result['sample_id']} status={result['status']}", flush=True)
    with (root / 'per_sample.jsonl').open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    ok = [x for x in records if x['status'] == 'ok']
    summary = {
        'samples_selected': len(selected), 'samples_ok': len(ok),
        'samples_skipped': len(records) - len(ok),
        'figures': [x['figure'] for x in ok],
        'per_sample': 'per_sample.jsonl', 'metadata': 'metadata.json',
    }
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'[history-novelty] complete: {root / "figures"}', flush=True)


if __name__ == '__main__':
    main()
