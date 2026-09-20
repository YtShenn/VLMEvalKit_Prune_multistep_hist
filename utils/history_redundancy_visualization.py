#!/usr/bin/env python3
"""Redundancy-only full-history visual analysis (no masking or action scoring).

This opt-in offline tool is intentionally separate from the causal utility
experiment. It extracts Qwen3-VL visual-encoder features once per sample, so
it can densely cover local visual regions at a fraction of masking cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

try:
    from history_attention_diagnostic import (complete_history_candidates,
        device_inputs, image_items, make_inputs)
    from history_attention_utils import grid_rows, locate_visual_segments
except ModuleNotFoundError:
    from .history_attention_diagnostic import (complete_history_candidates,
        device_inputs, image_items, make_inputs)
    from .history_attention_utils import grid_rows, locate_visual_segments


FIELDS = ['sample_id', 'task_key', 'history_step', 'relative_step', 'scale',
          'grid_box_yxyx', 'center_xy', 'redundancy']


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--model', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--num-tasks', type=int, default=500)
    p.add_argument('--all-eligible', action='store_true')
    p.add_argument('--history-steps', type=int, default=4)
    p.add_argument('--local-scale', type=int, default=4, help='Final visual-token width/height of local patches.')
    p.add_argument('--spatial-width', type=int, default=16); p.add_argument('--spatial-height', type=int, default=9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-shards', type=int, default=1); p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--progress', choices=['0', '1'], default='1')
    p.add_argument('--merge-workers', action='store_true')
    return p


def task_key(row, fallback):
    return str(row.get('_trajectory_key', row.get('task_id', row.get('episode_id', row.get('task', row.get('instruction', fallback))))))


def select_tasks(data, candidates, count, seed):
    groups = {}
    for idx in candidates: groups.setdefault(task_key(data.iloc[idx], idx), []).append(idx)
    rng = random.Random(seed); keys = sorted(groups); rng.shuffle(keys)
    for key in keys: rng.shuffle(groups[key])
    return [groups[key][0] for key in keys[:count]]


def grid_features(model, batch, segments):
    batch = device_inputs(batch, model)
    with torch.no_grad():
        output = model.get_image_features(batch['pixel_values'], batch['image_grid_thw'])
    pooled = output.pooler_output
    if isinstance(pooled, (list, tuple)): pooled = torch.cat(list(pooled), dim=0)
    pooled = pooled.detach().float().cpu().numpy()
    result, cursor = [], 0
    for seg in segments:
        n = int(seg['visual_tokens'])
        result.append(pooled[cursor:cursor+n].reshape(seg['grid_t'], seg['grid_height'], seg['grid_width'], -1).mean(0))
        cursor += n
    if cursor != len(pooled): raise ValueError(f'visual feature/token mismatch: {len(pooled)} != {cursor}')
    return result


def cosine(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def corresponding(previous, r0, r1, c0, c1, current_h, current_w):
    h, w = previous.shape[:2]
    y0, y1 = int(np.floor(r0*h/current_h)), int(np.ceil(r1*h/current_h))
    x0, x1 = int(np.floor(c0*w/current_w)), int(np.ceil(c1*w/current_w))
    return previous[y0:max(y0+1, y1), x0:max(x0+1, x1)].mean((0, 1))


def ecdf(ax, values, color, label):
    x = np.sort(np.asarray(values)); y = np.arange(1, len(x)+1) / len(x)
    ax.step(x, y, where='post', color=color, lw=2, label=f'{label} (n={len(x):,}, median={np.median(x):.3f})')
    ax.axvline(np.median(x), color=color, lw=1, ls='--', alpha=.7)


def figures(root, rows, spatial_w, spatial_h):
    import matplotlib.pyplot as plt
    local = [r['redundancy'] for r in rows if r['scale'] == 'local']
    global_ = [r['redundancy'] for r in rows if r['scale'] == 'global']
    if not local or not global_: raise RuntimeError('Need both local and global comparable regions to plot.')
    fig, ax = plt.subplots(figsize=(7, 5)); ecdf(ax, local, '#1f77b4', 'local patch'); ecdf(ax, global_, '#d62728', 'global/full-frame')
    ax.set(xlabel='Redundancy: cosine visual-feature similarity to preceding history frame', ylabel='Cumulative proportion', xlim=(-.05, 1.02))
    ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=9); fig.tight_layout(); fig.savefig(root/'figures'/'redundancy_ecdf_by_scale.png', dpi=220); plt.close(fig)

    sums = np.zeros((spatial_h, spatial_w)); counts = np.zeros((spatial_h, spatial_w), dtype=int)
    for r in rows:
        if r['scale'] != 'local': continue
        x, y = r['center_xy']; col = min(spatial_w-1, max(0, int(x*spatial_w))); row = min(spatial_h-1, max(0, int(y*spatial_h)))
        sums[row, col] += r['redundancy']; counts[row, col] += 1
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im = axes[0].imshow(mean, origin='upper', cmap='viridis', vmin=.0, vmax=1.0, aspect='auto')
    fig.colorbar(im, ax=axes[0], label='mean local redundancy'); axes[0].set(title='Spatial mean redundancy', xlabel='normalized screen x bin', ylabel='normalized screen y bin')
    im = axes[1].imshow(counts, origin='upper', cmap='magma', aspect='auto')
    fig.colorbar(im, ax=axes[1], label='region count'); axes[1].set(title='Spatial sampling coverage', xlabel='normalized screen x bin', ylabel='normalized screen y bin')
    fig.tight_layout(); fig.savefig(root/'figures'/'local_redundancy_spatial_heatmap.png', dpi=220); plt.close(fig)


def merge(root, spatial_w, spatial_h):
    root = Path(root); workers = sorted((root/'workers').glob('worker_*')); rows, samples = [], []
    for worker in workers:
        for name, target in [('redundancy_regions.jsonl', rows), ('per_sample.jsonl', samples)]:
            path = worker/name
            if path.exists():
                with path.open(encoding='utf-8') as fh: target.extend(json.loads(x) for x in fh if x.strip())
    if not rows: raise RuntimeError(f'No worker outputs below {root}/workers')
    (root/'figures').mkdir(parents=True, exist_ok=True)
    with (root/'redundancy_regions.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'redundancy_regions.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    with (root/'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for row in samples: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    figures(root, rows, spatial_w, spatial_h)
    local = [x['redundancy'] for x in rows if x['scale']=='local']; global_ = [x['redundancy'] for x in rows if x['scale']=='global']
    summary = {'workers_merged':len(workers), 'samples':len(samples), 'comparable_regions':len(rows),
               'local':{'count':len(local),'mean':float(np.mean(local)),'median':float(np.median(local))},
               'global':{'count':len(global_),'mean':float(np.mean(global_)),'median':float(np.median(global_))}}
    (root/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    a = args_parser().parse_args()
    if a.merge_workers: merge(a.work_dir, a.spatial_width, a.spatial_height); return
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards: raise ValueError('invalid shard index')
    root = Path(a.work_dir); root.mkdir(parents=True, exist_ok=True); (root/'figures').mkdir(exist_ok=True)
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE']='0'; os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES']=str(a.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE']='0'; os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS']='1'; os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES']=str(a.history_steps)
    from vlmeval.dataset import build_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dataset = build_dataset(a.data); data = dataset.data; candidates = complete_history_candidates(data, a.history_steps)
    selected = list(candidates) if a.all_eligible else select_tasks(data, candidates, a.num_tasks, a.seed)
    before = len(selected); selected = [i for i in selected if zlib.crc32(task_key(data.iloc[i], i).encode()) % a.num_shards == a.shard_index]
    (root/'metadata.json').write_text(json.dumps(vars(a)|{'eligible_complete_history_rows':len(candidates),'selected_before_sharding':before,'selected_rows':len(selected),'analysis':'visual redundancy only; no masking or action likelihood'}, indent=2, ensure_ascii=False))
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype='auto', device_map='auto', attn_implementation='sdpa').eval()
    image_id = model.config.image_token_id; merge_size = getattr(model.config.vision_config, 'spatial_merge_size', 1)
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError: tqdm = None
    bar = tqdm(total=len(selected), desc=f'worker {a.shard_index+1}/{a.num_shards} visual', unit='task', dynamic_ncols=True, disable=a.progress=='0') if tqdm else None
    rows, samples = [], []
    for idx in selected:
        row = data.iloc[idx]; sid = str(row.get('index', idx)); key = task_key(row, idx); sample = {'sample_id':sid,'task_key':key,'status':'skipped'}
        try:
            built = dataset.build_prompt(row); messages = built[0] if isinstance(built, tuple) else built
            records = image_items(messages); prompt, _, _ = make_inputs(processor, messages, '')
            segments = locate_visual_segments(prompt['input_ids'][0], image_id, grid_rows(prompt.get('image_grid_thw'), merge_size), records)
            history = [(seg, feat) for seg, feat in zip(segments, grid_features(model, prompt, segments)) if seg['kind']=='history']
            if len(history) < a.history_steps: raise ValueError('incomplete_history_after_prompt_build')
            for frame_i, (seg, features) in enumerate(history):
                if frame_i == 0: continue
                prev = history[frame_i-1][1]; h,w=seg['grid_height'],seg['grid_width']; rel=frame_i-len(history)
                rows.append({'sample_id':sid,'task_key':key,'history_step':seg['history_step'],'relative_step':rel,'scale':'global','grid_box_yxyx':[0,0,h,w],'center_xy':[.5,.5],'redundancy':cosine(features.mean((0,1)),prev.mean((0,1)) )})
                for r0 in range(0,h,a.local_scale):
                    for c0 in range(0,w,a.local_scale):
                        r1,c1=min(h,r0+a.local_scale),min(w,c0+a.local_scale)
                        rows.append({'sample_id':sid,'task_key':key,'history_step':seg['history_step'],'relative_step':rel,'scale':'local','grid_box_yxyx':[r0,c0,r1,c1],'center_xy':[(c0+c1)/(2*w),(r0+r1)/(2*h)],'redundancy':cosine(features[r0:r1,c0:c1].mean((0,1)),corresponding(prev,r0,r1,c0,c1,h,w))})
            sample['status']='ok'
        except Exception as exc: sample['skip_reason']=f'{type(exc).__name__}: {exc}'
        samples.append(sample)
        if bar: bar.update(1)
    if bar: bar.close()
    with (root/'redundancy_regions.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for row in samples: fh.write(json.dumps(row, ensure_ascii=False)+'\n')


if __name__ == '__main__': main()
