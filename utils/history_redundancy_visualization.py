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
import torch.nn.functional as F

try:
    from history_attention_diagnostic import (complete_history_candidates,
        device_inputs, gold_from_row, image_items, make_inputs)
    from history_attention_utils import grid_rows, locate_visual_segments
except ModuleNotFoundError:
    from .history_attention_diagnostic import (complete_history_candidates,
        device_inputs, gold_from_row, image_items, make_inputs)
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
    p.add_argument('--qualitative-examples', type=int, default=10,
                   help='Maximum examples to render for each bounded-similarity set (0 disables rendering).')
    p.add_argument('--qual-global-min', type=float, default=.90)
    p.add_argument('--qual-global-max', type=float, default=.95)
    p.add_argument('--qual-local-min', type=float, default=.60)
    p.add_argument('--qual-local-max', type=float, default=.70)
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


PLOT_COLORS = {'token': '#009E73', 'local': '#0072B2', 'global': '#D55E00'}  # Okabe--Ito, colour-blind safe.
PLOT_LABELS = {'token': 'Individual visual tokens', 'local': '4×4-token patches', 'global': 'Full frames'}


def tokenwise_similarity(previous, current):
    """Per-current-token cosine, using the same bilinear alignment as novelty."""
    prev = torch.from_numpy(previous).permute(2, 0, 1).unsqueeze(0)
    prev = F.interpolate(prev, size=current.shape[:2], mode='bilinear', align_corners=False)
    prev = prev[0].permute(1, 2, 0).numpy()
    numerator = (prev * current).sum(axis=-1)
    denominator = np.linalg.norm(prev, axis=-1) * np.linalg.norm(current, axis=-1)
    return np.clip(numerator / np.maximum(denominator, 1e-12), -1.0, 1.0)


def smoothed_density(values, bins=160, sigma_bins=2.5):
    """A lightweight KDE-like density on the fixed [0, 1] similarity domain."""
    density, edges = np.histogram(values, bins=bins, range=(0, 1), density=True)
    radius = int(np.ceil(4 * sigma_bins))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return (edges[:-1] + edges[1:]) / 2, np.convolve(density, kernel, mode='same')


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1, dtype=float) / len(values)


def figures(root, rows, spatial_w, spatial_h, token_similarity=None):
    import matplotlib.pyplot as plt
    local = [r['redundancy'] for r in rows if r['scale'] == 'local']
    global_ = [r['redundancy'] for r in rows if r['scale'] == 'global']
    if not local or not global_: raise RuntimeError('Need both local and global comparable regions to plot.')
    # Use an explicit, self-contained style so the figure is legible after it is
    # reduced to a single-column paper width and remains consistent across hosts.
    with plt.rc_context({
        # Retain the original canvas size.  These are the largest sizes that
        # leave every label and median callout inside that fixed canvas.
        'font.family': 'DejaVu Sans', 'font.size': 28,
        'axes.labelsize': 26, 'axes.titlesize': 26,
        'xtick.labelsize': 24, 'ytick.labelsize': 24,
        'legend.fontsize': 25, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    }):
        # The token-scale curve is deliberately omitted: the plot focuses on
        # the two region granularities used for the masking comparison.
        groups = [('local', local), ('global', global_)]
        fig, ax = plt.subplots(figsize=(7.3, 5.7), constrained_layout=True)
        # ECDF retains every observation and makes the median visually exact.
        # This is deliberately one shared axis so the three granularities can
        # be compared without interpreting density bandwidth.
        for index, (key, values) in enumerate(groups):
            x, y = ecdf(values)
            median = float(np.median(values))
            ax.plot(x, y, color=PLOT_COLORS[key], lw=2.8,
                    label=PLOT_LABELS[key])
            ax.axvline(median, color=PLOT_COLORS[key], lw=1.55, ls='--', alpha=.92)
            ax.text(median, .09 + index * .20, f'Median\n{median:.3f}',
                    ha='center', va='bottom', color=PLOT_COLORS[key], fontweight='bold',
                    fontsize=25,
                    bbox={'boxstyle': 'round,pad=.20', 'facecolor':'white',
                          'edgecolor':PLOT_COLORS[key], 'lw':1.25, 'alpha':.96}, zorder=5)
        from matplotlib.ticker import PercentFormatter
        ax.set(xlabel='Visual-feature similarity (cosine)', ylabel='Cumulative fraction',
               xlim=(-.02, 1.01), ylim=(0, 1.02))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(axis='x', pad=11)
        ax.grid(color='#D9D9D9', lw=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='upper left', frameon=False, handlelength=2.4)

        output = root / 'figures' / 'redundancy_ecdf_by_scale'
        fig.savefig(output.with_suffix('.png'), dpi=450, bbox_inches='tight', facecolor='white')
        fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
        plt.close(fig)

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


def _select_qualitative_examples(pairs, global_min, global_max, local_min, local_max, limit):
    """Select reproducible examples in bounded, visually discriminable ranges."""
    if limit <= 0:
        return {}
    ranked = {
        'global_0.90_to_0.95': [p for p in pairs if global_min <= p['global_redundancy'] <= global_max],
        'local_0.60_to_0.70': [p for p in pairs if any(local_min <= x['redundancy'] <= local_max
                                                        for x in p['local_regions'])],
    }
    selected = {}
    for name, candidates in ranked.items():
        if name.startswith('global'):
            midpoint = (global_min + global_max) / 2
            candidates.sort(key=lambda p: (abs(p['global_redundancy'] - midpoint), p['sample_id'], p['history_step']))
        else:
            midpoint = (local_min + local_max) / 2
            candidates.sort(key=lambda p: (-sum(local_min <= x['redundancy'] <= local_max for x in p['local_regions']),
                                            min(abs(x['redundancy'] - midpoint) for x in p['local_regions']),
                                            p['sample_id'], p['history_step']))
        used, keep = set(), []
        for pair in candidates:
            identity = (pair['sample_id'], pair['history_step'])
            if identity not in used:
                keep.append(pair); used.add(identity)
            if len(keep) == limit:
                break
        selected[name] = keep
    return selected


def _render_qualitative_panel(path, pair, local_min):
    """Render raw frames, a feature overlay, and a deep-gray redundancy mask."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from textwrap import fill

    previous = Image.open(pair['previous_image_path']).convert('RGB')
    current = Image.open(pair['current_image_path']).convert('RGB')
    if current.size != previous.size:
        current = current.resize(previous.size, Image.Resampling.BILINEAR)

    h, w = pair['grid_height'], pair['grid_width']
    heat = np.full((h, w), np.nan, dtype=float)
    for region in pair['local_regions']:
        r0, c0, r1, c1 = region['grid_box_yxyx']
        heat[r0:r1, c0:c1] = region['redundancy']
    # Rasterize the exact feature-grid score to image pixels. Regions at or
    # above local_min are shown as deep gray in the final panel; all other
    # regions retain their original current-frame appearance.
    heat_pixels = np.asarray(Image.fromarray(heat.astype(np.float32), mode='F').resize(current.size, Image.Resampling.NEAREST))
    masked = np.asarray(current).copy()
    masked[heat_pixels >= local_min] = (48, 48, 48)
    local_scores = np.asarray([x['redundancy'] for x in pair['local_regions']], dtype=float)
    previous_instruction = fill(str(pair.get('previous_instruction', 'Unavailable')), width=38)
    current_instruction = fill(str(pair.get('current_instruction', 'Unavailable')), width=38)
    previous_gt = fill(str(pair.get('previous_gt_action', 'Unavailable')), width=38)
    current_gt = fill(str(pair.get('current_gt_action', 'Unavailable')), width=38)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.0), constrained_layout=True,
                                 gridspec_kw={'width_ratios': [1, 1, 1, 1.08]})
        axes[0].imshow(previous); axes[0].set_title(f'Previous frame\nInstruction: {previous_instruction}\nGT: {previous_gt}', fontweight='bold', fontsize=7); axes[0].axis('off')
        axes[1].imshow(current); axes[1].set_title(f'Current frame\nInstruction: {current_instruction}\nGT: {current_gt}', fontweight='bold', fontsize=7); axes[1].axis('off')
        axes[2].imshow(current)
        im = axes[2].imshow(heat, extent=(0, current.width, current.height, 0), cmap='Blues_r',
                            vmin=.0, vmax=1., alpha=.72, interpolation='nearest')
        axes[2].set_title('Current + feature-similarity overlay', fontweight='bold', fontsize=8); axes[2].axis('off')
        axes[3].imshow(masked)
        axes[3].set_title(f'Current with redundant regions masked\n(deep gray: cosine ≥ {local_min:.2f})',
                          fontweight='bold', fontsize=8); axes[3].axis('off')
        fig.colorbar(im, ax=axes[2], fraction=.046, pad=.02, label='local cosine similarity')
        fig.suptitle(
            f"Global cosine = {pair['global_redundancy']:.3f}  |  "
            f"Local cosine: min / median / max = {local_scores.min():.3f} / {np.median(local_scores):.3f} / {local_scores.max():.3f}",
            fontsize=8.5, fontweight='bold')
        fig.savefig(path.with_suffix('.png'), dpi=350, facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)


def qualitative_figures(root, pairs, global_min, global_max, local_min, local_max, limit):
    root = Path(root) / 'figures' / 'qualitative_redundancy_examples'
    root.mkdir(parents=True, exist_ok=True)
    selected = _select_qualitative_examples(pairs, global_min, global_max, local_min, local_max, limit)
    manifest = {'criteria': {'global_redundancy_range': [global_min, global_max],
                             'local_redundancy_range': [local_min, local_max],
                             'redundancy_mask_rule': f'local cosine >= {local_min}'}, 'examples': {}}
    for category, examples in selected.items():
        folder = root / category; folder.mkdir(exist_ok=True)
        manifest['examples'][category] = []
        for index, pair in enumerate(examples, 1):
            stem = f'{index:02d}_sample_{pair["sample_id"]}_history_{pair["history_step"]}'
            try:
                _render_qualitative_panel(folder / stem, pair, local_min)
                manifest['examples'][category].append({'file': str((folder / stem).relative_to(root)), **pair})
            except (OSError, ValueError) as exc:
                manifest['examples'][category].append({'render_error': f'{type(exc).__name__}: {exc}', **pair})
    (root / 'selection.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def merge(root, spatial_w, spatial_h, global_min=.90, global_max=.95, local_min=.60, local_max=.70, qualitative_examples=10):
    root = Path(root); workers = sorted((root/'workers').glob('worker_*')); rows, samples, pairs, token_chunks = [], [], [], []
    for worker in workers:
        for name, target in [('redundancy_regions.jsonl', rows), ('per_sample.jsonl', samples),
                             ('qualitative_pairs.jsonl', pairs)]:
            path = worker/name
            if path.exists():
                with path.open(encoding='utf-8') as fh: target.extend(json.loads(x) for x in fh if x.strip())
        token_path = worker / 'token_similarity.npy'
        if token_path.exists():
            token_chunks.append(np.load(token_path))
    if not rows: raise RuntimeError(f'No worker outputs below {root}/workers')
    (root/'figures').mkdir(parents=True, exist_ok=True)
    with (root/'redundancy_regions.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'redundancy_regions.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    with (root/'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for row in samples: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    token_similarity = np.concatenate(token_chunks) if token_chunks else None
    if token_similarity is not None:
        np.save(root / 'token_similarity.npy', token_similarity)
    figures(root, rows, spatial_w, spatial_h, token_similarity)
    if pairs and qualitative_examples > 0:
        qualitative_figures(root, pairs, global_min, global_max, local_min, local_max, qualitative_examples)
    local = [x['redundancy'] for x in rows if x['scale']=='local']; global_ = [x['redundancy'] for x in rows if x['scale']=='global']
    summary = {'workers_merged':len(workers), 'samples':len(samples), 'comparable_regions':len(rows),
               'local':{'count':len(local),'mean':float(np.mean(local)),'median':float(np.median(local))},
               'global':{'count':len(global_),'mean':float(np.mean(global_)),'median':float(np.median(global_))}}
    if token_similarity is not None:
        summary['token'] = {'count':int(token_similarity.size), 'mean':float(np.mean(token_similarity)),
                            'median':float(np.median(token_similarity)),
                            'definition':'cosine(current token, bilinearly aligned preceding-frame token); equals 1 - novelty'}
    (root/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    a = args_parser().parse_args()
    if not (0 <= a.qual_global_min <= a.qual_global_max <= 1 and
            0 <= a.qual_local_min <= a.qual_local_max <= 1):
        raise ValueError('qualitative similarity ranges must satisfy 0 <= min <= max <= 1')
    if a.merge_workers:
        merge(a.work_dir, a.spatial_width, a.spatial_height, a.qual_global_min,
              a.qual_global_max, a.qual_local_min, a.qual_local_max, a.qualitative_examples)
        return
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards: raise ValueError('invalid shard index')
    root = Path(a.work_dir); root.mkdir(parents=True, exist_ok=True); (root/'figures').mkdir(exist_ok=True)
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE']='0'; os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES']=str(a.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE']='0'; os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS']='1'; os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES']=str(a.history_steps)
    from vlmeval.dataset import build_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dataset = build_dataset(a.data); data = dataset.data; candidates = complete_history_candidates(data, a.history_steps)
    # Attach per-frame task metadata so both screenshots in a pair can carry
    # their own ground-truth action in qualitative figures.
    frame_metadata = {}
    for _, source_row in data.iterrows():
        source_path = str(source_row.get('image_path', source_row.get('image', '')) or '')
        if not source_path:
            continue
        action, _ = gold_from_row(source_row)
        frame_metadata[os.path.abspath(source_path)] = {
            'instruction': str(source_row.get('instruction', source_row.get('question', source_row.get('description', 'Unavailable')))),
            'gt_action': str(action or 'Unavailable'),
        }
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
    rows, samples, qualitative_pairs, token_chunks = [], [], [], []
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
                previous_seg, prev = history[frame_i-1]; h,w=seg['grid_height'],seg['grid_width']; rel=frame_i-len(history)
                global_redundancy = cosine(features.mean((0,1)), prev.mean((0,1)))
                # This is exactly the complement of history_visual_novelty.py:
                # token similarity = 1 - token novelty.  It is retained in a
                # compact .npy stream rather than exploding the region JSONL.
                token_chunks.append(tokenwise_similarity(prev, features).astype(np.float32, copy=False).ravel())
                rows.append({'sample_id':sid,'task_key':key,'history_step':seg['history_step'],'relative_step':rel,'scale':'global','grid_box_yxyx':[0,0,h,w],'center_xy':[.5,.5],'redundancy':global_redundancy})
                local_regions = []
                for r0 in range(0,h,a.local_scale):
                    for c0 in range(0,w,a.local_scale):
                        r1,c1=min(h,r0+a.local_scale),min(w,c0+a.local_scale)
                        redundancy = cosine(features[r0:r1,c0:c1].mean((0,1)), corresponding(prev,r0,r1,c0,c1,h,w))
                        local = {'grid_box_yxyx':[r0,c0,r1,c1], 'redundancy':redundancy}
                        local_regions.append(local)
                        rows.append({'sample_id':sid,'task_key':key,'history_step':seg['history_step'],'relative_step':rel,'scale':'local','grid_box_yxyx':local['grid_box_yxyx'],'center_xy':[(c0+c1)/(2*w),(r0+r1)/(2*h)],'redundancy':redundancy})
                # Retain only compact numeric patch summaries and paths; visual files
                # are generated after worker merge so multi-GPU workers never collide.
                qualitative_pairs.append({'sample_id':sid, 'task_key':key, 'history_step':seg['history_step'],
                                          'relative_step':rel, 'previous_image_path':previous_seg['path'],
                                          'current_image_path':seg['path'], 'grid_height':h, 'grid_width':w,
                                          'global_redundancy':global_redundancy, 'local_regions':local_regions,
                                          'previous_gt_action':frame_metadata.get(os.path.abspath(previous_seg['path']), {}).get('gt_action', 'Unavailable'),
                                          'current_gt_action':frame_metadata.get(os.path.abspath(seg['path']), {}).get('gt_action', 'Unavailable'),
                                          'previous_instruction':frame_metadata.get(os.path.abspath(previous_seg['path']), {}).get('instruction', 'Unavailable'),
                                          'current_instruction':frame_metadata.get(os.path.abspath(seg['path']), {}).get('instruction', str(row.get('instruction', 'Unavailable')))})
            sample['status']='ok'
        except Exception as exc: sample['skip_reason']=f'{type(exc).__name__}: {exc}'
        samples.append(sample)
        if bar: bar.update(1)
    if bar: bar.close()
    with (root/'redundancy_regions.jsonl').open('w', encoding='utf-8') as fh:
        for row in rows: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for row in samples: fh.write(json.dumps(row, ensure_ascii=False)+'\n')
    with (root/'qualitative_pairs.jsonl').open('w', encoding='utf-8') as fh:
        for pair in qualitative_pairs: fh.write(json.dumps(pair, ensure_ascii=False)+'\n')
    np.save(root / 'token_similarity.npy', np.concatenate(token_chunks) if token_chunks else np.empty(0, dtype=np.float32))


if __name__ == '__main__': main()
