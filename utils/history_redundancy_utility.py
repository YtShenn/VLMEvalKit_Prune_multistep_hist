#!/usr/bin/env python3
"""Causal redundancy--utility study for full-history Qwen3-VL trajectories.

This is an offline, opt-in experiment.  For each selected *successful
trajectory* it computes visual-feature redundancy between corresponding regions
of adjacent history screenshots, then measures causal utility with either a
legacy pixel occlusion or a physical visual-token pruning intervention.
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
from PIL import Image, ImageStat

try:
    from history_attention_diagnostic import (complete_history_candidates,
        device_inputs, explicit_sample_indices, gold_from_row, image_items,
        make_inputs)
    from history_attention_utils import grid_rows, locate_visual_segments
except ModuleNotFoundError:
    from .history_attention_diagnostic import (complete_history_candidates,
        device_inputs, explicit_sample_indices, gold_from_row, image_items,
        make_inputs)
    from .history_attention_utils import grid_rows, locate_visual_segments


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--success-manifest', '--success-jsonl', dest='success_manifest', default=None,
                   help='Either a per_sample.jsonl from history_attention_freerun_diagnostic.py, or JSON '
                        '`{"successful_trajectory_keys": [...]}` produced by an external trajectory evaluator.')
    p.add_argument('--success-scope', choices=['trajectory', 'step'], default='trajectory',
                   help='trajectory: every dataset step must be free-run correct; step: retain individually correct free-run decision points.')
    p.add_argument('--outcome-scope', choices=['success', 'all'], default='success',
                   help='success filters to successful free-run points; all samples every complete-history point and only records an outcome label when available.')
    p.add_argument('--num-samples', type=int, default=20, help='0 with --all-eligible means no sample-count cap.')
    p.add_argument('--all-eligible', action='store_true', help='Run every eligible successful decision point.')
    p.add_argument('--sample-by', choices=['task', 'sequential'], default='task')
    p.add_argument('--samples-per-task', type=int, default=1)
    p.add_argument('--sample-ids', default=None)
    p.add_argument('--history-steps', type=int, default=4)
    p.add_argument('--region-scales', default='4,8,global',
                   help='Comma-separated visual-token block widths, plus optional `global`.')
    p.add_argument('--max-regions-per-frame', type=int, default=48,
                   help='Uniform deterministic cap after all scales are tiled; 0 means no cap (very expensive).')
    p.add_argument('--mask-mode', choices=['mean_rgb', 'gray'], default='mean_rgb')
    p.add_argument('--intervention-mode', choices=['pixel_mask', 'token_prune', 'drop_frame', 'thumbnail_frame'],
                   default='pixel_mask',
                   help='pixel_mask preserves the legacy image-corruption study; drop_frame removes a '
                        'full history image and its visual tokens; thumbnail_frame replaces that full '
                        'history image with the deployment thumbnail; token_prune physically removes the '
                        'selected region\'s post-merge visual tokens in the decoder. Image-level modes evaluate global regions only.')
    p.add_argument('--thumbnail-long-edge', type=int, default=192,
                   help='Long edge in pixels for thumbnail_frame; matches the AndroidControl deployment default.')
    p.add_argument('--token-prune-layer', type=int, default=0,
                   help='Decoder layer at which token_prune physically shortens the sequence; 0 removes selected tokens before the first LLM decoder layer.')
    p.add_argument('--action-score', choices=['sum_logprob', 'mean_logprob'], default='sum_logprob')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-shards', type=int, default=1, help='Independent worker count; launcher assigns one GPU per shard.')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--progress', choices=['0', '1'], default='1')
    p.add_argument('--merge-workers', action='store_true', help='Merge worker outputs under --work-dir; does not load a model.')
    return p


def read_success_by_sample_id(path):
    """Read free-run correctness keyed by the dataset's stable ``index`` value."""
    outcomes = {}
    source = Path(path)
    paths = sorted(source.rglob('per_sample.jsonl')) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(f'No per_sample.jsonl below success manifest directory: {source}')
    for record_path in paths:
        with record_path.open(encoding='utf-8') as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                ok = rec.get('correctness', {}).get('overall_success')
                if ok is None:
                    continue
                outcomes[str(rec.get('sample_id'))] = bool(ok)
    return outcomes


def read_success_manifest(path):
    """Prefer explicit trajectory labels when an evaluator provides them."""
    source = Path(path)
    if source.suffix.lower() == '.json':
        payload = json.loads(source.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and isinstance(payload.get('successful_trajectory_keys'), list):
            return {str(x) for x in payload['successful_trajectory_keys']}, None
    return None, read_success_by_sample_id(source)


def successful_trajectory_keys(data, outcomes):
    """Require every annotated step in a dataset trajectory to be free-run correct.

    Free-run reports historically used a per-row ``episode_id`` fallback on some
    datasets, so grouping their records directly can mistakenly call a single
    correct step a successful trajectory.  The dataset's precomputed trajectory
    key is authoritative here.  A partially evaluated trajectory is excluded.
    """
    key_col = '_trajectory_key' if '_trajectory_key' in data.columns else 'episode_id'
    groups = {}
    for i in range(len(data)):
        row = data.iloc[i]
        key = str(row.get(key_col, row.get('task_id', row.get('index', i))))
        groups.setdefault(key, []).append(str(row.get('index', i)))
    return {key for key, ids in groups.items() if ids and all(outcomes.get(x) is True for x in ids)}


def sample_indices_by_task(data, candidates, count, sample_by, samples_per_task, seed):
    """Sample AndroidControl task trajectories, not merely similarly worded rows."""
    candidates = list(candidates)
    if sample_by == 'sequential': return candidates[:count]
    groups = {}
    for idx in candidates:
        row = data.iloc[idx]
        key = str(row.get('_trajectory_key', row.get('task_id', row.get('episode_id', row.get('task', row.get('instruction', idx))))))
        groups.setdefault(key, []).append(idx)
    rng = random.Random(seed); keys = sorted(groups); rng.shuffle(keys)
    for key in keys: rng.shuffle(groups[key])
    selected = []
    for round_idx in range(max(1, samples_per_task)):
        for key in keys:
            if round_idx < len(groups[key]):
                selected.append(groups[key][round_idx])
                if len(selected) >= count: return selected
    return selected


def parse_scales(text):
    values = []
    for x in str(text).split(','):
        x = x.strip().lower()
        if not x:
            continue
        if x == 'global': values.append(x)
        else:
            n = int(x)
            if n < 1: raise ValueError('region scale must be >= 1')
            values.append(n)
    if not values: raise ValueError('--region-scales is empty')
    return values


def region_specs(seg, scales, cap, rng):
    """Tiled regions in final LLM visual-token coordinates."""
    h, w = int(seg['grid_height']), int(seg['grid_width'])
    out = []
    for scale in scales:
        if scale == 'global':
            out.append(dict(row0=0, row1=h, col0=0, col1=w, scale='global', region_type='global'))
            continue
        for row0 in range(0, h, scale):
            for col0 in range(0, w, scale):
                out.append(dict(row0=row0, row1=min(h, row0 + scale), col0=col0, col1=min(w, col0 + scale),
                                scale=str(scale), region_type='local'))
    if cap > 0 and len(out) > cap:
        # Keep global regions, sample the rest uniformly rather than biasing top-left.
        global_regions = [x for x in out if x['region_type'] == 'global']
        other = [x for x in out if x['region_type'] != 'global']
        keep = max(0, cap - len(global_regions))
        indices = rng.choice(len(other), size=min(keep, len(other)), replace=False)
        out = global_regions + [other[int(i)] for i in sorted(indices)]
    return out


def pixel_box(region, seg, image_size):
    iw, ih = image_size
    h, w = seg['grid_height'], seg['grid_width']
    x0, x1 = round(region['col0'] * iw / w), round(region['col1'] * iw / w)
    y0, y1 = round(region['row0'] * ih / h), round(region['row1'] * ih / h)
    return [int(x0), int(y0), int(max(x0 + 1, x1)), int(max(y0 + 1, y1))]


def mask_image(source, box, destination, mode):
    im = Image.open(source).convert('RGB')
    if mode == 'gray': color = (127, 127, 127)
    else: color = tuple(round(x) for x in ImageStat.Stat(im).mean[:3])
    im.paste(color, tuple(box))
    im.save(destination)


def action_logprob(model, full, prompt_len, score_mode, token_drop_positions=None, token_prune_layer=4):
    """Teacher-force ``gold`` after an optional physical post-merge token drop.

    ``full`` intentionally still contains the original gold continuation.  The
    custom decoder shortens only its hidden-state sequence, so the gold labels
    must be read from the original input while logits are indexed with the
    shorter effective prompt length.
    """
    batch = device_inputs(full, model)
    target = batch['input_ids'][0, prompt_len:]
    if not len(target): raise ValueError('gold action tokenized to an empty continuation')
    config = getattr(model.config, 'text_config', None)
    if token_drop_positions is not None:
        if config is None:
            raise RuntimeError('token_prune requires the custom Qwen3-VL decoder configuration')
        config._redundancy_utility_token_drop_positions = [int(x) for x in token_drop_positions]
        config._redundancy_utility_token_prune_layer = int(token_prune_layer)
    try:
        with torch.no_grad():
            logits = model(**batch, use_cache=False, return_dict=True).logits[0]
        stats = dict(getattr(config, '_redundancy_utility_last_stats', {}) or {}) if config is not None else {}
    finally:
        if config is not None:
            for name in ('_redundancy_utility_token_drop_positions', '_redundancy_utility_token_prune_layer'):
                if hasattr(config, name):
                    delattr(config, name)
    dropped = int(stats.get('dropped_visual_tokens', 0) or 0)
    effective_prompt_len = int(prompt_len) - dropped
    if effective_prompt_len <= 0:
        raise RuntimeError(f'token_prune removed too many prompt tokens: prompt={prompt_len}, dropped={dropped}')
    # Logits at position p predict token p+1; the prompt-final logit predicts gold[0].
    selected = logits[effective_prompt_len - 1:effective_prompt_len - 1 + len(target)].float()
    if len(selected) != len(target):
        raise RuntimeError(f'token-pruned logits cannot score complete gold action: {len(selected)} != {len(target)}')
    value = torch.log_softmax(selected, -1).gather(1, target[:, None]).sum().item()
    return (float(value / len(target) if score_mode == 'mean_logprob' else value), int(len(target)),
            {'effective_prompt_tokens': effective_prompt_len, 'pruned_visual_tokens': dropped, **stats})


def visual_features(model, full, segments):
    """Qwen visual encoder features f(r), aligned to exact placeholder segments."""
    batch = device_inputs(full, model)
    with torch.no_grad():
        # ``get_image_features`` follows the model config's return-dict default.
        # Passing return_dict through some Transformers versions forwards it a
        # second time into the visual tower, so deliberately use that default.
        output = model.get_image_features(batch['pixel_values'], batch['image_grid_thw'])
    pooled = output.pooler_output
    if isinstance(pooled, (tuple, list)): pooled = torch.cat(list(pooled), dim=0)
    pooled = pooled.detach().float().cpu().numpy()
    result = []
    cursor = 0
    for seg in segments:
        n = int(seg['visual_tokens'])
        result.append(pooled[cursor:cursor + n].reshape(seg['grid_t'], seg['grid_height'], seg['grid_width'], -1).mean(0))
        cursor += n
    if cursor != len(pooled): raise ValueError(f'visual features {len(pooled)} do not match mapped tokens {cursor}')
    return result


def region_feature(features, region):
    return features[region['row0']:region['row1'], region['col0']:region['col1']].mean((0, 1))


def corresponding_feature(previous, region, current_seg):
    """Same normalized screen rectangle in the preceding history grid."""
    h, w = previous.shape[:2]
    y0 = int(np.floor(region['row0'] * h / current_seg['grid_height']))
    y1 = int(np.ceil(region['row1'] * h / current_seg['grid_height']))
    x0 = int(np.floor(region['col0'] * w / current_seg['grid_width']))
    x1 = int(np.ceil(region['col1'] * w / current_seg['grid_width']))
    return previous[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)].mean((0, 1))


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def region_token_positions(segment, region):
    """Exact flattened post-merge visual-token positions for one grid region."""
    h, w, t = (int(segment['grid_height']), int(segment['grid_width']), int(segment['grid_t']))
    offsets = []
    for time_index in range(t):
        base = int(segment['sequence_start']) + time_index * h * w
        for row in range(int(region['row0']), int(region['row1'])):
            offsets.extend(range(base + row * w + int(region['col0']), base + row * w + int(region['col1'])))
    expected = (int(region['row1']) - int(region['row0'])) * (int(region['col1']) - int(region['col0'])) * t
    if len(offsets) != expected or not offsets:
        raise ValueError(f'Invalid region-to-token mapping: {region}')
    return offsets


def make_masked_messages(messages, target_path, masked_path):
    answer = []
    for item in messages:
        copied = dict(item)
        if copied.get('type') == 'image' and str(copied.get('value', '')).removeprefix('file://') == target_path:
            copied['value'] = str(masked_path)
        answer.append(copied)
    return answer


def _matches_image_path(item, target_path):
    if item.get('type') != 'image':
        return False
    value = str(item.get('value', '')).removeprefix('file://')
    return os.path.abspath(value) == os.path.abspath(str(target_path))


def make_dropped_frame_messages(messages, target_path):
    """Remove exactly one history image, so its visual tokens leave the prompt."""
    answer, removed = [], False
    for item in messages:
        if _matches_image_path(item, target_path):
            removed = True
            continue
        answer.append(dict(item))
    if not removed:
        raise ValueError(f'target history image was not present in prompt: {target_path}')
    return answer


def deployment_thumbnail_item(source_path, cache_dir, sample_id, history_index, long_edge):
    """Build the same thumbnail representation used by AndroidControl state packets.

    The causal experiment must replace a frame with the actual low-resolution
    visual input used at inference, not an average-colour proxy.  The shared
    state-packet builder also supplies aligned dimensions and ``resized_*``
    metadata expected by the Qwen processor.
    """
    from vlmeval.dataset.AndroidControl_Curated.state_packet import build_state_packet

    overrides = {
        'ANDROID_CONTROL_STATE_PACKET_CACHE_DIR': str(cache_dir),
        'ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE': str(long_edge),
        'ANDROID_CONTROL_STATE_PACKET_IMAGE_MODE': 'thumbnail',
    }
    original = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        images, meta = build_state_packet(
            image_path=str(source_path), action_packet={}, sample_index=str(sample_id),
            history_index=int(history_index),
        )
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    thumbnail = next((image for image in images if image.kind == 'thumbnail'), None)
    if thumbnail is None:
        raise RuntimeError('state-packet thumbnail builder returned no thumbnail image')
    return thumbnail.to_message_item(), {
        'thumbnail_path': thumbnail.path,
        'thumbnail_estimated_tokens': int(thumbnail.estimated_tokens),
        'thumbnail_width': int(thumbnail.width),
        'thumbnail_height': int(thumbnail.height),
        'thumbnail_builder': 'android_control_state_packet',
        'thumbnail_compression_ratio': float(meta['compression_ratio_vs_original']),
    }


def make_thumbnail_frame_messages(messages, target_path, thumbnail_item):
    """Replace exactly one source image while retaining its prompt position."""
    answer, replaced = [], False
    for item in messages:
        copied = dict(item)
        if _matches_image_path(copied, target_path):
            copied.update(thumbnail_item)
            replaced = True
        answer.append(copied)
    if not replaced:
        raise ValueError(f'target history image was not present in prompt: {target_path}')
    return answer


def save_figures(root, points, provenance_note=None):
    """Plot an immediately readable redundancy--utility summary.

    Each scale is split into its own redundancy tertiles, then shown as six
    mean-and-bootstrap-CI estimates. This keeps the primary comparison visible
    without asking readers to decode an overplotted point cloud.
    """
    import matplotlib.pyplot as plt
    valid = [x for x in points if x.get('redundancy') is not None]
    if not valid:
        (root / 'figures' / 'NO_COMPARABLE_REGIONS.txt').write_text('No region has a preceding history frame.\n')
        return

    # Keep the styling self-contained: it must also work on headless workers
    # without a seaborn installation or an external Matplotlib style file.
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 14,
        'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'legend.fontsize': 13, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })
    groups = [('4', 'Local regions', '#377eb8'),
              ('global', 'Full-frame regions', '#e76f51')]
    present = [(key, title, color) for key, title, color in groups if any(p['scale'] == key for p in valid)]
    # Preserve unexpected scales instead of silently dropping experimental data.
    known = {key for key, _, _ in groups}
    present.extend((key, f'{key}$\\times${key} token regions', '#6a4c93')
                   for key in sorted({p['scale'] for p in valid} - known) if key != 'global')

    # A narrow, title-free forest plot: the caption carries the interpretation.
    fig, ax = plt.subplots(figsize=(6.2, 4.15))
    fig.subplots_adjust(left=.265, right=.985, bottom=.20, top=.965)
    rng = np.random.default_rng(20260920)
    summaries = []
    for label, title, color in present:
        rows = [p for p in valid if p['scale'] == label]
        x = np.asarray([p['redundancy'] for p in rows], dtype=float)
        utility = np.asarray([p['utility'] for p in rows], dtype=float)
        # Local and full-frame regions use separate tertile cutoffs because
        # their raw redundancy ranges are not directly comparable.
        edges = np.quantile(x, [0, 1 / 3, 2 / 3, 1])
        for group, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (x >= left) & ((x < right) if group < 2 else (x <= right))
            values = utility[mask]
            boot = rng.choice(values, size=(1200, len(values)), replace=True).mean(axis=1)
            if group == 0:
                level, threshold = 'Low', '$r < {:.3f}$'.format(edges[1])
            elif group == 1:
                level, threshold = 'Medium', '$ {:.3f} \\leq r < {:.3f}$'.format(edges[1], edges[2])
            else:
                level, threshold = 'High', '$r \\geq {:.3f}$'.format(edges[2])
            summaries.append({
                'scale': title, 'level': level, 'threshold': threshold,
                'n': len(values), 'mean': float(values.mean()),
                'lo': float(np.quantile(boot, .025)), 'hi': float(np.quantile(boot, .975)),
                'color': color,
            })

    y_positions = np.arange(len(summaries))[::-1]
    for y_pos, item in zip(y_positions, summaries):
        ax.errorbar(item['mean'], y_pos,
                    xerr=[[item['mean'] - item['lo']], [item['hi'] - item['mean']]],
                    fmt='o', color=item['color'], ecolor=item['color'],
                    elinewidth=2.5, capsize=4, markersize=8,
                    markerfacecolor='white', markeredgewidth=2.2, zorder=3)

    ax.axvline(0, color='#4d4d4d', lw=1.25, ls='--', zorder=0)
    if len(present) > 1:
        ax.axhline(2.5, color='#d9d9d9', lw=.9, zorder=0)
    ax.set_yticks(y_positions, [''] * len(summaries))
    modes = {str(p.get('intervention_mode', 'pixel_mask')) for p in valid}
    if modes == {'token_prune'}:
        impact_label = 'Token-pruning impact on gold-action log-likelihood'
    elif modes == {'pixel_mask'}:
        impact_label = 'Masking impact on gold-action log-likelihood'
    else:
        impact_label = 'Input-intervention impact on gold-action log-likelihood'
    ax.set_xlabel(impact_label, labelpad=7)
    ax.grid(axis='x', color='#e0e0e0', lw=.85, zorder=0)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.tick_params(axis='y', length=0)
    for y_pos, item in zip(y_positions, summaries):
        ax.text(-.10, y_pos + .16, item['level'], transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=14, clip_on=False)
        ax.text(-.10, y_pos - .16, f"({item['threshold']})", transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=12.5, clip_on=False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', linestyle='None', color=color,
               markerfacecolor='white', markeredgewidth=2.1, markersize=8.5,
               label=title)
        for _, title, color in present
    ]
    ax.legend(handles=handles, loc='upper left', frameon=False, handletextpad=.35,
              borderpad=.1, labelspacing=.25)
    fig.savefig(root / 'figures' / 'redundancy_utility_density.png', dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(root / 'figures' / 'redundancy_utility_density.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(root / 'figures' / 'redundancy_utility_density.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)


FIELDS = ['sample_id','episode_id','outcome','history_step','relative_step','image_path','grid_box_yxyx','pixel_box_xyxy','scale','region_type','redundancy','utility','baseline_logprob','masked_logprob','intervention_mode','baseline_prompt_tokens','intervened_prompt_tokens','pruned_visual_tokens','token_prune_layer','dropped_visual_tokens','effective_prompt_tokens','drop_layer','prune_applied','sequence_length_before','sequence_length_after','thumbnail_path','thumbnail_estimated_tokens','thumbnail_width','thumbnail_height','thumbnail_builder','thumbnail_compression_ratio']


def merge_worker_outputs(root):
    """Create the single analysis table/figure from completed GPU workers only."""
    root = Path(root)
    all_workers = sorted((root / 'workers').glob('worker_*'))
    # A completed worker normally has summary.json.  Also accept the paired
    # durable JSONL outputs: this makes aggregation recoverable if a harmless
    # final CSV/figure write fails after every model forward has finished.
    workers = sorted(worker for worker in all_workers
                     if (worker / 'summary.json').is_file()
                     or ((worker / 'regions.jsonl').is_file() and (worker / 'per_sample.jsonl').is_file()))
    if not workers: raise RuntimeError(f'No worker directories under {root / "workers"}')
    points, samples = [], []
    for worker in workers:
        path = worker / 'regions.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as fh:
                points.extend(json.loads(x) for x in fh if x.strip())
        path = worker / 'per_sample.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as fh:
                samples.extend(json.loads(x) for x in fh if x.strip())
    if not samples: raise RuntimeError('Workers produced no per_sample.jsonl records.')
    # A worker may have been restarted after interruption.  Retain one record
    # per masked region rather than allowing its pre-restart tail to duplicate
    # a completed region in the final aggregate.
    unique_points = {}
    for point in points:
        key = (str(point['sample_id']), point['history_step'],
               tuple(point['grid_box_yxyx']), point['scale'])
        unique_points.setdefault(key, point)
    duplicate_points_dropped = len(points) - len(unique_points)
    points = list(unique_points.values())
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    with (root / 'regions.jsonl').open('w', encoding='utf-8') as fh:
        for point in points: fh.write(json.dumps(point, ensure_ascii=False) + '\n')
    with (root / 'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for sample in samples: fh.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with (root / 'regions.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(points)
    provenance_note = None
    if len(workers) != len(all_workers):
        provenance_note = f'Partial aggregation: {len(workers)}/{len(all_workers)} worker shards completed.'
    save_figures(root, points, provenance_note=provenance_note)
    valid = [x for x in points if x.get('redundancy') is not None]
    summary = {'workers_discovered': len(all_workers), 'workers_merged': len(workers), 'samples_ok': sum(x['status'] == 'ok' for x in samples),
               'samples_skipped': sum(x['status'] != 'ok' for x in samples), 'regions_total': len(points),
               'duplicate_region_records_dropped': duplicate_points_dropped,
               'regions_with_redundancy': len(valid),
               'mean_redundancy': float(np.mean([x['redundancy'] for x in valid])) if valid else None,
               'mean_utility': float(np.mean([x['utility'] for x in valid])) if valid else None,
               'outputs': {'points_jsonl': 'regions.jsonl', 'points_csv': 'regions.csv', 'density_figure': 'figures/redundancy_utility_density.png'}}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'[merge] workers={len(workers)} samples={len(samples)} regions={len(points)} -> {root}', flush=True)


def main():
    a = args_parser().parse_args(); scales = parse_scales(a.region_scales)
    if a.merge_workers:
        merge_worker_outputs(a.work_dir)
        return
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        raise ValueError('--shard-index must be in [0, --num-shards)')
    root = Path(a.work_dir); (root / 'masked_images').mkdir(parents=True, exist_ok=True); (root / 'figures').mkdir(exist_ok=True)
    thumbnail_cache_dir = root / 'thumbnail_images'
    if a.intervention_mode == 'thumbnail_frame':
        thumbnail_cache_dir.mkdir(exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    # Explicitly lock this experiment to uncompressed, full history.
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE'] = '0'; os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES'] = str(a.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE'] = '0'; os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS'] = '1'; os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES'] = str(a.history_steps)
    from vlmeval.dataset import build_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dataset = build_dataset(a.data); data = getattr(dataset, 'data', None)
    if data is None: raise RuntimeError(f'Cannot build dataset {a.data}')
    direct_success, outcomes = (None, {}) if not a.success_manifest else read_success_manifest(a.success_manifest)
    successful_episodes = direct_success if direct_success is not None else successful_trajectory_keys(data, outcomes)
    successful_ids = {sid for sid, ok in outcomes.items() if ok}
    if a.outcome_scope == 'success' and not a.success_manifest:
        raise RuntimeError('--success-manifest is required when --outcome-scope success.')
    if a.outcome_scope == 'success' and direct_success is None and a.success_scope == 'trajectory' and not successful_episodes:
        raise RuntimeError('No fully successful trajectories found. Provide successful_trajectory_keys, use a free-run file covering every step, or explicitly choose --success-scope step.')
    if a.outcome_scope == 'success' and direct_success is None and a.success_scope == 'step' and not successful_ids:
        raise RuntimeError('No successful free-run decision points found in the supplied manifest.')
    processor = AutoProcessor.from_pretrained(a.model)
    if a.intervention_mode == 'token_prune':
        # This decoder contains the repository's DeepStack/KV-safe physical
        # sequence surgery. It is dormant unless the experiment publishes a
        # region-specific drop list for this forward pass.
        from vlmeval.vlm.qwen3_vl.modeling_qwen3_vl_attn_prune import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            a.model, torch_dtype='auto', device_map='auto', attn_implementation='sdpa').eval()
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype='auto', device_map='auto', attn_implementation='sdpa').eval()
    image_id = getattr(model.config, 'image_token_id', None)
    if image_id is None: raise RuntimeError('Model config lacks image_token_id.')
    merge = getattr(getattr(model.config, 'vision_config', None), 'spatial_merge_size', 1)
    complete_history = complete_history_candidates(data, a.history_steps)
    if a.outcome_scope == 'all':
        eligible = complete_history
    else:
        eligible = [i for i in complete_history
                    if (str(data.iloc[i].get('_trajectory_key', data.iloc[i].get('episode_id', data.iloc[i].get('task_id', '')))) in successful_episodes
                        if direct_success is not None or a.success_scope == 'trajectory'
                        else str(data.iloc[i].get('index', i)) in successful_ids)]
    if a.sample_ids:
        selected = explicit_sample_indices(data, a.sample_ids)
    elif a.all_eligible:
        selected = list(eligible)
    else:
        selected = sample_indices_by_task(data, eligible, a.num_samples, a.sample_by, a.samples_per_task, a.seed)
    if a.sample_ids: selected = [i for i in selected if i in eligible]
    selected_before_sharding = len(selected)
    # Keep all decision points from one task/trajectory on the same worker for
    # reproducibility and balanced IO, independently of PYTHONHASHSEED.
    def shard_for(row, fallback):
        key = str(row.get('_trajectory_key', row.get('task_id', row.get('episode_id', row.get('index', fallback)))))
        return zlib.crc32(key.encode('utf-8')) % a.num_shards
    selected = [i for i in selected if shard_for(data.iloc[i], i) == a.shard_index]
    metadata = vars(a) | {'successful_episodes': len(successful_episodes), 'successful_decision_points': len(successful_ids), 'success_source': ('not_used_for_selection' if a.outcome_scope == 'all' and not a.success_manifest else ('explicit_trajectory_keys' if direct_success is not None else ('all_steps_free_run' if a.success_scope == 'trajectory' else 'individual_free_run_success'))), 'eligible_complete_history_rows': len(complete_history), 'eligible_rows': len(eligible), 'selected_before_sharding': selected_before_sharding, 'selected_rows': len(selected),
                          'full_history_only': a.intervention_mode == 'pixel_mask',
                          'compression_or_pruning': a.intervention_mode != 'pixel_mask',
                          'feature_definition': 'mean Qwen3-VL post-merge visual-encoder feature over each region',
                          'utility_definition': 'teacher-forced gold action log likelihood difference after the selected input intervention',
                          'intervention_definition': {
                              'pixel_mask': f'source-image region replaced with {a.mask_mode}',
                              'token_prune': f'post-merge visual tokens in selected region physically removed at decoder layer {a.token_prune_layer}',
                              'drop_frame': 'full source history image removed from prompt, including its visual tokens',
                              'thumbnail_frame': f'full source history image replaced in place by a {a.thumbnail_long_edge}px-long-edge deployment thumbnail',
                          }[a.intervention_mode]}
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    points, samples = [], []
    out_jsonl = root / 'regions.jsonl'; out_jsonl.write_text('')
    progress_enabled = a.progress == '1'
    try:
        from tqdm.auto import tqdm
    except ModuleNotFoundError:
        tqdm = None
    prefix = f'worker {a.shard_index + 1}/{a.num_shards}'
    sample_bar = tqdm(total=len(selected), desc=f'{prefix} samples', unit='task', dynamic_ncols=True, disable=not progress_enabled) if tqdm else None
    # This is exact when a region cap is active; otherwise it is deliberately
    # indeterminate because each UI resolution creates a different grid.
    if a.intervention_mode in {'drop_frame', 'thumbnail_frame'}:
        expected_regions = len(selected) * a.history_steps
    else:
        expected_regions = len(selected) * a.history_steps * a.max_regions_per_frame if a.max_regions_per_frame > 0 else None
    region_bar = tqdm(total=expected_regions, desc=f'{prefix} masks', unit='region', dynamic_ncols=True, disable=not progress_enabled) if tqdm else None
    for idx in selected:
        row = data.iloc[idx]; sid = str(row.get('index', idx)); ep = str(row.get('episode_id', row.get('task_id', row.get('_trajectory_key', sid))))
        outcome = ('success' if outcomes.get(sid) is True else 'failure' if outcomes.get(sid) is False else 'unknown')
        sample = {'sample_id': sid, 'episode_id': ep, 'outcome': outcome, 'status': 'skipped'}
        try:
            gold, why = gold_from_row(row)
            if why: raise ValueError(why)
            built = dataset.build_prompt(row); messages = built[0] if isinstance(built, tuple) else built
            records = image_items(messages); histories = [x for x in records if x['kind'] == 'history']
            if len(histories) < a.history_steps: raise ValueError(f'incomplete_history: {len(histories)} < {a.history_steps}')
            prompt, full, _ = make_inputs(processor, messages, gold); prompt_len = int(prompt['input_ids'].shape[1])
            baseline, token_count, _ = action_logprob(model, full, prompt_len, a.action_score)
            segments = locate_visual_segments(full['input_ids'][0], image_id, grid_rows(full.get('image_grid_thw'), merge), records)
            hist_segments = [x for x in segments if x['kind'] == 'history']
            feature_grids = visual_features(model, full, segments)
            hist_features = [feature_grids[segments.index(x)] for x in hist_segments]
            sample.update(status='ok', gold_action=gold, baseline_logprob=baseline, action_token_count=token_count, regions=0)
            rng = np.random.default_rng(a.seed + idx)
            for frame_i, (seg, features) in enumerate(zip(hist_segments, hist_features)):
                specs = region_specs(seg, scales, a.max_regions_per_frame, rng)
                if a.intervention_mode in {'drop_frame', 'thumbnail_frame'}:
                    # A real frame removal/replacement has a well-defined visual-token
                    # interpretation only for the global region.  Retain the legacy
                    # pixel-mask path for local patch interventions.
                    specs = [region for region in specs if region['region_type'] == 'global']
                size = Image.open(seg['path']).size
                for region_i, region in enumerate(specs):
                    box = pixel_box(region, seg, size)
                    intervention_meta = {}
                    if a.intervention_mode == 'pixel_mask':
                        masked = root / 'masked_images' / f'{sid}_h{frame_i}_r{region_i}.png'
                        mask_image(seg['path'], box, masked, a.mask_mode)
                        masked_messages = make_masked_messages(messages, seg['path'], masked)
                    elif a.intervention_mode == 'token_prune':
                        # Keep the pixels and processor input unchanged. The
                        # custom decoder removes only these exact post-merge
                        # visual positions, using the same physical sequence
                        # surgery as the repository's HistPrune path.
                        masked_messages = messages
                        intervention_meta = {
                            'pruned_visual_tokens': len(region_token_positions(seg, region)),
                            'token_prune_layer': a.token_prune_layer,
                        }
                    elif a.intervention_mode == 'drop_frame':
                        masked_messages = make_dropped_frame_messages(messages, seg['path'])
                    else:
                        thumbnail_item, intervention_meta = deployment_thumbnail_item(
                            seg['path'], thumbnail_cache_dir, sid, frame_i, a.thumbnail_long_edge)
                        masked_messages = make_thumbnail_frame_messages(messages, seg['path'], thumbnail_item)
                    mprompt, mfull, _ = make_inputs(processor, masked_messages, gold)
                    intervened_prompt_len = int(mprompt['input_ids'].shape[1])
                    if a.intervention_mode == 'pixel_mask' and intervened_prompt_len != prompt_len:
                        raise ValueError('pixel masking unexpectedly changed prompt token length')
                    if a.intervention_mode == 'token_prune':
                        if intervened_prompt_len != prompt_len:
                            raise RuntimeError('token_prune must retain the original processor prompt')
                        masked_score, _, score_meta = action_logprob(
                            model, mfull, prompt_len, a.action_score,
                            token_drop_positions=region_token_positions(seg, region),
                            token_prune_layer=a.token_prune_layer,
                        )
                        intervened_prompt_len = int(score_meta['effective_prompt_tokens'])
                        intervention_meta.update(score_meta)
                        if int(score_meta['pruned_visual_tokens']) != len(region_token_positions(seg, region)):
                            raise RuntimeError('custom decoder did not prune exactly the requested region tokens')
                    else:
                        masked_score, _, score_meta = action_logprob(model, mfull, intervened_prompt_len, a.action_score)
                    red = None if frame_i == 0 else cosine(region_feature(features, region), corresponding_feature(hist_features[frame_i - 1], region, seg))
                    rec = {'sample_id': sid, 'episode_id': ep, 'outcome': outcome, 'history_step': seg['history_step'], 'relative_step': frame_i - len(hist_segments),
                           'image_path': seg['path'], 'grid_box_yxyx': [region['row0'], region['col0'], region['row1'], region['col1']], 'pixel_box_xyxy': box,
                           'scale': region['scale'], 'region_type': region['region_type'], 'redundancy': red,
                           'utility': baseline - masked_score, 'baseline_logprob': baseline, 'masked_logprob': masked_score,
                           'intervention_mode': a.intervention_mode, 'baseline_prompt_tokens': prompt_len,
                           'intervened_prompt_tokens': intervened_prompt_len, **intervention_meta}
                    points.append(rec); sample['regions'] += 1
                    with out_jsonl.open('a') as fh: fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    if region_bar: region_bar.update(1)
        except Exception as exc:
            sample['skip_reason'] = f'{type(exc).__name__}: {exc}'
        samples.append(sample)
        if sample_bar: sample_bar.update(1)
    if sample_bar: sample_bar.close()
    if region_bar: region_bar.close()
    with (root / 'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for rec in samples: fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
    with (root / 'regions.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(points)
    save_figures(root, points)
    valid = [x for x in points if x['redundancy'] is not None]
    summary = {'samples_ok': sum(x['status'] == 'ok' for x in samples), 'samples_skipped': sum(x['status'] != 'ok' for x in samples),
               'regions_total': len(points), 'regions_with_redundancy': len(valid),
               'mean_redundancy': float(np.mean([x['redundancy'] for x in valid])) if valid else None,
               'mean_utility': float(np.mean([x['utility'] for x in valid])) if valid else None,
               'outputs': {'points_jsonl': 'regions.jsonl', 'points_csv': 'regions.csv', 'density_figure': 'figures/redundancy_utility_density.png'}}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__': main()
