#!/usr/bin/env python3
"""Causal redundancy--utility study for full-history Qwen3-VL trajectories.

This is an offline, opt-in experiment.  It never patches the model, dataset, or
normal inference path.  For each selected *successful trajectory* it computes
visual-feature redundancy between corresponding regions of adjacent history
screenshots, then measures causal utility by masking one source-image region
and teacher-forcing the gold next action.
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


def action_logprob(model, full, prompt_len, score_mode):
    batch = device_inputs(full, model)
    with torch.no_grad():
        logits = model(**batch, use_cache=False, return_dict=True).logits[0]
    target = batch['input_ids'][0, prompt_len:]
    if not len(target): raise ValueError('gold action tokenized to an empty continuation')
    # Logits at position p predict token p+1; the prompt-final logit predicts gold[0].
    selected = logits[prompt_len - 1:prompt_len - 1 + len(target)].float()
    value = torch.log_softmax(selected, -1).gather(1, target[:, None]).sum().item()
    return float(value / len(target) if score_mode == 'mean_logprob' else value), int(len(target))


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


def make_masked_messages(messages, target_path, masked_path):
    answer = []
    for item in messages:
        copied = dict(item)
        if copied.get('type') == 'image' and str(copied.get('value', '')).removeprefix('file://') == target_path:
            copied['value'] = str(masked_path)
        answer.append(copied)
    return answer


def save_figures(root, points):
    import matplotlib.pyplot as plt
    valid = [x for x in points if x.get('redundancy') is not None]
    if not valid:
        (root / 'figures' / 'NO_COMPARABLE_REGIONS.txt').write_text('No region has a preceding history frame.\n')
        return
    x = np.asarray([p['redundancy'] for p in valid]); y = np.asarray([p['utility'] for p in valid])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    density = ax.hexbin(x, y, gridsize=35, bins='log', mincnt=1, cmap='Greys', alpha=.82)
    fig.colorbar(density, ax=ax, label='log10 region count')
    palette = {'global': '#d62728', '4': '#1f77b4', '8': '#2ca02c', '1': '#9467bd'}
    for label in sorted({p['scale'] for p in valid}, key=str):
        q = [p for p in valid if p['scale'] == label]
        ax.scatter([p['redundancy'] for p in q], [p['utility'] for p in q], s=9, alpha=.38,
                   color=palette.get(label, '#ff7f0e'), label=('global/full-frame' if label == 'global' else f'{label}x{label} token region'))
    ax.axvline(0, color='black', lw=.6); ax.axhline(0, color='black', lw=.6)
    ax.set(xlabel='Redundancy: cosine visual-feature similarity to preceding history frame',
           ylabel='Utility: log p(gold action | H) - log p(gold action | H \\ region)')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(root / 'figures' / 'redundancy_utility_density.png', dpi=220); plt.close(fig)


FIELDS = ['sample_id','episode_id','outcome','history_step','relative_step','image_path','grid_box_yxyx','pixel_box_xyxy','scale','region_type','redundancy','utility','baseline_logprob','masked_logprob']


def merge_worker_outputs(root):
    """Create the single analysis table/figure after all GPU workers finish."""
    root = Path(root)
    workers = sorted((root / 'workers').glob('worker_*'))
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
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    with (root / 'regions.jsonl').open('w', encoding='utf-8') as fh:
        for point in points: fh.write(json.dumps(point, ensure_ascii=False) + '\n')
    with (root / 'per_sample.jsonl').open('w', encoding='utf-8') as fh:
        for sample in samples: fh.write(json.dumps(sample, ensure_ascii=False) + '\n')
    with (root / 'regions.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS); writer.writeheader(); writer.writerows(points)
    save_figures(root, points)
    valid = [x for x in points if x.get('redundancy') is not None]
    summary = {'workers_merged': len(workers), 'samples_ok': sum(x['status'] == 'ok' for x in samples),
               'samples_skipped': sum(x['status'] != 'ok' for x in samples), 'regions_total': len(points),
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
                          'full_history_only': True, 'compression_or_pruning': 'disabled',
                          'feature_definition': 'mean Qwen3-VL post-merge visual-encoder feature over each region',
                          'utility_definition': 'teacher-forced gold action log likelihood difference after source-pixel masking'}
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
            baseline, token_count = action_logprob(model, full, prompt_len, a.action_score)
            segments = locate_visual_segments(full['input_ids'][0], image_id, grid_rows(full.get('image_grid_thw'), merge), records)
            hist_segments = [x for x in segments if x['kind'] == 'history']
            feature_grids = visual_features(model, full, segments)
            hist_features = [feature_grids[segments.index(x)] for x in hist_segments]
            sample.update(status='ok', gold_action=gold, baseline_logprob=baseline, action_token_count=token_count, regions=0)
            rng = np.random.default_rng(a.seed + idx)
            for frame_i, (seg, features) in enumerate(zip(hist_segments, hist_features)):
                specs = region_specs(seg, scales, a.max_regions_per_frame, rng)
                size = Image.open(seg['path']).size
                for region_i, region in enumerate(specs):
                    box = pixel_box(region, seg, size); masked = root / 'masked_images' / f'{sid}_h{frame_i}_r{region_i}.png'
                    mask_image(seg['path'], box, masked, a.mask_mode)
                    masked_messages = make_masked_messages(messages, seg['path'], masked)
                    mprompt, mfull, _ = make_inputs(processor, masked_messages, gold)
                    if int(mprompt['input_ids'].shape[1]) != prompt_len: raise ValueError('masking unexpectedly changed prompt token length')
                    masked_score, _ = action_logprob(model, mfull, prompt_len, a.action_score)
                    red = None if frame_i == 0 else cosine(region_feature(features, region), corresponding_feature(hist_features[frame_i - 1], region, seg))
                    rec = {'sample_id': sid, 'episode_id': ep, 'outcome': outcome, 'history_step': seg['history_step'], 'relative_step': frame_i - len(hist_segments),
                           'image_path': seg['path'], 'grid_box_yxyx': [region['row0'], region['col0'], region['row1'], region['col1']], 'pixel_box_xyxy': box,
                           'scale': region['scale'], 'region_type': region['region_type'], 'redundancy': red,
                           'utility': baseline - masked_score, 'baseline_logprob': baseline, 'masked_logprob': masked_score}
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
