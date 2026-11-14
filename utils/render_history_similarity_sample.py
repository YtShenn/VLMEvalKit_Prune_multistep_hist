#!/usr/bin/env python3
"""Render all adjacent-history similarity panels for one stable dataset sample ID.

The token-level panel uses exactly ``1 - novelty`` from
``history_visual_novelty.py``.  The patch panel uses the 4×4-token aggregation
used by ``history_redundancy_visualization.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from textwrap import fill

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.history_attention_diagnostic import (  # noqa: E402
    device_inputs, gold_from_row, image_items, make_inputs,
)
from utils.history_attention_utils import grid_rows, locate_visual_segments  # noqa: E402
from utils.history_redundancy_visualization import (  # noqa: E402
    corresponding, cosine, grid_features, tokenwise_similarity,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample-ids', required=True,
                        help='Comma-separated stable dataset `index` values, in render order, e.g. 8173,319.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--history-steps', type=int, default=4)
    parser.add_argument('--local-scale', type=int, default=4)
    return parser.parse_args()


def frame_metadata(data):
    result = {}
    for _, row in data.iterrows():
        image_path = str(row.get('image_path', row.get('image', '')) or '')
        if not image_path:
            continue
        action, _ = gold_from_row(row)
        result[os.path.abspath(image_path)] = {
            'instruction': str(row.get('instruction', row.get('question', row.get('description', 'Unavailable')))),
            'gt_action': str(action or 'Unavailable'),
        }
    return result


def patch_similarity_map(previous, current, local_scale):
    """Return coarse patch similarity expanded to the current token grid."""
    h, w = current.shape[:2]
    heat = np.full((h, w), np.nan, dtype=np.float32)
    values = []
    for r0 in range(0, h, local_scale):
        for c0 in range(0, w, local_scale):
            r1, c1 = min(h, r0 + local_scale), min(w, c0 + local_scale)
            value = cosine(current[r0:r1, c0:c1].mean((0, 1)),
                           corresponding(previous, r0, r1, c0, c1, h, w))
            heat[r0:r1, c0:c1] = value
            values.append(value)
    return heat, np.asarray(values, dtype=np.float32)


def render_pair(output, pair, metadata):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from PIL import Image

    previous_image = Image.open(pair['previous_path']).convert('RGB')
    current_image = Image.open(pair['current_path']).convert('RGB')
    if current_image.size != previous_image.size:
        current_image = current_image.resize(previous_image.size, Image.Resampling.BILINEAR)
    previous_meta = metadata.get(os.path.abspath(pair['previous_path']), {})
    current_meta = metadata.get(os.path.abspath(pair['current_path']), {})
    previous_instruction = fill(previous_meta.get('instruction', 'Unavailable'), width=38)
    current_instruction = fill(current_meta.get('instruction', 'Unavailable'), width=38)
    previous_gt = fill(previous_meta.get('gt_action', 'Unavailable'), width=38)
    current_gt = fill(current_meta.get('gt_action', 'Unavailable'), width=38)
    # Preserve the red token-similarity encoding, but stop the high-similarity
    # end at pale pink rather than near-white so its contrast is less extreme.
    muted_token_cmap = LinearSegmentedColormap.from_list(
        'token_similarity_no_white', ['#B9553D', '#D87959', '#EEA17F', '#F5C7B8'])

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42}):
        fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.1), constrained_layout=True,
                                 gridspec_kw={'width_ratios': [1, 1, 1, 1.08]})
        axes[0].imshow(previous_image)
        axes[0].set_title(f'Previous frame\nInstruction: {previous_instruction}\nGT: {previous_gt}',
                          fontsize=7, fontweight='bold')
        axes[1].imshow(current_image)
        axes[1].set_title(f'Current frame\nInstruction: {current_instruction}\nGT: {current_gt}',
                          fontsize=7, fontweight='bold')
        axes[2].imshow(current_image)
        patch_im = axes[2].imshow(pair['patch_similarity'], extent=(0, current_image.width, current_image.height, 0),
                                  cmap='Blues_r', vmin=0, vmax=1, alpha=.72, interpolation='nearest')
        axes[2].set_title('Current + 4×4-token patch similarity', fontsize=8, fontweight='bold')
        axes[3].imshow(current_image)
        token_im = axes[3].imshow(pair['token_similarity'], extent=(0, current_image.width, current_image.height, 0),
                                  cmap=muted_token_cmap, vmin=0, vmax=1, alpha=.72, interpolation='nearest')
        axes[3].set_title('Current + token-wise similarity\n(= 1 − novelty)', fontsize=8, fontweight='bold')
        for axis in axes:
            axis.axis('off')
        patch_cbar = fig.colorbar(patch_im, ax=axes[2], fraction=.046, pad=.02)
        patch_cbar.set_label('patch cosine', fontsize=16)
        patch_cbar.ax.tick_params(labelsize=16)
        fig.colorbar(token_im, ax=axes[3], fraction=.046, pad=.02, label='token cosine')
        fig.suptitle(
            f"{pair['label']}  |  global cosine = {pair['global_similarity']:.3f}  |  "
            f"patch cosine (mean / median) = {pair['patch_mean']:.3f} / {pair['patch_median']:.3f}  |  "
            f"token cosine (mean / median) = {pair['token_mean']:.3f} / {pair['token_median']:.3f}",
            fontsize=8.2, fontweight='bold')
        fig.savefig(output.with_suffix('.png'), dpi=350, facecolor='white')
        fig.savefig(output.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)


def main():
    args = parse_args()
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE'] = '0'
    os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES'] = str(args.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE'] = '0'
    os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS'] = '1'
    os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES'] = str(args.history_steps)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from vlmeval.dataset import build_dataset

    dataset = build_dataset(args.data)
    data = dataset.data
    metadata = frame_metadata(data)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype='auto', device_map='auto', attn_implementation='sdpa').eval()
    image_id = model.config.image_token_id
    merge_size = getattr(model.config.vision_config, 'spatial_merge_size', 1)
    requested = [sample_id.strip() for sample_id in args.sample_ids.split(',') if sample_id.strip()]
    if not requested:
        raise ValueError('--sample-ids must contain at least one ID')
    row_by_id = {str(row.get('index', index)): index for index, (_, row) in enumerate(data.iterrows())}
    unknown = [sample_id for sample_id in requested if sample_id not in row_by_id]
    if unknown:
        raise ValueError(f'Unknown dataset sample IDs: {unknown}')
    output_root = Path(args.work_dir) / 'figures'
    for sample_id in requested:
        root = output_root / f'sample_{sample_id}_similarity'
        root.mkdir(parents=True, exist_ok=True)
        row = data.iloc[row_by_id[sample_id]]
        built = dataset.build_prompt(row)
        messages = built[0] if isinstance(built, tuple) else built
        prompt, _, _ = make_inputs(processor, messages, '')
        records = image_items(messages)
        segments = locate_visual_segments(prompt['input_ids'][0], image_id,
            grid_rows(prompt.get('image_grid_thw'), merge_size), records)
        history = [(segment, feature) for segment, feature in zip(segments, grid_features(model, prompt, segments))
                   if segment['kind'] == 'history']
        if len(history) < 2:
            raise ValueError(f'Sample {sample_id}: need at least two history frames, received {len(history)}')
        summary = {'sample_id': sample_id, 'history_pairs': []}
        for position in range(1, len(history)):
            previous_seg, previous = history[position - 1]
            current_seg, current = history[position]
            token = tokenwise_similarity(previous, current).astype(np.float32)
            patch, patch_values = patch_similarity_map(previous, current, args.local_scale)
            pair = {
                'label': f'History {position - len(history)} → {position + 1 - len(history)}',
                'previous_path': previous_seg['path'], 'current_path': current_seg['path'],
                'global_similarity': cosine(previous.mean((0, 1)), current.mean((0, 1))),
                'patch_similarity': patch, 'token_similarity': token,
                'patch_mean': float(patch_values.mean()), 'patch_median': float(np.median(patch_values)),
                'token_mean': float(token.mean()), 'token_median': float(np.median(token)),
            }
            stem = root / f'{position:02d}_history_{position - len(history)}_to_{position + 1 - len(history)}'
            render_pair(stem, pair, metadata)
            summary['history_pairs'].append({key: value for key, value in pair.items()
                                             if key not in {'patch_similarity', 'token_similarity'}} | {'file': str(stem.name)})
        (root / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(root, flush=True)


if __name__ == '__main__':
    main()
