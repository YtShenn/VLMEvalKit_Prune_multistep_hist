#!/usr/bin/env python3
import argparse
import json
import os
import time
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

from eval_android_control_attn_top4 import (
    AttentionCapture,
    COORD_PROMPT,
    TemporaryDecoderLayerSlice,
    _build_prompt,
    _extract_image_path,
    _get_gt_bbox,
    _make_grid9_prompt_text,
    _mean,
    _normalize_adapter_path,
    _resolve_core_model,
    _safe_literal_eval,
    _to_bbox_norm,
    aggregate_grid9_attention_scores,
    bbox_center_norm_xyxy,
    bbox_to_grid9,
    infer_visual_grid_size,
    parse_color,
    prepare_multimodal_prompt,
    resolve_language_model,
    resolve_spatial_merge_size,
    select_query_indices,
    topk_grids_from_ranking,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import AndroidControlCurated  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate AndroidControl attention top4 recall for all decoder layers and save summary + plot.'
    )
    parser.add_argument('--base_model', type=str, default='/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct')
    parser.add_argument('--adapter_path', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='AndroidControl_Curated_High_Task_Improved')
    parser.add_argument('--output_dir', type=str, default='OUTPUT/outputs_qwen3vl_android_control_attn_top4_all_layers')
    parser.add_argument('--subset_limit', type=int, default=0)
    parser.add_argument('--decoder_layers_to_run', type=int, default=16)
    parser.add_argument('--topk_eval', type=int, default=4)
    parser.add_argument('--attn_query_chunk_size', type=int, default=128)
    parser.add_argument(
        '--prompt_mode',
        type=str,
        default='grid9_coord',
        choices=['benchmark', 'step_instruction_only', 'grid9_coord'],
    )
    parser.add_argument('--only_click_longpress', action='store_true')
    parser.add_argument('--line_width', type=int, default=1)
    parser.add_argument('--line_color', type=str, default='255,64,64')
    parser.add_argument('--print_per_sample', action='store_true')
    parser.add_argument('--print_timing', action='store_true')
    return parser.parse_args()


def _build_layer_metric_dict(layer_indices: List[int]) -> Dict[int, Dict[str, List[float]]]:
    return {
        int(layer_idx): {
            'hit_top1': [],
            'hit_top2': [],
            'hit_top3': [],
            'hit_top4': [],
            'hit_topk': [],
        }
        for layer_idx in layer_indices
    }


def _make_record_base(args, sample_index: int, img_path: str, gt_action: str, prompt_style: str, prompt_text: str, line) -> dict:
    return {
        'dataset_name': args.dataset,
        'subset': args.dataset,
        'sample_index': sample_index,
        'image_path': img_path,
        'instruction': str(line.get('instruction', '')),
        'step_instruction': str(line.get('step_instruction', '')),
        'history': str(line.get('history', '')),
        'prompt_text': prompt_text,
        'prompt_mode': prompt_style,
        'coord_prompt': COORD_PROMPT if prompt_style == 'grid9_coord' else None,
        'gt_action': gt_action,
    }


def _write_plot(output_path: str, layer_summaries: List[dict]) -> None:
    if plt is None:
        return
    xs = [int(item['layer_number_1based']) for item in layer_summaries]
    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111)
    for key, label in (
        ('recall_top1', 'Top1 Recall'),
        ('recall_top2', 'Top2 Recall'),
        ('recall_top3', 'Top3 Recall'),
        ('recall_top4', 'Top4 Recall'),
    ):
        ys = [float(item.get(key, 0.0)) for item in layer_summaries]
        ax.plot(xs, ys, marker='o', linewidth=2, label=label)
    ax.set_xlabel('Layer Number (1-based)')
    ax.set_ylabel('Recall')
    ax.set_title('AndroidControl Attention Grid Recall by Layer')
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    wall_clock_start = time.perf_counter()
    line_color = parse_color(args.line_color)

    adapter_path = _normalize_adapter_path(args.adapter_path)
    if adapter_path is not None and PeftModel is None:
        raise RuntimeError('adapter_path was provided but `peft` is not installed in the active env.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    model = PeftModel.from_pretrained(base, adapter_path) if adapter_path is not None else base
    model = model.to(device).eval()

    core_model = _resolve_core_model(model)
    language_model = resolve_language_model(model)
    spatial_merge_size = resolve_spatial_merge_size(model)
    total_hidden_layers = int(language_model.config.num_hidden_layers)
    decoder_layers_to_run = min(max(1, int(args.decoder_layers_to_run)), total_hidden_layers)
    attn_layers = list(range(decoder_layers_to_run))

    ds = AndroidControlCurated(dataset=args.dataset)
    n = len(ds.data)
    if args.subset_limit > 0:
        n = min(n, args.subset_limit)

    per_sample_records: List[dict] = []
    sample_times_sec: List[float] = []
    processor_times_sec: List[float] = []
    visual_encode_times_sec: List[float] = []
    llm_forward_times_sec: List[float] = []
    topk_times_sec: List[float] = []
    layer_metrics = _build_layer_metric_dict(attn_layers)
    num_eligible_for_prune = 0
    num_valid_for_recall = 0
    num_skipped_non_click = 0
    num_skipped_no_bbox = 0
    num_click_longpress_total = 0
    num_click_longpress_with_bbox = 0

    with TemporaryDecoderLayerSlice(language_model, decoder_layers_to_run):
        with AttentionCapture(language_model, attn_layers, query_chunk_size=args.attn_query_chunk_size) as capture:
            for i in tqdm(range(n), desc=f'android attn top4 all-layers {args.dataset}'):
                t0 = time.perf_counter()
                line = ds.data.iloc[i]
                sample_index = int(line.get('index', i + 1))
                gt_action = str(line.get('gt_action', '')).strip().lower()
                gt_bbox = _get_gt_bbox(line)
                eligible = int((not args.only_click_longpress) or gt_action in ('click', 'long_press'))

                if gt_action in ('click', 'long_press'):
                    num_click_longpress_total += 1
                    if gt_bbox is not None:
                        num_click_longpress_with_bbox += 1

                msgs, prompt_style, render_payload = _build_prompt(
                    ds, line, prompt_mode=args.prompt_mode, line_width=args.line_width, line_color=line_color
                )
                img_path = _extract_image_path(msgs)
                prompt_text = ''
                for item in msgs:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        prompt_text = str(item.get('value', ''))
                        break

                if img_path is None:
                    record = _make_record_base(args, sample_index, None, gt_action, prompt_style, prompt_text, line)
                    record.update({'eligible_for_prune': 0, 'error': 'missing_image_path'})
                    per_sample_records.append(record)
                    continue

                if render_payload is not None:
                    raw_img = render_payload['raw_img']
                    model_image = render_payload['model_image']
                    messages = render_payload['messages']
                    image_for_processor = model_image
                else:
                    raw_img = Image.open(img_path).convert('RGB')
                    image_for_processor = raw_img
                    messages = [{'role': 'user', 'content': []}]
                    for item in msgs:
                        if item['type'] == 'image':
                            messages[0]['content'].append({'type': 'image', 'image': raw_img})
                        elif item['type'] == 'text':
                            messages[0]['content'].append({'type': 'text', 'text': item['value']})

                chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                t_proc0 = time.perf_counter()
                inputs = processor(text=[chat_text], images=[image_for_processor], return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                t_proc1 = time.perf_counter()

                capture.reset()
                t_vis0 = time.perf_counter()
                inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids = prepare_multimodal_prompt(
                    core_model,
                    inputs,
                )
                t_vis1 = time.perf_counter()

                if visual_pos_masks is None:
                    record = _make_record_base(args, sample_index, img_path, gt_action, prompt_style, prompt_text, line)
                    record.update({'eligible_for_prune': 0, 'error': 'visual_pos_masks_is_none'})
                    per_sample_records.append(record)
                    continue

                t_llm0 = time.perf_counter()
                with torch.no_grad():
                    _ = language_model(
                        attention_mask=inputs.get('attention_mask'),
                        position_ids=position_ids,
                        inputs_embeds=inputs_embeds,
                        use_cache=False,
                        visual_pos_masks=visual_pos_masks,
                        deepstack_visual_embeds=deepstack_visual_embeds,
                        return_dict=True,
                    )
                t_llm1 = time.perf_counter()

                selection = capture.selection if capture.selection is not None else select_query_indices(capture)
                if selection is None:
                    record = _make_record_base(args, sample_index, img_path, gt_action, prompt_style, prompt_text, line)
                    record.update({'eligible_for_prune': 0, 'error': 'query_selection_is_none'})
                    per_sample_records.append(record)
                    continue

                grid_size = infer_visual_grid_size(
                    inputs['image_grid_thw'].detach().cpu(),
                    spatial_merge_size=spatial_merge_size,
                )
                expected_tokens = int(grid_size[0] * grid_size[1])
                t_topk0 = time.perf_counter()

                gt_bbox_norm = None
                gt_grid = None
                gt_click_xy_norm = None
                if gt_bbox is not None:
                    w, h = raw_img.size
                    gt_bbox_norm = [float(x) for x in _to_bbox_norm(gt_bbox, w, h)]
                    gt_grid = int(bbox_to_grid9(gt_bbox_norm))
                    gt_cx, gt_cy = bbox_center_norm_xyxy(gt_bbox_norm)
                    gt_click_xy_norm = [float(gt_cx), float(gt_cy)]

                if eligible:
                    num_eligible_for_prune += 1
                    if gt_bbox is None:
                        num_skipped_no_bbox += 1
                    else:
                        num_valid_for_recall += 1
                else:
                    num_skipped_non_click += 1

                per_layer_predictions = []
                for layer_idx in attn_layers:
                    attn_weights = capture.layer_attn.get(layer_idx)
                    if attn_weights is None:
                        per_layer_predictions.append(
                            {
                                'layer_index': int(layer_idx),
                                'layer_number_1based': int(layer_idx + 1),
                                'error': 'attn_weights_is_none',
                            }
                        )
                        continue
                    if attn_weights.numel() != expected_tokens:
                        raise ValueError(
                            f'Attention map/token grid mismatch at layer {layer_idx}: '
                            f'{attn_weights.numel()} weights vs {expected_tokens} grid cells '
                            f'(grid_size={grid_size}, spatial_merge_size={spatial_merge_size}).'
                        )
                    attn_grid_ranking = aggregate_grid9_attention_scores(attn_weights, grid_size)
                    pred_grids_topk = [int(x) for x in topk_grids_from_ranking(attn_grid_ranking, args.topk_eval)]

                    hit_top1 = None
                    hit_top2 = None
                    hit_top3 = None
                    hit_top4 = None
                    hit_topk = None
                    if eligible and gt_grid is not None:
                        pred_grid = int(pred_grids_topk[0]) if pred_grids_topk else -1
                        hit_top1 = int(pred_grid == gt_grid)
                        hit_top2 = int(gt_grid in pred_grids_topk[:2]) if len(pred_grids_topk) >= 1 else 0
                        hit_top3 = int(gt_grid in pred_grids_topk[:3]) if len(pred_grids_topk) >= 1 else 0
                        hit_top4 = int(gt_grid in pred_grids_topk[:4]) if len(pred_grids_topk) >= 1 else 0
                        hit_topk = int(gt_grid in pred_grids_topk[: int(args.topk_eval)])
                        layer_metrics[layer_idx]['hit_top1'].append(float(hit_top1))
                        layer_metrics[layer_idx]['hit_top2'].append(float(hit_top2))
                        layer_metrics[layer_idx]['hit_top3'].append(float(hit_top3))
                        layer_metrics[layer_idx]['hit_top4'].append(float(hit_top4))
                        layer_metrics[layer_idx]['hit_topk'].append(float(hit_topk))

                    per_layer_predictions.append(
                        {
                            'layer_index': int(layer_idx),
                            'layer_number_1based': int(layer_idx + 1),
                            'pred_grid': int(pred_grids_topk[0]) if pred_grids_topk else -1,
                            'pred_grids_topk': pred_grids_topk,
                            'hit_top1': hit_top1,
                            'hit_top2': hit_top2,
                            'hit_top3': hit_top3,
                            'hit_top4': hit_top4,
                            'hit_topk': hit_topk,
                        }
                    )
                t_topk1 = time.perf_counter()

                processor_sec = float(t_proc1 - t_proc0)
                visual_encode_sec = float(t_vis1 - t_vis0)
                llm_forward_sec = float(t_llm1 - t_llm0)
                topk_select_sec = float(t_topk1 - t_topk0)
                elapsed_sec = float(time.perf_counter() - t0)

                record = _make_record_base(args, sample_index, img_path, gt_action, prompt_style, prompt_text, line)
                record.update(
                    {
                        'eligible_for_prune': int(eligible),
                        'gt_bbox': gt_bbox,
                        'gt_bbox_norm': gt_bbox_norm,
                        'gt_grid': gt_grid,
                        'gt_click_xy_norm': gt_click_xy_norm,
                        'decoder_layers_to_run': int(decoder_layers_to_run),
                        'visual_grid_size': [int(grid_size[0]), int(grid_size[1])],
                        'num_query_tokens': int(selection.q_indices.numel()),
                        'processor_sec': processor_sec,
                        'visual_encode_sec': visual_encode_sec,
                        'llm_forward_sec': llm_forward_sec,
                        'topk_select_sec': topk_select_sec,
                        'elapsed_sec': elapsed_sec,
                        'per_layer_predictions': per_layer_predictions,
                    }
                )
                if not eligible:
                    record['skip_reason'] = 'non_click_non_longpress'
                elif gt_bbox is None:
                    record['skip_reason'] = 'eligible_but_no_bbox'
                per_sample_records.append(record)

                sample_times_sec.append(elapsed_sec)
                processor_times_sec.append(processor_sec)
                visual_encode_times_sec.append(visual_encode_sec)
                llm_forward_times_sec.append(llm_forward_sec)
                topk_times_sec.append(topk_select_sec)

                if args.print_timing:
                    print(
                        f"[Timing] sample_index={sample_index} total_s={elapsed_sec:.6f} "
                        f"processor_s={processor_sec:.6f} visual_encode_s={visual_encode_sec:.6f} "
                        f"llm_forward_s={llm_forward_sec:.6f} topk_select_s={topk_select_sec:.6f}",
                        flush=True,
                    )
                if args.print_per_sample:
                    first_layer = per_layer_predictions[0] if per_layer_predictions else {}
                    last_layer = per_layer_predictions[-1] if per_layer_predictions else {}
                    print(
                        json.dumps(
                            {
                                'sample_index': sample_index,
                                'gt_action': gt_action,
                                'gt_grid': gt_grid,
                                'eligible_for_prune': int(eligible),
                                'first_layer_pred_grids_topk': first_layer.get('pred_grids_topk'),
                                'last_layer_pred_grids_topk': last_layer.get('pred_grids_topk'),
                                'num_layers_scanned': len(per_layer_predictions),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    layer_summaries = []
    for layer_idx in attn_layers:
        metrics = layer_metrics[layer_idx]
        layer_summaries.append(
            {
                'layer_index': int(layer_idx),
                'layer_number_1based': int(layer_idx + 1),
                'num_valid_for_recall': int(len(metrics['hit_topk'])),
                'recall_top1': _mean(metrics['hit_top1']),
                'recall_top2': _mean(metrics['hit_top2']),
                'recall_top3': _mean(metrics['hit_top3']),
                'recall_top4': _mean(metrics['hit_top4']),
                'recall_topk': _mean(metrics['hit_topk']),
            }
        )

    plot_path = os.path.join(args.output_dir, 'layer_recall_plot.png')
    _write_plot(plot_path, layer_summaries)

    summary = {
        'dataset': args.dataset,
        'base_model': args.base_model,
        'adapter_path': adapter_path or 'none',
        'prompt_mode': args.prompt_mode,
        'decoder_layers_to_run': int(decoder_layers_to_run),
        'decoder_layers_total': int(total_hidden_layers),
        'attn_mode': 'instruction_to_image',
        'layerwise_recall': layer_summaries,
        'topk_eval': int(args.topk_eval),
        'attn_query_chunk_size': int(args.attn_query_chunk_size),
        'num_records': len(per_sample_records),
        'num_click_longpress_total': int(num_click_longpress_total),
        'num_click_longpress_with_bbox': int(num_click_longpress_with_bbox),
        'num_click_longpress_without_bbox': int(num_click_longpress_total - num_click_longpress_with_bbox),
        'num_eligible_for_prune': int(num_eligible_for_prune),
        'num_valid_for_recall': int(num_valid_for_recall),
        'num_skipped_non_click': int(num_skipped_non_click),
        'num_skipped_no_bbox': int(num_skipped_no_bbox),
        'avg_sample_time_sec': _mean(sample_times_sec),
        'total_sample_time_sec': float(sum(sample_times_sec)) if sample_times_sec else 0.0,
        'avg_processor_sec': _mean(processor_times_sec),
        'total_processor_sec': float(sum(processor_times_sec)) if processor_times_sec else 0.0,
        'avg_visual_encode_sec': _mean(visual_encode_times_sec),
        'total_visual_encode_sec': float(sum(visual_encode_times_sec)) if visual_encode_times_sec else 0.0,
        'avg_llm_forward_sec': _mean(llm_forward_times_sec),
        'total_llm_forward_sec': float(sum(llm_forward_times_sec)) if llm_forward_times_sec else 0.0,
        'avg_topk_select_sec': _mean(topk_times_sec),
        'total_topk_select_sec': float(sum(topk_times_sec)) if topk_times_sec else 0.0,
        'num_samples_timed': len(sample_times_sec),
        'wall_clock_total_sec': float(time.perf_counter() - wall_clock_start),
        'plot_path': plot_path if plt is not None else None,
        'plot_backend_available': bool(plt is not None),
    }

    if layer_summaries:
        best_top1 = max(layer_summaries, key=lambda item: item['recall_top1'])
        best_top4 = max(layer_summaries, key=lambda item: item['recall_top4'])
        summary['best_layer_top1'] = best_top1
        summary['best_layer_top4'] = best_top4

    per_sample_path = os.path.join(args.output_dir, 'per_sample.json')
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(per_sample_path, 'w', encoding='utf-8') as f:
        json.dump(per_sample_records, f, ensure_ascii=False, indent=2)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'[saved] per-sample: {per_sample_path}')
    print(f'[saved] summary: {summary_path}')
    if plt is not None:
        print(f'[saved] plot: {plot_path}')


if __name__ == '__main__':
    main()
