#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
import time
import types
from contextlib import AbstractContextManager
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, repeat_kv

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from vlmeval.dataset import GUIOdyssey  # noqa: E402


COORD_PROMPT = (
    "You are a GUI grounding assistant. You are given a full-screen screenshot with a 3x3 grid overlay "
    "and a user instruction. Predict the target click by outputting the grid cell index first, followed by "
    "the normalized click coordinates. Output exactly in the format: <grid_index> <x> <y>. "
    "Here grid_index must be one integer from 1 to 9, and x and y must be normalized floats in [0, 1]."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate GUIOdyssey attention top4 candidates and save per_sample.json.'
    )
    parser.add_argument('--base_model', type=str, default='/mnt/storage/users/ytshen_data/Qwen3-VL-4B-Instruct')
    parser.add_argument('--adapter_path', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='GUIOdyssey_high_task_split')
    parser.add_argument('--output_dir', type=str, default='OUTPUT/outputs_qwen3vl_gui_odyssey_attn_top4')
    parser.add_argument('--subset_limit', type=int, default=0)
    parser.add_argument('--decoder_layers_to_run', type=int, default=16)
    parser.add_argument('--target_layer_index', type=int, default=-1)
    parser.add_argument('--topk_eval', type=int, default=4)
    parser.add_argument('--attn_query_chunk_size', type=int, default=128)
    parser.add_argument(
        '--prompt_mode',
        type=str,
        default='grid9_coord',
        choices=['benchmark', 'task_only', 'grid9_coord'],
        help='benchmark uses GUIOdyssey current prompt; task_only uses screenshot + task text only; grid9_coord fully mimics ScreenSpotPro grid9 coord prompt style.',
    )
    parser.add_argument('--only_click_longpress', action='store_true')
    parser.add_argument('--line_width', type=int, default=1)
    parser.add_argument('--line_color', type=str, default='255,64,64')
    parser.add_argument('--save_visualizations', action='store_true')
    parser.add_argument('--visualize_every', type=int, default=50)
    parser.add_argument('--visualize_limit', type=int, default=0)
    parser.add_argument('--print_per_sample', action='store_true')
    parser.add_argument('--print_timing', action='store_true')
    return parser.parse_args()


def parse_color(s: str) -> Tuple[int, int, int]:
    vals = [int(x) for x in str(s).split(',')]
    if len(vals) != 3:
        return (255, 64, 64)
    return tuple(vals)


def _normalize_adapter_path(adapter_path: Optional[str]) -> Optional[str]:
    if adapter_path is None:
        return None
    text = str(adapter_path).strip()
    if text == '' or text.lower() in {'none', 'null', 'nil'}:
        return None
    return text


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _safe_literal_eval(val):
    if isinstance(val, (list, tuple, dict)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return val


def _parse_answer(answer: str) -> Tuple[str, Optional[List[float]]]:
    raw = str(answer).strip()
    m = re.search(r'(CLICK|LONG_PRESS|SCROLL|TYPE|PRESS_HOME|PRESS_BACK|PRESS_RECENT|COMPLETE|IMPOSSIBLE)\s*:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    if m:
        action = m.group(1).upper()
        payload = m.group(2).strip()
        if action in ('CLICK', 'LONG_PRESS'):
            point = _safe_literal_eval(payload)
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    return action, [float(point[0]), float(point[1])]
                except Exception:
                    return action, None
        return action, None
    m2 = re.search(r'\b(CLICK|LONG_PRESS|SCROLL|TYPE|PRESS_HOME|PRESS_BACK|PRESS_RECENT|COMPLETE|IMPOSSIBLE)\b', raw, re.IGNORECASE)
    if m2:
        return m2.group(1).upper(), None
    return '', None


def _extract_image_path(msgs: List[dict]) -> Optional[str]:
    for item in msgs:
        if isinstance(item, dict) and item.get('type') == 'image':
            value = item.get('value')
            if value is not None:
                return str(value)
    return None


def bbox_center_norm_xyxy(bbox_norm: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(x) for x in bbox_norm]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_to_grid9(bbox_norm: Sequence[float]) -> int:
    cx, cy = bbox_center_norm_xyxy(bbox_norm)
    col = min(2, max(0, int(cx * 3.0)))
    row = min(2, max(0, int(cy * 3.0)))
    return int(row * 3 + col + 1)


def point_to_grid9(point_xy: Sequence[float], w: int, h: int) -> Tuple[int, List[float]]:
    px, py = [float(x) for x in point_xy[:2]]
    px = max(0.0, min(float(w), px))
    py = max(0.0, min(float(h), py))
    norm = [px / float(w), py / float(h)]
    col = min(2, max(0, int(norm[0] * 3.0)))
    row = min(2, max(0, int(norm[1] * 3.0)))
    return int(row * 3 + col + 1), norm


def _guiodyssey_coord_to_pixel_xy(point_xy, w: int, h: int) -> List[float]:
    px, py = [float(x) for x in point_xy[:2]]
    # GUIOdyssey action coordinates and sam2_bbox are stored in [0, 1000] space.
    if max(abs(px), abs(py)) <= 1000.0 + 1e-6:
        return [px / 1000.0 * float(w), py / 1000.0 * float(h)]
    return [px, py]


def _guiodyssey_bbox_to_pixel_xyxy(bbox_raw, w: int, h: int) -> List[float]:
    bbox = _safe_literal_eval(bbox_raw)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f'Invalid GUIOdyssey bbox: {bbox_raw}')
    x1, y1 = _guiodyssey_coord_to_pixel_xy([bbox[0], bbox[1]], w, h)
    x2, y2 = _guiodyssey_coord_to_pixel_xy([bbox[2], bbox[3]], w, h)
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(1.0, x2)
        y2 = y1 + max(1.0, y2)
    x1 = max(0.0, min(w - 1.0, x1))
    y1 = max(0.0, min(h - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(w), x2))
    y2 = max(y1 + 1.0, min(float(h), y2))
    return [x1, y1, x2, y2]


def _to_bbox_norm(bbox_raw, w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = _guiodyssey_bbox_to_pixel_xyxy(bbox_raw, w, h)
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(1.0, x2)
        y2 = y1 + max(1.0, y2)
    x1 = max(0.0, min(w - 1.0, x1))
    y1 = max(0.0, min(h - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(w), x2))
    y2 = max(y1 + 1.0, min(float(h), y2))
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def draw_grid9_overlay(
    image: Image.Image,
    line_width: int = 1,
    line_color: Tuple[int, int, int] = (255, 64, 64),
) -> Image.Image:
    img = image.convert('RGB')
    w, h = img.size
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x1 = int(round(w / 3.0))
    x2 = int(round(w * 2.0 / 3.0))
    y1 = int(round(h / 3.0))
    y2 = int(round(h * 2.0 / 3.0))
    draw.line([(x1, 0), (x1, h)], fill=line_color, width=line_width)
    draw.line([(x2, 0), (x2, h)], fill=line_color, width=line_width)
    draw.line([(0, y1), (w, y1)], fill=line_color, width=line_width)
    draw.line([(0, y2), (w, y2)], fill=line_color, width=line_width)
    return out


def grid_id_to_bounds(grid_id: int, width: int, height: int) -> Tuple[int, int, int, int]:
    idx = max(1, min(9, int(grid_id))) - 1
    row, col = divmod(idx, 3)
    xs = [0, int(round(width / 3.0)), int(round(width * 2.0 / 3.0)), width]
    ys = [0, int(round(height / 3.0)), int(round(height * 2.0 / 3.0)), height]
    return xs[col], ys[row], xs[col + 1], ys[row + 1]


def _load_font(image_height: int) -> ImageFont.ImageFont:
    font_size = max(14, int(round(image_height * 0.03)))
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size=font_size)
    except Exception:
        return ImageFont.load_default()


def parse_bbox_xyxy_pixels(bbox_raw, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = _guiodyssey_bbox_to_pixel_xyxy(bbox_raw, width, height)
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(1.0, x2)
        y2 = y1 + max(1.0, y2)
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _normalize_weights(weights: torch.Tensor) -> torch.Tensor:
    weights = weights.detach().float().cpu().flatten()
    if weights.numel() == 0:
        return weights
    finite_mask = torch.isfinite(weights)
    if not finite_mask.any():
        return torch.zeros_like(weights)
    weights = weights.clone()
    valid = weights[finite_mask]
    low = float(torch.quantile(valid, 0.70).item())
    high = float(torch.quantile(valid, 0.995).item())
    if high <= low:
        low = float(valid.min().item())
        high = float(valid.max().item())
    weights = weights.clamp(min=low, max=high)
    denom = high - low
    if denom <= 0:
        return torch.zeros_like(weights)
    return ((weights - low) / denom).clamp_(0.0, 1.0).pow_(1.8)


def _draw_heatmap_overlay(
    draw: ImageDraw.ImageDraw,
    image_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    weights: torch.Tensor,
    color: Tuple[int, int, int] = (220, 30, 30),
    max_alpha: int = 170,
) -> None:
    norm = _normalize_weights(weights)
    if norm.numel() == 0:
        return
    grid_h, grid_w = grid_size
    width, height = image_size
    patch_w = width / float(grid_w)
    patch_h = height / float(grid_h)
    for idx, value in enumerate(norm.tolist()):
        row, col = divmod(idx, grid_w)
        alpha = int(round(max_alpha * float(value)))
        if alpha <= 0:
            continue
        x0 = int(round(col * patch_w))
        y0 = int(round(row * patch_h))
        x1 = int(round((col + 1) * patch_w))
        y1 = int(round((row + 1) * patch_h))
        if x1 <= x0:
            x1 = min(width, x0 + 1)
        if y1 <= y0:
            y1 = min(height, y0 + 1)
        draw.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], alpha))


def render_attention_visualization(
    image: Image.Image,
    weights: torch.Tensor,
    grid_size: Tuple[int, int],
    gt_bbox_xyxy: Optional[Tuple[int, int, int, int]],
    ranked_grid_ids: Sequence[int],
) -> Image.Image:
    canvas = image.convert('RGBA').copy()
    overlay = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, mode='RGBA')
    _draw_heatmap_overlay(overlay_draw, canvas.size, grid_size, weights=weights)
    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas, mode='RGBA')
    if gt_bbox_xyxy is not None:
        draw.rectangle(list(gt_bbox_xyxy), outline=(0, 220, 80), width=4)
    font = _load_font(canvas.size[1])
    for rank, grid_id in enumerate(ranked_grid_ids, start=1):
        x1, y1, x2, y2 = grid_id_to_bounds(int(grid_id), canvas.size[0], canvas.size[1])
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=4)
        label = str(rank)
        label_box = draw.textbbox((0, 0), label, font=font)
        label_w = label_box[2] - label_box[0]
        label_h = label_box[3] - label_box[1]
        pad = max(4, label_h // 4)
        bg = [x1 + 4, y1 + 4, x1 + 4 + label_w + pad * 2, y1 + 4 + label_h + pad * 2]
        draw.rectangle(bg, fill=(0, 0, 0, 180))
        draw.text((bg[0] + pad, bg[1] + pad), label, fill=(255, 220, 0), font=font)
    return canvas.convert('RGB')


def build_output_path(output_dir: str, dataset_name: str, sample_index: int, image_path: str, layer_idx: int) -> str:
    save_dir = os.path.join(output_dir, 'visualizations', str(dataset_name).strip() or 'unknown')
    os.makedirs(save_dir, exist_ok=True)
    image_stem = os.path.splitext(os.path.basename(str(image_path)))[0] or 'image'
    return os.path.join(save_dir, f'{image_stem}__idx_{int(sample_index):05d}__layer_{int(layer_idx):02d}.png')


def infer_visual_grid_size(image_grid_thw: torch.Tensor, spatial_merge_size: int = 1) -> Tuple[int, int]:
    row = image_grid_thw[0].tolist()
    if len(row) < 3:
        raise ValueError(f'Unexpected image_grid_thw shape: {image_grid_thw}')
    grid_h = int(row[-2]) // max(1, int(spatial_merge_size))
    grid_w = int(row[-1]) // max(1, int(spatial_merge_size))
    return (grid_h, grid_w)


def aggregate_grid9_attention_scores(attn_weights: torch.Tensor, grid_size: Tuple[int, int]) -> List[dict]:
    grid_h, grid_w = [int(x) for x in grid_size]
    if attn_weights.numel() != grid_h * grid_w:
        raise ValueError(f'Attention numel {attn_weights.numel()} does not match grid size {grid_size}')
    weights_2d = attn_weights.view(grid_h, grid_w)
    row_edges = [0, grid_h / 3.0, grid_h * 2.0 / 3.0, float(grid_h)]
    col_edges = [0, grid_w / 3.0, grid_w * 2.0 / 3.0, float(grid_w)]
    ranking = []
    for grid_id in range(1, 10):
        idx = grid_id - 1
        row, col = divmod(idx, 3)
        r0 = int(round(row_edges[row]))
        r1 = max(r0 + 1, int(round(row_edges[row + 1])))
        c0 = int(round(col_edges[col]))
        c1 = max(c0 + 1, int(round(col_edges[col + 1])))
        score = float(weights_2d[r0:r1, c0:c1].sum().item())
        ranking.append({'grid': int(grid_id), 'score': score})
    ranking.sort(key=lambda item: item['score'], reverse=True)
    return ranking


def topk_grids_from_ranking(ranking: List[dict], topk: int) -> List[int]:
    out = []
    for item in ranking:
        gid = int(item['grid'])
        if 1 <= gid <= 9 and gid not in out:
            out.append(gid)
        if len(out) >= int(topk):
            break
    return out


def _resolve_core_model(model):
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if hasattr(base_model, 'model'):
        return base_model.model
    return base_model


def resolve_language_model(model) -> torch.nn.Module:
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'language_model'):
        return base_model.model.language_model
    if hasattr(base_model, 'language_model'):
        return base_model.language_model
    raise AttributeError('Could not resolve Qwen3-VL language model for attention hooks.')


def resolve_spatial_merge_size(model) -> int:
    base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'visual'):
        return int(getattr(base_model.model.visual, 'spatial_merge_size', 1))
    if hasattr(base_model, 'visual'):
        return int(getattr(base_model.visual, 'spatial_merge_size', 1))
    return 1


def _get_image_feature_outputs(core_model, pixel_values, image_grid_thw):
    try:
        return core_model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
    except TypeError:
        return core_model.get_image_features(pixel_values, image_grid_thw)


def _unpack_image_feature_outputs(image_outputs):
    if hasattr(image_outputs, 'pooler_output'):
        pooler_output = image_outputs.pooler_output
        deepstack_visual_embeds = getattr(image_outputs, 'deepstack_features', None)
        return pooler_output, deepstack_visual_embeds

    if isinstance(image_outputs, (tuple, list)):
        if not image_outputs:
            raise ValueError('Qwen3-VL get_image_features returned an empty tuple/list.')
        pooler_output = image_outputs[0]
        deepstack_visual_embeds = image_outputs[1] if len(image_outputs) > 1 else None
        return pooler_output, deepstack_visual_embeds

    raise TypeError(f'Unsupported image feature output type: {type(image_outputs)!r}')


def prepare_multimodal_prompt(core_model, inputs: Dict[str, torch.Tensor]):
    input_ids = inputs['input_ids']
    attention_mask = inputs.get('attention_mask')
    mm_token_type_ids = inputs.get('mm_token_type_ids')
    image_grid_thw = inputs.get('image_grid_thw')
    pixel_values = inputs.get('pixel_values')

    inputs_embeds = core_model.language_model.embed_tokens(input_ids)
    visual_pos_masks = None
    deepstack_visual_embeds = None

    if pixel_values is not None:
        image_outputs = _get_image_feature_outputs(core_model, pixel_values, image_grid_thw)
        pooler_output, deepstack_visual_embeds = _unpack_image_feature_outputs(image_outputs)
        if isinstance(pooler_output, (tuple, list)):
            image_embeds = torch.cat(pooler_output, dim=0)
        else:
            image_embeds = pooler_output
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = core_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

    position_ids = core_model.compute_3d_position_ids(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        video_grid_thw=inputs.get('video_grid_thw'),
        attention_mask=attention_mask,
        past_key_values=None,
        mm_token_type_ids=mm_token_type_ids,
    )
    return inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids


class QuerySelection:
    def __init__(self, q_indices: torch.Tensor, v_token_start: int, v_token_num: int):
        self.q_indices = q_indices
        self.v_token_start = int(v_token_start)
        self.v_token_num = int(v_token_num)


class AttentionCapture(AbstractContextManager):
    def __init__(self, language_model: torch.nn.Module, layer_indices: Sequence[int], query_chunk_size: int):
        self.language_model = language_model
        self.layer_indices = list(layer_indices)
        self.query_chunk_size = max(1, int(query_chunk_size))
        self.handles = []
        self.inputs_embeds = None
        self.visual_pos_masks = None
        self.layer_attn = {}
        self.selection = None
        self._orig_forwards = {}

    def __enter__(self):
        self.handles.append(
            self.language_model.register_forward_pre_hook(self._lm_pre_hook, with_kwargs=True)
        )
        for layer_idx in self.layer_indices:
            layer = self.language_model.layers[layer_idx].self_attn
            self._orig_forwards[layer_idx] = layer.forward
            layer.forward = types.MethodType(self._wrap_attn_forward(layer_idx, layer.forward), layer)
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        for layer_idx, orig_forward in self._orig_forwards.items():
            try:
                self.language_model.layers[layer_idx].self_attn.forward = orig_forward
            except Exception:
                pass
        self.handles = []
        self._orig_forwards = {}
        return False

    def reset(self):
        self.inputs_embeds = None
        self.visual_pos_masks = None
        self.layer_attn = {}
        self.selection = None

    def _lm_pre_hook(self, module, args, kwargs):
        self.inputs_embeds = kwargs.get('inputs_embeds')
        self.visual_pos_masks = kwargs.get('visual_pos_masks')
        self.selection = None

    def _ensure_selection(self) -> Optional[QuerySelection]:
        if self.selection is None:
            self.selection = select_query_indices(self)
        return self.selection

    def _compute_layer_attn_summary(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_values=None,
    ) -> Optional[torch.Tensor]:
        selection = self._ensure_selection()
        if selection is None:
            return None

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_states = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, module.num_key_value_groups)

        q_indices = selection.q_indices.to(device=query_states.device, dtype=torch.long)
        q_indices = q_indices[(q_indices >= 0) & (q_indices < query_states.shape[2])]
        if q_indices.numel() == 0:
            return None

        v_start = int(selection.v_token_start)
        v_end = v_start + int(selection.v_token_num)
        if v_end > key_states.shape[2]:
            return None

        total = None
        count = 0
        scaling = float(module.scaling)
        for q_chunk in q_indices.split(self.query_chunk_size):
            q_chunk_states = query_states.index_select(dim=2, index=q_chunk)
            scores = torch.matmul(q_chunk_states, key_states.transpose(2, 3)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask[:, :, q_chunk, :]
            probs = torch.softmax(scores.float(), dim=-1)
            visual_probs = probs[:, :, :, v_start:v_end].sum(dim=2).sum(dim=1)
            total = visual_probs if total is None else (total + visual_probs)
            count += int(q_chunk.numel() * probs.shape[1])

        if total is None or count <= 0:
            return None
        return (total[0] / float(count)).detach().float().cpu()

    def _wrap_attn_forward(self, layer_idx: int, orig_forward):
        def wrapped(module, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
            if layer_idx not in self.layer_attn:
                attn_map = self._compute_layer_attn_summary(
                    module=module,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                if attn_map is not None:
                    self.layer_attn[layer_idx] = attn_map
            return orig_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        return wrapped


def select_query_indices(capture: AttentionCapture) -> Optional[QuerySelection]:
    if capture.inputs_embeds is None or capture.visual_pos_masks is None:
        return None
    visual_mask = capture.visual_pos_masks[0].detach().bool()
    v_positions = torch.where(visual_mask)[0]
    if v_positions.numel() == 0:
        return None
    v_token_start = int(v_positions[0].item())
    v_token_num = int(v_positions.numel())
    text_token_start = v_token_start + v_token_num

    inputs_embeds = capture.inputs_embeds.detach().float()
    if inputs_embeds.shape[1] <= text_token_start:
        return None
    visual_tokens = inputs_embeds[:, v_token_start:text_token_start, :]
    text_tokens = inputs_embeds[:, text_token_start:, :]
    if text_tokens.shape[1] == 0:
        return None

    similarity = torch.matmul(visual_tokens, text_tokens.transpose(1, 2))
    token_scores = similarity.softmax(dim=2).mean(dim=1)
    selected_rel = torch.where(token_scores[0] > token_scores[0].mean())[0]
    if selected_rel.numel() == 0:
        selected_rel = torch.arange(text_tokens.shape[1], device=token_scores.device)
    q_indices = (selected_rel + text_token_start).cpu()
    return QuerySelection(q_indices=q_indices, v_token_start=v_token_start, v_token_num=v_token_num)


class TemporaryDecoderLayerSlice(AbstractContextManager):
    def __init__(self, language_model: torch.nn.Module, num_layers: int):
        self.language_model = language_model
        self.num_layers = int(num_layers)
        self._orig_layers = None
        self._orig_num_hidden_layers = None

    def __enter__(self):
        self._orig_layers = self.language_model.layers
        self._orig_num_hidden_layers = getattr(self.language_model.config, 'num_hidden_layers', None)
        kept = list(self._orig_layers[: self.num_layers])
        self.language_model.layers = torch.nn.ModuleList(kept)
        if self._orig_num_hidden_layers is not None:
            self.language_model.config.num_hidden_layers = len(kept)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig_layers is not None:
            self.language_model.layers = self._orig_layers
        if self._orig_num_hidden_layers is not None:
            self.language_model.config.num_hidden_layers = self._orig_num_hidden_layers
        return False


def _build_gui_history_text(line) -> str:
    his_actions = _safe_literal_eval(line.get('history_action', '[]'))
    if not isinstance(his_actions, list):
        his_actions = []
    if len(his_actions) > 0:
        return '\n'.join([f'{i + 1}. {a}' for i, a in enumerate(his_actions)])
    return 'None'


def _build_prompt(ds, line, prompt_mode: str, line_width: int, line_color: Tuple[int, int, int]):
    if isinstance(line, int):
        line = ds.data.iloc[line]

    if prompt_mode == 'benchmark':
        msgs = ds.build_prompt(line)
        if isinstance(msgs, tuple):
            msgs = msgs[0]
        return msgs, 'benchmark', None

    image_path = ds._resolve_image_path(line['image'])
    task_text = str(line.get('question', '') or '')
    if prompt_mode == 'task_only':
        msgs = [dict(type='image', value=image_path), dict(type='text', value=task_text)]
        return msgs, 'task_only', None

    raw_img = Image.open(image_path).convert('RGB')
    model_image = draw_grid9_overlay(raw_img, line_width=line_width, line_color=line_color)
    prompt_text = task_text
    history_text = _build_gui_history_text(line)
    user_text = f'Task: {prompt_text}\nPrevious Actions:\n{history_text}'
    msgs = [dict(type='image', value=image_path), dict(type='text', value=user_text)]
    messages = [
        {'role': 'system', 'content': [{'type': 'text', 'text': COORD_PROMPT}]},
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': model_image},
                {'type': 'text', 'text': user_text},
            ],
        },
    ]
    return msgs, 'grid9_coord', {'raw_img': raw_img, 'model_image': model_image, 'messages': messages}


def _should_visualize(args, sample_index_0based: int, saved_count: int) -> bool:
    if not args.save_visualizations:
        return False
    if args.visualize_limit > 0 and saved_count >= int(args.visualize_limit):
        return False
    every = max(1, int(args.visualize_every))
    return (sample_index_0based % every) == 0


def _summarize(records: List[dict], topk_eval: int) -> dict:
    valid = [r for r in records if r.get('eligible_for_prune', 0) == 1 and r.get('hit_topk') is not None]
    skipped = [r for r in records if r.get('eligible_for_prune', 0) != 1]
    missing_image = [r for r in records if r.get('error') == 'image_file_missing']
    unreadable_image = [r for r in records if r.get('error') == 'image_file_unreadable']
    summary = {
        'num_records': len(records),
        'num_valid': len(valid),
        'num_skipped': len(skipped),
        'num_missing_image': len(missing_image),
        'num_unreadable_image': len(unreadable_image),
        'num_skipped_bad_image': len(missing_image) + len(unreadable_image),
    }
    for k in range(1, max(1, int(topk_eval)) + 1):
        summary[f'recall_top{k}'] = _mean([float(r.get(f'hit_top{k}', 0)) for r in valid])
    return summary


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
    target_layer_index = int(args.target_layer_index)
    if target_layer_index < 0:
        target_layer_index = decoder_layers_to_run - 1
    if target_layer_index >= decoder_layers_to_run:
        raise ValueError(
            f'--target_layer_index={target_layer_index} must be within [0, {decoder_layers_to_run - 1}].'
        )
    attn_layers = [target_layer_index]

    ds = GUIOdyssey(dataset=args.dataset)
    n = len(ds.data)
    if args.subset_limit > 0:
        n = min(n, args.subset_limit)

    per_sample_records: List[dict] = []
    sample_times_sec: List[float] = []
    processor_times_sec: List[float] = []
    visual_encode_times_sec: List[float] = []
    llm_forward_times_sec: List[float] = []
    topk_times_sec: List[float] = []
    visualization_count = 0

    with TemporaryDecoderLayerSlice(language_model, decoder_layers_to_run):
        with AttentionCapture(language_model, attn_layers, query_chunk_size=args.attn_query_chunk_size) as capture:
            for i in tqdm(range(n), desc=f'gui odyssey attn top4 {args.dataset}'):
                t0 = time.perf_counter()
                line = ds.data.iloc[i]
                sample_index = int(line.get('index', i + 1))
                answer = str(line.get('answer', ''))
                gt_action, gt_point = _parse_answer(answer)
                gt_bbox = _safe_literal_eval(line.get('sam2_bbox', []))
                if not isinstance(gt_bbox, (list, tuple)) or len(gt_bbox) != 4:
                    gt_bbox = None
                eligible = int((not args.only_click_longpress) or gt_action in ('CLICK', 'LONG_PRESS'))

                try:
                    msgs, prompt_style, render_payload = _build_prompt(
                        ds, line, prompt_mode=args.prompt_mode, line_width=args.line_width, line_color=line_color
                    )
                except FileNotFoundError:
                    image_path = ds._resolve_image_path(line['image'])
                    record = {
                        'dataset_name': args.dataset,
                        'subset': args.dataset,
                        'sample_index': sample_index,
                        'image_path': image_path,
                        'gt_action': gt_action,
                        'eligible_for_prune': 0,
                        'prompt_mode': args.prompt_mode,
                        'error': 'image_file_missing',
                    }
                    per_sample_records.append(record)
                    if args.print_per_sample:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                    continue
                except UnidentifiedImageError:
                    image_path = ds._resolve_image_path(line['image'])
                    record = {
                        'dataset_name': args.dataset,
                        'subset': args.dataset,
                        'sample_index': sample_index,
                        'image_path': image_path,
                        'gt_action': gt_action,
                        'eligible_for_prune': 0,
                        'prompt_mode': args.prompt_mode,
                        'error': 'image_file_unreadable',
                    }
                    per_sample_records.append(record)
                    if args.print_per_sample:
                        print(json.dumps(record, ensure_ascii=False), flush=True)
                    continue
                img_path = _extract_image_path(msgs)
                if img_path is None:
                    per_sample_records.append(
                        {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': None,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'error': 'missing_image_path',
                        }
                    )
                    continue

                prompt_text = ''
                for item in msgs:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        prompt_text = str(item.get('value', ''))
                        break

                raw_img = None
                model_image = None
                if render_payload is not None:
                    raw_img = render_payload['raw_img']
                    model_image = render_payload['model_image']
                    messages = render_payload['messages']
                    image_for_processor = model_image
                else:
                    try:
                        raw_img = Image.open(img_path).convert('RGB')
                    except FileNotFoundError:
                        record = {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': img_path,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'prompt_mode': prompt_style,
                            'error': 'image_file_missing',
                        }
                        per_sample_records.append(record)
                        if args.print_per_sample:
                            print(json.dumps(record, ensure_ascii=False), flush=True)
                        continue
                    except UnidentifiedImageError:
                        record = {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': img_path,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'prompt_mode': prompt_style,
                            'error': 'image_file_unreadable',
                        }
                        per_sample_records.append(record)
                        if args.print_per_sample:
                            print(json.dumps(record, ensure_ascii=False), flush=True)
                        continue
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
                    per_sample_records.append(
                        {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': img_path,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'prompt_mode': prompt_style,
                            'error': 'visual_pos_masks_is_none',
                        }
                    )
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
                    per_sample_records.append(
                        {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': img_path,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'prompt_mode': prompt_style,
                            'error': 'query_selection_is_none',
                        }
                    )
                    continue

                grid_size = infer_visual_grid_size(
                    inputs['image_grid_thw'].detach().cpu(),
                    spatial_merge_size=spatial_merge_size,
                )
                expected_tokens = int(grid_size[0] * grid_size[1])
                t_topk0 = time.perf_counter()
                attn_weights = capture.layer_attn.get(target_layer_index)
                if attn_weights is None:
                    per_sample_records.append(
                        {
                            'dataset_name': args.dataset,
                            'subset': args.dataset,
                            'sample_index': sample_index,
                            'image_path': img_path,
                            'gt_action': gt_action,
                            'eligible_for_prune': 0,
                            'prompt_mode': prompt_style,
                            'error': 'attn_weights_is_none',
                        }
                    )
                    continue
                if attn_weights.numel() != expected_tokens:
                    raise ValueError(
                        f'Attention map/token grid mismatch at layer {target_layer_index}: '
                        f'{attn_weights.numel()} weights vs {expected_tokens} grid cells '
                        f'(grid_size={grid_size}, spatial_merge_size={spatial_merge_size}).'
                    )
                attn_grid_ranking = aggregate_grid9_attention_scores(attn_weights, grid_size)
                pred_grids_topk = [int(x) for x in topk_grids_from_ranking(attn_grid_ranking, args.topk_eval)]
                t_topk1 = time.perf_counter()

                gt_bbox_norm = None
                gt_grid = None
                gt_click_xy_norm = None
                gt_bbox_xyxy = None
                if gt_bbox is not None:
                    w, h = raw_img.size
                    gt_bbox_norm = [float(x) for x in _to_bbox_norm(gt_bbox, w, h)]
                    gt_grid = int(bbox_to_grid9(gt_bbox_norm))
                    gt_cx, gt_cy = bbox_center_norm_xyxy(gt_bbox_norm)
                    gt_click_xy_norm = [float(gt_cx), float(gt_cy)]
                    gt_bbox_xyxy = parse_bbox_xyxy_pixels(gt_bbox, w, h)
                elif gt_point is not None:
                    w, h = raw_img.size
                    gt_point_pixels = _guiodyssey_coord_to_pixel_xy(gt_point, w, h)
                    gt_grid, gt_click_xy_norm = point_to_grid9(gt_point_pixels, w, h)

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

                processor_sec = float(t_proc1 - t_proc0)
                visual_encode_sec = float(t_vis1 - t_vis0)
                llm_forward_sec = float(t_llm1 - t_llm0)
                topk_select_sec = float(t_topk1 - t_topk0)
                elapsed_sec = float(time.perf_counter() - t0)

                vis_path = None
                if _should_visualize(args, i, visualization_count):
                    vis_image = model_image if model_image is not None else draw_grid9_overlay(
                        raw_img, line_width=args.line_width, line_color=line_color
                    )
                    vis = render_attention_visualization(
                        image=vis_image,
                        weights=attn_weights,
                        grid_size=grid_size,
                        gt_bbox_xyxy=gt_bbox_xyxy,
                        ranked_grid_ids=pred_grids_topk,
                    )
                    vis_path = build_output_path(args.output_dir, args.dataset, sample_index, img_path, target_layer_index)
                    vis.save(vis_path)
                    visualization_count += 1

                record = {
                    'dataset_name': args.dataset,
                    'subset': args.dataset,
                    'sample_index': sample_index,
                    'image_path': img_path,
                    'question': str(line.get('question', '')),
                    'history_action': line.get('history_action', '[]'),
                    'history_screenshot': line.get('history_screenshot', '[]'),
                    'prompt_text': prompt_text,
                    'prompt_mode': prompt_style,
                    'coord_prompt': COORD_PROMPT if prompt_style == 'grid9_coord' else None,
                    'gt_action': gt_action,
                    'gt_answer': answer,
                    'eligible_for_prune': int(eligible),
                    'gt_bbox': gt_bbox,
                    'gt_bbox_norm': gt_bbox_norm,
                    'gt_grid': gt_grid,
                    'gt_click_xy_norm': gt_click_xy_norm,
                    'pred_grid': int(pred_grids_topk[0]) if pred_grids_topk else -1,
                    'pred_grids_topk': pred_grids_topk,
                    'pred_grids_topk_with_scores': [
                        {'rank': rank_idx + 1, 'grid': int(item['grid']), 'score': float(item['score'])}
                        for rank_idx, item in enumerate(attn_grid_ranking[: int(args.topk_eval)])
                    ],
                    'hit_top1': hit_top1,
                    'hit_top2': hit_top2,
                    'hit_top3': hit_top3,
                    'hit_top4': hit_top4,
                    'hit_topk': hit_topk,
                    'decoder_layers_to_run': int(decoder_layers_to_run),
                    'target_layer_index': int(target_layer_index),
                    'target_layer_number_1based': int(target_layer_index + 1),
                    'attn_grid_ranking': attn_grid_ranking,
                    'visual_grid_size': [int(grid_size[0]), int(grid_size[1])],
                    'num_query_tokens': int(selection.q_indices.numel()),
                    'processor_sec': processor_sec,
                    'visual_encode_sec': visual_encode_sec,
                    'llm_forward_sec': llm_forward_sec,
                    'topk_select_sec': topk_select_sec,
                    'elapsed_sec': elapsed_sec,
                    'visualization_path': vis_path,
                }
                if not eligible:
                    record['skip_reason'] = 'non_click_non_longpress'
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
                    print(
                        json.dumps(
                            {
                                'sample_index': sample_index,
                                'gt_action': gt_action,
                                'prompt_mode': prompt_style,
                                'gt_grid': gt_grid,
                                'pred_grids_topk': pred_grids_topk,
                                'decoder_layers_to_run': int(decoder_layers_to_run),
                                'target_layer_number_1based': int(target_layer_index + 1),
                                'hit_top1': hit_top1,
                                'hit_top2': hit_top2,
                                'hit_top3': hit_top3,
                                'hit_top4': hit_top4,
                                'hit_topk': hit_topk,
                                'visualization_path': vis_path,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    summary = _summarize(per_sample_records, topk_eval=args.topk_eval)
    summary.update(
        {
            'dataset': args.dataset,
            'base_model': args.base_model,
            'adapter_path': adapter_path or 'none',
            'prompt_mode': args.prompt_mode,
            'decoder_layers_to_run': int(decoder_layers_to_run),
            'decoder_layers_total': int(total_hidden_layers),
            'attn_mode': 'instruction_to_image',
            'target_layer_index': int(target_layer_index),
            'target_layer_number_1based': int(target_layer_index + 1),
            'topk_eval': int(args.topk_eval),
            'attn_query_chunk_size': int(args.attn_query_chunk_size),
            'save_visualizations': bool(args.save_visualizations),
            'visualize_every': int(args.visualize_every),
            'visualize_limit': int(args.visualize_limit),
            'num_visualizations_saved': int(visualization_count),
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
        }
    )
    summary['recall_contains_gt_topk'] = float(summary.get(f'recall_top{int(args.topk_eval)}', 0.0))

    per_sample_path = os.path.join(args.output_dir, 'per_sample.json')
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(per_sample_path, 'w', encoding='utf-8') as f:
        json.dump(per_sample_records, f, ensure_ascii=False, indent=2)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'[saved] per-sample: {per_sample_path}')
    print(f'[saved] summary: {summary_path}')


if __name__ == '__main__':
    main()
