import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    BaseModelOutputWithPast,
    Cache,
    DynamicCache,
    FlashAttentionKwargs,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration as Qwen3VLForConditionalGenerationOrigin,
    Qwen3VLModel as Qwen3VLModelOrigin,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
    Unpack,
    create_causal_mask,
)


def _norm_path(value: object) -> str:
    if value is None:
        return ""
    try:
        return os.path.normpath(str(value).strip())
    except Exception:
        return ""


def _basename(value: object) -> str:
    p = _norm_path(value)
    return os.path.basename(p) if p else ""


def _load_roi_records(json_path: str) -> List[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    raise TypeError(f"Unsupported ROI json format in {json_path}")


def _record_dataset_name(record: dict) -> str:
    for key in ("dataset_name", "dataset", "subset"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _record_sample_index(record: dict) -> str:
    for key in ("sample_index", "index"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _build_roi_lookup(json_path: str) -> dict:
    records = _load_roi_records(json_path)
    by_dataset_index: Dict[Tuple[str, str], dict] = {}
    by_image_path: Dict[str, dict] = {}
    by_image_basename: Dict[str, dict] = {}
    ambiguous_path = set()
    ambiguous_base = set()

    for record in records:
        ds_name = _record_dataset_name(record)
        sample_index = _record_sample_index(record)
        if ds_name and sample_index:
            by_dataset_index[(ds_name, sample_index)] = record

        image_path = ""
        for key in ("image_path", "image", "sample_image", "path"):
            image_path = _norm_path(record.get(key))
            if image_path:
                break
        if image_path:
            if image_path in by_image_path:
                ambiguous_path.add(image_path)
            else:
                by_image_path[image_path] = record

            base = _basename(image_path)
            if base:
                if base in by_image_basename:
                    ambiguous_base.add(base)
                else:
                    by_image_basename[base] = record

    for key in ambiguous_path:
        by_image_path.pop(key, None)
    for key in ambiguous_base:
        by_image_basename.pop(key, None)

    return {
        "records": records,
        "by_dataset_index": by_dataset_index,
        "by_image_path": by_image_path,
        "by_image_basename": by_image_basename,
    }


def _get_current_roi_record(cfg) -> Tuple[Optional[dict], Optional[str]]:
    lookup = getattr(cfg, "_roi_prune_lookup", None)
    json_path = getattr(cfg, "_roi_prune_json_path", None)
    if lookup is None:
        if not json_path:
            return None, "missing_json_path"
        lookup = _build_roi_lookup(str(json_path))
        setattr(cfg, "_roi_prune_lookup", lookup)

    dataset_name = str(getattr(cfg, "_vlmeval_current_dataset_name", "") or "")
    sample_index = str(getattr(cfg, "_vlmeval_current_sample_index", "") or "")
    image_path = _norm_path(getattr(cfg, "_vlmeval_current_image_path", None))
    image_base = _basename(image_path)

    if dataset_name and sample_index:
        record = lookup["by_dataset_index"].get((dataset_name, sample_index))
        if record is not None:
            return record, "dataset_index"

    if image_path:
        record = lookup["by_image_path"].get(image_path)
        if record is not None:
            return record, "image_path"

    if image_base:
        record = lookup["by_image_basename"].get(image_base)
        if record is not None:
            return record, "image_basename"

    return None, "no_match"


def _sanitize_topk_grids(values: Sequence[object], topk_keep: int) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values:
        try:
            grid_id = int(value)
        except Exception:
            continue
        if 1 <= grid_id <= 9 and grid_id not in seen:
            out.append(grid_id)
            seen.add(grid_id)
        if len(out) >= int(topk_keep):
            break
    return out


def _extract_grid_ids(record: Optional[dict], topk_keep: int) -> List[int]:
    if not isinstance(record, dict):
        return []
    candidates = []
    for key in ("pred_grids_topk", "top4_grid_ids", "grid_ids", "topk_grid_ids"):
        value = record.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    prune_info = record.get("prune")
    if isinstance(prune_info, dict):
        for key in ("pred_grids_topk", "top4_grid_ids", "grid_ids", "topk_grid_ids"):
            value = prune_info.get(key)
            if isinstance(value, list):
                candidates.extend(value)
    return _sanitize_topk_grids(candidates, topk_keep)


def _extract_bbox_candidates(record: Optional[dict], topk_keep: int) -> List[List[float]]:
    if not isinstance(record, dict):
        return []
    boxes: List[List[float]] = []

    def _collect(value):
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                boxes.append([float(x) for x in value])
            except Exception:
                return
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for box_key in ("bbox", "roi_bbox", "bbox_xyxy", "xyxy"):
                        if box_key in item:
                            _collect(item[box_key])
                            break
                else:
                    _collect(item)

    for key in ("bbox", "roi_bbox", "bbox_xyxy", "top4_bboxes", "rois", "roi_candidates"):
        if key in record:
            _collect(record[key])
    prune_info = record.get("prune")
    if isinstance(prune_info, dict):
        for key in ("bbox", "roi_bbox", "bbox_xyxy", "top4_bboxes", "rois", "roi_candidates"):
            if key in prune_info:
                _collect(prune_info[key])

    deduped: List[List[float]] = []
    seen = set()
    for box in boxes:
        key = tuple(round(float(x), 4) for x in box)
        if key not in seen:
            deduped.append(box)
            seen.add(key)
        if len(deduped) >= int(topk_keep):
            break
    return deduped


def _resolve_visual_grid_size(cfg, record: Optional[dict], total_visual_tokens: int) -> Optional[Tuple[int, int]]:
    candidates = []
    if isinstance(record, dict):
        for key in ("visual_grid_size", "grid_size", "visual_hw"):
            value = record.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                candidates.append((int(value[0]), int(value[1])))
        prune_info = record.get("prune")
        if isinstance(prune_info, dict):
            for key in ("visual_grid_size", "grid_size", "visual_hw"):
                value = prune_info.get(key)
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    candidates.append((int(value[0]), int(value[1])))

    current_hw = getattr(cfg, "_vlmeval_current_visual_grid_hw", None)
    if isinstance(current_hw, (list, tuple)) and len(current_hw) == 2:
        candidates.append((int(current_hw[0]), int(current_hw[1])))

    for grid_h, grid_w in candidates:
        if grid_h > 0 and grid_w > 0 and grid_h * grid_w == int(total_visual_tokens):
            return (grid_h, grid_w)
    return None


def _grid_ids_to_visual_keep_mask(grid_ids: Sequence[int], grid_size: Tuple[int, int], device) -> torch.Tensor:
    grid_h, grid_w = [int(x) for x in grid_size]
    keep = torch.zeros(grid_h * grid_w, dtype=torch.bool, device=device)
    row_edges = [0, grid_h / 3.0, grid_h * 2.0 / 3.0, float(grid_h)]
    col_edges = [0, grid_w / 3.0, grid_w * 2.0 / 3.0, float(grid_w)]
    for grid_id in grid_ids:
        idx = max(1, min(9, int(grid_id))) - 1
        row, col = divmod(idx, 3)
        r0 = int(round(row_edges[row]))
        r1 = max(r0 + 1, int(round(row_edges[row + 1])))
        c0 = int(round(col_edges[col]))
        c1 = max(c0 + 1, int(round(col_edges[col + 1])))
        for r in range(r0, min(r1, grid_h)):
            start = r * grid_w + c0
            end = r * grid_w + min(c1, grid_w)
            keep[start:end] = True
    return keep


def _bbox_coord_space(record: Optional[dict]) -> str:
    if not isinstance(record, dict):
        return "auto"
    for key in ("coord_space", "bbox_coord_space", "roi_coord_space"):
        value = record.get(key)
        if value is not None:
            return str(value).strip().lower()
    prune_info = record.get("prune")
    if isinstance(prune_info, dict):
        for key in ("coord_space", "bbox_coord_space", "roi_coord_space"):
            value = prune_info.get(key)
            if value is not None:
                return str(value).strip().lower()
    return "auto"


def _bbox_to_pixels(
    bbox: Sequence[float],
    image_wh: Tuple[int, int],
    coord_space: str,
) -> Optional[Tuple[float, float, float, float]]:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in bbox]
    except Exception:
        return None
    image_w, image_h = image_wh
    max_val = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if coord_space in ("normalized_0_1", "norm01", "unit"):
        scale_x = image_w
        scale_y = image_h
    elif coord_space in ("normalized_1000", "norm1000", "1k"):
        scale_x = image_w / 1000.0
        scale_y = image_h / 1000.0
    elif coord_space == "pixel":
        scale_x = 1.0
        scale_y = 1.0
    else:
        if max_val <= 1.5:
            scale_x = image_w
            scale_y = image_h
        elif max_val <= 1000.0 and max(image_w, image_h) > 1000:
            scale_x = image_w / 1000.0
            scale_y = image_h / 1000.0
        else:
            scale_x = 1.0
            scale_y = 1.0

    px1 = max(0.0, min(float(image_w), x1 * scale_x))
    py1 = max(0.0, min(float(image_h), y1 * scale_y))
    px2 = max(0.0, min(float(image_w), x2 * scale_x))
    py2 = max(0.0, min(float(image_h), y2 * scale_y))
    if px2 < px1:
        px1, px2 = px2, px1
    if py2 < py1:
        py1, py2 = py2, py1
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def _bbox_candidates_to_visual_keep_mask(
    bbox_candidates: Sequence[Sequence[float]],
    grid_size: Tuple[int, int],
    image_wh: Tuple[int, int],
    coord_space: str,
    device,
) -> torch.Tensor:
    grid_h, grid_w = [int(x) for x in grid_size]
    keep = torch.zeros(grid_h * grid_w, dtype=torch.bool, device=device)
    image_w, image_h = image_wh
    patch_w = float(image_w) / float(grid_w)
    patch_h = float(image_h) / float(grid_h)

    pixel_boxes = []
    for bbox in bbox_candidates:
        pixel_box = _bbox_to_pixels(bbox, image_wh=image_wh, coord_space=coord_space)
        if pixel_box is not None:
            pixel_boxes.append(pixel_box)

    if not pixel_boxes:
        return keep

    for row in range(grid_h):
        y0 = row * patch_h
        y1 = (row + 1) * patch_h
        for col in range(grid_w):
            x0 = col * patch_w
            x1 = (col + 1) * patch_w
            hit = False
            for bx1, by1, bx2, by2 in pixel_boxes:
                if min(x1, bx2) > max(x0, bx1) and min(y1, by2) > max(y0, by1):
                    hit = True
                    break
            if hit:
                keep[row * grid_w + col] = True
    return keep


def _build_uniform_keep_mask(total_visual_tokens: int, keep_every: int, offset: int, device) -> torch.Tensor:
    keep = torch.zeros(int(total_visual_tokens), dtype=torch.bool, device=device)
    keep_every = int(keep_every)
    if total_visual_tokens <= 0 or keep_every <= 0:
        return keep
    offset = int(offset) % keep_every
    keep[offset::keep_every] = True
    return keep


def _index_attention_mask(attention_mask, keep_indices: torch.Tensor):
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        return attention_mask.index_select(2, keep_indices).index_select(3, keep_indices)
    if attention_mask.dim() == 3:
        return attention_mask.index_select(1, keep_indices).index_select(2, keep_indices)
    if attention_mask.dim() == 2:
        return attention_mask.index_select(1, keep_indices)
    return attention_mask


def _prune_past_key_values_by_seq_indices(
    past_key_values: Cache | None,
    keep_indices: torch.Tensor,
    through_layer_idx: int,
    original_seq_len: int,
) -> None:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return

    max_layer = min(int(through_layer_idx), len(past_key_values.layers) - 1)
    for layer_idx in range(max_layer + 1):
        layer = past_key_values.layers[layer_idx]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            continue
        if not getattr(layer, "is_initialized", False):
            continue
        if keys.ndim < 3 or values.ndim < 3:
            continue
        if int(keys.shape[-2]) != int(original_seq_len) or int(values.shape[-2]) != int(original_seq_len):
            continue

        layer_keep_indices = keep_indices.to(device=keys.device)
        layer.keys = keys.index_select(-2, layer_keep_indices)
        layer.values = values.index_select(-2, layer_keep_indices.to(device=values.device))

        new_seq_len = int(layer.keys.shape[-2])
        if hasattr(layer, "cumulative_length"):
            if isinstance(layer.cumulative_length, int):
                layer.cumulative_length = new_seq_len
            elif torch.is_tensor(layer.cumulative_length):
                layer.cumulative_length.fill_(new_seq_len)


def _shape_tuple(value) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    try:
        return tuple(int(x) for x in value.shape)
    except Exception:
        return None


def _debug_enabled(cfg) -> bool:
    return bool(getattr(cfg, "_roi_prune_debug", False))


def _debug_sample_allowed(cfg) -> bool:
    limit = getattr(cfg, "_roi_prune_debug_max_sample_index", None)
    if limit is None:
        return True
    sample_index = getattr(cfg, "_vlmeval_current_sample_index", None)
    try:
        return int(sample_index) <= int(limit)
    except Exception:
        return False


def _should_print_debug(cfg) -> bool:
    return _debug_enabled(cfg) and _debug_sample_allowed(cfg)


def _cache_layer_shape_summary(
    past_key_values: Cache | None,
    layer_indices: Sequence[int],
) -> str:
    if past_key_values is None or not hasattr(past_key_values, "layers"):
        return "none"
    parts = []
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(past_key_values.layers):
            continue
        layer = past_key_values.layers[layer_idx]
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        parts.append(
            f"L{layer_idx}:K{_shape_tuple(keys)}:V{_shape_tuple(values)}"
        )
    return "|".join(parts) if parts else "none"


def _print_roi_prune_state(
    cfg,
    *,
    tag: str,
    layer_idx: int,
    prune_applied: bool,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    text_position_ids: torch.Tensor | None,
    position_embeddings,
    visual_pos_masks: torch.Tensor | None,
    deepstack_visual_embeds: list[torch.Tensor] | None,
    past_key_values: Cache | None,
    extra: Optional[dict] = None,
) -> None:
    if not _should_print_debug(cfg):
        return

    sample_index = getattr(cfg, "_vlmeval_current_sample_index", None)
    dataset_name = getattr(cfg, "_vlmeval_current_dataset_name", None)
    visual_count = int(visual_pos_masks[0].sum().item()) if visual_pos_masks is not None else -1
    deepstack_len = len(deepstack_visual_embeds) if deepstack_visual_embeds is not None else 0
    deepstack_shape = None
    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        deepstack_shape = _shape_tuple(deepstack_visual_embeds[layer_idx])
    rope_cos_shape = None
    rope_sin_shape = None
    if isinstance(position_embeddings, tuple) and len(position_embeddings) >= 2:
        rope_cos_shape = _shape_tuple(position_embeddings[0])
        rope_sin_shape = _shape_tuple(position_embeddings[1])
    cache_len = None
    try:
        cache_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
    except Exception:
        cache_len = None
    probe_layers = sorted({0, layer_idx - 1, layer_idx, layer_idx + 1})
    cache_shapes = _cache_layer_shape_summary(past_key_values, probe_layers)

    msg = (
        "[ROIPruneDebug] "
        f"tag={tag} "
        f"sample_index={sample_index} "
        f"dataset={dataset_name} "
        f"layer_idx={layer_idx} "
        f"layer_order={layer_idx + 1} "
        f"prune_applied={prune_applied} "
        f"hidden_shape={_shape_tuple(hidden_states)} "
        f"attention_mask_shape={_shape_tuple(attention_mask)} "
        f"text_position_ids_shape={_shape_tuple(text_position_ids)} "
        f"rope_cos_shape={rope_cos_shape} "
        f"rope_sin_shape={rope_sin_shape} "
        f"visual_count={visual_count} "
        f"deepstack_len={deepstack_len} "
        f"deepstack_shape={deepstack_shape} "
        f"cache_seq_len={cache_len} "
        f"cache_shapes={cache_shapes}"
    )
    if extra:
        for key, value in extra.items():
            msg += f" {key}={value}"
    print(msg, flush=True)


class Qwen3VLTextModelRoiPrune(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    input_modalities = ("text",)
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple | BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        forward_index = int(getattr(self.config, "_vlmeval_generate_forward_index", 0) or 0)
        setattr(self.config, "_vlmeval_generate_forward_index", forward_index + 1)

        # When use_cache=False, generation replays the full sequence on every step.
        # In that mode, `visual_pos_masks is not None and seq_len > 1` is also true for
        # later generation steps, so we cannot use it alone to distinguish prefill from decode.
        # Treat the first forward of each generate() call as prefill, and all later forwards
        # as decode-side generation work.
        is_prefill_forward = forward_index == 0 and visual_pos_masks is not None and inputs_embeds.shape[1] > 1 and not self.training
        prune_enabled = is_prefill_forward and bool(getattr(self.config, "_roi_prune_enabled", False))
        prune_layer_idx = int(getattr(self.config, "_roi_prune_layer_idx", 15))
        topk_keep = int(getattr(self.config, "_roi_prune_topk_keep", 4))
        uniform_keep_every = int(getattr(self.config, "_roi_prune_uniform_keep_every", 0))
        uniform_keep_offset = int(getattr(self.config, "_roi_prune_uniform_keep_offset", 0))
        debug_enabled = bool(getattr(self.config, "_roi_prune_debug", False))
        print_layer_attn_tokens = bool(getattr(self.config, "_print_layer_attn_tokens", False))

        prune_applied = False
        layer_attn_token_stats = []
        prune_stats = {
            "prune_applied": False,
            "visual_tokens_before": int(visual_pos_masks[0].sum().item()) if visual_pos_masks is not None else 0,
            "visual_tokens_after": int(visual_pos_masks[0].sum().item()) if visual_pos_masks is not None else 0,
            "seq_tokens_before": int(hidden_states.shape[1]),
            "seq_tokens_after": int(hidden_states.shape[1]),
            "keep_indices_count": int(hidden_states.shape[1]),
            "selected_top4_grids": [],
            "top4_visual_tokens": 0,
            "bbox_visual_tokens": 0,
            "uniform_keep_every": int(uniform_keep_every),
            "uniform_keep_offset": int(uniform_keep_offset),
            "uniform_candidate_visual_tokens": 0,
            "uniform_added_visual_tokens": 0,
            "kept_visual_rel_count": 0,
            "kept_visual_rel_indices": [],
            "top4_keep_visual_rel_indices": [],
            "bbox_keep_visual_rel_indices": [],
            "uniform_keep_visual_rel_indices": [],
            "prefill_split_layer_order_1based": int(prune_layer_idx + 1),
            "pre_prune_1_to_16_sec": 0.0,
            "lookup_align_key": None,
            "lookup_status": None,
            "prune_selection_sec": 0.0,
            "prune_op_sec": 0.0,
            "post_prune_to_end_sec": 0.0,
            "prune_layer_to_finish_sec": 0.0,
        }
        t_forward0 = time.perf_counter()
        split_boundary_time = None
        prune_finish_time = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if print_layer_attn_tokens:
                q_len = int(hidden_states.shape[1])
                cache_len_before = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
                kv_len = int(cache_len_before + q_len)
                layer_attn_token_stats.append(
                    {
                        "layer_idx_0based": int(layer_idx),
                        "layer_order_1based": int(layer_idx + 1),
                        "q_len": int(q_len),
                        "kv_len_including_cache": int(kv_len),
                        "cache_len_before": int(cache_len_before),
                    }
                )
            if prune_enabled and layer_idx in {max(0, prune_layer_idx - 1), prune_layer_idx, min(len(self.layers) - 1, prune_layer_idx + 1)}:
                _print_roi_prune_state(
                    self.config,
                    tag="layer_enter",
                    layer_idx=layer_idx,
                    prune_applied=prune_applied,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    text_position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    past_key_values=past_key_values,
                )
            try:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            except RuntimeError as err:
                visual_count = int(visual_pos_masks[0].sum().item()) if visual_pos_masks is not None else -1
                attn_shape = tuple(attention_mask.shape) if attention_mask is not None else None
                pos_shape = tuple(text_position_ids.shape) if text_position_ids is not None else None
                rope_shape = None
                if isinstance(position_embeddings, tuple) and len(position_embeddings) >= 1:
                    try:
                        rope_shape = tuple(position_embeddings[0].shape)
                    except Exception:
                        rope_shape = None
                print(
                    "[ROIPruneError] "
                    f"layer_idx={layer_idx} "
                    f"layer_order={layer_idx + 1} "
                    f"prune_applied={prune_applied} "
                    f"hidden_shape={tuple(hidden_states.shape)} "
                    f"visual_count={visual_count} "
                    f"attention_mask_shape={attn_shape} "
                    f"text_position_ids_shape={pos_shape} "
                    f"rope_cos_shape={rope_shape} "
                    f"use_cache={bool(use_cache)} "
                    f"error={err}",
                    flush=True,
                )
                _print_roi_prune_state(
                    self.config,
                    tag="layer_error",
                    layer_idx=layer_idx,
                    prune_applied=prune_applied,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    text_position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    past_key_values=past_key_values,
                    extra={"error": repr(err)},
                )
                raise
            hidden_states = layer_outputs

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if is_prefill_forward and split_boundary_time is None and layer_idx == prune_layer_idx:
                split_boundary_time = time.perf_counter()

            if not prune_enabled or prune_applied or layer_idx != prune_layer_idx or visual_pos_masks is None:
                continue

            t_select0 = time.perf_counter()
            record, align_key = _get_current_roi_record(self.config)
            prune_stats["lookup_align_key"] = align_key
            prune_stats["lookup_status"] = "matched" if record is not None else "no_match"
            selected_top4 = _extract_grid_ids(record, topk_keep=topk_keep)
            bbox_candidates = _extract_bbox_candidates(record, topk_keep=topk_keep)

            visual_positions = torch.where(visual_pos_masks[0].bool())[0]
            total_visual_tokens = int(visual_positions.numel())
            grid_size = _resolve_visual_grid_size(self.config, record, total_visual_tokens)
            if grid_size is None:
                prune_stats["lookup_status"] = "grid_size_mismatch_or_missing"
                continue

            top4_keep_mask = _grid_ids_to_visual_keep_mask(selected_top4, grid_size, hidden_states.device)
            image_wh = getattr(self.config, "_vlmeval_current_image_size_wh", None)
            bbox_keep_mask = torch.zeros_like(top4_keep_mask)
            if isinstance(image_wh, (list, tuple)) and len(image_wh) == 2 and bbox_candidates:
                bbox_keep_mask = _bbox_candidates_to_visual_keep_mask(
                    bbox_candidates=bbox_candidates,
                    grid_size=grid_size,
                    image_wh=(int(image_wh[0]), int(image_wh[1])),
                    coord_space=_bbox_coord_space(record),
                    device=hidden_states.device,
                )
            uniform_keep_mask = _build_uniform_keep_mask(
                total_visual_tokens,
                uniform_keep_every,
                uniform_keep_offset,
                hidden_states.device,
            )

            roi_keep_mask = top4_keep_mask | bbox_keep_mask
            keep_visual_rel_mask = roi_keep_mask | uniform_keep_mask
            prune_stats["prune_selection_sec"] = float(time.perf_counter() - t_select0)
            if not bool(keep_visual_rel_mask.any().item()):
                prune_stats["lookup_status"] = "empty_keep_mask"
                continue

            keep_visual_rel_idx = torch.where(keep_visual_rel_mask)[0]
            keep_visual_abs = visual_positions.index_select(0, keep_visual_rel_idx)
            seq_keep_mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            seq_keep_mask[visual_positions] = False
            seq_keep_mask[keep_visual_abs] = True
            keep_indices = torch.where(seq_keep_mask)[0]
            original_seq_len = int(hidden_states.shape[1])
            _print_roi_prune_state(
                self.config,
                tag="pre_prune",
                layer_idx=layer_idx,
                prune_applied=prune_applied,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                text_position_ids=text_position_ids,
                position_embeddings=position_embeddings,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                past_key_values=past_key_values,
                extra={
                    "original_seq_len": original_seq_len,
                    "keep_indices_count": int(keep_indices.numel()),
                    "keep_visual_rel_count": int(keep_visual_rel_idx.numel()),
                    "top4_keep_count": int(top4_keep_mask.sum().item()),
                    "bbox_keep_count": int(bbox_keep_mask.sum().item()),
                    "uniform_keep_count": int(uniform_keep_mask.sum().item()),
                },
            )

            t_prune0 = time.perf_counter()
            _prune_past_key_values_by_seq_indices(
                past_key_values=past_key_values,
                keep_indices=keep_indices,
                through_layer_idx=layer_idx,
                original_seq_len=original_seq_len,
            )
            hidden_states = hidden_states.index_select(1, keep_indices)
            batch_size = hidden_states.shape[0]
            new_seq_len = hidden_states.shape[1]
            dense_pos = torch.arange(new_seq_len, device=hidden_states.device, dtype=torch.long)
            text_position_ids = dense_pos.view(1, new_seq_len).expand(batch_size, new_seq_len)
            position_ids = dense_pos.view(1, 1, new_seq_len).expand(3, batch_size, new_seq_len)
            visual_pos_masks = visual_pos_masks.index_select(1, keep_indices)
            if deepstack_visual_embeds is not None:
                deepstack_visual_embeds = [
                    emb.index_select(0, keep_visual_rel_idx.to(emb.device)) for emb in deepstack_visual_embeds
                ]
            # After mid-prefill pruning, rebuild a fresh causal mask that exactly matches
            # the shortened sequence length, instead of relying on any pre-prune mask state.
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=torch.ones(
                    (batch_size, new_seq_len), dtype=torch.bool, device=hidden_states.device
                ),
                past_key_values=None,
                position_ids=text_position_ids,
            )
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            prune_finish_time = time.perf_counter()
            prune_applied = True
            _print_roi_prune_state(
                self.config,
                tag="post_prune",
                layer_idx=layer_idx,
                prune_applied=prune_applied,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                text_position_ids=text_position_ids,
                position_embeddings=position_embeddings,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                past_key_values=past_key_values,
                extra={
                    "new_seq_len": int(hidden_states.shape[1]),
                    "new_visual_count": int(visual_pos_masks[0].sum().item()),
                },
            )

            top4_keep_idx = torch.where(top4_keep_mask)[0]
            bbox_keep_idx = torch.where(bbox_keep_mask)[0]
            uniform_keep_idx = torch.where(uniform_keep_mask)[0]
            prune_stats.update(
                {
                    "prune_applied": True,
                    "visual_tokens_before": int(total_visual_tokens),
                    "visual_tokens_after": int(visual_pos_masks[0].sum().item()),
                    "seq_tokens_before": int(seq_keep_mask.numel()),
                    "seq_tokens_after": int(hidden_states.shape[1]),
                    "keep_indices_count": int(keep_indices.numel()),
                    "selected_top4_grids": [int(x) for x in selected_top4],
                    "top4_visual_tokens": int(top4_keep_mask.sum().item()),
                    "bbox_visual_tokens": int(bbox_keep_mask.sum().item()),
                    "uniform_candidate_visual_tokens": int(uniform_keep_mask.sum().item()),
                    "uniform_added_visual_tokens": int((uniform_keep_mask & (~roi_keep_mask)).sum().item()),
                    "kept_visual_rel_count": int(keep_visual_rel_idx.numel()),
                    "kept_visual_rel_indices": [int(x) for x in keep_visual_rel_idx.detach().cpu().tolist()],
                    "top4_keep_visual_rel_indices": [int(x) for x in top4_keep_idx.detach().cpu().tolist()],
                    "bbox_keep_visual_rel_indices": [int(x) for x in bbox_keep_idx.detach().cpu().tolist()],
                    "uniform_keep_visual_rel_indices": [int(x) for x in uniform_keep_idx.detach().cpu().tolist()],
                    "prune_op_sec": float(prune_finish_time - t_prune0),
                    "prune_layer_to_finish_sec": float(prune_finish_time - split_boundary_time) if split_boundary_time is not None else 0.0,
                }
            )
            if debug_enabled:
                sample_index = getattr(self.config, "_vlmeval_current_sample_index", None)
                print(
                    f"[ROIPrune] sample_index={sample_index} align={align_key} "
                    f"grid_size={grid_size} top4={selected_top4} bbox_count={len(bbox_candidates)} "
                    f"visual_before={total_visual_tokens} visual_after={int(visual_pos_masks[0].sum().item())}",
                    flush=True,
                )

        hidden_states = self.norm(hidden_states)
        t_forward1 = time.perf_counter()
        if is_prefill_forward:
            if split_boundary_time is None:
                prune_stats["pre_prune_1_to_16_sec"] = float(t_forward1 - t_forward0)
                prune_stats["post_prune_to_end_sec"] = 0.0
            else:
                prune_stats["pre_prune_1_to_16_sec"] = float(split_boundary_time - t_forward0)
                if prune_finish_time is None:
                    prune_stats["post_prune_to_end_sec"] = float(t_forward1 - split_boundary_time)
                else:
                    prune_stats["post_prune_to_end_sec"] = float(t_forward1 - prune_finish_time)
            prune_stats["layer_attn_token_stats"] = layer_attn_token_stats
            setattr(self.config, "_roi_prune_last_stats", prune_stats)

        runtime = dict(getattr(self.config, "_vlmeval_generate_timing_accum", {}) or {})
        runtime.setdefault("prefill_s", 0.0)
        runtime.setdefault("decode_s", 0.0)
        runtime.setdefault("decode_steps", 0)
        runtime.setdefault("prefill_before_prune_layer_s", 0.0)
        runtime.setdefault("prefill_split_to_prune_start_s", 0.0)
        runtime.setdefault("prune_layer_to_prefill_end_s", 0.0)
        runtime.setdefault("split_layer_to_prefill_end_without_prune_s", 0.0)
        runtime.setdefault("prune_selection_s", 0.0)
        runtime.setdefault("prune_op_s", 0.0)
        runtime.setdefault("prune_layer_to_finish_s", 0.0)
        forward_elapsed = float(t_forward1 - t_forward0)
        if is_prefill_forward:
            runtime["prefill_s"] += forward_elapsed
            runtime["prefill_before_prune_layer_s"] += float(prune_stats.get("pre_prune_1_to_16_sec", 0.0) or 0.0)
            if split_boundary_time is None:
                runtime["prefill_split_to_prune_start_s"] += 0.0
                runtime["prune_layer_to_prefill_end_s"] += 0.0
                runtime["split_layer_to_prefill_end_without_prune_s"] += 0.0
            elif prune_finish_time is None:
                runtime["prefill_split_to_prune_start_s"] += 0.0
                runtime["prune_layer_to_prefill_end_s"] += 0.0
                runtime["split_layer_to_prefill_end_without_prune_s"] += float(t_forward1 - split_boundary_time)
            else:
                split_to_prune_start = float(
                    prune_stats.get("prune_selection_sec", 0.0) or 0.0
                ) + float(
                    prune_stats.get("prune_op_sec", 0.0) or 0.0
                )
                split_to_prune_start = max(
                    0.0,
                    float(prune_finish_time - split_boundary_time) - split_to_prune_start,
                )
                runtime["prefill_split_to_prune_start_s"] += split_to_prune_start
                runtime["prune_layer_to_prefill_end_s"] += float(t_forward1 - prune_finish_time)
                runtime["split_layer_to_prefill_end_without_prune_s"] += 0.0
            runtime["prune_selection_s"] += float(prune_stats.get("prune_selection_sec", 0.0) or 0.0)
            runtime["prune_op_s"] += float(prune_stats.get("prune_op_sec", 0.0) or 0.0)
            runtime["prune_layer_to_finish_s"] += float(prune_stats.get("prune_layer_to_finish_sec", 0.0) or 0.0)
        else:
            runtime["decode_s"] += forward_elapsed
            runtime["decode_steps"] += 1
        setattr(self.config, "_vlmeval_generate_timing_accum", runtime)
        setattr(self.config, "_vlmeval_generate_timing_last", dict(runtime))

        if print_layer_attn_tokens:
            sample_index = getattr(self.config, "_vlmeval_current_sample_index", None)
            phase = "prefill" if is_prefill_forward else "decode"
            print(
                f"[LayerAttnTokens] sample_index={sample_index} phase={phase} forward_index={forward_index} seq_len={int(inputs_embeds.shape[1])}",
                flush=True,
            )
            for stat in layer_attn_token_stats:
                print(
                    "[LayerAttnTokens] "
                    f"layer={stat['layer_order_1based']} "
                    f"q_len={stat['q_len']} "
                    f"kv_len={stat['kv_len_including_cache']} "
                    f"cache_len_before={stat['cache_len_before']}",
                    flush=True,
                )

        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = "model"
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModelRoiPrune._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    if hasattr(Qwen3VLModelOrigin, "get_vision_position_ids"):
        get_vision_position_ids = Qwen3VLModelOrigin.get_vision_position_ids
    if hasattr(Qwen3VLModelOrigin, "compute_3d_position_ids"):
        compute_3d_position_ids = Qwen3VLModelOrigin.compute_3d_position_ids
    if hasattr(Qwen3VLModelOrigin, "get_rope_index"):
        get_rope_index = Qwen3VLModelOrigin.get_rope_index
    if hasattr(Qwen3VLModelOrigin, "get_video_features"):
        get_video_features = Qwen3VLModelOrigin.get_video_features
    if hasattr(Qwen3VLModelOrigin, "get_image_features"):
        get_image_features = Qwen3VLModelOrigin.get_image_features
    if hasattr(Qwen3VLModelOrigin, "get_placeholder_mask"):
        get_placeholder_mask = Qwen3VLModelOrigin.get_placeholder_mask
    forward = Qwen3VLModelOrigin.forward


class Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGenerationOrigin):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Keep parity with upstream Qwen3-VL generate-time multimodal kwargs handling,
        # while ensuring mm_token_type_ids survives unused-kwargs validation.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if mm_token_type_ids is not None:
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        return model_inputs
