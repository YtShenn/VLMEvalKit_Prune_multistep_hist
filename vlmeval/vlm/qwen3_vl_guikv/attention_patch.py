from __future__ import annotations

import importlib
import inspect
import os
import types
from typing import Any

import torch

from .gui_kv_utils import GUIKVCluster, GUIKVConfig


def _as_int_list(x) -> list[int]:
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    return [int(v) for v in x]


def _image_token_ranges_from_inputs(
    input_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    image_token_id: int | None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[int], list[int]]:
    if input_ids is None:
        return [], []

    if attention_mask is not None and attention_mask.ndim == 2:
        valid = attention_mask[0].bool()
    else:
        valid = torch.ones_like(input_ids[0], dtype=torch.bool)

    if mm_token_type_ids is not None:
        modality = mm_token_type_ids[0].to(input_ids.device)
        mask = (modality == 1) & valid
    elif image_token_id is not None:
        mask = (input_ids[0] == int(image_token_id)) & valid
    else:
        return [], []

    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        return [], []

    starts = []
    ends = []
    start = int(positions[0])
    prev = int(positions[0])
    for pos_t in positions[1:]:
        pos = int(pos_t)
        if pos != prev + 1:
            starts.append(start)
            ends.append(prev + 1)
            start = pos
        prev = pos
    starts.append(start)
    ends.append(prev + 1)

    counts = []
    if image_grid_thw is not None:
        try:
            raw_counts = _as_int_list(image_grid_thw.prod(-1))
            total_observed = sum(e - s for s, e in zip(starts, ends))
            raw_total = sum(raw_counts)
            if raw_total == total_observed:
                counts = raw_counts
            elif total_observed > 0 and raw_total % total_observed == 0:
                divisor = max(1, raw_total // total_observed)
                counts = [max(1, int(c) // divisor) for c in raw_counts]
            else:
                counts = []
            if sum(counts) != total_observed and len(counts) == len(starts):
                counts = [e - s for s, e in zip(starts, ends)]
        except Exception:
            counts = []

    if counts and len(starts) == 1 and len(counts) > 1 and sum(counts) == (ends[0] - starts[0]):
        base = starts[0]
        split_starts, split_ends = [], []
        for c in counts:
            split_starts.append(base)
            base += int(c)
            split_ends.append(base)
        starts, ends = split_starts, split_ends

    return starts, ends


def _find_language_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    candidates = [
        "model.language_model.layers",
        "language_model.layers",
        "model.model.layers",
        "model.layers",
    ]
    for path in candidates:
        cur: Any = model
        ok = True
        for name in path.split("."):
            cur = getattr(cur, name, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, (torch.nn.ModuleList, list, tuple)):
            return list(cur)
    return []


def _cache_update(cache, key_states, value_states, layer_idx: int, cache_kwargs=None):
    try:
        if cache_kwargs is not None:
            return cache.update(key_states, value_states, layer_idx, cache_kwargs)
    except TypeError:
        pass
    return cache.update(key_states, value_states, layer_idx)


def _set_cache_layer(cache, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
    layers = getattr(cache, "layers", None)
    if layers is not None and layer_idx < len(layers):
        layer = layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            layer.keys = key_states
            layer.values = value_states


def _set_seen_tokens(cache, seen_tokens: int) -> None:
    if hasattr(cache, "_seen_tokens"):
        try:
            cache._seen_tokens = int(seen_tokens)
        except Exception:
            pass
    try:
        cache._guikv_original_seq_length = int(seen_tokens)
    except Exception:
        pass


def _record_guikv_decode_stats(attn_module, *, original_seq_len: int, actual_kv_seq_len: int) -> None:
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        return
    try:
        cfg._guikv_last_actual_kv_seq_len = int(actual_kv_seq_len)
        cfg._guikv_last_original_seq_len = int(original_seq_len)
    except Exception:
        pass


def _record_guikv_prefill_stats(
    attn_module,
    *,
    original_seq_len: int,
    compressed_seq_len: int,
    window_size: int,
    starts: list[int],
    ends: list[int],
    kept_indices: torch.Tensor | None,
) -> None:
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        return
    visual_before = int(sum(max(0, int(e) - int(s)) for s, e in zip(starts, ends)))
    visual_after = 0.0
    if visual_before > 0:
        if int(compressed_seq_len) >= int(original_seq_len):
            visual_after = float(visual_before)
        else:
            ranges = [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) > int(s)]
            recent_start = max(0, int(original_seq_len) - int(window_size))
            recent_positions = torch.arange(
                recent_start,
                int(original_seq_len),
                device=kept_indices.device if kept_indices is not None else None,
            )
            if kept_indices is not None and kept_indices.numel() > 0:
                kept_rows = kept_indices.reshape(-1, kept_indices.shape[-1])
                recent_rows = recent_positions.reshape(1, -1).expand(kept_rows.shape[0], -1)
                retained = torch.cat([kept_rows, recent_rows], dim=-1)
                counts = []
                for row in retained:
                    unique_pos = torch.unique(row)
                    count = 0
                    for s, e in ranges:
                        count += int(((unique_pos >= s) & (unique_pos < e)).sum().item())
                    counts.append(count)
                visual_after = float(sum(counts) / len(counts)) if counts else 0.0
            else:
                count = 0
                for s, e in ranges:
                    count += int(((recent_positions >= s) & (recent_positions < e)).sum().item())
                visual_after = float(count)
    try:
        cfg._guikv_last_original_prompt_len = int(original_seq_len)
        cfg._guikv_last_compressed_seq_len = int(compressed_seq_len)
        cfg._guikv_last_actual_kv_seq_len = int(compressed_seq_len)
        cfg._guikv_last_visual_tokens_before = int(visual_before)
        cfg._guikv_last_visual_tokens_after = float(visual_after)
    except Exception:
        pass


def _attention_interface(attn_module, fallback):
    all_attention = getattr(importlib.import_module(attn_module.__class__.__module__), "ALL_ATTENTION_FUNCTIONS", None)
    if all_attention is None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS as all_attention
    impl = getattr(attn_module.config, "_attn_implementation", "eager")
    if hasattr(all_attention, "get_interface"):
        return all_attention.get_interface(impl, fallback)
    return all_attention[impl] if impl != "eager" else fallback


def _make_guikv_attention_forward(attn_module):
    module_globals = importlib.import_module(attn_module.__class__.__module__)
    apply_rotary_pos_emb = getattr(module_globals, "apply_rotary_pos_emb")
    eager_attention_forward = getattr(module_globals, "eager_attention_forward")

    def guikv_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        past_key_value=None,
        **kwargs,
    ):
        cache = past_key_values if past_key_values is not None else past_key_value
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_len = int(hidden_states.shape[1])

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if q_len == 1 and cos.shape[-2] > 1:
            cos = cos[..., -1:, :]
            sin = sin[..., -1:, :]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        cache_kwargs = {"sin": sin, "cos": cos}
        if "cache_position" in kwargs:
            cache_kwargs["cache_position"] = kwargs["cache_position"]

        if q_len > 1:
            self._guikv_original_prompt_len = q_len
            self._guikv_seen_tokens = q_len

        guikv_prefill_cache = None
        guikv_prefill_seen = None
        if cache is not None:
            original_seen = int(getattr(self, "_guikv_seen_tokens", getattr(self, "_guikv_original_prompt_len", q_len)))
            if q_len > 1:
                original_seen = q_len
                guikv_prefill_cache = cache
                guikv_prefill_seen = original_seen
                self._guikv_seen_tokens = original_seen
            else:
                key_states, value_states = _cache_update(
                    cache, key_states, value_states, self.layer_idx, cache_kwargs
                )
                self._guikv_seen_tokens = original_seen + q_len
                _set_seen_tokens(cache, self._guikv_seen_tokens)
                _record_guikv_decode_stats(
                    self,
                    original_seq_len=int(self._guikv_seen_tokens),
                    actual_kv_seq_len=int(key_states.shape[-2]),
                )

        attention_interface = _attention_interface(self, eager_attention_forward)
        if os.getenv("GUIKV_DEBUG_SHAPES", "0") == "1" and int(getattr(self, "layer_idx", 0)) < 3:
            mask_shape = tuple(attention_mask.shape) if attention_mask is not None else None
            cache_len = None
            try:
                cache_len = cache.get_seq_length(self.layer_idx) if cache is not None else None
            except Exception:
                cache_len = "err"
            print(
                "[GUIKVShape] "
                f"layer={getattr(self, 'layer_idx', None)} q_len={q_len} "
                f"hidden={tuple(hidden_states.shape)} cos={tuple(cos.shape)} sin={tuple(sin.shape)} "
                f"q={tuple(query_states.shape)} k={tuple(key_states.shape)} v={tuple(value_states.shape)} "
                f"mask={mask_shape} cache_len={cache_len}",
                flush=True,
            )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if guikv_prefill_cache is not None:
            cfg = getattr(self, "_guikv_config", GUIKVConfig())
            text_cfg = self.config
            starts = list(getattr(text_cfg, "_guikv_vision_start_idx", []) or [])
            ends = list(getattr(text_cfg, "_guikv_vision_end_idx", []) or [])
            max_capacity_raw = getattr(text_cfg, "_guikv_max_capacity_prompt", cfg.max_capacity_prompt)
            max_capacity = int(max_capacity_raw) if max_capacity_raw is not None else None
            window_size = int(getattr(text_cfg, "_guikv_window_size", cfg.window_size))
            window_size = min(window_size, max_capacity - 2) if max_capacity is not None and max_capacity > 2 else window_size
            if not hasattr(self, "kv_cluster"):
                self.kv_cluster = GUIKVCluster()
            self.kv_cluster.reset(
                window_size=window_size,
                max_capacity_prompt=max_capacity,
                kernel_size=int(getattr(text_cfg, "_guikv_kernel_size", cfg.kernel_size)),
                pooling=str(getattr(text_cfg, "_guikv_pooling", cfg.pooling)),
                merge=getattr(text_cfg, "_guikv_merge", cfg.merge),
                alpha=float(getattr(text_cfg, "_guikv_alpha", cfg.alpha)),
                temperature=float(getattr(text_cfg, "_guikv_temperature", cfg.temperature)),
                total_keep_ratio=float(getattr(text_cfg, "_guikv_total_keep_ratio", cfg.total_keep_ratio)),
                vision_start_idx=starts,
                vision_end_idx=ends,
            )
            cache_key_states, cache_value_states = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
                hidden_states=hidden_states,
            )
            self.kept_indices = self.kv_cluster.kept_indices
            _cache_update(guikv_prefill_cache, cache_key_states, cache_value_states, self.layer_idx, cache_kwargs)
            _set_cache_layer(guikv_prefill_cache, self.layer_idx, cache_key_states, cache_value_states)
            _set_seen_tokens(guikv_prefill_cache, int(guikv_prefill_seen or q_len))
            _record_guikv_prefill_stats(
                self,
                original_seq_len=int(guikv_prefill_seen or q_len),
                compressed_seq_len=int(cache_key_states.shape[-2]),
                window_size=int(window_size),
                starts=starts,
                ends=ends,
                kept_indices=self.kept_indices,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return guikv_attention_forward


def _patch_outer_forward(model: torch.nn.Module) -> None:
    if getattr(model, "_guikv_outer_forward_patched", False):
        return
    orig_forward = model.forward

    def patched_forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", None)
        if input_ids is None and args:
            input_ids = args[0]
        attention_mask = kwargs.get("attention_mask", None)
        mm_token_type_ids = kwargs.get("mm_token_type_ids", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        image_token_id = getattr(getattr(self, "config", None), "image_token_id", None)
        starts, ends = _image_token_ranges_from_inputs(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            attention_mask=attention_mask,
        )
        text_cfg = getattr(getattr(self, "config", None), "text_config", None)
        if text_cfg is None:
            text_cfg = getattr(self, "config", None)
        if text_cfg is not None:
            text_cfg._guikv_vision_start_idx = starts
            text_cfg._guikv_vision_end_idx = ends
            text_cfg._guikv_image_count = len(starts)
        return orig_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    model._guikv_outer_forward_patched = True


def _patch_prepare_inputs_for_guikv(model: torch.nn.Module) -> None:
    if getattr(model, "_guikv_prepare_inputs_patched", False):
        return
    orig_prepare = getattr(model, "prepare_inputs_for_generation", None)
    if orig_prepare is None:
        return

    def patched_prepare_inputs_for_generation(
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
        model_inputs = orig_prepare(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if past_key_values is None:
            return model_inputs
        original_seen = getattr(past_key_values, "_guikv_original_seq_length", None)
        if original_seen is None:
            return model_inputs
        cur_input_ids = model_inputs.get("input_ids", None)
        if cur_input_ids is not None and getattr(cur_input_ids, "ndim", 0) == 2 and cur_input_ids.shape[1] > 1:
            model_inputs["input_ids"] = cur_input_ids[:, -1:]
        cur_inputs_embeds = model_inputs.get("inputs_embeds", None)
        if cur_inputs_embeds is not None and getattr(cur_inputs_embeds, "ndim", 0) == 3 and cur_inputs_embeds.shape[1] > 1:
            model_inputs["inputs_embeds"] = cur_inputs_embeds[:, -1:, :]
        cur_pos = model_inputs.get("position_ids", None)
        if cur_pos is not None and getattr(cur_pos, "ndim", 0) >= 2 and cur_pos.shape[-1] > 1:
            model_inputs["position_ids"] = cur_pos[..., -1:]
        cur_cache_position = model_inputs.get("cache_position", None)
        if cur_cache_position is not None and getattr(cur_cache_position, "ndim", 0) >= 1 and cur_cache_position.shape[-1] > 1:
            model_inputs["cache_position"] = cur_cache_position[-1:]
        cur_mm = model_inputs.get("mm_token_type_ids", None)
        if cur_mm is not None and getattr(cur_mm, "ndim", 0) == 2 and cur_mm.shape[1] > 1:
            model_inputs["mm_token_type_ids"] = cur_mm[:, -1:]
        if os.getenv("GUIKV_DEBUG_SHAPES", "0") == "1":
            def _shape(name):
                value = model_inputs.get(name, None)
                return tuple(value.shape) if hasattr(value, "shape") else None
            print(
                "[GUIKVPrepare] "
                f"input_ids={_shape('input_ids')} inputs_embeds={_shape('inputs_embeds')} "
                f"position_ids={_shape('position_ids')} attention_mask={_shape('attention_mask')} "
                f"cache_position={_shape('cache_position')} mm_token_type_ids={_shape('mm_token_type_ids')}",
                flush=True,
            )
        return model_inputs

    patched_prepare_inputs_for_generation.__signature__ = inspect.signature(patched_prepare_inputs_for_generation)
    model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, model)
    model._guikv_prepare_inputs_patched = True


def init_gui_kv(model: torch.nn.Module, config: GUIKVConfig | None = None) -> None:
    """Attach GUI-KV to a single Qwen3-VL model instance."""
    if getattr(model, "_guikv_initialized", False):
        return
    config = config or GUIKVConfig()
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        text_cfg = getattr(model, "config", None)
    if text_cfg is not None:
        text_cfg._guikv_max_capacity_prompt = (
            int(config.max_capacity_prompt) if config.max_capacity_prompt is not None else None
        )
        text_cfg._guikv_total_keep_ratio = float(config.total_keep_ratio)
        text_cfg._guikv_window_size = int(config.window_size)
        text_cfg._guikv_kernel_size = int(config.kernel_size)
        text_cfg._guikv_pooling = str(config.pooling)
        text_cfg._guikv_alpha = float(config.alpha)
        text_cfg._guikv_temperature = float(config.temperature)
        text_cfg._guikv_merge = config.merge

    layers = _find_language_layers(model)
    if not layers:
        raise RuntimeError("Could not find Qwen3-VL language layers for GUI-KV patching.")
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_guikv_forward_patched", False):
            continue
        attn._guikv_config = config
        attn.forward = types.MethodType(_make_guikv_attention_forward(attn), attn)
        attn._guikv_forward_patched = True
    _patch_outer_forward(model)
    _patch_prepare_inputs_for_guikv(model)
    model._guikv_initialized = True


def update_gui_kv_config(model: torch.nn.Module, **kwargs) -> None:
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        text_cfg = getattr(model, "config", None)
    if text_cfg is None:
        return
    mapping = {
        "max_capacity_prompt": "_guikv_max_capacity_prompt",
        "total_keep_ratio": "_guikv_total_keep_ratio",
        "window_size": "_guikv_window_size",
        "kernel_size": "_guikv_kernel_size",
        "pooling": "_guikv_pooling",
        "alpha": "_guikv_alpha",
        "temperature": "_guikv_temperature",
        "merge": "_guikv_merge",
    }
    for key, attr in mapping.items():
        if key in kwargs and kwargs[key] is not None:
            setattr(text_cfg, attr, kwargs[key])
