from __future__ import annotations

import importlib
import inspect
import os
import time
import types
from typing import Any

import torch

from .config import STLiteConfig
from .kv_cache import STLiteKVCluster, image_token_ranges_from_inputs


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
        cache._stlite_original_seq_length = int(seen_tokens)
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


def _record_layer_stats(attn_module, stats: dict, original_seq_len: int, prefill_s: float, compression_s: float) -> None:
    cfg = getattr(attn_module, "config", None)
    if cfg is None:
        return
    layer_idx = int(getattr(attn_module, "layer_idx", 0))
    all_stats = dict(getattr(cfg, "_stlite_last_stats", {}) or {})
    per_layer = dict(all_stats.get("ST_LITE_per_layer", {}) or {})
    layer_stats = dict(stats or {})
    layer_stats["ST_LITE_layer_idx"] = layer_idx
    per_layer[str(layer_idx)] = layer_stats
    layer_tokens = [int(v.get("layer_keep_tokens", 0) or 0) for v in per_layer.values()]
    all_stats.update(
        {
            "ST_LITE_original_prefill_tokens": int(original_seq_len),
            "ST_LITE_kv_tokens_before": int(original_seq_len),
            "ST_LITE_kv_tokens_after": int(layer_stats.get("compressed_seq_len", original_seq_len)),
            "ST_LITE_layer_keep_tokens": layer_tokens,
            "ST_LITE_history_visual_tokens": int(layer_stats.get("history_visual_tokens", 0) or 0),
            "ST_LITE_history_visual_tokens_after": int(
                layer_stats.get("history_visual_tokens_after", 0) or 0
            ),
            "ST_LITE_history_visual_retention_rate": float(
                layer_stats.get("history_visual_retention_rate", 0.0) or 0.0
            ),
            "ST_LITE_current_visual_tokens": int(layer_stats.get("current_visual_tokens", 0) or 0),
            "ST_LITE_text_action_tokens": int(layer_stats.get("text_action_tokens", 0) or 0),
            "ST_LITE_prefill_latency_s": float(prefill_s),
            "ST_LITE_compression_latency_s": float(compression_s),
            "ST_LITE_scoring_overhead_s": float(layer_stats.get("scoring_overhead_s", 0.0) or 0.0),
            "ST_LITE_compression_ratio": float(
                layer_stats.get("compressed_seq_len", original_seq_len) / max(1, original_seq_len)
            ),
            "ST_LITE_per_layer": per_layer,
        }
    )
    cfg._stlite_last_stats = all_stats


def _make_stlite_attention_forward(attn_module):
    module_globals = importlib.import_module(attn_module.__class__.__module__)
    apply_rotary_pos_emb = getattr(module_globals, "apply_rotary_pos_emb")
    eager_attention_forward = getattr(module_globals, "eager_attention_forward")

    def stlite_attention_forward(
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
        if q_len == 1 and cache is not None:
            # Qwen3-VL uses cache_position both for RoPE construction and for
            # cache writes. After compression those positions differ: keep
            # the logical value for RoPE, but append at the physical cache end.
            try:
                physical_position = int(cache.get_seq_length(self.layer_idx))
                cache_kwargs["cache_position"] = torch.tensor(
                    [physical_position], device=key_states.device, dtype=torch.long
                )
            except Exception:
                pass

        stlite_prefill_cache = None
        original_seen = int(getattr(self, "_stlite_seen_tokens", q_len))
        if cache is not None:
            if q_len > 1:
                original_seen = q_len
                stlite_prefill_cache = cache
                self._stlite_seen_tokens = q_len
                self._stlite_prefill_start_s = time.perf_counter()
            else:
                key_states, value_states = _cache_update(cache, key_states, value_states, self.layer_idx, cache_kwargs)
                self._stlite_seen_tokens = original_seen + q_len
                _set_seen_tokens(cache, self._stlite_seen_tokens)

        attention_interface = _attention_interface(self, eager_attention_forward)
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

        if stlite_prefill_cache is not None:
            text_cfg = self.config
            st_cfg = getattr(text_cfg, "_stlite_config", STLiteConfig())
            ranges = list(getattr(text_cfg, "_stlite_vision_ranges", []) or [])
            if not hasattr(self, "kv_cluster"):
                self.kv_cluster = STLiteKVCluster(st_cfg, ranges)
            self.kv_cluster.reset(st_cfg, ranges)
            compress_start = time.perf_counter()
            cache_key_states, cache_value_states = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
                hidden_states=hidden_states,
            )
            compression_s = time.perf_counter() - compress_start
            self.kept_indices = self.kv_cluster.kept_indices
            _cache_update(stlite_prefill_cache, cache_key_states, cache_value_states, self.layer_idx, cache_kwargs)
            _set_cache_layer(stlite_prefill_cache, self.layer_idx, cache_key_states, cache_value_states)
            _set_seen_tokens(stlite_prefill_cache, int(original_seen))
            try:
                stlite_prefill_cache._stlite_decode_steps = 0
            except Exception:
                pass
            prefill_s = max(0.0, compress_start - float(getattr(self, "_stlite_prefill_start_s", compress_start)))
            _record_layer_stats(self, self.kv_cluster.last_stats, int(original_seen), prefill_s, compression_s)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return stlite_attention_forward


def _patch_outer_forward(model: torch.nn.Module) -> None:
    if getattr(model, "_stlite_outer_forward_patched", False):
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
        ranges = image_token_ranges_from_inputs(
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
            text_cfg._stlite_vision_ranges = ranges
            text_cfg._stlite_image_count = len(ranges)
        return orig_forward(*args, **kwargs)

    model.forward = types.MethodType(patched_forward, model)
    model._stlite_outer_forward_patched = True


def _patch_prepare_inputs_for_stlite(model: torch.nn.Module) -> None:
    if getattr(model, "_stlite_prepare_inputs_patched", False):
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
        if past_key_values is None or getattr(past_key_values, "_stlite_original_seq_length", None) is None:
            return model_inputs
        for key in ("input_ids", "mm_token_type_ids"):
            val = model_inputs.get(key, None)
            if val is not None and getattr(val, "ndim", 0) == 2 and val.shape[1] > 1:
                model_inputs[key] = val[:, -1:]
        embeds = model_inputs.get("inputs_embeds", None)
        if embeds is not None and getattr(embeds, "ndim", 0) == 3 and embeds.shape[1] > 1:
            model_inputs["inputs_embeds"] = embeds[:, -1:, :]
        pos = model_inputs.get("position_ids", None)
        if pos is not None and getattr(pos, "ndim", 0) >= 2 and pos.shape[-1] > 1:
            model_inputs["position_ids"] = pos[..., -1:]
        return model_inputs

    patched_prepare_inputs_for_generation.__signature__ = inspect.signature(patched_prepare_inputs_for_generation)
    model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, model)
    model._stlite_prepare_inputs_patched = True


def init_stlite(model: torch.nn.Module, config: STLiteConfig | None = None) -> None:
    if getattr(model, "_stlite_initialized", False):
        return
    config = config or STLiteConfig()
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        text_cfg = getattr(model, "config", None)
    if text_cfg is not None:
        text_cfg._stlite_config = config
        text_cfg._stlite_last_stats = {}

    layers = _find_language_layers(model)
    if not layers:
        raise RuntimeError("Could not find Qwen3-VL language layers for ST-Lite patching.")
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_stlite_forward_patched", False):
            continue
        attn.forward = types.MethodType(_make_stlite_attention_forward(attn), attn)
        attn._stlite_forward_patched = True
    _patch_outer_forward(model)
    _patch_prepare_inputs_for_stlite(model)
    model._stlite_initialized = True


def update_stlite_config(model: torch.nn.Module, **kwargs) -> None:
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        text_cfg = getattr(model, "config", None)
    if text_cfg is None:
        return
    cfg = getattr(text_cfg, "_stlite_config", STLiteConfig())
    for key, value in kwargs.items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    text_cfg._stlite_config = STLiteConfig.from_env(**cfg.__dict__)


def get_stlite_stats(model: torch.nn.Module) -> dict:
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    if text_cfg is None:
        text_cfg = getattr(model, "config", None)
    return dict(getattr(text_cfg, "_stlite_last_stats", {}) or {}) if text_cfg is not None else {}
