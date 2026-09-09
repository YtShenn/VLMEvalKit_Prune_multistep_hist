"""Qwen3-VL-specific, fail-closed layer-2 SSP patch.

This module is intentionally self contained.  It neither imports nor calls any
other pruning backend in this repository.
"""

from __future__ import annotations

import types
from typing import Any

import torch

from .config import GUIPrunerConfig, resolve_current_keep_ratio
from .ssp import foreground_mask, select_stratified


def _current_grid(model: torch.nn.Module) -> tuple[int, int, int]:
    grid = getattr(model.config.text_config, "_vlmeval_current_image_grid_thw", None)
    if grid is None:
        raise RuntimeError("GUIPruner SSP needs image_grid_thw from the Qwen3-VL processor.")
    row = grid[-1].detach().cpu().tolist() if isinstance(grid, torch.Tensor) else grid[-1]
    merge = int(model.config.vision_config.spatial_merge_size)
    t, h, w = (int(x) for x in row)
    if h % merge or w % merge:
        raise RuntimeError("GUIPruner SSP cannot map a non-merged Qwen3-VL visual grid safely.")
    return t, h // merge, w // merge


def _trim_prefill_cache(cache: Any, keep: torch.Tensor, layer_count: int) -> None:
    """Trim the first two layers' prompt KV states after their full-context pass."""
    if cache is None:
        return
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) < layer_count:
        raise RuntimeError("Unsupported Qwen3-VL cache object: cannot preserve SSP generation KV cache.")
    for layer in layers[:layer_count]:
        for name in ("keys", "values"):
            tensor = getattr(layer, name, None)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise RuntimeError("Unsupported Qwen3-VL cache layout: expected [B,H,S,D] keys and values.")
            setattr(layer, name, tensor.index_select(2, keep.to(tensor.device)))


def _capture_attention(layer: torch.nn.Module) -> None:
    original = layer.self_attn.forward

    def wrapped(*args, **kwargs):
        output, weights = original(*args, **kwargs)
        layer._guipruner_last_attention = weights
        return output, weights

    layer.self_attn.forward = wrapped


def _patch_compute_positions(outer_model: torch.nn.Module) -> None:
    original = outer_model.compute_3d_position_ids

    def wrapped(self, *args, **kwargs):
        past = kwargs.get("past_key_values")
        embeds = kwargs.get("inputs_embeds")
        next_pos = getattr(self, "_guipruner_next_position_ids", None)
        if past is not None and next_pos is not None and embeds is not None:
            length = int(embeds.shape[1])
            result = next_pos.to(embeds.device).expand(-1, embeds.shape[0], length)
            self._guipruner_next_position_ids = result[:, :1, -1:] + 1
            return result
        return original(*args, **kwargs)

    outer_model.compute_3d_position_ids = types.MethodType(wrapped, outer_model)


def init_guipruner(model: torch.nn.Module, config: GUIPrunerConfig) -> None:
    """Install the exact layer-2 patch once, or fail before any inference happens."""
    outer = getattr(model, "model", None)
    text = getattr(outer, "language_model", None)
    layers = getattr(text, "layers", None)
    if outer is None or text is None or layers is None or len(layers) < config.prune_layer:
        raise RuntimeError("GUIPruner-reproduction requires the HuggingFace Qwen3-VL model layout.")
    if getattr(text, "_guipruner_installed", False):
        return
    # Qwen3-VL only returns attention weights through eager attention.  This is
    # deliberately explicit: SSP must use shallow-layer attention, never a proxy.
    model.config.text_config._attn_implementation = "eager"
    text.config._attn_implementation = "eager"
    layer_index = config.prune_layer - 1  # paper layer 2 -> zero-based index 1
    _capture_attention(layers[layer_index])
    original_forward = text.forward

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        **kwargs,
    ):
        # Decode tokens must use the original implementation.  The outer patch
        # supplies position ids continued from the retained prefill sequence.
        if past_key_values is not None or inputs_embeds is None or inputs_embeds.shape[0] != 1:
            return original_forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds, **kwargs,
            )
        if input_ids is not None:
            raise RuntimeError("GUIPruner expected Qwen3-VL multimodal inputs_embeds prefill path.")
        # This is the upstream Qwen3VLTextModel forward loop, split after layer 2.
        try:
            from transformers.cache_utils import DynamicCache
            from transformers.masking_utils import create_causal_mask
            from transformers.modeling_outputs import BaseModelOutputWithPast
        except Exception as exc:
            raise RuntimeError("GUIPruner requires the installed transformers Qwen3-VL internals.") from exc
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            raise RuntimeError("GUIPruner cannot preserve M-RoPE without explicit Qwen3-VL position_ids.")
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids, rope_position_ids = position_ids[0], position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            text_position_ids, rope_position_ids = None, position_ids
        else:
            raise RuntimeError("Unexpected Qwen3-VL M-RoPE position_ids shape.")
        causal = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=text_position_ids,
        )
        hidden = inputs_embeds
        pos_emb = self.rotary_emb(hidden, rope_position_ids)
        for idx in range(layer_index + 1):
            hidden = self.layers[idx](hidden, attention_mask=causal, position_ids=text_position_ids,
                                      past_key_values=past_key_values, position_embeddings=pos_emb, **kwargs)
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                hidden = self._deepstack_process(hidden, visual_pos_masks, deepstack_visual_embeds[idx])
        weights = getattr(self.layers[layer_index], "_guipruner_last_attention", None)
        if not isinstance(weights, torch.Tensor):
            raise RuntimeError("GUIPruner requires eager layer-2 attention weights; none were returned.")
        if visual_pos_masks is None or visual_pos_masks.ndim != 2:
            raise RuntimeError("GUIPruner requires Qwen3-VL visual_pos_masks for exact current-frame isolation.")
        t, gh, gw = _current_grid(model)
        current_n = t * gh * gw
        visual_positions = torch.where(visual_pos_masks[0])[0]
        if visual_positions.numel() < current_n:
            raise RuntimeError("Current visual token count exceeds Qwen3-VL visual placeholder count.")
        current_positions = visual_positions[-current_n:]
        image = getattr(model.config.text_config, "_vlmeval_current_vis_image_pil", None)
        if image is None:
            raise RuntimeError("GUIPruner needs the current PIL image cached by Qwen3-VL input preparation.")
        fg = foreground_mask(image, gh, gw).to(hidden.device)
        if t != 1:
            fg = fg.repeat(t)
        # Aggregate layer-2 attention over heads and all query positions.
        scores = weights[0, :, :, current_positions].float().mean(dim=(0, 1))
        history_original = int(getattr(model.config.text_config, "_guipruner_history_original_visual_tokens", 0) or 0)
        if config.overall_keep_ratio is None:
            current_keep_ratio = config.current_keep_ratio
            budget_mode = "paper_separate"
        else:
            current_keep_ratio = resolve_current_keep_ratio(
                history_original_tokens=history_original,
                current_original_tokens=current_n,
                history_keep_ratio=config.history_keep_ratio,
                overall_keep_ratio=config.overall_keep_ratio,
            )
            budget_mode = "global_history_plus_current"
        selection = select_stratified(scores, fg, current_keep_ratio, config.background_saliency)
        retained_current = current_positions[selection.final]
        keep_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        keep_mask[current_positions] = False
        keep_mask[retained_current] = True
        keep = torch.where(keep_mask)[0]
        _trim_prefill_cache(past_key_values, keep, config.prune_layer)
        hidden = hidden.index_select(1, keep)
        rope_position_ids = rope_position_ids.index_select(2, keep)
        text_position_ids = text_position_ids.index_select(1, keep) if text_position_ids is not None else None
        raw_mask = attention_mask.index_select(1, keep) if attention_mask is not None else None
        visual_keep = visual_pos_masks.index_select(1, keep)
        if deepstack_visual_embeds is not None:
            visual_selected = keep_mask[visual_positions]
            deepstack_visual_embeds = [x.index_select(0, torch.where(visual_selected)[0].to(x.device)) for x in deepstack_visual_embeds]
        causal = create_causal_mask(config=self.config, inputs_embeds=hidden, attention_mask=raw_mask,
                                    past_key_values=past_key_values, position_ids=text_position_ids)
        pos_emb = self.rotary_emb(hidden, rope_position_ids)
        for idx in range(layer_index + 1, len(self.layers)):
            hidden = self.layers[idx](hidden, attention_mask=causal, position_ids=text_position_ids,
                                      past_key_values=past_key_values, position_embeddings=pos_emb, **kwargs)
            if deepstack_visual_embeds is not None and idx < len(deepstack_visual_embeds):
                hidden = self._deepstack_process(hidden, visual_keep, deepstack_visual_embeds[idx])
        # Preserve the M-RoPE continuation independently of compressed KV length.
        outer._guipruner_next_position_ids = rope_position_ids[:, :, -1:] + 1
        model.config.text_config._guipruner_last_stats = {
            "implementation": "GUIPruner-reproduction (unofficial reproduction)",
            "prune_layer_one_based": config.prune_layer,
            "budget_mode": budget_mode,
            "history_original_visual_tokens": history_original,
            "history_keep_ratio": config.history_keep_ratio,
            "overall_keep_ratio": config.overall_keep_ratio,
            "effective_current_keep_ratio": current_keep_ratio,
            "current_visual_tokens_before": int(current_n),
            "current_visual_tokens_after": int(selection.final.numel()),
            "foreground_tokens": int(selection.foreground.numel()),
            "background_tokens": int(selection.background.numel()),
            "uniform_tokens": int(selection.uniform.numel()),
        }
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden), past_key_values=past_key_values)

    text.forward = types.MethodType(patched_forward, text)
    _patch_compute_positions(outer)
    text._guipruner_installed = True
