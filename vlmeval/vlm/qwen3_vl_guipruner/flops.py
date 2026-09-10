"""Auditable analytical FLOPs correction for GUIPruner-reproduction.

The repository runtime tracker observes one Qwen3-VL text-model forward at
prefill, whereas GUIPruner executes that forward in two sequence-length
segments.  This module replaces only that prefill component; the tracker keeps
its observed vision encoder and cached decode components.
"""

from __future__ import annotations

from typing import Mapping


def _text_config(model):
    config = getattr(model, "config", None)
    return getattr(config, "text_config", config)


def decoder_flops(model, *, query_tokens: int, key_tokens: int, layer_count: int) -> float:
    """Decoder FLOPs for the supplied number of identical Qwen3-VL layers."""
    cfg = _text_config(model)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    intermediate = int(getattr(cfg, "intermediate_size", 0) or 0)
    heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    kv_heads = int(getattr(cfg, "num_key_value_heads", heads) or heads)
    head_dim = int(getattr(cfg, "head_dim", hidden // max(heads, 1)) or 0)
    query_tokens, key_tokens, layer_count = int(query_tokens), int(key_tokens), int(layer_count)
    if min(query_tokens, key_tokens, layer_count, hidden, intermediate, heads, kv_heads, head_dim) <= 0:
        return 0.0
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    attention_linear = 2.0 * query_tokens * hidden * (q_out + kv_out + kv_out + q_out)
    attention_kernel = 4.0 * heads * query_tokens * key_tokens * head_dim
    mlp = (
        2.0 * query_tokens * hidden * intermediate
        + 2.0 * query_tokens * hidden * intermediate
        + 2.0 * query_tokens * intermediate * hidden
    )
    return float(layer_count) * float(attention_linear + attention_kernel + mlp)


def lm_head_flops(model, *, token_count: int) -> float:
    cfg = _text_config(model)
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    vocab = int(getattr(cfg, "vocab_size", 0) or 0)
    token_count = int(token_count)
    if min(token_count, hidden, vocab) <= 0:
        return 0.0
    return float(2.0 * token_count * hidden * vocab)


def correct_split_prefill_flops(
    model,
    generic_flops: Mapping[str, float],
    *,
    prompt_tokens_before_ssp: int,
    prompt_tokens_after_ssp: int,
    prune_layer_one_based: int,
) -> dict[str, float | int | str]:
    """Replace full-length prefill estimates with GUIPruner's split execution.

    ``generic_flops`` is the repository analytical runtime result.  It already
    observes decode using the physically trimmed cache.  The prefill decoder
    term is always replaced.  Older trackers estimated LM-head work from the
    input sequence length and require the same replacement; the current
    tracker hooks the actual LM-head input, which is already post-SSP.
    """
    cfg = _text_config(model)
    layer_total = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    before = int(prompt_tokens_before_ssp)
    after = int(prompt_tokens_after_ssp)
    split = int(prune_layer_one_based)
    if min(layer_total, before, after, split) <= 0 or split > layer_total or after > before:
        raise RuntimeError("Invalid GUIPruner split FLOPs dimensions.")

    generic_llm = float(generic_flops.get("llm_flops", 0.0) or 0.0)
    generic_head = float(generic_flops.get("lm_head_flops", 0.0) or 0.0)
    vision = float(generic_flops.get("vision_flops", 0.0) or 0.0)
    full_prefill_llm = decoder_flops(model, query_tokens=before, key_tokens=before, layer_count=layer_total)
    split_prefill_llm = (
        decoder_flops(model, query_tokens=before, key_tokens=before, layer_count=split)
        + decoder_flops(model, query_tokens=after, key_tokens=after, layer_count=layer_total - split)
    )
    full_prefill_head = lm_head_flops(model, token_count=before)
    split_prefill_head = lm_head_flops(model, token_count=after)
    corrected_llm = max(0.0, generic_llm - full_prefill_llm + split_prefill_llm)
    lm_head_measured = bool(generic_flops.get("lm_head_measured", False))
    corrected_head = (
        generic_head
        if lm_head_measured
        else max(0.0, generic_head - full_prefill_head + split_prefill_head)
    )
    corrected_e2e = vision + corrected_llm + corrected_head
    return {
        "vision_flops": vision,
        "llm_flops": corrected_llm,
        "lm_head_flops": corrected_head,
        "e2e_flops": corrected_e2e,
        "forward_steps": int(generic_flops.get("forward_steps", 0) or 0),
        "lm_head_measured": lm_head_measured,
        "estimation": "GUIPruner-reproduction analytical split-prefill; vision/decode from common runtime tracker",
        "prefill_full_length_llm_flops_replaced": full_prefill_llm,
        "prefill_split_llm_flops": split_prefill_llm,
        "prefill_full_length_lm_head_flops_replaced": full_prefill_head,
        "prefill_split_lm_head_flops": split_prefill_head,
        "prefill_tokens_before_ssp": before,
        "prefill_tokens_after_ssp": after,
        "prune_layer_one_based": split,
    }
