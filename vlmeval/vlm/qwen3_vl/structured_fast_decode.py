from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import torch


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)


def _empty_meta(reason: str) -> dict[str, Any]:
    return {
        "template_prefill_enabled": False,
        "template_prefill_fallback_reason": reason,
        "template_prefill_impl": "structured_fast_decode_v1",
        "template_prefill_backend_impl": "direct_forward_kv_cache",
        "template_prefill_requested_impl": "structured_fast_decode_v1",
        "template_schema": None,
        "template_static_parts": [],
        "template_slot_stats": [],
        "template_static_token_count": 0,
        "template_static_decode_steps": 0,
        "template_unknown_decode_steps": 0,
        "template_decode_tokens": 0,
    }


GUI_ACTION_HEADS = (
    "CLICK",
    "LONG_PRESS",
    "SCROLL",
    "TYPE",
    "PRESS_HOME",
    "PRESS_BACK",
    "PRESS_RECENT",
    "COMPLETE",
    "IMPOSSIBLE",
)

GUI_SCROLL_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
GUI_TERMINAL_ACTIONS = ("PRESS_HOME", "PRESS_BACK", "PRESS_RECENT", "COMPLETE", "IMPOSSIBLE")
GUI_COORD_ACTIONS = ("CLICK", "LONG_PRESS")
AITW_SEMANTIC_ACTIONS = (
    "click",
    "input_text",
    "input text",
    "type",
    "scroll down",
    "scroll up",
    "scroll left",
    "scroll right",
    "scroll_down",
    "scroll_up",
    "scroll_left",
    "scroll_right",
    "press back",
    "press home",
    "press_back",
    "press_home",
    "navigate_back",
    "navigate_home",
    "navigate back",
    "navigate home",
    "enter",
    "complete",
)


def _dataset_family(dataset: str | None) -> str:
    name = str(dataset or "")
    if name.startswith("AndroidControl"):
        return "androidcontrol"
    if name.startswith("AITW"):
        return "aitw"
    if name.startswith("GUIOdyssey"):
        return "guiodyssey"
    if name.startswith("Mind2Web"):
        return "mind2web"
    return ""


def _batch_to_dict(inputs) -> dict[str, Any]:
    if isinstance(inputs, dict):
        return dict(inputs)
    try:
        return dict(inputs)
    except Exception:
        return {k: getattr(inputs, k) for k in dir(inputs) if not k.startswith("_")}


def _get_input_ids(inputs) -> torch.Tensor | None:
    if isinstance(inputs, dict):
        return inputs.get("input_ids", None)
    return getattr(inputs, "input_ids", None)


def _get_attention_mask(inputs) -> torch.Tensor | None:
    if isinstance(inputs, dict):
        return inputs.get("attention_mask", None)
    return getattr(inputs, "attention_mask", None)


def _tokenize(tokenizer, text: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    return ids.to(device=device)


def _continuation_token_ids(tokenizer, prefix_text: str, append_text: str) -> tuple[list[int], bool]:
    prefix_text = str(prefix_text or "")
    append_text = str(append_text or "")
    if not append_text:
        return [], True
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix_text + append_text, add_special_tokens=False).input_ids
    if len(full_ids) >= len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        return list(full_ids[len(prefix_ids):]), True
    return list(tokenizer(append_text, add_special_tokens=False).input_ids), False


def _decode_ids(tokenizer, ids: list[int]) -> str:
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)



def _cache_seq_len(past_key_values, fallback: int) -> int:
    if past_key_values is None:
        return int(fallback)
    try:
        return int(past_key_values.get_seq_length())
    except Exception:
        pass
    try:
        return int(past_key_values[0][0].shape[-2])
    except Exception:
        return int(fallback)


def _align_attention_mask_to_past(
    attention_mask: torch.Tensor,
    past_key_values,
    device: torch.device,
) -> torch.Tensor:
    past_len = _cache_seq_len(past_key_values, int(attention_mask.shape[1]))
    if past_len <= 0 or int(attention_mask.shape[1]) == past_len:
        return attention_mask
    return torch.ones((attention_mask.shape[0], past_len), dtype=attention_mask.dtype, device=device)


def _select_next_token(logits: torch.Tensor, generate_kwargs: dict[str, Any]) -> int:
    next_logits = logits[:, -1, :]
    do_sample = bool(generate_kwargs.get("do_sample", False))
    if not do_sample:
        return int(torch.argmax(next_logits, dim=-1).item())

    temperature = float(generate_kwargs.get("temperature", 1.0) or 1.0)
    if temperature > 0:
        next_logits = next_logits / temperature

    top_k = int(generate_kwargs.get("top_k", 0) or 0)
    if top_k > 0 and top_k < next_logits.shape[-1]:
        values, _ = torch.topk(next_logits, top_k)
        min_values = values[:, -1, None]
        next_logits = torch.where(next_logits < min_values, torch.full_like(next_logits, -float("inf")), next_logits)

    top_p = float(generate_kwargs.get("top_p", 1.0) or 1.0)
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        next_logits = torch.full_like(next_logits, -float("inf"))
        next_logits.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(next_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _model_forward(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    extra_inputs: dict[str, Any] | None = None,
    logits_to_keep: int | None = None,
):
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "return_dict": True,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    if extra_inputs:
        kwargs.update(extra_inputs)
    return model(**kwargs)


def _decode_position_kwargs(model, past_key_values, q_len: int, device: torch.device) -> dict[str, torch.Tensor]:
    past_len = _cache_seq_len(past_key_values, 0)
    positions = torch.arange(past_len, past_len + int(q_len), dtype=torch.long, device=device)
    kwargs = {"cache_position": positions}
    prune_offset = 0

    # Qwen3-VL uses multimodal RoPE deltas after the first prefill. If we omit
    # position_ids here, the model recomputes them from the full attention_mask
    # and can produce a full-context position tensor for a q_len continuation.
    # Build only the continuation segment while preserving the cached delta.
    qwen_model = getattr(model, "model", None)
    rope_deltas = getattr(qwen_model, "rope_deltas", None)
    if torch.is_tensor(rope_deltas):
        text_model = getattr(qwen_model, "language_model", None)
        text_config = getattr(text_model, "config", None)
        prune_offset = int(getattr(text_config, "_attn_prune_cache_position_offset", 0) or 0)
        logical_positions = positions + prune_offset
        delta = rope_deltas.to(device=device, dtype=torch.long)
        batch = int(delta.shape[0]) if delta.ndim >= 1 else 1
        base = logical_positions.view(1, 1, -1).expand(3, batch, -1)
        delta = delta.view(1, batch, 1)
        kwargs["position_ids"] = base + delta
    if _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG_POS", "0"):
        print(
            f"[StructuredFastDecodePos] past_len={past_len} q_len={int(q_len)} prune_offset={prune_offset}",
            flush=True,
        )
    return kwargs


def _continuation_forward(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
    device: torch.device,
):
    pos_kwargs = _decode_position_kwargs(model, past_key_values, int(input_ids.shape[1]), device)
    if hasattr(model, "prepare_inputs_for_generation"):
        try:
            prepared = model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                is_first_iteration=False,
                **pos_kwargs,
            )
            prepared["use_cache"] = True
            prepared["return_dict"] = True
            prepared["logits_to_keep"] = 1
            return model(**prepared)
        except Exception:
            pass
    return _model_forward(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        logits_to_keep=1,
        **pos_kwargs,
    )


def _forward_token_ids(
    model,
    *,
    token_ids: list[int],
    past_key_values,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
    device: torch.device,
    tokenwise: bool = False,
) -> tuple[Any, torch.Tensor, torch.Tensor, int, float]:
    if not token_ids:
        return past_key_values, attention_mask, logits, 0, 0.0
    elapsed = 0.0
    forwarded = 0
    chunks = ([int(x)] for x in token_ids) if tokenwise else (list(int(x) for x in token_ids),)
    for chunk in chunks:
        ids = torch.tensor([list(chunk)], dtype=torch.long, device=device)
        n_tok = int(ids.shape[1])
        if n_tok == 0:
            continue
        attention_mask = _align_attention_mask_to_past(attention_mask, past_key_values, device)
        new_mask = torch.ones((attention_mask.shape[0], n_tok), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)
        start = time.perf_counter()
        out = _continuation_forward(
            model,
            input_ids=ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            device=device,
        )
        elapsed += time.perf_counter() - start
        past_key_values = out.past_key_values
        logits = out.logits
        forwarded += n_tok
    return past_key_values, attention_mask, logits, forwarded, elapsed


def _append_static(
    model,
    tokenizer,
    *,
    text: str,
    past_key_values,
    attention_mask: torch.Tensor,
    logits: torch.Tensor,
    device: torch.device,
    prefix_text: str | None = None,
) -> tuple[Any, torch.Tensor, torch.Tensor, int, float]:
    if prefix_text is not None:
        token_ids, ok = _continuation_token_ids(tokenizer, prefix_text, text)
        if not ok:
            raise RuntimeError("static_token_boundary_shift")
        ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        ids = _tokenize(tokenizer, text, device)
    token_ids = [int(x) for x in ids[0].tolist()]
    tokenwise = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_STATIC_TOKENWISE", "0")
    return _forward_token_ids(
        model,
        token_ids=token_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        logits=logits,
        device=device,
        tokenwise=tokenwise,
    )


def _decode_until(
    model,
    tokenizer,
    *,
    logits: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    device: torch.device,
    generate_kwargs: dict[str, Any],
    stop_chars: tuple[str, ...],
    max_tokens: int,
    slot_name: str,
) -> tuple[str, str, list[int], Any, torch.Tensor, torch.Tensor, int, bool, str | None, float]:
    ids: list[int] = []
    rendered = ""
    raw = ""
    steps = 0
    elapsed = 0.0
    eos_ids = generate_kwargs.get("eos_token_id", None)
    if eos_ids is None:
        eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    elif isinstance(eos_ids, (list, tuple, set)):
        eos_set = {int(x) for x in eos_ids if x is not None}
    else:
        eos_set = set()

    for _ in range(max_tokens):
        token_id = _select_next_token(logits, generate_kwargs)
        ids.append(token_id)
        steps += 1
        raw = _decode_ids(tokenizer, ids)
        stop_idx = None
        for ch in stop_chars:
            idx = raw.find(ch)
            if idx >= 0:
                stop_idx = idx if stop_idx is None else min(stop_idx, idx)
        rendered = raw if stop_idx is None else raw[:stop_idx]

        past_key_values, attention_mask, logits, _, step_elapsed = _forward_token_ids(
            model,
            token_ids=[token_id],
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            logits=logits,
            device=device,
            tokenwise=False,
        )
        elapsed += step_elapsed

        if stop_idx is not None:
            return rendered.strip(), raw, ids, past_key_values, attention_mask, logits, steps, True, None, elapsed
        if token_id in eos_set:
            return rendered.strip(), raw, ids, past_key_values, attention_mask, logits, steps, False, f"{slot_name}_hit_eos", elapsed

    return rendered.strip(), raw, ids, past_key_values, attention_mask, logits, steps, False, f"{slot_name}_max_tokens", elapsed


def _normalize_action(action: str) -> str:
    text = str(action or "").strip().strip('"').strip("'").strip().lower()
    if "tap" in text:
        text = text.replace("tap", "click")
    if "long press" in text:
        text = text.replace("long press", "long_press")
    return text


def _canonical_aitw_semantic_action(action: str) -> str:
    text = str(action or "").strip().strip('"').strip("'").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    if text in {"tap", "click"}:
        return "click"
    if text in {"input text", "input_text", "type"}:
        return "input_text"
    if text in {"swipe up", "scroll down"}:
        return "scroll down"
    if text in {"swipe down", "scroll up"}:
        return "scroll up"
    if text in {"swipe right", "scroll left"}:
        return "scroll left"
    if text in {"swipe left", "scroll right"}:
        return "scroll right"
    if text in {"back", "press back", "navigate back", "navigate_back"}:
        return "navigate_back"
    if text in {"home", "press home", "navigate home", "navigate_home"}:
        return "navigate_home"
    if text in {"enter", "press enter"}:
        return "enter"
    if text in {"complete", "done"}:
        return "complete"
    return text


def _looks_like_aitw_action_id_payload(text: str) -> bool:
    stripped = str(text or "").strip().strip('"').strip("'").strip()
    if stripped in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}:
        return True
    nums = re.findall(r"\b(?:0|1|2|3|4|5|6|7|8|9|10)\b", stripped)
    tokens = re.findall(r"\w+", stripped)
    return bool(tokens) and len(nums) == len(tokens)


def _suffix_after_first_stop(text: str, stop_chars: tuple[str, ...]) -> str | None:
    raw = str(text or "")
    best = None
    best_ch = None
    for ch in stop_chars:
        idx = raw.find(ch)
        if idx >= 0 and (best is None or idx < best):
            best = idx
            best_ch = ch
    if best is None or best_ch is None:
        return None
    return raw[best + len(best_ch):]


def _tail_after_cached_suffix(suffix: str, default_tail: str) -> str | None:
    suffix = str(suffix or "")
    if suffix == "":
        return default_tail
    if suffix.startswith("</answer>"):
        return ""
    if suffix.startswith("}"):
        return "</answer>"
    return None


def _normalize_gui_closed_set(text: str) -> str:
    return str(text or "").strip().upper()


def _match_gui_closed_set(
    raw: str,
    candidates: tuple[str, ...],
    allow_boundary_chars: str,
    *,
    accept_exact: bool = True,
) -> tuple[bool, str, str, str | None]:
    text = str(raw or "").lstrip()
    upper = text.upper()
    stripped_upper = upper.strip().upper()
    for cand in candidates:
        cand_upper = cand.upper()
        if upper == cand_upper:
            if accept_exact:
                return True, cand, text[: len(cand)], ""
            return False, "", "", None
        if upper.startswith(cand_upper):
            suffix = text[len(cand):]
            if suffix and suffix[0] in allow_boundary_chars and suffix.strip(allow_boundary_chars + " \n\r\t") == "":
                return True, cand, text[: len(cand)], suffix
    if any(cand.upper().startswith(stripped_upper) for cand in candidates):
        return False, "", "", None
    return False, "", "", "closed_set_no_prefix"


def _decode_gui_closed_set(
    model,
    tokenizer,
    *,
    logits: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    device: torch.device,
    generate_kwargs: dict[str, Any],
    candidates: tuple[str, ...],
    allow_boundary_chars: str,
    max_tokens: int,
    slot_name: str,
    accept_exact: bool = True,
) -> tuple[str, str, str, list[int], Any, torch.Tensor, torch.Tensor, int, bool, str | None, float]:
    ids: list[int] = []
    raw = ""
    elapsed = 0.0
    eos_ids = generate_kwargs.get("eos_token_id", None)
    if eos_ids is None:
        eos_ids = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_ids, int):
        eos_set = {eos_ids}
    elif isinstance(eos_ids, (list, tuple, set)):
        eos_set = {int(x) for x in eos_ids if x is not None}
    else:
        eos_set = set()

    for _ in range(max_tokens):
        token_id = _select_next_token(logits, generate_kwargs)
        ids.append(token_id)
        raw = _decode_ids(tokenizer, ids)

        past_key_values, attention_mask, logits, _, step_elapsed = _forward_token_ids(
            model,
            token_ids=[token_id],
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            logits=logits,
            device=device,
            tokenwise=False,
        )
        elapsed += step_elapsed

        done, canonical, rendered, suffix_or_reason = _match_gui_closed_set(
            raw, candidates, allow_boundary_chars, accept_exact=accept_exact
        )
        if done:
            return canonical, raw, suffix_or_reason or "", ids, past_key_values, attention_mask, logits, len(ids), True, None, elapsed
        if suffix_or_reason == "closed_set_no_prefix":
            return "", raw, "", ids, past_key_values, attention_mask, logits, len(ids), False, f"{slot_name}_no_prefix", elapsed
        if token_id in eos_set:
            return "", raw, "", ids, past_key_values, attention_mask, logits, len(ids), False, f"{slot_name}_hit_eos", elapsed

    return "", raw, "", ids, past_key_values, attention_mask, logits, len(ids), False, f"{slot_name}_max_tokens", elapsed


def _gui_static_tail_from_suffix(suffix: str, expected: str) -> str | None:
    suffix = str(suffix or "")
    if suffix == "":
        return expected
    if expected.startswith(suffix):
        return expected[len(suffix):]
    return None


def _gui_rendered_prefix(raw: str, suffix: str, fallback: str) -> str:
    text = str(raw or "").lstrip()
    suffix = str(suffix or "")
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    text = text.strip()
    return text or str(fallback or "")


def _valid_gui_coord_pair(text: str) -> bool:
    parts = [p.strip() for p in str(text or "").split(",")]
    if len(parts) != 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
    except Exception:
        return False
    return True


def _normalize_mind2web_coord_pair(text: str) -> tuple[str, bool]:
    cleaned = str(text or "").strip()
    cleaned = cleaned.strip("[]() ")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2:
        return cleaned, False
    out = []
    try:
        for part in parts:
            value = float(part)
            if not (0.0 <= value <= 1000.0):
                return cleaned, False
            if abs(value - round(value)) < 1e-6:
                out.append(str(int(round(value))))
            else:
                out.append(str(value))
    except Exception:
        return cleaned, False
    return ",".join(out), True


def _escape_mind2web_value(text: str) -> str:
    return json.dumps(str(text or "").strip(), ensure_ascii=False)



def _common_inputs_or_fallback(processor, inputs):
    tokenizer = processor.tokenizer
    input_ids = _get_input_ids(inputs)
    attention_mask = _get_attention_mask(inputs)
    if input_ids is None:
        return tokenizer, None, None, None, None, _empty_meta("missing_input_ids")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    if int(input_ids.shape[0]) != 1:
        return tokenizer, input_ids, attention_mask, input_ids.device, None, _empty_meta("batch_size_not_one")
    inputs_dict = _batch_to_dict(inputs)
    extra_inputs = {
        k: v
        for k, v in inputs_dict.items()
        if k not in ("input_ids", "attention_mask", "position_ids", "cache_position", "past_key_values")
    }
    return tokenizer, input_ids, attention_mask, input_ids.device, extra_inputs, None


def _aitw_structured_fast_decode(
    *,
    dataset: str | None,
    model,
    processor,
    inputs,
    generate_kwargs: dict[str, Any],
    sample_meta: dict | None = None,
) -> tuple[str | None, dict[str, Any]]:
    tokenizer, input_ids, attention_mask, device, extra_inputs, fallback_meta = _common_inputs_or_fallback(processor, inputs)
    if fallback_meta is not None:
        return None, fallback_meta

    static_parts = [
        '{"action_type": "',
        ', "click_point": [',
        '"}',
    ]
    template_plan = '{"action_type": "{SEMANTIC_ACTION}", ...}'
    meta = _empty_meta("not_attempted")
    meta.update(
        {
            "template_schema": "aitw_semantic_json",
            "template_static_parts": static_parts,
            "template_plan": template_plan,
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": None,
        }
    )

    action_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS", 16)
    coord_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_AITW_COORD_MAX_TOKENS", 24)
    text_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_AITW_TEXT_MAX_TOKENS", 32)
    debug = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG", "0")
    sample_index = str((sample_meta or {}).get("sample_index", ""))

    static_token_count = 0
    static_steps = 0
    dynamic_steps = 0
    static_s = 0.0
    dynamic_s = 0.0
    slot_stats: list[dict[str, Any]] = []
    coord_pair = ""
    typed_text = ""
    action_head = ""

    try:
        with torch.no_grad():
            prefill_start = time.perf_counter()
            out = _model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                extra_inputs=extra_inputs,
            )
            prefill_s = time.perf_counter() - prefill_start
            past = out.past_key_values
            logits = out.logits

            past, attention_mask, logits, n_tok, elapsed = _append_static(
                model,
                tokenizer,
                text=static_parts[0],
                past_key_values=past,
                attention_mask=attention_mask,
                logits=logits,
                device=device,
            )
            static_token_count += n_tok
            static_steps += int(n_tok > 0)
            static_s += elapsed

            action_head, raw_action, action_suffix, action_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_gui_closed_set(
                model,
                tokenizer,
                logits=logits,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
                generate_kwargs=generate_kwargs,
                candidates=AITW_SEMANTIC_ACTIONS,
                allow_boundary_chars='":,}\n',
                max_tokens=action_max_tokens,
                slot_name="aitw_action_head",
            )
            dynamic_steps += steps
            dynamic_s += elapsed
            semantic_action = _canonical_aitw_semantic_action(action_head)
            slot_stats.append(
                {
                    "slot": "action_head",
                    "decode_tokens": steps,
                    "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                    "done": ok,
                    "fallback": not ok,
                    "reason": reason,
                    "raw_text": raw_action,
                    "rendered_text": semantic_action or action_head,
                }
            )
            if not ok or not semantic_action:
                meta.update(
                    {
                        "template_slot_stats": slot_stats,
                        "template_prefill_fallback_reason": reason or "aitw_action_head_failed",
                    }
                )
                return None, meta

            if semantic_action == "click":
                tail = _gui_static_tail_from_suffix(action_suffix, '", "click_point": [')
                if tail is None:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": f"aitw_action_suffix_unexpected:{action_suffix!r}",
                        }
                    )
                    return None, meta
                past, attention_mask, logits, n_tok, elapsed = _append_static(
                    model,
                    tokenizer,
                    text=tail,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    logits=logits,
                    device=device,
                )
                static_token_count += n_tok
                static_steps += int(n_tok > 0)
                static_s += elapsed

                coord_pair, raw_coord, coord_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    stop_chars=("]", ")"),
                    max_tokens=coord_max_tokens,
                    slot_name="aitw_click_point",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                valid_coord = _valid_gui_coord_pair(coord_pair)
                slot_stats.append(
                    {
                        "slot": "click_point",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": bool(ok and valid_coord),
                        "fallback": not bool(ok and valid_coord),
                        "reason": None if valid_coord else (reason or "aitw_click_point_invalid"),
                        "raw_text": raw_coord,
                        "rendered_text": coord_pair,
                    }
                )
                if not ok or not valid_coord:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": reason or "aitw_click_point_invalid",
                        }
                    )
                    return None, meta
                coords = []
                for part in coord_pair.split(","):
                    value = float(part.strip())
                    coords.append(int(round(value)) if abs(value - round(value)) < 1e-6 else value)
                final_text = json.dumps({"action_type": "click", "bbox_2d": coords}, ensure_ascii=False)

            elif semantic_action == "input_text":
                tail = _gui_static_tail_from_suffix(action_suffix, ": ")
                if tail is None:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": f"aitw_input_text_suffix_unexpected:{action_suffix!r}",
                        }
                    )
                    return None, meta
                past, attention_mask, logits, n_tok, elapsed = _append_static(
                    model,
                    tokenizer,
                    text=tail,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    logits=logits,
                    device=device,
                )
                static_token_count += n_tok
                static_steps += int(n_tok > 0)
                static_s += elapsed

                typed_text, raw_text, text_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    stop_chars=('"', "\n"),
                    max_tokens=text_max_tokens,
                    slot_name="aitw_typed_text",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                slot_stats.append(
                    {
                        "slot": "typed_text",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": ok,
                        "fallback": not ok,
                        "reason": reason,
                        "raw_text": raw_text,
                        "rendered_text": typed_text,
                    }
                )
                if not ok:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "aitw_typed_text_failed"})
                    return None, meta
                if _looks_like_aitw_action_id_payload(typed_text):
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": "aitw_typed_text_looks_like_action_id",
                        }
                    )
                    return None, meta
                final_text = json.dumps({"action_type": f"input_text: {typed_text}"}, ensure_ascii=False)

            else:
                tail = _gui_static_tail_from_suffix(action_suffix, '"}')
                if tail is None:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": f"aitw_terminal_suffix_unexpected:{action_suffix!r}",
                        }
                    )
                    return None, meta
                past, attention_mask, logits, n_tok, elapsed = _append_static(
                    model,
                    tokenizer,
                    text=tail,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    logits=logits,
                    device=device,
                )
                static_token_count += n_tok
                static_steps += int(n_tok > 0)
                static_s += elapsed
                final_text = json.dumps({"action_type": semantic_action}, ensure_ascii=False)

    except Exception as exc:
        meta["template_prefill_fallback_reason"] = f"structured_fast_decode_exception:{type(exc).__name__}:{exc}"
        return None, meta

    meta.update(
        {
            "template_prefill_enabled": True,
            "template_prefill_fallback_reason": None,
            "template_slot_stats": slot_stats,
            "template_static_token_count": int(static_token_count),
            "template_static_decode_steps": int(static_steps),
            "template_unknown_decode_steps": int(dynamic_steps),
            "template_decode_tokens": int(dynamic_steps),
            "template_final_text": final_text,
            "template_generated_text": "".join(str(s.get("raw_text", "")) for s in slot_stats),
            "structured_fast_decode_static_forward_steps": int(static_steps),
            "structured_fast_decode_dynamic_forward_steps": int(dynamic_steps),
            "structured_fast_decode_static_s": float(static_s),
            "structured_fast_decode_dynamic_s": float(dynamic_s),
            "structured_fast_decode_prefill_s": float(prefill_s),
            "structured_fast_decode_action": semantic_action,
            "structured_fast_decode_action_normalized": semantic_action,
            "structured_fast_decode_bbox": coord_pair,
            "structured_fast_decode_typed_text": typed_text,
        }
    )
    if debug:
        print(
            "[StructuredFastDecode] "
            f"sample_index={sample_index} "
            f"enabled=1 schema=aitw_semantic_json "
            f"template_plan={template_plan!r} "
            f"static_parts={static_parts!r} "
            f"static_forward_steps={static_steps} "
            f"static_tokens={static_token_count} "
            f"dynamic_forward_steps={dynamic_steps} "
            f"action={semantic_action!r} "
            f"coord_pair={coord_pair!r} "
            f"typed_text={typed_text!r} "
            f"final={final_text!r}",
            flush=True,
        )
    return final_text, meta


def _mind2web_structured_fast_decode(
    *,
    dataset: str | None,
    model,
    processor,
    inputs,
    generate_kwargs: dict[str, Any],
    sample_meta: dict | None = None,
) -> tuple[str | None, dict[str, Any]]:
    tokenizer, input_ids, attention_mask, device, extra_inputs, fallback_meta = _common_inputs_or_fallback(processor, inputs)
    if fallback_meta is not None:
        return None, fallback_meta

    static_parts = [
        '{"action_type": ',
        ', "click_point": (',
        ', "value": "',
        '}',
    ]
    template_plan = '{"action_type": ACTION_ID, "click_point": (x,y), "value": optional_text}'
    meta = _empty_meta("not_attempted")
    meta.update(
        {
            "template_schema": "mind2web_action_dict",
            "template_static_parts": static_parts,
            "template_plan": template_plan,
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": None,
        }
    )

    action_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_ACTION_MAX_TOKENS", 4)
    coord_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_COORD_MAX_TOKENS", 24)
    value_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_MIND2WEB_VALUE_MAX_TOKENS", 32)
    debug = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG", "0")
    sample_index = str((sample_meta or {}).get("sample_index", ""))

    static_token_count = 0
    static_steps = 0
    dynamic_steps = 0
    static_s = 0.0
    dynamic_s = 0.0
    slot_stats: list[dict[str, Any]] = []
    action_id = ""
    coord_pair = ""
    value_text = ""

    try:
        with torch.no_grad():
            prefill_start = time.perf_counter()
            out = _model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                extra_inputs=extra_inputs,
            )
            prefill_s = time.perf_counter() - prefill_start
            past = out.past_key_values
            logits = out.logits

            past, attention_mask, logits, n_tok, elapsed = _append_static(
                model,
                tokenizer,
                text=static_parts[0],
                past_key_values=past,
                attention_mask=attention_mask,
                logits=logits,
                device=device,
            )
            static_token_count += n_tok
            static_steps += int(n_tok > 0)
            static_s += elapsed

            action_id, raw_action, action_suffix, action_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_gui_closed_set(
                model,
                tokenizer,
                logits=logits,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
                generate_kwargs=generate_kwargs,
                candidates=("2", "3", "4"),
                allow_boundary_chars=",}\n ",
                max_tokens=action_max_tokens,
                slot_name="mind2web_action_type",
            )
            dynamic_steps += steps
            dynamic_s += elapsed
            slot_stats.append(
                {
                    "slot": "action_type",
                    "decode_tokens": steps,
                    "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                    "done": ok,
                    "fallback": not ok,
                    "reason": reason,
                    "raw_text": raw_action,
                    "rendered_text": action_id,
                }
            )
            if not ok or action_id not in {"2", "3", "4"}:
                meta.update(
                    {
                        "template_slot_stats": slot_stats,
                        "template_prefill_fallback_reason": reason or "mind2web_action_type_failed",
                    }
                )
                return None, meta

            tail = _gui_static_tail_from_suffix(action_suffix, static_parts[1])
            if tail is None:
                meta.update(
                    {
                        "template_slot_stats": slot_stats,
                        "template_prefill_fallback_reason": f"mind2web_action_suffix_unexpected:{action_suffix!r}",
                    }
                )
                return None, meta
            past, attention_mask, logits, n_tok, elapsed = _append_static(
                model,
                tokenizer,
                text=tail,
                past_key_values=past,
                attention_mask=attention_mask,
                logits=logits,
                device=device,
            )
            static_token_count += n_tok
            static_steps += int(n_tok > 0)
            static_s += elapsed

            coord_raw_pair, raw_coord, coord_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                model,
                tokenizer,
                logits=logits,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
                generate_kwargs=generate_kwargs,
                stop_chars=(")", "]", "\n"),
                max_tokens=coord_max_tokens,
                slot_name="mind2web_click_point",
            )
            dynamic_steps += steps
            dynamic_s += elapsed
            coord_pair, valid_coord = _normalize_mind2web_coord_pair(coord_raw_pair)
            slot_stats.append(
                {
                    "slot": "click_point",
                    "decode_tokens": steps,
                    "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                    "done": bool(ok and valid_coord),
                    "fallback": not bool(ok and valid_coord),
                    "reason": None if valid_coord else (reason or "mind2web_click_point_invalid"),
                    "raw_text": raw_coord,
                    "rendered_text": coord_pair,
                }
            )
            if not ok or not valid_coord:
                meta.update(
                    {
                        "template_slot_stats": slot_stats,
                        "template_prefill_fallback_reason": reason or "mind2web_click_point_invalid",
                    }
                )
                return None, meta

            if action_id in {"2", "3"}:
                coord_suffix = _suffix_after_first_stop(raw_coord, (")", "]", "\n")) or ""
                value_prefix_tail = _gui_static_tail_from_suffix(coord_suffix, static_parts[2])
                if value_prefix_tail is None:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": f"mind2web_coord_suffix_unexpected:{coord_suffix!r}",
                        }
                    )
                    return None, meta
                past, attention_mask, logits, n_tok, elapsed = _append_static(
                    model,
                    tokenizer,
                    text=value_prefix_tail,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    logits=logits,
                    device=device,
                )
                static_token_count += n_tok
                static_steps += int(n_tok > 0)
                static_s += elapsed

                value_text, raw_value, value_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    stop_chars=('"', "\n", "}"),
                    max_tokens=value_max_tokens,
                    slot_name="mind2web_value",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                slot_stats.append(
                    {
                        "slot": "value",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": ok,
                        "fallback": not ok,
                        "reason": reason,
                        "raw_text": raw_value,
                        "rendered_text": value_text,
                    }
                )
                if not ok:
                    meta.update(
                        {
                            "template_slot_stats": slot_stats,
                            "template_prefill_fallback_reason": reason or "mind2web_value_failed",
                        }
                    )
                    return None, meta
                final_text = (
                    f'{{"action_type": {action_id}, "click_point": ({coord_pair}), '
                    f'"value": {_escape_mind2web_value(value_text)}}}'
                )
            else:
                final_text = f'{{"action_type": {action_id}, "click_point": ({coord_pair})}}'

    except Exception as exc:
        meta["template_prefill_fallback_reason"] = f"structured_fast_decode_exception:{type(exc).__name__}:{exc}"
        return None, meta

    meta.update(
        {
            "template_prefill_enabled": True,
            "template_prefill_fallback_reason": None,
            "template_slot_stats": slot_stats,
            "template_static_token_count": int(static_token_count),
            "template_static_decode_steps": int(static_steps),
            "template_unknown_decode_steps": int(dynamic_steps),
            "template_decode_tokens": int(dynamic_steps),
            "template_final_text": final_text,
            "template_generated_text": "".join(str(s.get("raw_text", "")) for s in slot_stats),
            "structured_fast_decode_static_forward_steps": int(static_steps),
            "structured_fast_decode_dynamic_forward_steps": int(dynamic_steps),
            "structured_fast_decode_static_s": float(static_s),
            "structured_fast_decode_dynamic_s": float(dynamic_s),
            "structured_fast_decode_prefill_s": float(prefill_s),
            "structured_fast_decode_action": action_id,
            "structured_fast_decode_action_normalized": action_id,
            "structured_fast_decode_bbox": coord_pair,
            "structured_fast_decode_typed_text": value_text,
        }
    )
    if debug:
        print(
            "[StructuredFastDecode] "
            f"sample_index={sample_index} "
            "enabled=1 schema=mind2web_action_dict "
            f"template_plan={template_plan!r} "
            f"static_parts={static_parts!r} "
            f"static_forward_steps={static_steps} "
            f"static_tokens={static_token_count} "
            f"dynamic_forward_steps={dynamic_steps} "
            f"action={action_id!r} "
            f"coord_pair={coord_pair!r} "
            f"value={value_text!r} "
            f"final={final_text!r}",
            flush=True,
        )
    return final_text, meta


def _android_structured_fast_decode(
    *,
    dataset: str | None,
    model,
    processor,
    inputs,
    generate_kwargs: dict[str, Any],
    sample_meta: dict | None = None,
) -> tuple[str | None, dict[str, Any]]:
    tokenizer, input_ids, attention_mask, device, extra_inputs, fallback_meta = _common_inputs_or_fallback(processor, inputs)
    if fallback_meta is not None:
        return None, fallback_meta

    static_parts = [
        '<answer>{"action_type": "',
        ', "bbox_2d": [',
        '}</answer>',
        "}</answer>",
    ]
    template_plan = '<answer>{"action_type": "{ACTION_TYPE}", "bbox_2d": [{BBOX_2D}]}</answer>'
    meta = _empty_meta("not_attempted")
    meta.update(
        {
            "template_schema": "android_action_first_json",
            "template_static_parts": static_parts,
            "template_plan": template_plan,
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": None,
        }
    )

    action_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS", 16)
    bbox_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_BBOX_MAX_TOKENS", 64)
    debug = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG", "0")
    sample_index = str((sample_meta or {}).get("sample_index", ""))

    static_token_count = 0
    static_steps = 0
    dynamic_steps = 0
    static_s = 0.0
    dynamic_s = 0.0
    slot_stats: list[dict[str, Any]] = []

    try:
        with torch.no_grad():
            prefill_start = time.perf_counter()
            out = _model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                extra_inputs=extra_inputs,
            )
            prefill_s = time.perf_counter() - prefill_start
            past = out.past_key_values
            logits = out.logits

            past, attention_mask, logits, n_tok, elapsed = _append_static(
                model,
                tokenizer,
                text=static_parts[0],
                past_key_values=past,
                attention_mask=attention_mask,
                logits=logits,
                device=device,
            )
            static_token_count += n_tok
            static_steps += int(n_tok > 0)
            static_s += elapsed

            action, raw_action, action_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                model,
                tokenizer,
                logits=logits,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
                generate_kwargs=generate_kwargs,
                stop_chars=('"',),
                max_tokens=action_max_tokens,
                slot_name="action",
            )
            dynamic_steps += steps
            dynamic_s += elapsed
            action_norm = _normalize_action(action)
            slot_stats.append(
                {
                    "slot": "action_type",
                    "decode_tokens": steps,
                    "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                    "done": ok,
                    "fallback": not ok,
                    "reason": reason,
                    "raw_text": raw_action,
                    "rendered_text": action,
                }
            )
            action_suffix = _suffix_after_first_stop(raw_action, ('"',))
            if not ok or not action or action_suffix is None:
                meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "empty_action"})
                return None, meta

            needs_bbox = action_norm in ("click", "long_press")
            if needs_bbox:
                if action_suffix.startswith(","):
                    suffix_after_comma = action_suffix[1:]
                    if suffix_after_comma.strip():
                        bbox_prefix = suffix_after_comma
                    else:
                        bbox_prefix = " \"bbox_2d\": ["
                elif action_suffix == "":
                    bbox_prefix = static_parts[1]
                else:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"action_suffix_unexpected:{action_suffix!r}"})
                    return None, meta
                past, attention_mask, logits, n_tok, elapsed = _append_static(
                    model,
                    tokenizer,
                    text=bbox_prefix,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    logits=logits,
                    device=device,
                )
                static_token_count += n_tok
                static_steps += int(n_tok > 0)
                static_s += elapsed

                bbox, raw_bbox, bbox_ids, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    stop_chars=("]",),
                    max_tokens=bbox_max_tokens,
                    slot_name="bbox",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                slot_stats.append(
                    {
                        "slot": "bbox_2d",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": ok,
                        "fallback": not ok,
                        "reason": reason,
                        "raw_text": raw_bbox,
                        "rendered_text": bbox,
                    }
                )
                bbox_suffix = _suffix_after_first_stop(raw_bbox, ("]",))
                if not ok or not bbox or bbox_suffix is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "empty_bbox"})
                    return None, meta
                tail = _tail_after_cached_suffix(bbox_suffix, static_parts[3])
                if tail is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"bbox_suffix_unexpected:{bbox_suffix!r}"})
                    return None, meta
                final_text = f'<answer>{{"action_type": "{action}", "bbox_2d": [{bbox}]}}</answer>'
            else:
                bbox = ""
                raw_bbox = ""
                tail = _tail_after_cached_suffix(action_suffix, static_parts[2])
                if tail is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"action_suffix_unexpected:{action_suffix!r}"})
                    return None, meta
                final_text = f'<answer>{{"action_type": "{action}"}}</answer>'

            past, attention_mask, logits, n_tok, elapsed = _append_static(
                model,
                tokenizer,
                text=tail,
                past_key_values=past,
                attention_mask=attention_mask,
                logits=logits,
                device=device,
            )
            static_token_count += n_tok
            static_steps += int(n_tok > 0)
            static_s += elapsed
    except Exception as exc:
        meta["template_prefill_fallback_reason"] = f"structured_fast_decode_exception:{type(exc).__name__}:{exc}"
        return None, meta

    meta.update(
        {
            "template_prefill_enabled": True,
            "template_prefill_fallback_reason": None,
            "template_slot_stats": slot_stats,
            "template_static_token_count": int(static_token_count),
            "template_static_decode_steps": int(static_steps),
            "template_unknown_decode_steps": int(dynamic_steps),
            "template_decode_tokens": int(dynamic_steps),
            "template_final_text": final_text,
            "template_generated_text": "".join(str(s.get("raw_text", "")) for s in slot_stats),
            "structured_fast_decode_static_forward_steps": int(static_steps),
            "structured_fast_decode_dynamic_forward_steps": int(dynamic_steps),
            "structured_fast_decode_static_s": float(static_s),
            "structured_fast_decode_dynamic_s": float(dynamic_s),
            "structured_fast_decode_prefill_s": float(prefill_s),
            "structured_fast_decode_action": action,
            "structured_fast_decode_action_normalized": action_norm,
            "structured_fast_decode_bbox": bbox,
        }
    )
    if debug:
        print(
            "[StructuredFastDecode] "
            f"sample_index={sample_index} "
            f"enabled=1 schema=android_action_first_json "
            f"template_plan={template_plan!r} "
            f"static_parts={static_parts!r} "
            f"static_forward_steps={static_steps} "
            f"static_tokens={static_token_count} "
            f"dynamic_forward_steps={dynamic_steps} "
            f"action={action!r} "
            f"bbox={bbox!r} "
            f"final={final_text!r}",
            flush=True,
        )
    return final_text, meta



def _gui_structured_fast_decode(
    *,
    dataset: str | None,
    model,
    processor,
    inputs,
    generate_kwargs: dict[str, Any],
    sample_meta: dict | None = None,
) -> tuple[str | None, dict[str, Any]]:
    tokenizer, input_ids, attention_mask, device, extra_inputs, fallback_meta = _common_inputs_or_fallback(processor, inputs)
    if fallback_meta is not None:
        return None, fallback_meta

    static_parts = [": ", ": (", ")"]
    template_plan = "{ACTION_HEAD}{STATIC_PREFIX}{SLOT}{STATIC_SUFFIX}"
    meta = _empty_meta("not_attempted")
    meta.update(
        {
            "template_schema": "gui_command",
            "template_static_parts": static_parts,
            "template_plan": template_plan,
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": None,
        }
    )

    action_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_ACTION_MAX_TOKENS", 16)
    gui_direction_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_GUI_DIRECTION_MAX_TOKENS", 8)
    gui_coord_max_tokens = _env_int("QWEN3VL_STRUCTURED_FAST_DECODE_GUI_COORD_MAX_TOKENS", 24)
    gui_require_action_separator = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_GUI_REQUIRE_ACTION_SEPARATOR", "1")
    debug = _env_flag("QWEN3VL_STRUCTURED_FAST_DECODE_DEBUG", "0")
    sample_index = str((sample_meta or {}).get("sample_index", ""))

    static_token_count = 0
    static_steps = 0
    dynamic_steps = 0
    static_s = 0.0
    dynamic_s = 0.0
    slot_stats: list[dict[str, Any]] = []
    coord_pair = ""
    direction = ""

    try:
        with torch.no_grad():
            prefill_start = time.perf_counter()
            out = _model_forward(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                extra_inputs=extra_inputs,
            )
            prefill_s = time.perf_counter() - prefill_start
            past = out.past_key_values
            logits = out.logits

            action_head, raw_head, head_suffix, _, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_gui_closed_set(
                model,
                tokenizer,
                logits=logits,
                past_key_values=past,
                attention_mask=attention_mask,
                device=device,
                generate_kwargs=generate_kwargs,
                candidates=GUI_ACTION_HEADS,
                allow_boundary_chars=": ()\n\r\t",
                max_tokens=action_max_tokens,
                slot_name="gui_action_head",
                accept_exact=not gui_require_action_separator,
            )
            dynamic_steps += steps
            dynamic_s += elapsed
            slot_stats.append(
                {
                    "slot": "action_head",
                    "decode_tokens": steps,
                    "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                    "done": ok,
                    "fallback": not ok,
                    "reason": reason,
                    "raw_text": raw_head,
                    "rendered_text": action_head,
                }
            )
            if not ok or not action_head:
                meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "gui_empty_action_head"})
                return None, meta
            action_rendered = _gui_rendered_prefix(raw_head, head_suffix, action_head)
            if action_head == "TYPE":
                meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": "gui_type_free_text_fallback"})
                return None, meta
            if action_head in GUI_TERMINAL_ACTIONS:
                if head_suffix.strip():
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_terminal_suffix_unexpected:{head_suffix!r}"})
                    return None, meta
                final_text = action_rendered
                template_schema = "gui_terminal_command"
            elif action_head == "SCROLL":
                tail = _gui_static_tail_from_suffix(head_suffix, ": ")
                if tail is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_scroll_suffix_unexpected:{head_suffix!r}"})
                    return None, meta
                if tail:
                    past, attention_mask, logits, n_tok, elapsed = _append_static(
                        model,
                        tokenizer,
                        text=tail,
                        past_key_values=past,
                        attention_mask=attention_mask,
                        logits=logits,
                        device=device,
                        prefix_text=action_rendered + head_suffix,
                    )
                    static_token_count += n_tok
                    static_steps += int(n_tok > 0)
                    static_s += elapsed
                direction, raw_direction, direction_suffix, _, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_gui_closed_set(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    candidates=GUI_SCROLL_DIRECTIONS,
                    allow_boundary_chars="\n\r\t ",
                    max_tokens=gui_direction_max_tokens,
                    slot_name="gui_direction",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                slot_stats.append(
                    {
                        "slot": "direction",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": ok,
                        "fallback": not ok,
                        "reason": reason,
                        "raw_text": raw_direction,
                        "rendered_text": direction,
                    }
                )
                if not ok or not direction:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "gui_empty_direction"})
                    return None, meta
                if direction_suffix.strip():
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_direction_suffix_unexpected:{direction_suffix!r}"})
                    return None, meta
                direction_rendered = _gui_rendered_prefix(raw_direction, direction_suffix, direction)
                final_text = f"{action_rendered}: {direction_rendered}"
                template_schema = "gui_scroll_command"
            elif action_head in GUI_COORD_ACTIONS:
                tail = _gui_static_tail_from_suffix(head_suffix, ": (")
                if tail is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_coord_suffix_unexpected:{head_suffix!r}"})
                    return None, meta
                if tail:
                    past, attention_mask, logits, n_tok, elapsed = _append_static(
                        model,
                        tokenizer,
                        text=tail,
                        past_key_values=past,
                        attention_mask=attention_mask,
                        logits=logits,
                        device=device,
                        prefix_text=action_rendered + head_suffix,
                    )
                    static_token_count += n_tok
                    static_steps += int(n_tok > 0)
                    static_s += elapsed
                coord_pair, raw_coord_pair, _, past, attention_mask, logits, steps, ok, reason, elapsed = _decode_until(
                    model,
                    tokenizer,
                    logits=logits,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    device=device,
                    generate_kwargs=generate_kwargs,
                    stop_chars=(")",),
                    max_tokens=gui_coord_max_tokens,
                    slot_name="gui_coord_pair",
                )
                dynamic_steps += steps
                dynamic_s += elapsed
                slot_stats.append(
                    {
                        "slot": "coord_pair",
                        "decode_tokens": steps,
                        "prompt_tokens": _cache_seq_len(past, int(attention_mask.shape[1])),
                        "done": ok,
                        "fallback": not ok,
                        "reason": reason,
                        "raw_text": raw_coord_pair,
                        "rendered_text": coord_pair,
                    }
                )
                coord_suffix = _suffix_after_first_stop(raw_coord_pair, (")",))
                if not ok or not coord_pair or coord_suffix is None:
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": reason or "gui_empty_coord_pair"})
                    return None, meta
                if coord_suffix.strip():
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_coord_tail_unexpected:{coord_suffix!r}"})
                    return None, meta
                if not _valid_gui_coord_pair(coord_pair):
                    meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"gui_coord_invalid:{coord_pair!r}"})
                    return None, meta
                final_text = f"{action_rendered}: ({coord_pair})"
                template_schema = "gui_coord_command"
            else:
                meta.update({"template_slot_stats": slot_stats, "template_prefill_fallback_reason": f"unsupported_gui_action:{action_head}"})
                return None, meta
    except Exception as exc:
        meta["template_prefill_fallback_reason"] = f"structured_fast_decode_exception:{type(exc).__name__}:{exc}"
        return None, meta

    meta.update(
        {
            "template_prefill_enabled": True,
            "template_prefill_fallback_reason": None,
            "template_schema": template_schema,
            "template_slot_stats": slot_stats,
            "template_static_token_count": int(static_token_count),
            "template_static_decode_steps": int(static_steps),
            "template_unknown_decode_steps": int(dynamic_steps),
            "template_decode_tokens": int(dynamic_steps),
            "template_final_text": final_text,
            "template_generated_text": "".join(str(s.get("raw_text", "")) for s in slot_stats),
            "structured_fast_decode_static_forward_steps": int(static_steps),
            "structured_fast_decode_dynamic_forward_steps": int(dynamic_steps),
            "structured_fast_decode_static_s": float(static_s),
            "structured_fast_decode_dynamic_s": float(dynamic_s),
            "structured_fast_decode_prefill_s": float(prefill_s),
            "structured_fast_decode_action": action_head,
            "structured_fast_decode_action_normalized": action_head.lower(),
            "structured_fast_decode_gui_require_action_separator": bool(gui_require_action_separator),
            "structured_fast_decode_bbox": coord_pair,
            "structured_fast_decode_direction": direction,
        }
    )
    if debug:
        print(
            "[StructuredFastDecode] "
            f"sample_index={sample_index} "
            f"enabled=1 schema={template_schema} "
            f"template_plan={template_plan!r} "
            f"static_parts={static_parts!r} "
            f"static_forward_steps={static_steps} "
            f"static_tokens={static_token_count} "
            f"dynamic_forward_steps={dynamic_steps} "
            f"require_action_separator={int(gui_require_action_separator)} "
            f"action={action_head!r} "
            f"coord_pair={coord_pair!r} "
            f"direction={direction!r} "
            f"final={final_text!r}",
            flush=True,
        )
    return final_text, meta


def maybe_generate_with_structured_fast_decode(
    *,
    dataset: str | None,
    model,
    processor,
    inputs,
    generate_kwargs: dict[str, Any],
    sample_meta: dict | None = None,
) -> tuple[str | None, dict[str, Any]]:
    if not _env_flag("QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE", "0"):
        return None, _empty_meta("disabled")
    if generate_kwargs.get("use_cache", True) is False:
        return None, _empty_meta("requires_use_cache")

    family = _dataset_family(dataset)
    if family == "androidcontrol":
        return _android_structured_fast_decode(
            dataset=dataset,
            model=model,
            processor=processor,
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            sample_meta=sample_meta,
        )
    if family == "aitw":
        return _aitw_structured_fast_decode(
            dataset=dataset,
            model=model,
            processor=processor,
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            sample_meta=sample_meta,
        )
    if family == "guiodyssey":
        return _gui_structured_fast_decode(
            dataset=dataset,
            model=model,
            processor=processor,
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            sample_meta=sample_meta,
        )
    if family == "mind2web":
        return _mind2web_structured_fast_decode(
            dataset=dataset,
            model=model,
            processor=processor,
            inputs=inputs,
            generate_kwargs=generate_kwargs,
            sample_meta=sample_meta,
        )
    return None, _empty_meta(f"unsupported_dataset:{dataset}")
