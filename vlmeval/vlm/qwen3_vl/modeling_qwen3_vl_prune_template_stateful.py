from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

from . import modeling_qwen3_vl_prune_template as legacy

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover - fallback for older transformers
    DynamicCache = None


def _env_impl() -> str:
    return os.getenv("QWEN3VL_TEMPLATE_PREFILL_IMPL", "stateful").strip().lower()


def _eos_token_ids(model, tokenizer) -> set[int]:
    eos_ids = set()
    for source in (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            for item in source:
                try:
                    eos_ids.add(int(item))
                except Exception:
                    continue
        else:
            try:
                eos_ids.add(int(source))
            except Exception:
                continue
    return eos_ids


def _clone_past_key_values(cache):
    if cache is None:
        return None
    if DynamicCache is None or not hasattr(cache, "layers"):
        return cache
    cloned = DynamicCache()
    for layer_idx, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise RuntimeError(f"cache layer {layer_idx} has no keys/values")
        cloned.update(keys.clone(), values.clone(), layer_idx, None)
    return cloned


def _clone_logits(logits: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if logits is None:
        return None
    return logits.detach().clone()


def _tokenizer_encode(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _continuation_token_ids(tokenizer, prefix_text: str, append_text: str) -> tuple[list[int], bool]:
    prefix_ids = _tokenizer_encode(tokenizer, prefix_text)
    full_ids = _tokenizer_encode(tokenizer, prefix_text + append_text)
    if len(full_ids) >= len(prefix_ids) and full_ids[: len(prefix_ids)] == prefix_ids:
        return full_ids[len(prefix_ids) :], True
    return [], False


def _extract_cache_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    candidates: list[int] = []
    try:
        candidates.append(int(past_key_values.get_seq_length()))
    except Exception:
        pass
    try:
        layers = getattr(past_key_values, "layers", None) or []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is not None and hasattr(keys, "shape") and len(keys.shape) >= 3:
                candidates.append(int(keys.shape[-2]))
    except Exception:
        pass
    return max([0] + [x for x in candidates if x > 0])


class _BoundaryAndroidBBoxSlotParser:
    def parse(self, raw_text: str) -> legacy.SlotParseResult:
        text = str(raw_text or "")
        match = legacy.re.match(
            r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(.*)$",
            text,
            legacy.re.DOTALL,
        )
        if not match:
            nums = legacy.re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) > 4:
                return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_too_many_numbers")
            return legacy.SlotParseResult(done=False)

        try:
            values = [float(match.group(i)) for i in range(1, 5)]
        except Exception:
            return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_parse_failure")
        if values[2] < values[0] or values[3] < values[1]:
            return legacy.SlotParseResult(done=False)

        remainder = match.group(5)
        if remainder and not remainder.lstrip().startswith("]") and remainder.strip() not in ("", ",", ", ", " ]"):
            if len(legacy.re.findall(r"-?\d+(?:\.\d+)?", text)) > 4:
                return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_extra_content")
            return legacy.SlotParseResult(done=False)

        rendered = ", ".join(legacy._format_number(v) for v in values)
        return legacy.SlotParseResult(done=True, rendered=rendered, canonical=rendered)


def _build_android_final_text(mode: str, action: str, bbox: str = "") -> str:
    if action in ("click", "long_press"):
        if str(mode or "") == "action_first_json":
            return f'<answer>{{"action_type": "{action}", "bbox_2d": [{bbox}]}}</answer>'
        return f'<answer>{{"bbox_2d": [{bbox}], "action_type": "{action}"}}</answer>'
    return f'<answer>{{"action_type": "{action}"}}</answer>'


def _sampling_next_token(logits: torch.Tensor, generate_kwargs: dict) -> int:
    do_sample = bool(generate_kwargs.get("do_sample", False))
    temperature = generate_kwargs.get("temperature", None)
    top_p = float(generate_kwargs.get("top_p", 1.0) or 1.0)

    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())

    temp = float(temperature) if temperature not in (None, 0, 0.0) else 1.0
    if temp <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = torch.softmax(logits / temp, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep_mask = cumulative <= top_p
        keep_mask[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_indices, sorted_probs * keep_mask)
        total = filtered.sum(dim=-1, keepdim=True)
        probs = torch.where(total > 0, filtered / total.clamp_min(1e-12), probs)
    sample = torch.multinomial(probs, num_samples=1)
    return int(sample.item())


def _sampling_next_token_from_allowed(logits: torch.Tensor, allowed_ids: set[int], generate_kwargs: dict) -> int:
    if not allowed_ids:
        return _sampling_next_token(logits, generate_kwargs)
    allowed = sorted(int(x) for x in allowed_ids if int(x) >= 0)
    if not allowed:
        return _sampling_next_token(logits, generate_kwargs)

    allowed_tensor = torch.tensor(allowed, device=logits.device, dtype=torch.long)
    allowed_logits = torch.index_select(logits, dim=-1, index=allowed_tensor)
    chosen_local = _sampling_next_token(allowed_logits, generate_kwargs)
    chosen_local = max(0, min(int(chosen_local), len(allowed) - 1))
    return int(allowed[chosen_local])


def _closed_set_allowed_next_token_ids(tokenizer, sampled_ids: Sequence[int], parser) -> set[int]:
    if not isinstance(parser, legacy.ClosedSetSlotParser):
        return set()
    prefix = [int(x) for x in sampled_ids]
    allowed: set[int] = set()
    for candidate in getattr(parser, "candidates", []) or []:
        try:
            cand_ids = list(tokenizer.encode(str(candidate), add_special_tokens=False))
        except Exception:
            continue
        if len(cand_ids) <= len(prefix):
            continue
        if cand_ids[: len(prefix)] == prefix:
            allowed.add(int(cand_ids[len(prefix)]))
    return allowed


@dataclass
class _SessionSnapshot:
    current_text: str
    total_tokens: int
    attention_dtype: torch.dtype
    attention_device: torch.device
    attention_mask: Optional[torch.Tensor]
    past_key_values: object
    last_logits: Optional[torch.Tensor]


class StatefulTemplateDecodeSession:
    def __init__(
        self,
        *,
        model,
        processor,
        base_text: str,
        images,
        videos,
        video_metadatas,
        video_kwargs,
        generate_kwargs: dict,
        configure_context_fn: Optional[Callable],
    ) -> None:
        self.model = model
        self.processor = processor
        self.base_text = str(base_text)
        self.images = images
        self.videos = videos
        self.video_metadatas = video_metadatas
        self.video_kwargs = video_kwargs
        self.generate_kwargs = dict(generate_kwargs)
        self.configure_context_fn = configure_context_fn
        self.current_text = ""
        self.total_tokens = 0
        self.attention_dtype = torch.long
        self.attention_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.attention_mask: Optional[torch.Tensor] = None
        self.prepare_static_kwargs: dict = {}
        self.past_key_values = None
        self.last_logits = None
        self.eos_token_ids = _eos_token_ids(model, processor.tokenizer)

    def _model_forward(self, **kwargs):
        with torch.no_grad():
            return self.model(**kwargs)

    def prefill_initial_text(self, text: str) -> int:
        local_inputs = legacy._build_processor_inputs(
            processor=self.processor,
            model=self.model,
            text=text,
            images=self.images,
            videos=self.videos,
            video_metadatas=self.video_metadatas,
            video_kwargs=self.video_kwargs,
        )
        if self.configure_context_fn is not None:
            self.configure_context_fn(local_inputs)
        prompt_token_len = legacy._decode_prompt_token_len(local_inputs)
        attention_mask = getattr(local_inputs, "attention_mask", None)
        if attention_mask is None and hasattr(local_inputs, "get"):
            attention_mask = local_inputs.get("attention_mask", None)
        if attention_mask is not None:
            self.attention_dtype = attention_mask.dtype
            self.attention_device = attention_mask.device
            self.attention_mask = attention_mask.detach().clone()
        else:
            self.attention_mask = None
        self.prepare_static_kwargs = {}
        for key in (
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "mm_token_type_ids",
        ):
            value = getattr(local_inputs, key, None)
            if value is None and hasattr(local_inputs, "get"):
                value = local_inputs.get(key, None)
            if value is not None:
                self.prepare_static_kwargs[key] = value
        model_inputs = dict(local_inputs)
        model_inputs["use_cache"] = True
        model_inputs["return_dict"] = True
        outputs = self._model_forward(**model_inputs)
        past = getattr(outputs, "past_key_values", None)
        logits = getattr(outputs, "logits", None)
        if past is None or logits is None:
            raise RuntimeError("initial template prefill produced no cache/logits")
        self.current_text = str(text)
        cache_len = _extract_cache_len(past)
        self.total_tokens = int(max(cache_len, prompt_token_len))
        if self.attention_mask is None or int(self.attention_mask.shape[-1]) != int(self.total_tokens):
            self.attention_mask = torch.ones(
                (1, int(self.total_tokens)),
                dtype=self.attention_dtype,
                device=self.attention_device,
            )
        self.past_key_values = past
        self.last_logits = logits
        return prompt_token_len

    def snapshot(self) -> _SessionSnapshot:
        if self.last_logits is None:
            raise RuntimeError("cannot snapshot session before initial prefill")
        return _SessionSnapshot(
            current_text=self.current_text,
            total_tokens=int(self.total_tokens),
            attention_dtype=self.attention_dtype,
            attention_device=self.attention_device,
            attention_mask=self.attention_mask.detach().clone() if self.attention_mask is not None else None,
            past_key_values=_clone_past_key_values(self.past_key_values),
            last_logits=_clone_logits(self.last_logits),
        )

    def restore(self, snapshot: _SessionSnapshot) -> None:
        self.current_text = snapshot.current_text
        self.total_tokens = int(snapshot.total_tokens)
        self.attention_dtype = snapshot.attention_dtype
        self.attention_device = snapshot.attention_device
        self.attention_mask = snapshot.attention_mask
        self.past_key_values = snapshot.past_key_values
        self.last_logits = snapshot.last_logits

    def _forward_token_ids(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        if self.past_key_values is None or self.last_logits is None:
            raise RuntimeError("session is not initialized")
        device = self.attention_device
        new_input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        cache_len = _extract_cache_len(self.past_key_values)
        self.total_tokens = int(max(self.total_tokens, cache_len))
        if self.attention_mask is None or int(self.attention_mask.shape[-1]) != int(self.total_tokens):
            self.attention_mask = torch.ones((1, int(self.total_tokens)), dtype=self.attention_dtype, device=device)
        append_mask = torch.ones((1, int(new_input_ids.shape[1])), dtype=self.attention_dtype, device=device)
        self.attention_mask = torch.cat([self.attention_mask.to(device=device), append_mask], dim=-1)
        cache_position = torch.arange(
            int(self.total_tokens),
            int(self.total_tokens) + int(new_input_ids.shape[1]),
            dtype=torch.long,
            device=device,
        )
        prepared = dict(
            input_ids=new_input_ids,
            past_key_values=self.past_key_values,
            attention_mask=self.attention_mask,
            use_cache=True,
            cache_position=cache_position,
            logits_to_keep=1,
            return_dict=True,
        )
        try:
            outputs = self._model_forward(**prepared)
        except Exception as exc:
            cache_len = _extract_cache_len(self.past_key_values)
            raise RuntimeError(
                "stateful_continuation_forward_failed:"
                f"{type(exc).__name__}:{exc}; "
                f"new_input_ids_shape={tuple(new_input_ids.shape)} "
                f"attention_mask_shape={tuple(self.attention_mask.shape) if self.attention_mask is not None else None} "
                f"cache_len={cache_len} "
                f"total_tokens={self.total_tokens} "
                f"cache_position_shape={tuple(cache_position.shape)} "
                f"cache_position_first={int(cache_position[0].item()) if cache_position.numel() else None} "
                f"cache_position_last={int(cache_position[-1].item()) if cache_position.numel() else None}"
            ) from exc
        past = getattr(outputs, "past_key_values", None)
        logits = getattr(outputs, "logits", None)
        if past is None or logits is None:
            raise RuntimeError("token append produced no cache/logits")
        self.past_key_values = past
        self.last_logits = logits
        self.total_tokens = int(self.attention_mask.shape[-1])

    def append_known_text(self, append_text: str) -> None:
        append_text = str(append_text or "")
        if not append_text:
            return
        token_ids, ok = _continuation_token_ids(self.processor.tokenizer, self.current_text, append_text)
        if not ok:
            raise RuntimeError("token_boundary_shift")
        self._forward_token_ids(token_ids)
        self.current_text += append_text

    def generate_slot_in_place(self, *, parser, max_new_tokens: int):
        if self.past_key_values is None or self.last_logits is None:
            raise RuntimeError("session is not initialized")
        prompt_tokens = int(self.total_tokens)
        sampled_ids: list[int] = []
        raw_text = ""
        parsed = legacy.SlotParseResult(done=False)

        for _ in range(int(max_new_tokens)):
            if self.last_logits is None:
                raise RuntimeError("missing slot logits")
            allowed_ids = _closed_set_allowed_next_token_ids(self.processor.tokenizer, sampled_ids, parser)
            next_token_id = _sampling_next_token_from_allowed(
                self.last_logits[:, -1, :], allowed_ids, self.generate_kwargs
            )
            if self.eos_token_ids and next_token_id in self.eos_token_ids:
                parsed = legacy.SlotParseResult(done=False, fallback=True, reason="slot_eos_before_done")
                break
            sampled_ids.append(int(next_token_id))
            self._forward_token_ids([next_token_id])
            raw_text = self.processor.tokenizer.decode(
                sampled_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = parser.parse(raw_text)
            if parsed.done or parsed.fallback:
                break
        else:
            parsed = legacy.SlotParseResult(done=False, fallback=True, reason="slot_max_new_tokens")

        if parsed.done and raw_text:
            self.current_text += raw_text
        return parsed, raw_text, len(sampled_ids), prompt_tokens

    def generate_free_text_in_place(self, *, max_new_tokens: int) -> tuple[str, int]:
        if self.past_key_values is None or self.last_logits is None:
            raise RuntimeError("session is not initialized")
        sampled_ids: list[int] = []

        for _ in range(int(max_new_tokens)):
            if self.last_logits is None:
                raise RuntimeError("missing free-text logits")
            next_token_id = _sampling_next_token(self.last_logits[:, -1, :], self.generate_kwargs)
            if self.eos_token_ids and next_token_id in self.eos_token_ids:
                break
            sampled_ids.append(int(next_token_id))
            self._forward_token_ids([next_token_id])

        text = self.processor.tokenizer.decode(
            sampled_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if text:
            self.current_text += text
        return text, len(sampled_ids)

    def generate_slot(self, *, parser, max_new_tokens: int):
        snapshot = self.snapshot()
        prompt_tokens = int(snapshot.total_tokens)
        sampled_ids: list[int] = []
        raw_text = ""
        parsed = legacy.SlotParseResult(done=False)

        for _ in range(int(max_new_tokens)):
            if self.last_logits is None:
                raise RuntimeError("missing slot logits")
            allowed_ids = _closed_set_allowed_next_token_ids(self.processor.tokenizer, sampled_ids, parser)
            next_token_id = _sampling_next_token_from_allowed(
                self.last_logits[:, -1, :], allowed_ids, self.generate_kwargs
            )
            if self.eos_token_ids and next_token_id in self.eos_token_ids:
                parsed = legacy.SlotParseResult(done=False, fallback=True, reason="slot_eos_before_done")
                break
            sampled_ids.append(int(next_token_id))
            self._forward_token_ids([next_token_id])
            raw_text = self.processor.tokenizer.decode(
                sampled_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = parser.parse(raw_text)
            if parsed.done or parsed.fallback:
                break
        else:
            parsed = legacy.SlotParseResult(done=False, fallback=True, reason="slot_max_new_tokens")

        decode_tokens = len(sampled_ids)
        self.restore(snapshot)
        if parsed.done and parsed.rendered:
            self.append_known_text(parsed.rendered)
        return parsed, raw_text, decode_tokens, prompt_tokens


def _android_template_prefill_stateful(
    *,
    dataset,
    model,
    processor,
    base_text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    generate_kwargs: dict,
    configure_context_fn: Optional[Callable],
):
    mode = legacy._android_mode()
    if mode not in ("action_first_json", "bbox_first_json"):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"android_mode:{mode}"}
    if not bool(generate_kwargs.get("use_cache", True)):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "stateful_requires_use_cache"}

    initial_static = '<answer>{"action_type": "'
    static_parts = [initial_static]
    static_token_total = len(processor.tokenizer.encode(initial_static, add_special_tokens=False))
    slot_stats = []
    session = StatefulTemplateDecodeSession(
        model=model,
        processor=processor,
        base_text=base_text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        generate_kwargs=generate_kwargs,
        configure_context_fn=configure_context_fn,
    )
    prompt_tokens = session.prefill_initial_text(base_text + initial_static)

    action_parser = legacy.ClosedSetSlotParser(
        candidates=legacy.ANDROID_CLOSED_ACTIONS,
        ignore_case=False,
        allow_boundary_chars='"',
        open_prefixes=legacy.ANDROID_OPEN_ACTION_PREFIXES,
    )
    parsed_action, raw_action_text, action_decode_tokens, prompt_tokens = session.generate_slot_in_place(
        parser=action_parser,
        max_new_tokens=20,
    )
    slot_stats.append(
        {
            "slot": "action_type",
            "raw_text": raw_action_text,
            "rendered_text": parsed_action.rendered,
            "decode_tokens": action_decode_tokens,
            "prompt_tokens": prompt_tokens,
            "done": parsed_action.done,
            "fallback": parsed_action.fallback,
            "reason": parsed_action.reason,
        }
    )
    if not parsed_action.done:
        return None, {
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": parsed_action.reason or "android_action_type_slot_failed",
            "template_slot_stats": slot_stats,
            "template_static_parts": static_parts,
            "template_prefill_impl": "stateful_slot_cache",
        }

    built = initial_static + parsed_action.rendered
    action_value = parsed_action.canonical
    if action_value in ("click", "long_press"):
        static_mid = '", "bbox_2d": ['
        session.append_known_text(static_mid)
        built += static_mid
        static_parts.append(static_mid)
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        bbox_parser = _BoundaryAndroidBBoxSlotParser()
        parsed_bbox, raw_bbox_text, bbox_decode_tokens, bbox_prompt_tokens = session.generate_slot_in_place(
            parser=bbox_parser,
            max_new_tokens=40,
        )
        slot_stats.append(
            {
                "slot": "bbox_2d",
                "raw_text": raw_bbox_text,
                "rendered_text": parsed_bbox.rendered,
                "decode_tokens": bbox_decode_tokens,
                "prompt_tokens": bbox_prompt_tokens,
                "done": parsed_bbox.done,
                "fallback": parsed_bbox.fallback,
                "reason": parsed_bbox.reason,
            }
        )
        if not parsed_bbox.done:
            return None, {
                "template_prefill_enabled": False,
                "template_prefill_fallback_reason": parsed_bbox.reason or "android_bbox_slot_failed",
                "template_slot_stats": slot_stats,
                "template_static_parts": static_parts,
                "template_prefill_impl": "stateful_slot_cache",
            }
        bbox_value = parsed_bbox.canonical
        built += parsed_bbox.rendered
        suffix = "]}</answer>"
    else:
        bbox_value = ""
        suffix = '"}</answer>'

    built += suffix
    static_parts.append(suffix)
    static_token_total += len(processor.tokenizer.encode(suffix, add_special_tokens=False))
    total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats)
    final_text = _build_android_final_text(mode, action_value, bbox_value)
    return final_text, {
        "template_prefill_enabled": True,
        "template_schema": f"android_{mode}",
        "template_slot_stats": slot_stats,
        "template_static_parts": static_parts,
        "template_static_token_count": int(static_token_total),
        "template_static_decode_steps": 0,
        "template_unknown_decode_steps": int(total_decode_tokens),
        "template_decode_tokens": int(total_decode_tokens),
        "template_final_text": final_text,
        "template_prefill_impl": "stateful_slot_cache",
    }


def _gui_template_prefill_stateful(
    *,
    dataset,
    model,
    processor,
    base_text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    generate_kwargs: dict,
    configure_context_fn: Optional[Callable],
):
    mode = legacy._gui_mode()
    if mode not in ("command",):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"gui_mode:{mode}"}
    if not bool(generate_kwargs.get("use_cache", True)):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "stateful_requires_use_cache"}

    slot_stats = []
    static_token_total = 0
    session = StatefulTemplateDecodeSession(
        model=model,
        processor=processor,
        base_text=base_text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        generate_kwargs=generate_kwargs,
        configure_context_fn=configure_context_fn,
    )
    session.prefill_initial_text(base_text)

    head_parser = legacy.ClosedSetSlotParser(
        candidates=legacy.GUI_ACTION_HEADS,
        ignore_case=True,
        allow_boundary_chars="",
    )
    parsed_head, raw_head_text, head_decode_tokens, head_prompt_tokens = session.generate_slot_in_place(
        parser=head_parser,
        max_new_tokens=12,
    )
    slot_stats.append(
        {
            "slot": "action_head",
            "raw_text": raw_head_text,
            "rendered_text": parsed_head.rendered,
            "decode_tokens": head_decode_tokens,
            "prompt_tokens": head_prompt_tokens,
            "done": parsed_head.done,
            "fallback": parsed_head.fallback,
            "reason": parsed_head.reason,
        }
    )
    if not parsed_head.done:
        return None, {
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": parsed_head.reason or "gui_action_head_slot_failed",
            "template_slot_stats": slot_stats,
            "template_static_parts": [],
            "template_prefill_impl": "stateful_slot_cache",
        }

    action_head = parsed_head.canonical.upper()
    built = action_head
    if action_head in legacy.GUI_TERMINAL_ACTIONS:
        total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats)
        return built, {
            "template_prefill_enabled": True,
            "template_schema": "gui_terminal_command",
            "template_slot_stats": slot_stats,
            "template_static_parts": [],
            "template_static_token_count": 0,
            "template_static_decode_steps": 0,
            "template_unknown_decode_steps": int(total_decode_tokens),
            "template_decode_tokens": int(total_decode_tokens),
            "template_final_text": built,
            "template_prefill_impl": "stateful_slot_cache",
        }

    if action_head == "TYPE":
        static_mid = ": "
        static_parts = [static_mid]
        session.append_known_text(static_mid)
        built += static_mid
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        free_text_budget = int(
            os.getenv(
                "QWEN3VL_TEMPLATE_PREFILL_GUI_TYPE_MAX_NEW_TOKENS",
                str(generate_kwargs.get("max_new_tokens", 32) or 32),
            )
        )
        free_text, free_text_decode_tokens = session.generate_free_text_in_place(max_new_tokens=free_text_budget)
        built += free_text
        total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats) + int(free_text_decode_tokens)
        return built, {
            "template_prefill_enabled": True,
            "template_schema": "gui_type_command",
            "template_slot_stats": slot_stats,
            "template_static_parts": static_parts,
            "template_static_token_count": int(static_token_total),
            "template_static_decode_steps": 0,
            "template_unknown_decode_steps": int(total_decode_tokens),
            "template_decode_tokens": int(total_decode_tokens),
            "template_final_text": built,
            "template_prefill_impl": "stateful_slot_cache",
        }

    if action_head == "SCROLL":
        static_mid = ": "
        static_parts = [static_mid]
        session.append_known_text(static_mid)
        built += static_mid
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        direction_parser = legacy.ClosedSetSlotParser(
            candidates=legacy.GUI_SCROLL_DIRECTIONS,
            ignore_case=True,
            allow_boundary_chars="\n\r\t ",
        )
        parsed_dir, raw_dir_text, dir_decode_tokens, dir_prompt_tokens = session.generate_slot_in_place(
            parser=direction_parser,
            max_new_tokens=8,
        )
        slot_stats.append(
            {
                "slot": "direction",
                "raw_text": raw_dir_text,
                "rendered_text": parsed_dir.rendered,
                "decode_tokens": dir_decode_tokens,
                "prompt_tokens": dir_prompt_tokens,
                "done": parsed_dir.done,
                "fallback": parsed_dir.fallback,
                "reason": parsed_dir.reason,
            }
        )
        if not parsed_dir.done:
            return None, {
                "template_prefill_enabled": False,
                "template_prefill_fallback_reason": parsed_dir.reason or "gui_scroll_slot_failed",
                "template_slot_stats": slot_stats,
                "template_static_parts": static_parts,
                "template_prefill_impl": "stateful_slot_cache",
            }
        built += parsed_dir.rendered
        total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats)
        return built, {
            "template_prefill_enabled": True,
            "template_schema": "gui_scroll_command",
            "template_slot_stats": slot_stats,
            "template_static_parts": static_parts,
            "template_static_token_count": int(static_token_total),
            "template_static_decode_steps": 0,
            "template_unknown_decode_steps": int(total_decode_tokens),
            "template_decode_tokens": int(total_decode_tokens),
            "template_final_text": built,
            "template_prefill_impl": "stateful_slot_cache",
        }

    if action_head in legacy.GUI_COORD_ACTIONS:
        static_mid = ": ("
        static_parts = [static_mid]
        session.append_known_text(static_mid)
        built += static_mid
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        coord_parser = legacy.GuiCoordPairSlotParser()
        parsed_coords, raw_coords_text, coord_decode_tokens, coord_prompt_tokens = session.generate_slot_in_place(
            parser=coord_parser,
            max_new_tokens=16,
        )
        slot_stats.append(
            {
                "slot": "coord_pair",
                "raw_text": raw_coords_text,
                "rendered_text": parsed_coords.rendered,
                "decode_tokens": coord_decode_tokens,
                "prompt_tokens": coord_prompt_tokens,
                "done": parsed_coords.done,
                "fallback": parsed_coords.fallback,
                "reason": parsed_coords.reason,
            }
        )
        if not parsed_coords.done:
            return None, {
                "template_prefill_enabled": False,
                "template_prefill_fallback_reason": parsed_coords.reason or "gui_coord_slot_failed",
                "template_slot_stats": slot_stats,
                "template_static_parts": static_parts,
                "template_prefill_impl": "stateful_slot_cache",
            }
        built += parsed_coords.rendered
        suffix = ")"
        built += suffix
        static_parts.append(suffix)
        static_token_total += len(processor.tokenizer.encode(suffix, add_special_tokens=False))
        total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats)
        return built, {
            "template_prefill_enabled": True,
            "template_schema": "gui_coord_command",
            "template_slot_stats": slot_stats,
            "template_static_parts": static_parts,
            "template_static_token_count": int(static_token_total),
            "template_static_decode_steps": 0,
            "template_unknown_decode_steps": int(total_decode_tokens),
            "template_decode_tokens": int(total_decode_tokens),
            "template_final_text": built,
            "template_prefill_impl": "stateful_slot_cache",
        }

    return None, {
        "template_prefill_enabled": False,
        "template_prefill_fallback_reason": f"unsupported_gui_action:{action_head}",
        "template_slot_stats": slot_stats,
        "template_static_parts": [],
        "template_prefill_impl": "stateful_slot_cache",
    }


def _maybe_generate_with_template_prefill_stateful(
    *,
    dataset,
    model,
    processor,
    base_text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    generate_kwargs: dict,
    configure_context_fn: Optional[Callable] = None,
):
    if not legacy._env_flag("QWEN3VL_ENABLE_TEMPLATE_PREFILL", "0"):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "disabled"}
    if not legacy._dataset_enabled(dataset):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "dataset_not_enabled"}

    family = legacy._dataset_family(dataset)
    legacy._debug_print(dataset, f"enabled family={family} impl=stateful_slot_cache")
    if family == "androidcontrol":
        return _android_template_prefill_stateful(
            dataset=dataset,
            model=model,
            processor=processor,
            base_text=base_text,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            generate_kwargs=generate_kwargs,
            configure_context_fn=configure_context_fn,
        )
    if family == "guiodyssey":
        return _gui_template_prefill_stateful(
            dataset=dataset,
            model=model,
            processor=processor,
            base_text=base_text,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            generate_kwargs=generate_kwargs,
            configure_context_fn=configure_context_fn,
        )
    return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"unsupported_dataset:{dataset}"}


def maybe_generate_with_template_prefill(
    *,
    dataset,
    model,
    processor,
    base_text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    generate_kwargs: dict,
    configure_context_fn: Optional[Callable] = None,
):
    impl = _env_impl()
    if impl in {"legacy", "old", "original"}:
        return legacy.maybe_generate_with_template_prefill(
            dataset=dataset,
            model=model,
            processor=processor,
            base_text=base_text,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            generate_kwargs=generate_kwargs,
            configure_context_fn=configure_context_fn,
        )

    try:
        response, meta = _maybe_generate_with_template_prefill_stateful(
            dataset=dataset,
            model=model,
            processor=processor,
            base_text=base_text,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            generate_kwargs=generate_kwargs,
            configure_context_fn=configure_context_fn,
        )
        if response is None:
            legacy._debug_print(dataset, f"stateful fallback reason={meta.get('template_prefill_fallback_reason')}")
        else:
            legacy._debug_print(
                dataset,
                "success "
                f"schema={meta.get('template_schema')} "
                f"impl={meta.get('template_prefill_impl')} "
                f"static_tokens={meta.get('template_static_token_count', 0)} "
                f"decode_tokens={meta.get('template_decode_tokens', 0)} "
                f"final={response!r}",
            )
        return response, meta
    except Exception as exc:
        legacy._debug_print(dataset, f"stateful_exception={type(exc).__name__}:{exc}")
        if os.getenv("QWEN3VL_TEMPLATE_PREFILL_STATEFUL_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}:
            response, meta = legacy.maybe_generate_with_template_prefill(
                dataset=dataset,
                model=model,
                processor=processor,
                base_text=base_text,
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                video_kwargs=video_kwargs,
                generate_kwargs=generate_kwargs,
                configure_context_fn=configure_context_fn,
            )
            if isinstance(meta, dict):
                meta = dict(meta)
                meta.setdefault("template_prefill_impl", "legacy_after_stateful_exception")
                meta.setdefault("template_prefill_stateful_exception", f"{type(exc).__name__}:{exc}")
            return response, meta
        return None, {
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": f"stateful_exception:{type(exc).__name__}:{exc}",
            "template_prefill_impl": "stateful_slot_cache",
            "template_prefill_stateful_exception": f"{type(exc).__name__}:{exc}",
        }
