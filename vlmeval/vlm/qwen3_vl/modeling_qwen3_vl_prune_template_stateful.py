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


@dataclass
class _SessionSnapshot:
    current_text: str
    total_tokens: int
    attention_dtype: torch.dtype
    attention_device: torch.device
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
        model_inputs = dict(local_inputs)
        model_inputs["use_cache"] = True
        model_inputs["return_dict"] = True
        outputs = self._model_forward(**model_inputs)
        past = getattr(outputs, "past_key_values", None)
        logits = getattr(outputs, "logits", None)
        if past is None or logits is None:
            raise RuntimeError("initial template prefill produced no cache/logits")
        self.current_text = str(text)
        self.total_tokens = int(prompt_token_len)
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
            past_key_values=_clone_past_key_values(self.past_key_values),
            last_logits=_clone_logits(self.last_logits),
        )

    def restore(self, snapshot: _SessionSnapshot) -> None:
        self.current_text = snapshot.current_text
        self.total_tokens = int(snapshot.total_tokens)
        self.attention_dtype = snapshot.attention_dtype
        self.attention_device = snapshot.attention_device
        self.past_key_values = snapshot.past_key_values
        self.last_logits = snapshot.last_logits

    def _forward_token_ids(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return
        if self.past_key_values is None or self.last_logits is None:
            raise RuntimeError("session is not initialized")
        device = self.attention_device
        new_input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
        total_len = int(self.total_tokens) + int(new_input_ids.shape[1])
        attention_mask = torch.ones((1, total_len), dtype=self.attention_dtype, device=device)
        outputs = self._model_forward(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past = getattr(outputs, "past_key_values", None)
        logits = getattr(outputs, "logits", None)
        if past is None or logits is None:
            raise RuntimeError("token append produced no cache/logits")
        self.past_key_values = past
        self.last_logits = logits
        self.total_tokens = total_len

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
            next_token_id = _sampling_next_token(self.last_logits[:, -1, :], self.generate_kwargs)
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
            next_token_id = _sampling_next_token(self.last_logits[:, -1, :], self.generate_kwargs)
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
    if mode not in ("action_first_json",):
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
    parsed_action, raw_action_text, action_decode_tokens, prompt_tokens = session.generate_slot(
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
        bbox_parser = legacy.AndroidBBoxSlotParser()
        parsed_bbox, raw_bbox_text, bbox_decode_tokens, bbox_prompt_tokens = session.generate_slot(
            parser=bbox_parser,
            max_new_tokens=32,
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
        built += parsed_bbox.rendered
        suffix = "]}</answer>"
    else:
        suffix = '"}</answer>'

    session.append_known_text(suffix)
    built += suffix
    static_parts.append(suffix)
    static_token_total += len(processor.tokenizer.encode(suffix, add_special_tokens=False))
    total_decode_tokens = sum(int(item["decode_tokens"]) for item in slot_stats)
    return built, {
        "template_prefill_enabled": True,
        "template_schema": "android_action_first_json",
        "template_slot_stats": slot_stats,
        "template_static_parts": static_parts,
        "template_static_token_count": int(static_token_total),
        "template_static_decode_steps": 0,
        "template_unknown_decode_steps": int(total_decode_tokens),
        "template_decode_tokens": int(total_decode_tokens),
        "template_final_text": built,
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
        session.append_known_text(suffix)
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
        if os.getenv("QWEN3VL_TEMPLATE_PREFILL_STATEFUL_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}:
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
        raise
