from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from . import modeling_qwen3_vl_prune_template as legacy


@dataclass
class _StaticSegment:
    text: str
    token_ids: list[int]
    progress: int = 0


@dataclass
class _SlotSegment:
    name: str
    parser: object
    max_new_tokens: int
    token_ids: list[int]
    prompt_tokens: int
    raw_text: str = ""
    rendered_text: str = ""
    canonical: str = ""
    done: bool = False
    fallback: bool = False
    reason: str = ""


class _BoundaryAndroidBBoxSlotParser:
    _pattern = re.compile(
        r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(.*)$",
        re.DOTALL,
    )

    def parse(self, raw_text: str) -> legacy.SlotParseResult:
        text = str(raw_text or "")
        match = self._pattern.match(text)
        if not match:
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) > 4:
                return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_too_many_numbers")
            return legacy.SlotParseResult(done=False)

        remainder = match.group(5)
        if not remainder:
            # The fourth coordinate may still be streaming, e.g. "2040" after "2".
            return legacy.SlotParseResult(done=False)
        if not remainder.lstrip().startswith("]"):
            if len(re.findall(r"-?\d+(?:\.\d+)?", text)) > 4:
                return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_extra_content")
            return legacy.SlotParseResult(done=False)

        try:
            values = [float(match.group(i)) for i in range(1, 5)]
        except Exception:
            return legacy.SlotParseResult(done=False, fallback=True, reason="bbox_parse_failure")

        rendered = ", ".join(legacy._format_number(v) for v in values)
        return legacy.SlotParseResult(done=True, rendered=rendered, canonical=rendered)


class _AndroidActionThenBBoxState:
    def __init__(
        self,
        *,
        tokenizer,
        prompt_token_len: int,
        initial_text: str,
        eos_token_ids: set[int],
        output_mode: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_token_len = int(prompt_token_len)
        self.initial_text = str(initial_text)
        self.eos_token_ids = set(int(x) for x in eos_token_ids)
        self.output_mode = str(output_mode or "action_first_json")
        self.static_parts = [self.initial_text]
        self.total_static_token_count = int(
            sum(len(self.tokenizer.encode(part, add_special_tokens=False)) for part in self.static_parts)
        )
        self.segments: list[object] = [
            _SlotSegment(
                name="action_type",
                parser=legacy.ClosedSetSlotParser(
                    candidates=legacy.ANDROID_CLOSED_ACTIONS,
                    ignore_case=False,
                    allow_boundary_chars='"',
                    open_prefixes=legacy.ANDROID_OPEN_ACTION_PREFIXES,
                ),
                max_new_tokens=20,
                token_ids=[],
                prompt_tokens=int(prompt_token_len),
            ),
        ]
        self.current_index = 0
        self.processed_generated_tokens = 0
        self.generated_ids: list[int] = []
        self.generated_text = ""
        self.slot_values: dict[str, str] = {}
        self.slot_stats: list[dict] = []
        self.static_decode_steps = 0
        self.finished = False
        self.failed = False
        self.failed_reason = ""

    def _append_static_segment(self, text: str) -> None:
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        self.static_parts.append(text)
        self.total_static_token_count += len(token_ids)
        self.segments.append(_StaticSegment(text=text, token_ids=token_ids))

    def _append_slot_segment(self, name: str, parser, max_new_tokens: int) -> None:
        self.segments.append(
            _SlotSegment(
                name=name,
                parser=parser,
                max_new_tokens=int(max_new_tokens),
                token_ids=[],
                prompt_tokens=int(self.prompt_token_len + self.processed_generated_tokens),
            )
        )

    def _expand_after_slot(self, slot: _SlotSegment) -> None:
        if slot.name == "action_type":
            if slot.canonical in ("click", "long_press"):
                self._append_static_segment('", "bbox_2d": [')
                self._append_slot_segment("bbox_2d", _BoundaryAndroidBBoxSlotParser(), 40)
                self._append_static_segment("}</answer>")
            else:
                self._append_static_segment('"}</answer>')

    def _current_segment(self):
        if self.current_index >= len(self.segments):
            self.finished = True
            return None
        return self.segments[self.current_index]

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.finished = True
        self.failed_reason = str(reason)

    def process(self, input_ids: torch.LongTensor) -> None:
        if self.finished:
            return
        generated = input_ids[0, self.prompt_token_len :].tolist()
        while self.processed_generated_tokens < len(generated) and not self.finished:
            token_id = int(generated[self.processed_generated_tokens])
            self.generated_ids.append(token_id)
            seg = self._current_segment()
            if seg is None:
                self._fail("unexpected_extra_token_after_finish")
                break
            if isinstance(seg, _StaticSegment):
                if seg.progress >= len(seg.token_ids):
                    self.current_index += 1
                    continue
                expected_id = int(seg.token_ids[seg.progress])
                if token_id != expected_id:
                    self._fail(f"static_token_mismatch:expected={expected_id},got={token_id}")
                    break
                seg.progress += 1
                self.static_decode_steps += 1
                self.processed_generated_tokens += 1
                if seg.progress >= len(seg.token_ids):
                    self.current_index += 1
                continue

            seg.token_ids.append(token_id)
            seg.raw_text = self.tokenizer.decode(
                seg.token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = seg.parser.parse(seg.raw_text)
            self.processed_generated_tokens += 1
            if parsed.fallback:
                seg.fallback = True
                seg.reason = parsed.reason or f"{seg.name}_fallback"
                self._fail(seg.reason)
                break
            if len(seg.token_ids) >= seg.max_new_tokens and not parsed.done:
                seg.fallback = True
                seg.reason = f"{seg.name}_max_new_tokens"
                self._fail(seg.reason)
                break
            if parsed.done:
                seg.done = True
                seg.rendered_text = parsed.rendered
                seg.canonical = parsed.canonical
                self.slot_values[seg.name] = parsed.canonical
                self.slot_stats.append(
                    {
                        "slot": seg.name,
                        "raw_text": seg.raw_text,
                        "rendered_text": parsed.rendered,
                        "decode_tokens": len(seg.token_ids),
                        "prompt_tokens": seg.prompt_tokens,
                        "done": True,
                        "fallback": False,
                        "reason": parsed.reason,
                    }
                )
                self.current_index += 1
                self._expand_after_slot(seg)

        self.generated_text = self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if self.current_index >= len(self.segments):
            self.finished = True

    def force_token_id(self) -> Optional[int]:
        if self.finished:
            return None
        seg = self._current_segment()
        if seg is None or isinstance(seg, _SlotSegment):
            return None
        if seg.progress >= len(seg.token_ids):
            self.current_index += 1
            return self.force_token_id()
        return int(seg.token_ids[seg.progress])

    def build_final_text(self) -> str:
        action = self.slot_values.get("action_type", "")
        if action in ("click", "long_press"):
            bbox = self.slot_values.get("bbox_2d", "")
            if self.output_mode == "action_first_json":
                return f'<answer>{{"action_type": "{action}", "bbox_2d": [{bbox}]}}</answer>'
            return f'<answer>{{"bbox_2d": [{bbox}], "action_type": "{action}"}}</answer>'
        return f'<answer>{{"action_type": "{action}"}}</answer>'


class _TemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, state: _AndroidActionThenBBoxState, eos_token_ids: set[int]) -> None:
        self.state = state
        self.eos_token_ids = set(int(x) for x in eos_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.state.process(input_ids)
        if self.state.failed:
            masked = torch.full_like(scores, float("-inf"))
            eos_id = next(iter(self.eos_token_ids), None)
            if eos_id is not None and 0 <= eos_id < masked.shape[-1]:
                masked[:, eos_id] = 0
                return masked
            return scores
        forced_id = self.state.force_token_id()
        if forced_id is None:
            return scores
        masked = torch.full_like(scores, float("-inf"))
        if 0 <= forced_id < masked.shape[-1]:
            masked[:, forced_id] = 0
        return masked


class _TemplateStoppingCriteria(StoppingCriteria):
    def __init__(self, state: _AndroidActionThenBBoxState) -> None:
        self.state = state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.state.process(input_ids)
        return bool(self.state.finished)


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


def _build_android_final_text(mode: str, action: str, bbox: str = "") -> str:
    if action in ("click", "long_press"):
        if str(mode or "") == "action_first_json":
            return f'<answer>{{"action_type": "{action}", "bbox_2d": [{bbox}]}}</answer>'
        return f'<answer>{{"bbox_2d": [{bbox}], "action_type": "{action}"}}</answer>'
    return f'<answer>{{"action_type": "{action}"}}</answer>'


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
    mode = legacy._android_mode()
    if mode not in ("action_first_json", "bbox_first_json"):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"android_mode:{mode}"}

    built = '<answer>{"action_type": "'
    static_parts = ['<answer>{"action_type": "']
    static_token_total = len(processor.tokenizer.encode(static_parts[0], add_special_tokens=False))
    slot_stats: list[dict] = []

    action_parser = legacy.ClosedSetSlotParser(
        candidates=legacy.ANDROID_CLOSED_ACTIONS,
        ignore_case=False,
        allow_boundary_chars='"',
        open_prefixes=legacy.ANDROID_OPEN_ACTION_PREFIXES,
    )
    parsed_action, raw_action_text, action_decode_tokens, action_prompt_tokens = legacy._run_slot_generate(
        model=model,
        processor=processor,
        text=base_text + built,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        parser=action_parser,
        generate_kwargs=generate_kwargs,
        max_new_tokens=20,
        configure_context_fn=configure_context_fn,
    )
    slot_stats.append(
        {
            "slot": "action_type",
            "raw_text": raw_action_text,
            "rendered_text": parsed_action.rendered,
            "decode_tokens": int(action_decode_tokens),
            "prompt_tokens": int(action_prompt_tokens),
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
            "template_static_token_count": int(static_token_total),
            "template_static_decode_steps": 0,
            "template_unknown_decode_steps": int(action_decode_tokens),
            "template_decode_tokens": int(action_decode_tokens),
            "template_prefill_impl": "multi_generate_android_slot_prefill",
            "template_generated_text": raw_action_text,
        }

    built += parsed_action.rendered
    action_value = parsed_action.canonical
    raw_slot_texts = [raw_action_text]
    bbox_value = ""

    if action_value in ("click", "long_press"):
        static_mid = '", "bbox_2d": ['
        built += static_mid
        static_parts.append(static_mid)
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        parsed_bbox, raw_bbox_text, bbox_decode_tokens, bbox_prompt_tokens = legacy._run_slot_generate(
            model=model,
            processor=processor,
            text=base_text + built,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            parser=_BoundaryAndroidBBoxSlotParser(),
            generate_kwargs=generate_kwargs,
            max_new_tokens=40,
            configure_context_fn=configure_context_fn,
        )
        slot_stats.append(
            {
                "slot": "bbox_2d",
                "raw_text": raw_bbox_text,
                "rendered_text": parsed_bbox.rendered,
                "decode_tokens": int(bbox_decode_tokens),
                "prompt_tokens": int(bbox_prompt_tokens),
                "done": parsed_bbox.done,
                "fallback": parsed_bbox.fallback,
                "reason": parsed_bbox.reason,
            }
        )
        raw_slot_texts.append(raw_bbox_text)
        if not parsed_bbox.done:
            total_decode_tokens = int(sum(int(item["decode_tokens"]) for item in slot_stats))
            return None, {
                "template_prefill_enabled": False,
                "template_prefill_fallback_reason": parsed_bbox.reason or "android_bbox_slot_failed",
                "template_slot_stats": slot_stats,
                "template_static_parts": static_parts,
                "template_static_token_count": int(static_token_total),
                "template_static_decode_steps": 0,
                "template_unknown_decode_steps": total_decode_tokens,
                "template_decode_tokens": total_decode_tokens,
                "template_prefill_impl": "multi_generate_android_slot_prefill",
                "template_generated_text": "".join(raw_slot_texts),
            }
        bbox_value = parsed_bbox.canonical
        built += parsed_bbox.rendered
        suffix = "]}</answer>"
    else:
        suffix = '"}</answer>'

    built += suffix
    static_parts.append(suffix)
    static_token_total += len(processor.tokenizer.encode(suffix, add_special_tokens=False))
    total_decode_tokens = int(sum(int(item["decode_tokens"]) for item in slot_stats))
    final_text = _build_android_final_text(mode, action_value, bbox_value)
    return final_text, {
        "template_prefill_enabled": True,
        "template_schema": f"android_{mode}",
        "template_slot_stats": slot_stats,
        "template_static_parts": static_parts,
        "template_static_token_count": int(static_token_total),
        "template_static_decode_steps": 0,
        "template_unknown_decode_steps": total_decode_tokens,
        "template_decode_tokens": total_decode_tokens,
        "template_final_text": final_text,
        "template_generated_text": "".join(raw_slot_texts),
        "template_prefill_impl": "multi_generate_android_slot_prefill",
    }
