from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList

from . import modeling_qwen3_vl_prune_template as legacy
from . import modeling_qwen3_vl_prune_template_stateful as stateful


def _env_impl() -> str:
    return os.getenv("QWEN3VL_TEMPLATE_PREFILL_IMPL", "single_generate").strip().lower()


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
    token_ids: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    raw_text: str = ""
    rendered_text: str = ""
    done: bool = False
    fallback: bool = False
    reason: str = ""
    canonical: str = ""


@dataclass
class _FreeTextSegment:
    name: str
    max_new_tokens: int
    token_ids: list[int] = field(default_factory=list)
    raw_text: str = ""
    done: bool = False


class _TemplateConstraintState:
    def __init__(
        self,
        *,
        dataset: str | None,
        family: str,
        tokenizer,
        prompt_token_len: int,
        schema_name: str,
        initial_text: str,
        initial_segments: list[object],
        initial_static_parts: list[str],
        total_static_token_count: int,
        eos_token_ids: Optional[set[int]] = None,
        gui_type_max_new_tokens: int = 32,
    ) -> None:
        self.dataset = dataset
        self.family = family
        self.tokenizer = tokenizer
        self.prompt_token_len = int(prompt_token_len)
        self.schema_name = schema_name
        self.initial_text = initial_text
        self.segments = list(initial_segments)
        self.static_parts = list(initial_static_parts)
        self.total_static_token_count = int(total_static_token_count)
        self.processed_generated_tokens = 0
        self.current_index = 0
        self.finished = False
        self.failed = False
        self.failed_reason = ""
        self.static_decode_steps = 0
        self.slot_stats: list[dict] = []
        self.slot_values: dict[str, str] = {}
        self.generated_ids: list[int] = []
        self.generated_text = ""
        self.eos_token_ids = set(int(x) for x in (eos_token_ids or set()))
        self.gui_type_max_new_tokens = int(gui_type_max_new_tokens)
        self._advance_empty_static_segments()

    def _advance_empty_static_segments(self) -> None:
        while self.current_index < len(self.segments):
            seg = self.segments[self.current_index]
            if isinstance(seg, _StaticSegment) and not seg.token_ids:
                self.current_index += 1
                continue
            break
        if self.current_index >= len(self.segments):
            self.finished = True

    def _append_static_segment(self, text: str) -> None:
        text = str(text or "")
        token_ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        self.segments.append(_StaticSegment(text=text, token_ids=token_ids))
        self.static_parts.append(text)
        self.total_static_token_count += len(token_ids)

    def _append_slot_segment(self, name: str, parser, max_new_tokens: int) -> None:
        self.segments.append(
            _SlotSegment(
                name=name,
                parser=parser,
                max_new_tokens=int(max_new_tokens),
                prompt_tokens=int(self.prompt_token_len + self.processed_generated_tokens),
            )
        )

    def _append_free_text_segment(self, name: str, max_new_tokens: int) -> None:
        self.segments.append(_FreeTextSegment(name=name, max_new_tokens=int(max_new_tokens)))

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.finished = True
        self.failed_reason = str(reason)

    def _strict_closed_parser(self, candidates, *, ignore_case=False, open_prefixes=()):
        return legacy.ClosedSetSlotParser(
            candidates=candidates,
            ignore_case=ignore_case,
            allow_boundary_chars="",
            open_prefixes=open_prefixes,
        )

    def _expand_after_slot(self, slot: _SlotSegment) -> None:
        self.slot_values[slot.name] = slot.canonical
        if self.family == "androidcontrol":
            if slot.name == "action_type":
                if slot.canonical in ("click", "long_press"):
                    self._append_static_segment('", "bbox_2d": [')
                    self._append_slot_segment("bbox_2d", legacy.AndroidBBoxSlotParser(), 32)
                    self._append_static_segment("]}</answer>")
                else:
                    self._append_static_segment('}</answer>' if False else '"}</answer>')
                return
            if slot.name == "bbox_2d":
                return
        if self.family == "guiodyssey":
            if slot.name == "action_head":
                head = slot.canonical.upper()
                if head in legacy.GUI_TERMINAL_ACTIONS:
                    return
                if head == "TYPE":
                    self._append_static_segment(": ")
                    self._append_free_text_segment("free_text", self.gui_type_max_new_tokens)
                    return
                if head == "SCROLL":
                    self._append_static_segment(": ")
                    self._append_slot_segment(
                        "direction",
                        self._strict_closed_parser(legacy.GUI_SCROLL_DIRECTIONS, ignore_case=True),
                        8,
                    )
                    return
                if head in legacy.GUI_COORD_ACTIONS:
                    self._append_static_segment(": (")
                    self._append_slot_segment("coord_pair", legacy.GuiCoordPairSlotParser(), 16)
                    self._append_static_segment(")")
                    return
                self._fail(f"unsupported_gui_action:{head}")
                return
            if slot.name in ("direction", "coord_pair"):
                return

    def _current_segment(self):
        self._advance_empty_static_segments()
        if self.current_index >= len(self.segments):
            self.finished = True
            return None
        return self.segments[self.current_index]

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

            if isinstance(seg, _FreeTextSegment):
                seg.token_ids.append(token_id)
                seg.raw_text = self.tokenizer.decode(
                    seg.token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                self.processed_generated_tokens += 1
                if token_id in self.eos_token_ids or len(seg.token_ids) >= seg.max_new_tokens:
                    seg.done = True
                    self.slot_values[seg.name] = seg.raw_text
                    self.current_index += 1
                    self.finished = True
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
        self._advance_empty_static_segments()

    def force_token_id(self) -> Optional[int]:
        if self.finished:
            return None
        seg = self._current_segment()
        if seg is None or isinstance(seg, (_SlotSegment, _FreeTextSegment)):
            return None
        if seg.progress >= len(seg.token_ids):
            self.current_index += 1
            return self.force_token_id()
        return int(seg.token_ids[seg.progress])

    def build_final_text(self) -> str:
        if self.family == "androidcontrol":
            action = self.slot_values.get("action_type", "")
            if action in ("click", "long_press"):
                bbox = self.slot_values.get("bbox_2d", "")
                return f'{self.initial_text}{action}", "bbox_2d": [{bbox}]}}</answer>'
            return f'{self.initial_text}{action}"}}</answer>'
        if self.family == "guiodyssey":
            head = self.slot_values.get("action_head", "")
            if head.upper() in legacy.GUI_TERMINAL_ACTIONS:
                return head
            if head.upper() == "TYPE":
                return f"{head}: {self.slot_values.get('free_text', '')}"
            if head.upper() == "SCROLL":
                return f"{head}: {self.slot_values.get('direction', '')}"
            if head.upper() in legacy.GUI_COORD_ACTIONS:
                return f"{head}: ({self.slot_values.get('coord_pair', '')})"
            return head
        return self.initial_text + self.generated_text


class _TemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, state: _TemplateConstraintState, eos_token_ids: set[int]) -> None:
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
    def __init__(self, state: _TemplateConstraintState) -> None:
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


def _run_single_generate(
    *,
    dataset,
    family: str,
    schema_name: str,
    model,
    processor,
    text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    generate_kwargs: dict,
    configure_context_fn: Optional[Callable],
    state: _TemplateConstraintState,
):
    local_inputs = legacy._build_processor_inputs(
        processor=processor,
        model=model,
        text=text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
    )
    if configure_context_fn is not None:
        configure_context_fn(local_inputs)
    eos_ids = _eos_token_ids(model, processor.tokenizer)
    logits_processor = LogitsProcessorList([_TemplateLogitsProcessor(state, eos_ids)])
    stopping_criteria = StoppingCriteriaList([_TemplateStoppingCriteria(state)])
    run_kwargs = dict(generate_kwargs)
    run_kwargs["max_new_tokens"] = int(sum(getattr(seg, "max_new_tokens", 0) for seg in state.segments) + 64)
    run_kwargs["logits_processor"] = logits_processor
    run_kwargs["stopping_criteria"] = stopping_criteria
    outputs = model.generate(**local_inputs, **run_kwargs)
    state.process(local_inputs.input_ids.new_tensor(outputs))
    if state.failed:
        return None, {
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": state.failed_reason or "single_generate_failed",
            "template_slot_stats": list(state.slot_stats),
            "template_static_parts": list(state.static_parts),
            "template_prefill_impl": "single_generate_constrained",
            "template_generated_text": state.generated_text,
        }
    total_decode_tokens = int(state.processed_generated_tokens)
    return state.build_final_text(), {
        "template_prefill_enabled": True,
        "template_schema": schema_name,
        "template_slot_stats": list(state.slot_stats),
        "template_static_parts": list(state.static_parts),
        "template_static_token_count": int(state.total_static_token_count),
        "template_static_decode_steps": int(state.static_decode_steps),
        "template_unknown_decode_steps": int(sum(int(item["decode_tokens"]) for item in state.slot_stats)),
        "template_decode_tokens": total_decode_tokens,
        "template_final_text": state.build_final_text(),
        "template_generated_text": state.generated_text,
        "template_prefill_impl": "single_generate_constrained",
    }


def _android_single_generate(
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

    initial_static = '<answer>{"action_type": "'
    initial_static_ids = list(processor.tokenizer.encode(initial_static, add_special_tokens=False))
    initial_segments = [
        _SlotSegment(
            name="action_type",
            parser=legacy.ClosedSetSlotParser(
                candidates=legacy.ANDROID_CLOSED_ACTIONS,
                ignore_case=False,
                allow_boundary_chars="",
                open_prefixes=legacy.ANDROID_OPEN_ACTION_PREFIXES,
            ),
            max_new_tokens=20,
            prompt_tokens=0,
        )
    ]
    state = _TemplateConstraintState(
        dataset=dataset,
        family="androidcontrol",
        tokenizer=processor.tokenizer,
        prompt_token_len=0,
        schema_name="android_action_first_json",
        initial_text=initial_static,
        initial_segments=initial_segments,
        initial_static_parts=[initial_static],
        total_static_token_count=len(initial_static_ids),
        eos_token_ids=_eos_token_ids(model, processor.tokenizer),
    )
    text = base_text + initial_static
    tmp_inputs = legacy._build_processor_inputs(
        processor=processor,
        model=model,
        text=text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
    )
    state.prompt_token_len = legacy._decode_prompt_token_len(tmp_inputs)
    state.segments[0].prompt_tokens = state.prompt_token_len
    return _run_single_generate(
        dataset=dataset,
        family="androidcontrol",
        schema_name="android_action_first_json",
        model=model,
        processor=processor,
        text=text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        generate_kwargs=generate_kwargs,
        configure_context_fn=configure_context_fn,
        state=state,
    )


def _gui_single_generate(
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
    initial_segments = [
        _SlotSegment(
            name="action_head",
            parser=legacy.ClosedSetSlotParser(
                candidates=legacy.GUI_ACTION_HEADS,
                ignore_case=True,
                allow_boundary_chars="",
            ),
            max_new_tokens=12,
            prompt_tokens=0,
        )
    ]
    state = _TemplateConstraintState(
        dataset=dataset,
        family="guiodyssey",
        tokenizer=processor.tokenizer,
        prompt_token_len=0,
        schema_name="gui_command",
        initial_text="",
        initial_segments=initial_segments,
        initial_static_parts=[],
        total_static_token_count=0,
        eos_token_ids=_eos_token_ids(model, processor.tokenizer),
        gui_type_max_new_tokens=int(
            os.getenv(
                "QWEN3VL_TEMPLATE_PREFILL_GUI_TYPE_MAX_NEW_TOKENS",
                str(generate_kwargs.get("max_new_tokens", 32) or 32),
            )
        ),
    )
    tmp_inputs = legacy._build_processor_inputs(
        processor=processor,
        model=model,
        text=base_text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
    )
    state.prompt_token_len = legacy._decode_prompt_token_len(tmp_inputs)
    state.segments[0].prompt_tokens = state.prompt_token_len
    return _run_single_generate(
        dataset=dataset,
        family="guiodyssey",
        schema_name="gui_command",
        model=model,
        processor=processor,
        text=base_text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        generate_kwargs=generate_kwargs,
        configure_context_fn=configure_context_fn,
        state=state,
    )


def _maybe_generate_single(
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
    legacy._debug_print(dataset, f"enabled family={family} impl=single_generate_constrained")
    if family == "androidcontrol":
        return _android_single_generate(
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
        return _gui_single_generate(
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
    if impl in {"stateful", "slot_cache"}:
        return stateful.maybe_generate_with_template_prefill(
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
        response, meta = _maybe_generate_single(
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
            legacy._debug_print(dataset, f"single_generate fallback reason={meta.get('template_prefill_fallback_reason')}")
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
        legacy._debug_print(dataset, f"single_generate_exception={type(exc).__name__}:{exc}")
        if os.getenv("QWEN3VL_TEMPLATE_PREFILL_SINGLE_GENERATE_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}:
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
                meta.setdefault("template_prefill_impl", "legacy_after_single_generate_exception")
                meta.setdefault("template_prefill_single_generate_exception", f"{type(exc).__name__}:{exc}")
            return response, meta
        raise
