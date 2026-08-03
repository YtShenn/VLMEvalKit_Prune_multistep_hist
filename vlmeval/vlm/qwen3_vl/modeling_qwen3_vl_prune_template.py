from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


ANDROID_CLOSED_ACTIONS = (
    "click",
    "long_press",
    "swipe:up",
    "swipe:down",
    "swipe:left",
    "swipe:right",
    "wait",
    "navigate_back",
    "navigate_home",
)

ANDROID_OPEN_ACTION_PREFIXES = (
    "input_text:",
    "open_app:",
)

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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _dataset_family(dataset: str | None) -> str:
    name = str(dataset or "")
    if name.startswith("AndroidControl"):
        return "androidcontrol"
    if name.startswith("GUIOdyssey"):
        return "guiodyssey"
    return ""


def _dataset_enabled(dataset: str | None) -> bool:
    enabled = os.getenv("QWEN3VL_TEMPLATE_PREFILL_DATASETS", "androidcontrol,guiodyssey")
    families = {x.strip().lower() for x in enabled.split(",") if x.strip()}
    fam = _dataset_family(dataset)
    return bool(fam and fam in families)


def _android_mode() -> str:
    return os.getenv("QWEN3VL_TEMPLATE_PREFILL_ANDROID_MODE", "action_first_json").strip().lower()


def _gui_mode() -> str:
    return os.getenv("QWEN3VL_TEMPLATE_PREFILL_GUI_MODE", "command").strip().lower()


def _debug_enabled() -> bool:
    return _env_flag("QWEN3VL_TEMPLATE_PREFILL_DEBUG", "0")


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


@dataclass
class SlotParseResult:
    done: bool
    rendered: str = ""
    canonical: str = ""
    fallback: bool = False
    reason: str = ""


class _GeneratedTextStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_token_len: int, parser) -> None:
        self.tokenizer = tokenizer
        self.prompt_token_len = int(prompt_token_len)
        self.parser = parser

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        try:
            generated = input_ids[0, self.prompt_token_len :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            result = self.parser.parse(text)
            return bool(result.done or result.fallback)
        except Exception:
            return False


class ClosedSetSlotParser:
    def __init__(self, candidates: Sequence[str], ignore_case: bool = False, allow_boundary_chars: str = "", open_prefixes: Sequence[str] = ()) -> None:
        self.candidates = list(candidates)
        self.ignore_case = bool(ignore_case)
        self.allow_boundary_chars = allow_boundary_chars
        self.open_prefixes = tuple(open_prefixes)

    def _normalize(self, text: str) -> str:
        return text.upper() if self.ignore_case else text

    def parse(self, raw_text: str) -> SlotParseResult:
        text = str(raw_text or "")
        text = text.lstrip()
        if not text:
            return SlotParseResult(done=False)

        normalized_text = self._normalize(text)
        normalized_candidates = [self._normalize(x) for x in self.candidates]

        for prefix in self.open_prefixes:
            norm_prefix = self._normalize(prefix)
            if normalized_text.startswith(norm_prefix):
                return SlotParseResult(done=False, fallback=True, reason=f"open_prefix:{prefix}")

        exact_match = None
        for idx, candidate in enumerate(normalized_candidates):
            if normalized_text == candidate:
                exact_match = self.candidates[idx]
                break
            if normalized_text.startswith(candidate):
                remainder = text[len(self.candidates[idx]) :]
                if remainder and remainder[0] in self.allow_boundary_chars and remainder.strip(self.allow_boundary_chars + " \n\r\t") == "":
                    exact_match = self.candidates[idx]
                    break

        if exact_match is not None:
            return SlotParseResult(done=True, rendered=exact_match, canonical=exact_match)

        prefix_matches = [self.candidates[i] for i, candidate in enumerate(normalized_candidates) if candidate.startswith(normalized_text)]
        if prefix_matches:
            return SlotParseResult(done=False)
        return SlotParseResult(done=False, fallback=True, reason=f"invalid_text:{text[:64]!r}")


class AndroidBBoxSlotParser:
    _pattern = re.compile(
        r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    )

    def parse(self, raw_text: str) -> SlotParseResult:
        text = str(raw_text or "")
        match = self._pattern.match(text)
        if not match:
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) > 4:
                return SlotParseResult(done=False, fallback=True, reason="bbox_too_many_numbers")
            return SlotParseResult(done=False)

        try:
            values = [float(match.group(i)) for i in range(1, 5)]
        except Exception:
            return SlotParseResult(done=False, fallback=True, reason="bbox_parse_failure")

        remainder = text[match.end() :]
        if remainder and not remainder.lstrip().startswith("]") and remainder.strip() not in ("", ",", ", ", " ]"):
            if len(re.findall(r"-?\d+(?:\.\d+)?", text)) > 4:
                return SlotParseResult(done=False, fallback=True, reason="bbox_extra_content")
            return SlotParseResult(done=False)

        rendered = ", ".join(_format_number(v) for v in values)
        return SlotParseResult(done=True, rendered=rendered, canonical=rendered)


class GuiCoordPairSlotParser:
    _pattern = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")

    def parse(self, raw_text: str) -> SlotParseResult:
        text = str(raw_text or "")
        match = self._pattern.match(text)
        if not match:
            nums = re.findall(r"-?\d+(?:\.\d+)?", text)
            if len(nums) > 2:
                return SlotParseResult(done=False, fallback=True, reason="coord_pair_too_many_numbers")
            return SlotParseResult(done=False)

        try:
            x = float(match.group(1))
            y = float(match.group(2))
        except Exception:
            return SlotParseResult(done=False, fallback=True, reason="coord_pair_parse_failure")

        remainder = text[match.end() :]
        if remainder and not remainder.lstrip().startswith(")") and remainder.strip() not in ("", ",", ", ", " )"):
            if len(re.findall(r"-?\d+(?:\.\d+)?", text)) > 2:
                return SlotParseResult(done=False, fallback=True, reason="coord_pair_extra_content")
            return SlotParseResult(done=False)

        rendered = f"{_format_number(x)}, {_format_number(y)}"
        return SlotParseResult(done=True, rendered=rendered, canonical=rendered)


def _move_inputs_to_model(inputs, model):
    try:
        inputs = inputs.to(model.device)
        if hasattr(model, "dtype"):
            inputs = inputs.to(model.dtype)
    except Exception:
        inputs = inputs.to("cuda")
    return inputs


def _decode_prompt_token_len(local_inputs) -> int:
    if hasattr(local_inputs, "input_ids"):
        prompt_lengths = [int(x.shape[0]) for x in local_inputs.input_ids]
    else:
        prompt_tensor = local_inputs.get("input_ids", None)
        prompt_lengths = [int(x.shape[0]) for x in prompt_tensor] if prompt_tensor is not None else []
    return int(prompt_lengths[0]) if prompt_lengths else 0


def _build_processor_inputs(
    processor,
    model,
    text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
):
    local_inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    return _move_inputs_to_model(local_inputs, model)


def _run_slot_generate(
    *,
    model,
    processor,
    text: str,
    images,
    videos,
    video_metadatas,
    video_kwargs,
    parser,
    generate_kwargs: dict,
    max_new_tokens: int,
    configure_context_fn: Optional[Callable],
):
    local_inputs = _build_processor_inputs(
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
    prompt_token_len = _decode_prompt_token_len(local_inputs)
    stopping = StoppingCriteriaList(
        [_GeneratedTextStoppingCriteria(processor.tokenizer, prompt_token_len=prompt_token_len, parser=parser)]
    )

    slot_generate_kwargs = dict(generate_kwargs)
    slot_generate_kwargs["max_new_tokens"] = int(max_new_tokens)
    outputs = model.generate(
        **local_inputs,
        stopping_criteria=stopping,
        **slot_generate_kwargs,
    )
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(local_inputs.input_ids, outputs)]
    text_out = processor.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parser.parse(text_out)
    decode_tokens = int(generated_ids[0].shape[0]) if generated_ids else 0
    return parsed, text_out, decode_tokens, prompt_token_len


def _debug_print(dataset: str | None, message: str) -> None:
    if _debug_enabled():
        print(f"[TemplatePrefill] dataset={dataset} {message}", flush=True)


def _android_template_prefill(
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
    mode = _android_mode()
    if mode not in ("action_first_json",):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"android_mode:{mode}"}

    built = '<answer>{"action_type": "'
    slot_stats = []
    static_token_total = 0

    static_token_total += len(processor.tokenizer.encode(built, add_special_tokens=False))
    action_parser = ClosedSetSlotParser(
        candidates=ANDROID_CLOSED_ACTIONS,
        ignore_case=False,
        allow_boundary_chars='"',
        open_prefixes=ANDROID_OPEN_ACTION_PREFIXES,
    )
    parsed_action, raw_action_text, action_decode_tokens, prompt_tokens = _run_slot_generate(
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
            "template_static_parts": ['<answer>{"action_type": "'],
        }

    built += parsed_action.rendered
    action_value = parsed_action.canonical

    static_parts = ['<answer>{"action_type": "']
    if action_value in ("click", "long_press"):
        static_mid = '", "bbox_2d": ['
        built += static_mid
        static_parts.append(static_mid)
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        bbox_parser = AndroidBBoxSlotParser()
        parsed_bbox, raw_bbox_text, bbox_decode_tokens, bbox_prompt_tokens = _run_slot_generate(
            model=model,
            processor=processor,
            text=base_text + built,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            parser=bbox_parser,
            generate_kwargs=generate_kwargs,
            max_new_tokens=32,
            configure_context_fn=configure_context_fn,
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
            }
        built += parsed_bbox.rendered
        suffix = "]}</answer>"
    else:
        suffix = '"}</answer>'

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
    }


def _gui_template_prefill(
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
    mode = _gui_mode()
    if mode not in ("command",):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"gui_mode:{mode}"}

    slot_stats = []
    static_token_total = 0

    head_parser = ClosedSetSlotParser(candidates=GUI_ACTION_HEADS, ignore_case=True, allow_boundary_chars=": \n\r\t")
    parsed_head, raw_head_text, head_decode_tokens, head_prompt_tokens = _run_slot_generate(
        model=model,
        processor=processor,
        text=base_text,
        images=images,
        videos=videos,
        video_metadatas=video_metadatas,
        video_kwargs=video_kwargs,
        parser=head_parser,
        generate_kwargs=generate_kwargs,
        max_new_tokens=12,
        configure_context_fn=configure_context_fn,
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
        }

    action_head = parsed_head.canonical.upper()
    built = action_head

    if action_head in GUI_TERMINAL_ACTIONS:
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
        }

    if action_head == "TYPE":
        return None, {
            "template_prefill_enabled": False,
            "template_prefill_fallback_reason": "gui_type_free_text_fallback",
            "template_slot_stats": slot_stats,
            "template_static_parts": [],
        }

    if action_head == "SCROLL":
        static_mid = ": "
        built += static_mid
        static_parts = [static_mid]
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        direction_parser = ClosedSetSlotParser(candidates=GUI_SCROLL_DIRECTIONS, ignore_case=True, allow_boundary_chars="\n\r\t ")
        parsed_dir, raw_dir_text, dir_decode_tokens, dir_prompt_tokens = _run_slot_generate(
            model=model,
            processor=processor,
            text=base_text + built,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            parser=direction_parser,
            generate_kwargs=generate_kwargs,
            max_new_tokens=8,
            configure_context_fn=configure_context_fn,
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
        }

    if action_head in GUI_COORD_ACTIONS:
        static_mid = ": ("
        built += static_mid
        static_parts = [static_mid]
        static_token_total += len(processor.tokenizer.encode(static_mid, add_special_tokens=False))
        coord_parser = GuiCoordPairSlotParser()
        parsed_coords, raw_coords_text, coord_decode_tokens, coord_prompt_tokens = _run_slot_generate(
            model=model,
            processor=processor,
            text=base_text + built,
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            video_kwargs=video_kwargs,
            parser=coord_parser,
            generate_kwargs=generate_kwargs,
            max_new_tokens=16,
            configure_context_fn=configure_context_fn,
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
        }

    return None, {
        "template_prefill_enabled": False,
        "template_prefill_fallback_reason": f"unsupported_gui_action:{action_head}",
        "template_slot_stats": slot_stats,
        "template_static_parts": [],
    }


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
    if not _env_flag("QWEN3VL_ENABLE_TEMPLATE_PREFILL", "0"):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "disabled"}
    if not _dataset_enabled(dataset):
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": "dataset_not_enabled"}

    family = _dataset_family(dataset)
    _debug_print(dataset, f"enabled family={family}")
    if family == "androidcontrol":
        response, meta = _android_template_prefill(
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
    elif family == "guiodyssey":
        response, meta = _gui_template_prefill(
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
    else:
        return None, {"template_prefill_enabled": False, "template_prefill_fallback_reason": f"unsupported_dataset:{dataset}"}

    if response is None:
        _debug_print(dataset, f"fallback reason={meta.get('template_prefill_fallback_reason')}")
    else:
        _debug_print(
            dataset,
            "success "
            f"schema={meta.get('template_schema')} "
            f"static_tokens={meta.get('template_static_token_count', 0)} "
            f"decode_tokens={meta.get('template_decode_tokens', 0)} "
            f"final={response!r}",
        )
    return response, meta
