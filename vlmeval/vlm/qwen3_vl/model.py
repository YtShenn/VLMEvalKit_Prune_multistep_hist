from __future__ import annotations

import logging
import os
import time
import warnings
import json
import re
import types
import inspect

import torch
from PIL import Image
import ast
from ..base import BaseModel
from .prompt import Qwen3VLPromptMixin
from .structured_fast_decode import maybe_generate_with_structured_fast_decode
from ...smp import get_gpu_memory, listinstr
from ..qwen3_vl_flops import enable_qwen3vl_flops_profiling


VLLM_MAX_IMAGE_INPUT_NUM = 24


def _read_first_image_size_from_message(message: list[dict]) -> tuple[int, int] | None:
    for s in message:
        if not isinstance(s, dict):
            continue
        if s.get('type') != 'image':
            continue
        p = str(s.get('value', ''))
        if p.startswith('file://'):
            p = p[len('file://'):]
        if not p:
            continue
        try:
            with Image.open(p) as img:
                return img.size
        except Exception:
            continue
    return None


def _read_image_size_from_item(item: dict) -> tuple[int, int] | None:
    if not isinstance(item, dict) or item.get('type') != 'image':
        return None
    p = str(item.get('value', ''))
    if p.startswith('file://'):
        p = p[len('file://'):]
    if not p:
        return None
    try:
        with Image.open(p) as img:
            return img.size
    except Exception:
        return None


def _read_current_image_size_from_message(message: list[dict]) -> tuple[int, int] | None:
    """AndroidControl predicts coordinates on the final current screenshot, not history images."""
    saw_current_label = False
    last_image_size = None
    for s in message:
        if not isinstance(s, dict):
            continue
        if s.get('type') == 'text':
            text_raw = str(s.get('value', '') or '').strip()
            text = text_raw.lower()
            if (
                text.startswith('current screenshot:')
                or text.startswith('current image:')
                or '[current_image]' in text
            ):
                saw_current_label = True
            continue
        if s.get('type') != 'image':
            continue
        img_size = _read_image_size_from_item(s)
        if img_size is None:
            continue
        if saw_current_label:
            return img_size
        last_image_size = img_size
    return last_image_size


def _denorm_android_coord_values(vals, img_wh: tuple[int, int], base: float):
    if not isinstance(vals, list) or len(vals) not in (2, 4):
        return vals
    try:
        nums = [float(x) for x in vals]
    except Exception:
        return vals
    if any(v < 0 or v > base for v in nums):
        return vals
    w, h = img_wh
    if len(nums) == 4:
        out = [nums[0] / base * w, nums[1] / base * h, nums[2] / base * w, nums[3] / base * h]
    else:
        out = [nums[0] / base * w, nums[1] / base * h]
    return [int(round(x)) for x in out]


def _postprocess_androidcontrol_response(response: str, message: list[dict], base: float, denorm: bool = True) -> str:
    img_wh = _read_current_image_size_from_message(message)

    raw = str(response)
    tags = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    tag_name = "answer"
    if not tags:
        tags = re.findall(r"<action>(.*?)</action>", raw, re.DOTALL)
        tag_name = "action"
    if not tags:
        return response

    candidate = tags[-1].strip()
    try:
        payload = ast.literal_eval(candidate)
    except Exception:
        return response
    if not isinstance(payload, dict):
        return response

    changed = False
    action_type = str(payload.get("action_type", "") or payload.get("type", "") or "").strip()
    action_name = _normalize_action_name(action_type)
    if action_name not in ("click", "long_press"):
        for key in ("bbox_2d", "bbox", "point", "coordinate"):
            if key in payload:
                payload.pop(key, None)
                changed = True
    elif denorm and img_wh is not None:
        for key in ("bbox_2d", "bbox", "point", "coordinate"):
            if key in payload:
                new_vals = _denorm_android_coord_values(payload[key], img_wh, base=base)
                if new_vals != payload[key]:
                    payload[key] = new_vals
                    changed = True
                break
    if not changed:
        return response

    new_block = f"<{tag_name}>{json.dumps(payload, ensure_ascii=False)}</{tag_name}>"
    if tag_name == "answer":
        return re.sub(r"<answer>.*?</answer>", new_block, raw, count=1, flags=re.DOTALL)
    return re.sub(r"<action>.*?</action>", new_block, raw, count=1, flags=re.DOTALL)


def is_moe_model(model_path: str) -> bool:
    """Check if the model is a Mixture of Experts model."""
    path_parts = model_path.split('/')
    non_moe_patterns = ['2B','4B','8B','32B']
    for part in path_parts:
        if any(pattern in part for pattern in non_moe_patterns):
            return False
    return True


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')

#=======================timer================================
class _StageTimer:
    def __init__(self, use_cuda_events: bool = True, sync_cuda: bool = False) -> None:
        self.use_cuda_events = bool(use_cuda_events) and torch.cuda.is_available()
        self.sync_cuda = bool(sync_cuda)
        self._handles = []
        self._cpu_stacks = {}
        self._cuda_events = {}
        self.seconds = {}

    def add_module(self, key: str, module: torch.nn.Module | None) -> None:
        if module is None:
            return
        if self.use_cuda_events:
            def pre_hook(_, __):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                self._cuda_events.setdefault(key, []).append((start, end))

            def post_hook(_, __, ___):
                evs = self._cuda_events.get(key)
                if not evs:
                    return
                evs[-1][1].record()

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        else:
            def pre_hook(_, __):
                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self._cpu_stacks.setdefault(key, []).append(time.perf_counter())

            def post_hook(_, __, ___):
                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                st = self._cpu_stacks.get(key)
                if not st:
                    return
                start_t = st.pop()
                self.seconds[key] = self.seconds.get(key, 0.0) + (time.perf_counter() - start_t)

            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))

    def finalize(self) -> None:
        if not self.use_cuda_events:
            return
        torch.cuda.synchronize()
        for key, evs in self._cuda_events.items():
            total_ms = 0.0
            for start, end in evs:
                total_ms += float(start.elapsed_time(end))
            self.seconds[key] = self.seconds.get(key, 0.0) + total_ms / 1000.0

    def close(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []


def _pick_first_attr(module: torch.nn.Module, names: list[str]) -> torch.nn.Module | None:
    for n in names:
        m = getattr(module, n, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def _pick_nested_attr(module: torch.nn.Module, path: str) -> torch.nn.Module | None:
    cur = module
    for name in path.split('.'):
        cur = getattr(cur, name, None)
        if cur is None:
            return None
    return cur if isinstance(cur, torch.nn.Module) else None


def _pick_vision_and_llm_modules(model: torch.nn.Module) -> tuple[torch.nn.Module | None, torch.nn.Module | None, str, str]:
    vision = _pick_first_attr(model, ['vision_model', 'visual', 'vision_tower', 'vision_encoder', 'image_encoder', 'vision'])
    llm = _pick_first_attr(model, ['language_model', 'text_model', 'llm', 'transformer', 'decoder'])

    if vision is None:
        for path in (
            'model.visual',
            'model.vision_model',
            'model.vision',
            'model.vision_tower',
            'model.vision_encoder',
            'visual',
        ):
            vision = _pick_nested_attr(model, path)
            if vision is not None:
                break

    if llm is None:
        for path in (
            'model.language_model',
            'model.text_model',
            'model.llm',
            'model.transformer',
            'model.decoder',
            'model',
        ):
            llm = _pick_nested_attr(model, path)
            if llm is not None:
                break

    vision_name = vision.__class__.__name__ if vision is not None else 'None'
    llm_name = llm.__class__.__name__ if llm is not None else 'None'
    return vision, llm, vision_name, llm_name
#=======================timer================================


def _env_flag(name: str, default: str = '0') -> bool:
    return os.getenv(name, default).strip().lower() in ('1', 'true', 'yes', 'on')


def _use_qwen3vl_timing_model() -> bool:
    # The custom Qwen3-VL implementation is needed for ROI-prune experiments,
    # but it is not a requirement for ordinary stage timing / FLOPs profiling.
    # Keeping timing-only runs on the upstream HF model avoids shape mismatches
    # for multi-image history prompts while preserving the external timing hooks.
    return any(
        (
            _env_flag('QWEN3VL_ENABLE_ROI_PRUNE', '0'),
            _env_flag('QWEN3VL_USE_TIMING_MODEL', '0'),
        )
    )


def _use_qwen3vl_attn_prune_model(explicit: bool = False) -> bool:
    return bool(
        explicit
        or _env_flag('QWEN3VL_ENABLE_ATTN_PRUNE', '0')
        or _env_flag('QWEN3VL_USE_ATTN_PRUNE_MODEL', '0')
    )


def _sanitize_generate_inputs_for_model(model, inputs):
    # Keep multimodal helper fields intact. Compatibility with upstream HF Qwen3-VL
    # is handled by `_patch_upstream_qwen3vl_prepare_inputs_for_generation`.
    return inputs


def _patch_upstream_qwen3vl_prepare_inputs_for_generation(model) -> None:
    if getattr(model, '_vlmeval_accepts_mm_token_type_ids', False):
        return
    if not isinstance(model, torch.nn.Module):
        return
    if model.__class__.__module__.endswith('modeling_qwen3_vl_roi_prune'):
        model._vlmeval_accepts_mm_token_type_ids = True
        return
    module_name = str(getattr(model.__class__, '__module__', '') or '')
    class_name = str(getattr(model.__class__, '__name__', '') or '')
    if 'qwen3_vl' not in module_name.lower() and 'Qwen3VL' not in class_name:
        return

    orig_prepare = getattr(model, 'prepare_inputs_for_generation', None)
    orig_validate = getattr(model, '_validate_model_kwargs', None)
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
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if mm_token_type_ids is not None:
            model_inputs['mm_token_type_ids'] = mm_token_type_ids
        return model_inputs

    patched_prepare_inputs_for_generation.__signature__ = inspect.signature(patched_prepare_inputs_for_generation)
    model.prepare_inputs_for_generation = types.MethodType(patched_prepare_inputs_for_generation, model)

    if orig_validate is not None:
        def patched_validate_model_kwargs(self, model_kwargs):
            try:
                return orig_validate(model_kwargs)
            except ValueError as exc:
                if 'mm_token_type_ids' not in str(exc):
                    raise
                filtered = dict(model_kwargs)
                filtered.pop('mm_token_type_ids', None)
                return orig_validate(filtered)

        model._validate_model_kwargs = types.MethodType(patched_validate_model_kwargs, model)

    model._vlmeval_accepts_mm_token_type_ids = True


def _extract_seq_len(input_ids=None, inputs_embeds=None) -> int:
    if input_ids is not None:
        try:
            return int(input_ids.shape[1])
        except Exception:
            pass
    if inputs_embeds is not None:
        try:
            return int(inputs_embeds.shape[1])
        except Exception:
            pass
    return 0


def _extract_cache_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    try:
        return int(past_key_values.get_seq_length())
    except Exception:
        return 0


def _is_prefill_step_runtime(cache_position, past_key_values, input_ids, inputs_embeds) -> bool:
    try:
        if cache_position is not None:
            if torch.is_tensor(cache_position):
                if cache_position.numel() > 0:
                    return int(cache_position.flatten()[0].item()) == 0
            else:
                return int(cache_position) == 0
    except Exception:
        pass
    if past_key_values is None:
        return True
    try:
        if past_key_values.get_seq_length() == 0:
            return True
    except Exception:
        pass
    return _extract_seq_len(input_ids=input_ids, inputs_embeds=inputs_embeds) > 1


def _grid_token_count(grid_thw) -> int:
    if grid_thw is None:
        return 0
    try:
        data = grid_thw.detach().cpu().tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        if not data:
            return 0
        rows = data if isinstance(data[0], (list, tuple)) else [data]
        total = 0
        for row in rows:
            if row is None or len(row) < 3:
                continue
            total += int(row[0]) * int(row[1]) * int(row[2])
        return int(total)
    except Exception:
        return 0


def _estimate_visual_tokens(image_grid_thw=None, video_grid_thw=None) -> int:
    return int(_grid_token_count(image_grid_thw) + _grid_token_count(video_grid_thw))


def _estimate_llm_forward_flops(model, q_len: int, kv_len: int) -> float:
    cfg = getattr(model, 'config', None)
    text_cfg = getattr(cfg, 'text_config', cfg)
    if text_cfg is None:
        return 0.0
    hidden = int(getattr(text_cfg, 'hidden_size', 0) or 0)
    inter = int(getattr(text_cfg, 'intermediate_size', 0) or 0)
    layers = int(getattr(text_cfg, 'num_hidden_layers', 0) or 0)
    heads = int(getattr(text_cfg, 'num_attention_heads', 0) or 0)
    kv_heads = int(getattr(text_cfg, 'num_key_value_heads', heads) or heads)
    head_dim = int(getattr(text_cfg, 'head_dim', hidden // max(heads, 1)) or 0)
    if min(q_len, kv_len, hidden, inter, layers, heads, kv_heads, head_dim) <= 0:
        return 0.0
    q_proj_out = heads * head_dim
    kv_proj_out = kv_heads * head_dim
    attn_linear = 2.0 * q_len * hidden * (q_proj_out + kv_proj_out + kv_proj_out + q_proj_out)
    attn_kernel = 4.0 * heads * q_len * kv_len * head_dim
    mlp = 2.0 * q_len * hidden * inter + 2.0 * q_len * hidden * inter + 2.0 * q_len * inter * hidden
    return float(layers) * float(attn_linear + attn_kernel + mlp)


def _estimate_lm_head_flops(model, q_len: int) -> float:
    cfg = getattr(model, 'config', None)
    text_cfg = getattr(cfg, 'text_config', cfg)
    if text_cfg is None:
        return 0.0
    hidden = int(getattr(text_cfg, 'hidden_size', 0) or 0)
    vocab = int(getattr(text_cfg, 'vocab_size', 0) or 0)
    if min(q_len, hidden, vocab) <= 0:
        return 0.0
    return float(2.0 * q_len * hidden * vocab)


def _estimate_vision_forward_flops(model, visual_tokens: int) -> float:
    if visual_tokens <= 0:
        return 0.0
    cfg = getattr(model, 'config', None)
    vision_cfg = getattr(cfg, 'vision_config', None)
    if vision_cfg is None:
        return 0.0
    hidden = int(getattr(vision_cfg, 'hidden_size', 0) or 0)
    inter = int(getattr(vision_cfg, 'intermediate_size', 0) or 0)
    layers = int(
        getattr(vision_cfg, 'num_hidden_layers', None)
        or getattr(vision_cfg, 'depth', None)
        or 0
    )
    heads = int(getattr(vision_cfg, 'num_heads', 0) or 0)
    head_dim = hidden // max(heads, 1) if heads > 0 else 0
    if min(hidden, inter, layers, heads, head_dim) <= 0:
        return 0.0
    attn_linear = 8.0 * visual_tokens * hidden * hidden
    attn_kernel = 4.0 * heads * visual_tokens * visual_tokens * head_dim
    mlp = 4.0 * visual_tokens * hidden * inter
    return float(layers) * float(attn_linear + attn_kernel + mlp)


def _patch_qwen3vl_runtime_tracking(model) -> None:
    # Deprecated wrapper-based tracking path. Kept as a no-op because wrapping the
    # official HF Qwen3-VL forward/generate chain can perturb cache-sensitive decode.
    return


class _RuntimeTrackingHooks:
    def __init__(self, model, llm_module: torch.nn.Module | None, *, visual_tokens: int, prompt_seq_tokens: int, use_cuda_events: bool, sync_cuda: bool) -> None:
        self.model = model
        self.llm_module = llm_module
        self.visual_tokens = int(visual_tokens or 0)
        self.prompt_seq_tokens = int(prompt_seq_tokens or 0)
        self.use_cuda_events = bool(use_cuda_events) and torch.cuda.is_available()
        self.sync_cuda = bool(sync_cuda)
        self._handles = []
        self._stack = []
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.decode_steps = 0
        self.forward_steps = 0
        self.prefill_seen = False
        self.seq_tokens_before = int(prompt_seq_tokens or 0)
        self.seq_tokens_after = int(prompt_seq_tokens or 0)
        self.visual_tokens_before = int(visual_tokens or 0)
        self.visual_tokens_after = int(visual_tokens or 0)
        self.vision_flops = _estimate_vision_forward_flops(model, self.visual_tokens)
        self.llm_flops = 0.0
        self.lm_head_flops = 0.0
        self._pending_prefill_events = []
        self._pending_decode_events = []

    def _pre_hook(self, _module, args, kwargs):
        kwargs = kwargs or {}
        input_ids = kwargs.get('input_ids', None)
        past_key_values = kwargs.get('past_key_values', None)
        inputs_embeds = kwargs.get('inputs_embeds', None)
        cache_position = kwargs.get('cache_position', None)
        if len(args) >= 1 and input_ids is None:
            input_ids = args[0]
        if len(args) >= 4 and past_key_values is None:
            past_key_values = args[3]
        if len(args) >= 5 and inputs_embeds is None:
            inputs_embeds = args[4]
        q_len = _extract_seq_len(input_ids=input_ids, inputs_embeds=inputs_embeds)
        cache_len = _extract_cache_len(past_key_values)
        is_prefill = _is_prefill_step_runtime(
            cache_position=cache_position,
            past_key_values=past_key_values,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        if self.use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            self._stack.append((start, end, q_len, cache_len, is_prefill))
        else:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            self._stack.append((time.perf_counter(), None, q_len, cache_len, is_prefill))

    def _post_hook(self, _module, args, kwargs, output):
        if not self._stack:
            return output
        start_obj, end_obj, q_len, cache_len, is_prefill = self._stack.pop()
        if self.use_cuda_events:
            end_obj.record()
            elapsed = ('cuda_event', start_obj, end_obj)
        else:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = float(time.perf_counter() - start_obj)
        self.forward_steps += 1
        kv_len = max(int(q_len), int(cache_len + q_len))
        self.llm_flops += _estimate_llm_forward_flops(self.model, q_len=int(q_len), kv_len=int(kv_len))
        self.lm_head_flops += _estimate_lm_head_flops(self.model, q_len=max(1, int(q_len)))
        if is_prefill:
            self.prefill_seen = True
            if q_len > 0:
                self.seq_tokens_before = int(q_len)
                self.seq_tokens_after = int(q_len)
            self._accumulate_prefill_elapsed(elapsed)
        else:
            self.decode_steps += 1
            if q_len > 0:
                self.seq_tokens_after = int(max(self.seq_tokens_after, self.seq_tokens_before + self.decode_steps))
            self._accumulate_decode_elapsed(elapsed)
        return output

    def _accumulate_prefill_elapsed(self, elapsed):
        if isinstance(elapsed, tuple):
            self.prefill_s += 0.0
            self._pending_prefill_events = getattr(self, '_pending_prefill_events', [])
            self._pending_prefill_events.append((elapsed[1], elapsed[2]))
        else:
            self.prefill_s += float(elapsed)

    def _accumulate_decode_elapsed(self, elapsed):
        if isinstance(elapsed, tuple):
            self.decode_s += 0.0
            self._pending_decode_events = getattr(self, '_pending_decode_events', [])
            self._pending_decode_events.append((elapsed[1], elapsed[2]))
        else:
            self.decode_s += float(elapsed)

    def attach(self):
        if self.llm_module is None:
            return
        self._handles.append(self.llm_module.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        self._handles.append(self.llm_module.register_forward_hook(self._post_hook, with_kwargs=True))

    def finalize(self):
        try:
            if self.use_cuda_events:
                torch.cuda.synchronize()
                for start, end in self._pending_prefill_events:
                    self.prefill_s += float(start.elapsed_time(end)) / 1000.0
                for start, end in self._pending_decode_events:
                    self.decode_s += float(start.elapsed_time(end)) / 1000.0
        finally:
            for h in self._handles:
                try:
                    h.remove()
                except Exception:
                    pass
            self._handles = []

    def to_runtime_dict(self) -> dict:
        return {
            'prefill_s': float(self.prefill_s),
            'decode_s': float(self.decode_s),
            'decode_steps': int(self.decode_steps),
            'prefill_before_prune_layer_s': float(self.prefill_s),
            'prefill_split_to_prune_start_s': 0.0,
            'prune_layer_to_prefill_end_s': 0.0,
            'split_layer_to_prefill_end_without_prune_s': 0.0,
            'prune_selection_s': 0.0,
            'prune_op_s': 0.0,
            'prune_layer_to_finish_s': 0.0,
            'seq_tokens_before': int(self.seq_tokens_before),
            'seq_tokens_after': int(self.seq_tokens_after),
            'visual_tokens_before': int(self.visual_tokens_before),
            'visual_tokens_after': int(self.visual_tokens_after),
            'prompt_seq_tokens': int(self.seq_tokens_before),
            'decode_tokens': int(self.decode_steps),
            'timing_fallback': False,
            'timing_source': 'runtime_hooks',
        }

    def to_flops_dict(self) -> dict:
        e2e = float(self.vision_flops + self.llm_flops + self.lm_head_flops)
        return {
            'vision_flops': float(self.vision_flops),
            'llm_flops': float(self.llm_flops),
            'lm_head_flops': float(self.lm_head_flops),
            'e2e_flops': float(e2e),
            'forward_steps': int(self.forward_steps),
        }


def _extract_visual_grid_hw(inputs) -> tuple[int, int] | None:
    image_grid_thw = getattr(inputs, 'image_grid_thw', None)
    if image_grid_thw is None and isinstance(inputs, dict):
        image_grid_thw = inputs.get('image_grid_thw')
    if image_grid_thw is None:
        return None
    try:
        row = image_grid_thw[0].detach().cpu().tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw[0]
        if len(row) < 3:
            return None
        return int(row[-2]), int(row[-1])
    except Exception:
        return None


def _normalize_action_name(value) -> str:
    if value is None:
        return ''
    text = str(value).strip().lower()
    if not text:
        return ''
    if ':' in text:
        text = text.split(':', 1)[0]
    aliases = {
        'tap': 'click',
        'click': 'click',
        'long_press': 'long_press',
        'long press': 'long_press',
        'press': 'long_press',
        'scroll': 'scroll',
        'swipe': 'swipe',
        'type': 'type',
        'input_text': 'type',
    }
    for key, norm in aliases.items():
        if key in text:
            return norm
    return text


def _sample_allows_roi_prune(dataset: str | None, sample_meta: dict | None) -> bool:
    if _env_flag('QWEN3VL_ROI_PRUNE_ALLOW_NONCLICK', '0'):
        return True
    if not isinstance(sample_meta, dict):
        return True
    if isinstance(dataset, str) and dataset.startswith('AndroidControl'):
        return _normalize_action_name(sample_meta.get('gt_action')) in ('click', 'long_press')
    if isinstance(dataset, str) and dataset.startswith('GUIOdyssey'):
        answer = str(sample_meta.get('answer', '') or '')
        head = answer.split(':', 1)[0] if ':' in answer else answer
        return _normalize_action_name(head) in ('click', 'long_press')
    return True


def _configure_roi_prune_context(model, dataset: str | None, message: list[dict], sample_meta: dict | None, inputs) -> None:
    cfg = model.config.text_config
    enabled = _env_flag('QWEN3VL_ENABLE_ROI_PRUNE', '0')
    json_path = os.getenv('QWEN3VL_ROI_PRUNE_JSON', '').strip()

    message_image_paths = []
    message_sample_index = ''
    message_question = ''
    for item in message or []:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'image':
            value = item.get('value')
            if isinstance(value, str) and value:
                p = value[len('file://'):] if value.startswith('file://') else value
                if os.path.exists(p):
                    message_image_paths.append(p)
            if not message_sample_index and item.get('sample_index') is not None:
                message_sample_index = str(item.get('sample_index'))
            if not message_question and item.get('question') is not None:
                message_question = str(item.get('question'))
        elif item.get('type') == 'text' and not message_question:
            text_value = str(item.get('value', '') or item.get('text', '') or '').strip()
            if text_value:
                message_question = text_value

    cfg._vlmeval_current_dataset_name = str(dataset or '')
    cfg._vlmeval_current_sample_index = str((sample_meta or {}).get('sample_index', '') or message_sample_index)
    cfg._vlmeval_current_image_path = str((sample_meta or {}).get('image_path', '') or (message_image_paths[-1] if message_image_paths else ''))
    cfg._vlmeval_current_image_paths = list((sample_meta or {}).get('image_paths', []) or message_image_paths)
    cfg._vlmeval_current_question = str(
        (sample_meta or {}).get('step_instruction', '')
        or (sample_meta or {}).get('instruction', '')
        or (sample_meta or {}).get('question', '')
        or message_question
    )
    cfg._vlmeval_current_gt_meta = dict(sample_meta or {})
    img_wh = _read_current_image_size_from_message(message)
    cfg._vlmeval_current_image_size_wh = list(img_wh) if img_wh is not None else None
    visual_hw = _extract_visual_grid_hw(inputs)
    cfg._vlmeval_current_visual_grid_hw = list(visual_hw) if visual_hw is not None else None
    cfg._vlmeval_generate_timing_accum = {}
    cfg._vlmeval_generate_timing_last = {}
    cfg._vlmeval_generate_forward_index = 0
    cfg._roi_prune_last_stats = {}
    cfg._roi_prune_json_path = json_path or None
    cfg._roi_prune_debug = _env_flag('QWEN3VL_ROI_PRUNE_DEBUG', '0')
    cfg._print_layer_attn_tokens = _env_flag('QWEN3VL_ROI_PRUNE_PRINT_LAYER_ATTN_TOKENS', '0')

    prune_layer_order = os.getenv('QWEN3VL_ROI_PRUNE_LAYER_ORDER', '').strip()
    prune_layer_idx = os.getenv('QWEN3VL_ROI_PRUNE_LAYER_IDX', '').strip()
    if prune_layer_order:
        try:
            cfg._roi_prune_layer_idx = max(0, int(prune_layer_order) - 1)
        except Exception:
            cfg._roi_prune_layer_idx = 15
    elif prune_layer_idx:
        try:
            cfg._roi_prune_layer_idx = max(0, int(prune_layer_idx))
        except Exception:
            cfg._roi_prune_layer_idx = 15
    else:
        cfg._roi_prune_layer_idx = 15

    try:
        cfg._roi_prune_topk_keep = max(1, int(os.getenv('QWEN3VL_ROI_PRUNE_TOPK_KEEP', '4')))
    except Exception:
        cfg._roi_prune_topk_keep = 4
    try:
        cfg._roi_prune_uniform_keep_every = max(0, int(os.getenv('QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_EVERY', '0')))
    except Exception:
        cfg._roi_prune_uniform_keep_every = 0
    try:
        cfg._roi_prune_uniform_keep_offset = int(os.getenv('QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_OFFSET', '0'))
    except Exception:
        cfg._roi_prune_uniform_keep_offset = 0

    cfg._roi_prune_enabled = bool(enabled and json_path and _sample_allows_roi_prune(dataset, sample_meta))


def _record_attn_prune_stats(owner, model, sample_meta: dict | None) -> None:
    if not _use_qwen3vl_attn_prune_model(getattr(owner, 'use_attn_prune', False)):
        return
    cfg = model.config.text_config
    stats = dict(getattr(cfg, '_attn_prune_last_stats', {}) or {})
    if not stats:
        return
    stats['sample_index'] = str((sample_meta or {}).get('sample_index', '') or stats.get('sample_index', ''))
    stats['dataset_name'] = str((sample_meta or {}).get('dataset_name', '') or getattr(cfg, '_vlmeval_current_dataset_name', ''))
    if not hasattr(owner, '_vlmeval_prune_records'):
        owner._vlmeval_prune_records = []
    owner._vlmeval_prune_records.append(
        {
            'sample_index': stats.get('sample_index'),
            'dataset_name': stats.get('dataset_name'),
            'sort_s': float(stats.get('prune_selection_sec', 0.0) or 0.0),
            'recycle_s': float(stats.get('prune_op_sec', 0.0) or 0.0),
            'stats': stats,
        }
    )
    if _env_flag('QWEN3VL_ATTN_PRUNE_DEBUG', '0') or _env_flag('QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE', '0'):
        print(
            '[AttnPruneSample] '
            f"sample_index={stats.get('sample_index')} "
            f"dataset={stats.get('dataset_name')} "
            f"applied={int(bool(stats.get('prune_applied', False)))} "
            f"prune_layers={stats.get('prune_layers', stats.get('layers'))} "
            f"vis_layers={stats.get('vis_layers')} "
            f"keep_ratio={float(stats.get('keep_ratio', 1.0) or 1.0):.3f} "
            f"visual_before={stats.get('visual_tokens_before')} "
            f"visual_after={stats.get('visual_tokens_after')} "
            f"seq_before={stats.get('seq_tokens_before')} "
            f"seq_after={stats.get('seq_tokens_after')} "
            f"selection_s={float(stats.get('prune_selection_sec', 0.0) or 0.0):.6f} "
            f"prune_op_s={float(stats.get('prune_op_sec', 0.0) or 0.0):.6f}",
            flush=True,
        )


def _record_roi_prune_stats(owner, model, sample_meta: dict | None) -> None:
    cfg = model.config.text_config
    stats = dict(getattr(cfg, '_roi_prune_last_stats', {}) or {})
    if not stats:
        return
    stats['sample_index'] = str((sample_meta or {}).get('sample_index', ''))
    stats['dataset_name'] = str((sample_meta or {}).get('dataset_name', ''))
    if not hasattr(owner, '_vlmeval_prune_records'):
        owner._vlmeval_prune_records = []
    owner._vlmeval_prune_records.append(
        {
            'sample_index': stats.get('sample_index'),
            'dataset_name': stats.get('dataset_name'),
            'sort_s': float(stats.get('prune_selection_sec', 0.0) or 0.0),
            'recycle_s': float(stats.get('prune_op_sec', 0.0) or 0.0),
            'stats': stats,
        }
    )
    if _env_flag('QWEN3VL_ROI_PRUNE_DEBUG', '0'):
        print(
            '[ROIPruneSummary] '
            f"sample_index={stats.get('sample_index')} "
            f"applied={stats.get('prune_applied')} "
            f"align={stats.get('lookup_align_key')} "
            f"visual_before={stats.get('visual_tokens_before')} "
            f"visual_after={stats.get('visual_tokens_after')} "
            f"prune_op_s={float(stats.get('prune_op_sec', 0.0)):.6f}",
            flush=True,
        )


def _record_generate_timing(owner, model, sample_meta: dict | None, stage_record: dict | None) -> None:
    cfg = model.config.text_config
    runtime = dict(getattr(cfg, '_vlmeval_generate_timing_last', {}) or {})
    if not runtime and not stage_record:
        return

    merged = {}
    if isinstance(stage_record, dict):
        merged.update(stage_record)
    merged.update(runtime)
    flops_stats = dict(getattr(model, '_vlmeval_last_sample_flops', {}) or {})
    if flops_stats and any(
        float(flops_stats.get(key, 0.0) or 0.0) > 0.0
        for key in ('vision_flops', 'llm_flops', 'lm_head_flops', 'e2e_flops')
    ):
        merged.update(
            {
                'vision_flops': float(flops_stats.get('vision_flops', 0.0) or 0.0),
                'llm_flops': float(flops_stats.get('llm_flops', 0.0) or 0.0),
                'lm_head_flops': float(flops_stats.get('lm_head_flops', 0.0) or 0.0),
                'e2e_flops': float(flops_stats.get('e2e_flops', 0.0) or 0.0),
                'flops_forward_steps': int(flops_stats.get('forward_steps', 0) or 0),
            }
        )
    merged['sample_index'] = str((sample_meta or {}).get('sample_index', ''))
    merged['dataset_name'] = str((sample_meta or {}).get('dataset_name', ''))

    if not hasattr(owner, '_vlmeval_generate_timing_records'):
        owner._vlmeval_generate_timing_records = []
    owner._vlmeval_generate_timing_records.append(merged)

    if not (_env_flag('QWEN3VL_ROI_PRUNE_PRINT_PER_SAMPLE', '0') or _env_flag('QWEN3VL_PRINT_PER_SAMPLE', '0')):
        return

    prune_records = getattr(owner, '_vlmeval_prune_records', None) or []
    prune_stats = {}
    if prune_records and isinstance(prune_records[-1], dict):
        prune_stats = dict(prune_records[-1].get('stats', {}) or {})

    top4 = prune_stats.get('selected_top4_grids', [])
    seq_before = prune_stats.get('seq_tokens_before', merged.get('seq_tokens_before', None))
    seq_after = prune_stats.get('seq_tokens_after', merged.get('seq_tokens_after', None))
    visual_before = prune_stats.get('visual_tokens_before', merged.get('visual_tokens_before', None))
    visual_after = prune_stats.get('visual_tokens_after', merged.get('visual_tokens_after', None))
    prune_applied = bool(prune_stats.get('prune_applied', False))
    lookup_status = prune_stats.get('lookup_status', None)
    log_tag = '[ROIPruneSample]' if prune_applied or _env_flag('QWEN3VL_ENABLE_ROI_PRUNE', '0') else '[GenerateSample]'
    template_enabled = bool(merged.get('template_prefill_enabled', False))
    template_schema = merged.get('template_schema', None)
    template_impl = merged.get('template_prefill_impl', None)
    template_backend_impl = merged.get('template_prefill_backend_impl', None)
    template_requested_impl = merged.get('template_prefill_requested_impl', None)
    template_static_tokens = int(merged.get('template_static_token_count', 0) or 0)
    template_decode_tokens = int(merged.get('template_decode_tokens', 0) or 0)
    template_static_decode_steps = int(merged.get('template_static_decode_steps', 0) or 0)
    template_unknown_decode_steps = int(merged.get('template_unknown_decode_steps', 0) or 0)
    template_fallback_reason = merged.get('template_prefill_fallback_reason', None)
    print(
        f'{log_tag} '
        f"sample_index={merged.get('sample_index')} "
        f"dataset={merged.get('dataset_name')} "
        f"prune_applied={int(prune_applied)} "
        f"lookup_status={lookup_status} "
        f"top4_grids={top4} "
        f"prompt_seq_tokens={merged.get('prompt_seq_tokens', 0)} "
        f"decode_tokens={merged.get('decode_tokens', 0)} "
        f"decode_steps={merged.get('decode_steps', 0)} "
        f"encode_s={float(merged.get('vision_s', 0.0) or 0.0):.6f} "
        f"prefill_s={float(merged.get('prefill_s', 0.0) or 0.0):.6f} "
        f"decode_s={float(merged.get('decode_s', 0.0) or 0.0):.6f} "
        f"prefill_before_prune_layer_s={float(merged.get('prefill_before_prune_layer_s', 0.0) or 0.0):.6f} "
        f"prefill_split_to_prune_start_s={float(merged.get('prefill_split_to_prune_start_s', 0.0) or 0.0):.6f} "
        f"prune_layer_to_prefill_end_s={float(merged.get('prune_layer_to_prefill_end_s', 0.0) or 0.0):.6f} "
        f"split_layer_to_prefill_end_without_prune_s={float(merged.get('split_layer_to_prefill_end_without_prune_s', 0.0) or 0.0):.6f} "
        f"prune_selection_s={float(merged.get('prune_selection_s', 0.0) or 0.0):.6f} "
        f"prune_op_s={float(merged.get('prune_op_s', 0.0) or 0.0):.6f} "
        f"seq_tokens_before={seq_before} "
        f"seq_tokens_after={seq_after} "
        f"visual_tokens_before={visual_before} "
        f"visual_tokens_after={visual_after} "
        f"template_prefill_enabled={int(template_enabled)} "
        f"template_schema={template_schema} "
        f"template_prefill_impl={template_impl} "
        f"template_prefill_backend_impl={template_backend_impl} "
        f"template_prefill_requested_impl={template_requested_impl} "
        f"template_static_tokens={template_static_tokens} "
        f"template_decode_tokens={template_decode_tokens} "
        f"template_static_decode_steps={template_static_decode_steps} "
        f"template_unknown_decode_steps={template_unknown_decode_steps} "
        f"template_fallback_reason={template_fallback_reason} "
        f"vision_flops={float(merged.get('vision_flops', 0.0) or 0.0):.6e} "
        f"llm_flops={float(merged.get('llm_flops', 0.0) or 0.0):.6e} "
        f"e2e_flops={float(merged.get('e2e_flops', 0.0) or 0.0):.6e} "
        f"total_generate_s={float(merged.get('total_s', 0.0) or 0.0):.6f}",
        flush=True,
    )


def _populate_generate_timing_fallback(
    model,
    *,
    prompt_seq_tokens: int,
    decode_tokens: int,
    template_meta: dict | None,
    stage_record: dict | None,
) -> None:
    cfg = model.config.text_config
    runtime = dict(getattr(cfg, '_vlmeval_generate_timing_last', {}) or {})
    if runtime.get('prefill_s') is not None or runtime.get('decode_s') is not None:
        return

    stage_record = dict(stage_record or {})
    template_meta = dict(template_meta or {})
    total_s = float(stage_record.get('total_s', 0.0) or 0.0)
    vision_s = float(stage_record.get('vision_s', 0.0) or 0.0)
    llm_s = max(0.0, float(stage_record.get('llm_s', total_s - vision_s) or 0.0))
    decode_tokens = max(0, int(decode_tokens or 0))
    runtime.update(
        {
            'vision_s': vision_s,
            'prefill_s': llm_s,
            'decode_s': 0.0,
            'decode_steps': decode_tokens,
            'prefill_before_prune_layer_s': llm_s,
            'prefill_split_to_prune_start_s': 0.0,
            'prune_layer_to_prefill_end_s': 0.0,
            'split_layer_to_prefill_end_without_prune_s': 0.0,
            'prune_selection_s': 0.0,
            'prune_op_s': 0.0,
            'prune_layer_to_finish_s': 0.0,
            'total_s': total_s,
            'prompt_seq_tokens': int(prompt_seq_tokens or 0),
            'decode_tokens': decode_tokens,
            'timing_fallback': True,
            'timing_fallback_reason': 'non_timing_model_or_missing_forward_stats',
            'template_prefill_enabled': bool(template_meta.get('template_prefill_enabled', False)),
            'template_prefill_fallback_reason': template_meta.get('template_prefill_fallback_reason'),
            'template_static_token_count': int(template_meta.get('template_static_token_count', 0) or 0),
            'template_static_decode_steps': int(template_meta.get('template_static_decode_steps', 0) or 0),
            'template_unknown_decode_steps': int(template_meta.get('template_unknown_decode_steps', 0) or 0),
            'template_decode_tokens': int(template_meta.get('template_decode_tokens', 0) or 0),
            'template_schema': template_meta.get('template_schema'),
            'template_prefill_impl': template_meta.get('template_prefill_impl'),
            'template_prefill_backend_impl': template_meta.get('template_prefill_backend_impl'),
            'template_prefill_requested_impl': template_meta.get('template_prefill_requested_impl'),
        }
    )
    setattr(cfg, '_vlmeval_generate_timing_last', runtime)


def _roi_prune_generate_use_cache(model) -> bool:
    roi_prune_model_active = bool(_env_flag('QWEN3VL_ENABLE_ROI_PRUNE', '0'))
    try:
        roi_prune_model_active = roi_prune_model_active or ('roi_prune' in type(model.model).__module__)
    except Exception:
        pass
    if not roi_prune_model_active:
        return True
    # After mid-layer visual-token pruning, generation cache can retain pre-prune sequence
    # lengths across samples or decode steps. Keep it off unless explicitly re-enabled.
    return _env_flag('QWEN3VL_ROI_PRUNE_USE_CACHE', '0')


def _attn_prune_generate_use_cache(model) -> bool:
    attn_prune_model_active = bool(_env_flag('QWEN3VL_ENABLE_ATTN_PRUNE', '0') or _env_flag('QWEN3VL_USE_ATTN_PRUNE_MODEL', '0'))
    try:
        attn_prune_model_active = attn_prune_model_active or ('attn_prune' in type(model.model).__module__)
    except Exception:
        pass
    if not attn_prune_model_active:
        return True
    return _env_flag('QWEN3VL_ATTN_PRUNE_USE_CACHE', '0')


def _print_template_parts_sample(sample_meta: dict | None, template_meta: dict, template_response: str | None) -> None:
    if not _env_flag('QWEN3VL_TEMPLATE_PREFILL_DEBUG', '0'):
        return
    sample_index = str((sample_meta or {}).get('sample_index', ''))
    dataset_name = str((sample_meta or {}).get('dataset_name', ''))
    template_decode_tokens = int(template_meta.get('template_decode_tokens', 0) or 0)
    template_static_decode_steps = int(template_meta.get('template_static_decode_steps', 0) or 0)
    template_unknown_decode_steps = int(template_meta.get('template_unknown_decode_steps', 0) or 0)
    template_static_token_count = int(template_meta.get('template_static_token_count', 0) or 0)
    template_enabled = bool(template_meta.get('template_prefill_enabled', False))
    template_schema = template_meta.get('template_schema', None)
    template_impl = template_meta.get('template_prefill_impl', None)
    template_backend_impl = template_meta.get('template_prefill_backend_impl', None)
    template_requested_impl = template_meta.get('template_prefill_requested_impl', None)
    fallback_reason = template_meta.get('template_prefill_fallback_reason', None)
    static_parts = template_meta.get('template_static_parts', None) or []
    template_plan = template_meta.get('template_plan', None)
    print(
        '[TemplatePartsSample] '
        f'sample_index={sample_index} '
        f'dataset={dataset_name} '
        f'template_enabled={int(template_enabled)} '
        f'template_schema={template_schema} '
        f'template_impl={template_impl} '
        f'template_backend_impl={template_backend_impl} '
        f'template_requested_impl={template_requested_impl} '
        f'template_decode_steps={template_decode_tokens} '
        f'template_static_decode_steps={template_static_decode_steps} '
        f'template_unknown_decode_steps={template_unknown_decode_steps} '
        f'template_static_token_count={template_static_token_count} '
        f'template_fallback_reason={fallback_reason}',
        flush=True,
    )
    print(f'[TemplatePartsSample] sample_index={sample_index} template_output={template_response!r}', flush=True)
    print(f'[TemplatePartsSample] sample_index={sample_index} template_plan={template_plan!r}', flush=True)
    print(f'[TemplatePartsSample] sample_index={sample_index} static_parts={static_parts!r}', flush=True)
    slot_stats = template_meta.get('template_slot_stats', None)
    if slot_stats:
        for idx, slot in enumerate(slot_stats):
            print(
                '[TemplatePartsSlot] '
                f'sample_index={sample_index} '
                f'slot_index={idx} '
                f"slot={slot.get('slot')} "
                f"decode_steps={int(slot.get('decode_tokens', 0) or 0)} "
                f"prompt_tokens={int(slot.get('prompt_tokens', 0) or 0)} "
                f"done={int(bool(slot.get('done', False)))} "
                f"fallback={int(bool(slot.get('fallback', False)))} "
                f"reason={slot.get('reason')} "
                f"raw_text={slot.get('raw_text')!r} "
                f"rendered_text={slot.get('rendered_text')!r}",
                flush=True,
            )

class Qwen3VLChat(Qwen3VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        use_audio_in_video: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        if self.total_pixels and self.total_pixels > 24576 * 32 * 32:
            print('The total number of video tokens might too large, resulting in an overly long input sequence.')
        self.generate_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.use_attn_prune = bool(kwargs.pop('use_attn_prune', False) or kwargs.pop('attn_prune', False))
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.use_audio_in_video = use_audio_in_video

        assert model_path is not None
        self.model_path = model_path
        from transformers import AutoProcessor, AutoModelForImageTextToText
        # Use official Qwen3-Omni classes when model_path indicates omni
        if listinstr(['omni'], model_path.lower()):
            try:
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
            except Exception as err:
                logging.critical("pip install git+https://github.com/huggingface/transformers")
                raise err
            self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            self.processor = AutoProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        self.use_vllm = kwargs.get('use_vllm', False)
        self.use_lmdeploy = kwargs.get('use_lmdeploy', False)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        assert self.use_vllm + self.use_lmdeploy <= 1, "You can only set one flag `use_vllm` to True"
        if self.use_vllm:
            if listinstr(['omni'], self.model_path.lower()):
                os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM
            gpu_count = torch.cuda.device_count()
            tp_size = gpu_count if gpu_count > 0 else 1
            logging.info(
                f'Using vLLM for {self.model_path} inference with {tp_size} GPUs (available: {gpu_count})'
            )
            if os.environ.get('VLLM_WORKER_MULTIPROC_METHOD') != 'spawn':
                logging.warning(
                    "VLLM_WORKER_MULTIPROC_METHOD is not set to spawn. Use 'export VLLM_WORKER_MULTIPROC_METHOD=spawn'"
                )
            enable_expert_parallel = is_moe_model(self.model_path)
            # For Qwen3-Omni, vLLM engine v1 is not supported yet
            if listinstr(['omni'], self.model_path.lower()):
                limit_mm = {"image": 3, "video": 3, "audio": 3}
            else:
                limit_mm = {"image": self.limit_mm_per_prompt}
            self.llm = LLM(
                model=self.model_path,
                max_num_seqs=8,
                limit_mm_per_prompt=limit_mm,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
                seed=0,
                gpu_memory_utilization=kwargs.get("gpu_utils", 0.9),
                trust_remote_code=True,
            )
        else:
            if listinstr(['omni'], model_path.lower()):
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    model_path, dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                if _use_qwen3vl_attn_prune_model(self.use_attn_prune):
                    from .modeling_qwen3_vl_attn_prune import Qwen3VLForConditionalGeneration

                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype='auto', device_map='auto', attn_implementation='sdpa'
                    )
                elif _use_qwen3vl_timing_model():
                    from .modeling_qwen3_vl_roi_prune import Qwen3VLForConditionalGeneration

                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype='auto', device_map='auto', attn_implementation='sdpa'
                    )
                else:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_path, torch_dtype='auto', device_map='auto', attn_implementation='sdpa'#'flash_attention_2'
                    )
                    _patch_upstream_qwen3vl_prepare_inputs_for_generation(self.model)
            self.model.eval()
            _patch_qwen3vl_runtime_tracking(self.model)
            if _env_flag('QWEN3VL_ENABLE_ROI_PRUNE', '0') and not _env_flag('QWEN3VL_ROI_PRUNE_USE_CACHE', '0'):
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
                try:
                    self.model.generation_config.use_cache = False
                except Exception:
                    pass
            if _use_qwen3vl_attn_prune_model(self.use_attn_prune) and not _env_flag('QWEN3VL_ATTN_PRUNE_USE_CACHE', '0'):
                try:
                    self.model.config.use_cache = False
                except Exception:
                    pass
                try:
                    self.model.generation_config.use_cache = False
                except Exception:
                    pass
            if _env_flag('QWEN3VL_PROFILE_FLOPS', '0'):
                enable_qwen3vl_flops_profiling(self.model)

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner_transformers(self, message, dataset=None, **kwargs):
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("Please install it via 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        #====================加入图片大小缩放===================================
        bbox = kwargs.get("bbox", None)
        img_size = kwargs.get("img_size", None)
        if bbox is None or img_size is None:
            for s in message:
                if isinstance(s, dict):
                    if bbox is None and "bbox" in s:
                        bbox = s.get("bbox")
                    if img_size is None and "img_size" in s:
                        img_size = s.get("img_size")
        self.model.config.text_config._vlmeval_current_bbox = bbox
        self.model.config.text_config._vlmeval_current_img_size = img_size
        # print("before scale, img_size: ", self.model.config.text_config._vlmeval_current_img_size)
        #================加入图片大小缩放结束===================================

        if is_omni:
            # For Qwen3-Omni, messages is a list of dicts
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors='pt',
                padding=True,
                use_audio_in_video=self.use_audio_in_video,
            )
        else:
            # print("self.processor.image_processor.size['shortest_edge']", self.processor.image_processor.size['shortest_edge'])
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            for img in images:
                if hasattr(img, "size"):
                    w, h = img.size
                    # print(f"original image size: {(w, h)}")
                    break

            #====================加入图片大小缩放===================================
            input_scale = os.environ.get("VLMPRUNE_INPUT_SCALE", "1")
            # print("input_scale: ", input_scale)
            try:
                input_scale = float(input_scale)
            except Exception:
                input_scale = 1.0
            if input_scale <= 0:
                input_scale = 1.0
            if images is not None and input_scale != 1.0:
                print("in scale")
                patch_size = 16
                merge_size = 1
                if hasattr(self, "model") and hasattr(self.model, "config"):
                    merge_size = getattr(self.model.config.vision_config, "spatial_merge_size", 1) or 1
                align_base = max(1, int(patch_size * merge_size))
                first_scale = None
                scaled_images = []
                for img in images:
                    if hasattr(img, "size"):
                        w, h = img.size
                        new_w = max(1, int(round(w * input_scale)))
                        new_h = max(1, int(round(h * input_scale)))
                        new_w = max(align_base, (new_w // align_base) * align_base)
                        new_h = max(align_base, (new_h // align_base) * align_base)
                        if first_scale is None:
                            first_scale = (new_w / w, new_h / h)
                        scaled_images.append(img.resize((new_w, new_h), Image.BILINEAR))
                        # print("after scale, img_size: ", (new_w, new_h))
                    else:
                        scaled_images.append(img)
                images = scaled_images
                if hasattr(self, "model") and hasattr(self.model, "config"):
                    # print("in 1")
                    cfg = self.model.config.text_config
                    img_size = getattr(cfg, "_vlmeval_current_img_size", None)
                    bbox = getattr(cfg, "_vlmeval_current_bbox", None)
                    if first_scale is None:
                        first_scale = (input_scale, input_scale)
                    scale_x, scale_y = first_scale
                    # print("image_size type:",type(img_size))
                    if isinstance(img_size, str):
                        try:
                            img_size = ast.literal_eval(img_size)
                        except Exception:
                            img_size = None
                    # print("img_size: ", img_size)

                    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                        # print("in 2")
                        scaled_size = [img_size[0] * scale_x, img_size[1] * scale_y]
                        cfg._vlmeval_current_img_size = scaled_size
                        # print("after scale, img_size: ", cfg._vlmeval_current_img_size)
                        if isinstance(bbox, str):
                            try:
                                bbox = ast.literal_eval(bbox)
                            except Exception:
                                bbox = None
                        # print("bbox: ", bbox)
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            # print("in 3")
                            max_val = max(bbox)
                            if max_val > 1.5 and max_val <= max(img_size) + 1:
                                cfg._vlmeval_current_bbox = [
                                    bbox[0] * scale_x,
                                    bbox[1] * scale_y,
                                    bbox[2] * scale_x,
                                    bbox[3] * scale_y,
                                ]
            #====================加入图片大小缩放===================================

            video_metadatas = None
            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)

            inputs = self.processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                do_resize=False,
                return_tensors='pt',
                **(video_kwargs or {}),
            )
            try:
                if hasattr(self, 'model') and hasattr(self.model, 'config'):
                    cfg = self.model.config.text_config
                    if images is not None and len(images) > 0:
                        cfg._vlmeval_current_vis_image_pil = images[-1]
                    if hasattr(inputs, 'get'):
                        cfg._vlmeval_current_image_grid_thw = inputs.get("image_grid_thw", None)
                    else:
                        cfg._vlmeval_current_image_grid_thw = getattr(inputs, "image_grid_thw", None)
            except Exception:
                pass
        sample_meta = kwargs.get('sample_meta') if isinstance(kwargs, dict) else None
        if not is_omni and hasattr(self, 'model'):
            _configure_roi_prune_context(self.model, dataset, message, sample_meta, inputs)
        try:
            inputs = inputs.to(self.model.device)
            if hasattr(self.model, 'dtype'):
                inputs = inputs.to(self.model.dtype)
        except Exception:
            inputs = inputs.to('cuda')
        inputs = _sanitize_generate_inputs_for_model(self.model, inputs)

        if is_omni:
            try:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=self.use_audio_in_video,
                )
            except TypeError:
                text_ids, _ = self.model.generate(
                    **inputs,
                    return_audio=False,
                    use_audio_in_video=self.use_audio_in_video,
                )
            response = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        else:
            #=======================timer================================
            stage_timing = os.getenv('VLM_STAGE_TIMING', '0') == '1'
            stage_device = os.getenv('VLM_STAGE_TIMING_DEVICE', 'auto')
            use_cuda_events = (stage_device in ['auto', 'cuda']) and torch.cuda.is_available()
            sync_cuda = os.getenv('VLM_STAGE_TIMING_SYNC', '0') == '1'
            timer = None
            total_start = None
            vision_name = 'None'
            llm_name = 'None'
            if stage_timing:
                timer = _StageTimer(use_cuda_events=use_cuda_events, sync_cuda=sync_cuda)
                vision_m, llm_m, vision_name, llm_name = _pick_vision_and_llm_modules(self.model)
                timer.add_module('vision', vision_m)
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_start = time.perf_counter()
            #=======================timer================================
            stage_record = None
            generate_kwargs = dict(self.generate_kwargs)
            if not _roi_prune_generate_use_cache(self.model) or not _attn_prune_generate_use_cache(self.model):
                generate_kwargs['use_cache'] = False
            template_meta = {
                'template_prefill_enabled': False,
                'template_prefill_fallback_reason': 'not_attempted',
            }
            response = None
            prompt_seq_tokens = 0
            decode_tokens = 0
            runtime_tracking_enabled = (
                _env_flag('QWEN3VL_RUNTIME_TRACKING', '0')
                and not _use_qwen3vl_timing_model()
            )
            runtime_tracker = None
            if runtime_tracking_enabled:
                try:
                    prompt_tensor = getattr(inputs, 'input_ids', None)
                    if prompt_tensor is None and isinstance(inputs, dict):
                        prompt_tensor = inputs.get('input_ids', None)
                    if prompt_tensor is not None:
                        try:
                            prompt_seq_tokens = int(prompt_tensor.shape[1])
                        except Exception:
                            prompt_seq_tokens = int(prompt_seq_tokens or 0)
                    visual_tokens = _estimate_visual_tokens(
                        image_grid_thw=getattr(inputs, 'image_grid_thw', None) if not isinstance(inputs, dict) else inputs.get('image_grid_thw'),
                        video_grid_thw=getattr(inputs, 'video_grid_thw', None) if not isinstance(inputs, dict) else inputs.get('video_grid_thw'),
                    )
                    _, llm_m, _, _ = _pick_vision_and_llm_modules(self.model)
                    runtime_tracker = _RuntimeTrackingHooks(
                        self.model,
                        llm_m,
                        visual_tokens=visual_tokens,
                        prompt_seq_tokens=prompt_seq_tokens,
                        use_cuda_events=use_cuda_events,
                        sync_cuda=sync_cuda,
                    )
                    runtime_tracker.attach()
                except Exception as exc:
                    runtime_tracker = None
                    print(f'[RuntimeTracking] disabled_due_to_setup_error={exc}', flush=True)
            template_response, template_meta = maybe_generate_with_structured_fast_decode(
                dataset=dataset,
                model=self.model,
                processor=self.processor,
                inputs=inputs,
                generate_kwargs=generate_kwargs,
                sample_meta=sample_meta,
            )
            try:
                if template_response is not None:
                    response = template_response
                    try:
                        if hasattr(inputs, 'input_ids'):
                            prompt_lengths = [int(x.shape[0]) for x in inputs.input_ids]
                        else:
                            prompt_tensor = inputs.get('input_ids', None)
                            prompt_lengths = [int(x.shape[0]) for x in prompt_tensor] if prompt_tensor is not None else []
                        prompt_seq_tokens = int(prompt_lengths[0]) if prompt_lengths else 0
                        decode_tokens = int(template_meta.get('template_decode_tokens', 0) or 0)
                    except Exception:
                        prompt_seq_tokens = 0
                        decode_tokens = int(template_meta.get('template_decode_tokens', 0) or 0)
                else:
                    generated_ids = self.model.generate(
                        **inputs,
                        **generate_kwargs,
                    )
                    try:
                        if hasattr(inputs, 'input_ids'):
                            prompt_lengths = [int(x.shape[0]) for x in inputs.input_ids]
                        else:
                            prompt_tensor = inputs.get('input_ids', None)
                            prompt_lengths = [int(x.shape[0]) for x in prompt_tensor] if prompt_tensor is not None else []
                        prompt_seq_tokens = int(prompt_lengths[0]) if prompt_lengths else 0
                        if isinstance(generated_ids, torch.Tensor):
                            full_lengths = [int(x.shape[0]) for x in generated_ids]
                        else:
                            full_lengths = [int(x.shape[0]) for x in list(generated_ids)]
                        if prompt_lengths and full_lengths:
                            decode_tokens = max(0, int(full_lengths[0] - prompt_lengths[0]))
                    except Exception:
                        prompt_seq_tokens = 0
                        decode_tokens = 0
                    generated_ids_trimmed = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    out = self.processor.tokenizer.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    response = out[0]
            finally:
                if runtime_tracker is not None:
                    try:
                        runtime_tracker.seq_tokens_before = int(prompt_seq_tokens or runtime_tracker.seq_tokens_before)
                        runtime_tracker.seq_tokens_after = int(
                            max(runtime_tracker.seq_tokens_after, (prompt_seq_tokens or 0) + (decode_tokens or 0))
                        )
                        runtime_tracker.decode_steps = int(max(runtime_tracker.decode_steps, decode_tokens or 0))
                        runtime_tracker.finalize()
                        runtime_dict = runtime_tracker.to_runtime_dict()
                        runtime_dict['prompt_seq_tokens'] = int(prompt_seq_tokens or runtime_dict.get('prompt_seq_tokens', 0) or 0)
                        runtime_dict['decode_tokens'] = int(decode_tokens or runtime_dict.get('decode_tokens', 0) or 0)
                        runtime_dict.update(dict(template_meta or {}))
                        cfg = self.model.config.text_config
                        setattr(cfg, '_vlmeval_generate_timing_last', runtime_dict)
                        self.model._vlmeval_last_sample_flops = runtime_tracker.to_flops_dict()
                        if _env_flag('QWEN3VL_RUNTIME_TRACKING_DEBUG', '0'):
                            print(
                                '[RuntimeTracking] '
                                f'prefill_s={runtime_dict.get("prefill_s", 0.0):.6f} '
                                f'decode_s={runtime_dict.get("decode_s", 0.0):.6f} '
                                f'decode_steps={runtime_dict.get("decode_steps", 0)} '
                                f'seq_tokens_before={runtime_dict.get("seq_tokens_before")} '
                                f'seq_tokens_after={runtime_dict.get("seq_tokens_after")} '
                                f'visual_tokens={runtime_dict.get("visual_tokens_before")} '
                                f'llm_flops={self.model._vlmeval_last_sample_flops.get("llm_flops", 0.0):.0f}',
                                flush=True,
                            )
                    except Exception as exc:
                        print(f'[RuntimeTracking] finalize_error={exc}', flush=True)
            #=======================timer================================
            if stage_timing and timer is not None and total_start is not None:
                if sync_cuda and torch.cuda.is_available() and not use_cuda_events:
                    torch.cuda.synchronize()
                total_s = time.perf_counter() - total_start
                timer.finalize()
                vision_s = float(timer.seconds.get('vision', 0.0))
                llm_s = max(0.0, total_s - vision_s)
                other_s = max(0.0, total_s - vision_s - llm_s)
                sample_index = None
                try:
                    sample_index = getattr(self.model.config.text_config, "_vlmeval_current_sample_index", None)
                except Exception:
                    sample_index = None
                prefix = f'[StageTiming] index={sample_index} ' if sample_index is not None else '[StageTiming] '
                print(
                    f'{prefix}total_s={total_s:.6f} vision_s={vision_s:.6f} llm_s={llm_s:.6f} other_s={other_s:.6f} '
                    f'vision_mod={vision_name} llm_mod={llm_name}',
                    flush=True,
                )
                stage_record = dict(
                    sample_index=sample_index,
                    total_s=total_s,
                    vision_s=vision_s,
                    llm_s=llm_s,
                    other_s=other_s,
                    prompt_seq_tokens=prompt_seq_tokens,
                    decode_tokens=decode_tokens,
                    vision_mod=vision_name,
                    llm_mod=llm_name,
                )
                stage_record.update(template_meta)
                if not hasattr(self, '_vlmeval_stage_records'):
                    self._vlmeval_stage_records = []
                self._vlmeval_stage_records.append(stage_record)
                timer.close()
            #=======================timer================================
            _print_template_parts_sample(
                sample_meta=sample_meta,
                template_meta=template_meta,
                template_response=template_response,
            )
            _populate_generate_timing_fallback(
                self.model,
                prompt_seq_tokens=prompt_seq_tokens,
                decode_tokens=decode_tokens,
                template_meta=template_meta,
                stage_record=stage_record if 'stage_record' in locals() else None,
            )
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        if not is_omni and hasattr(self, 'model'):
            _record_roi_prune_stats(self, self.model, sample_meta)
            _record_attn_prune_stats(self, self.model, sample_meta)
            _record_generate_timing(self, self.model, sample_meta, stage_record if 'stage_record' in locals() else None)
        return response

    def generate_inner_vllm(self, message, dataset=None):
        from vllm import SamplingParams
        is_omni = listinstr(['omni'], self.model_path.lower())
        if is_omni:
            try:
                from qwen_omni_utils import process_mm_info
            except Exception as err:
                logging.critical("qwen_omni_utils not found, 'pip install qwen-omni-utils[decord]'")
                raise err
        else:
            try:
                from qwen_vl_utils import process_vision_info
            except Exception as err:
                logging.critical("qwen_vl_utils not found, 'pip install qwen-vl-utils'")
                raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if is_omni:
            audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=self.use_audio_in_video)
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            stop_token_ids=None
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        if is_omni and 'audios' in locals() and audios is not None:
            mm_data['audio'] = audios

        req = {'prompt': text}
        if mm_data:
            req['multi_modal_data'] = mm_data
        if is_omni:
            req['mm_processor_kwargs'] = {"use_audio_in_video": self.use_audio_in_video}
        elif video_kwargs is not None:
            req['mm_processor_kwargs'] = video_kwargs

        outputs = self.llm.generate([req], sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text

        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]

        if self.verbose:
            print(f'\033[32m{generated_text}\033[0m')
        return generated_text

    def generate_inner(self, message, dataset=None, **kwargs):
        enable_android_denorm = os.getenv('QWEN3VL_ANDROID_DENORM_ON_INFER', '0') == '1'
        denorm_base = os.getenv('QWEN3VL_ANDROID_DENORM_BASE', '1000')
        try:
            denorm_base = float(denorm_base)
        except Exception:
            denorm_base = 1000.0
        if denorm_base <= 0:
            denorm_base = 1000.0

        if self.use_vllm:
            response = self.generate_inner_vllm(message, dataset=dataset)
        else:
            response = self.generate_inner_transformers(message, dataset=dataset, **kwargs)

        if isinstance(dataset, str) and dataset.startswith('AndroidControl'):
            response = _postprocess_androidcontrol_response(
                response,
                message,
                base=denorm_base,
                denorm=enable_android_denorm,
            )
        return response
