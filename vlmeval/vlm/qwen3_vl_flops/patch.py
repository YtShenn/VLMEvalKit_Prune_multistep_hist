from __future__ import annotations

import os
import types
from typing import Any, Callable, Optional

import torch

try:
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as torch_profile
    from torch.profiler import record_function as torch_record_function

    _TORCH_PROFILER_AVAILABLE = True
except Exception:
    ProfilerActivity = None
    torch_profile = None
    torch_record_function = None
    _TORCH_PROFILER_AVAILABLE = False


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _flops_profile_enabled() -> bool:
    return _env_flag("QWEN3VL_PROFILE_FLOPS", "0")


def _flops_safe_mode_enabled() -> bool:
    return _env_flag("QWEN3VL_PROFILE_FLOPS_SAFE", "1")


def _cuda_ready() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _format_flops(flops: float) -> str:
    abs_flops = abs(flops)
    if abs_flops >= 1e12:
        return f"{flops / 1e12:.3f} TFLOPs"
    if abs_flops >= 1e9:
        return f"{flops / 1e9:.3f} GFLOPs"
    if abs_flops >= 1e6:
        return f"{flops / 1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _maybe_profile_flops(stage: str, fn: Callable[[], Any]) -> tuple[Any, Optional[float]]:
    if not _flops_profile_enabled():
        return fn(), None
    if _flops_safe_mode_enabled() and stage in {"llm_decoder", "lm_head"}:
        return fn(), None
    if not _TORCH_PROFILER_AVAILABLE:
        _log("[Efficiency INFO] FLOPs profiling requested but torch.profiler is unavailable.")
        return fn(), None

    activities = [ProfilerActivity.CPU]
    if _cuda_ready():
        activities.append(ProfilerActivity.CUDA)

    try:
        if _cuda_ready():
            torch.cuda.synchronize()
        with torch_profile(activities=activities, with_flops=True, profile_memory=False, record_shapes=False) as prof:
            with torch_record_function(f"qwen3vl.{stage}"):
                out = fn()
        if _cuda_ready():
            torch.cuda.synchronize()

        total_flops = 0.0
        for item in prof.key_averages():
            if item.flops is not None:
                total_flops += float(item.flops)
        return out, total_flops
    except Exception as err:
        _log(f"[Efficiency INFO] FLOPs profiling failed at stage `{stage}`: {err}")
        return fn(), None


def _to_int(value) -> Optional[int]:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.flatten()[0].item())
    try:
        return int(value)
    except Exception:
        return None


def _is_prefill_step(
    cache_position,
    past_key_values,
    input_ids,
    inputs_embeds,
) -> bool:
    cache_start = _to_int(cache_position)
    if cache_start is not None:
        return cache_start == 0
    if past_key_values is None:
        return True
    try:
        if past_key_values.get_seq_length() == 0:
            return True
    except Exception:
        pass
    if input_ids is not None:
        return input_ids.shape[1] > 1
    if inputs_embeds is not None:
        return inputs_embeds.shape[1] > 1
    return False


def _reset_sample_flops(model) -> None:
    model._sample_flops_steps = 0
    model._sample_llm_flops_sum = 0.0
    model._sample_e2e_flops_sum = 0.0
    model._sample_vision_flops_sum = 0.0
    model._sample_lm_head_flops_sum = 0.0


def _flush_sample_flops(model) -> None:
    if getattr(model, "_sample_flops_steps", 0) <= 0:
        return
    avg_vision = model._sample_vision_flops_sum / model._sample_flops_steps
    avg_llm = model._sample_llm_flops_sum / model._sample_flops_steps
    avg_e2e = model._sample_e2e_flops_sum / model._sample_flops_steps
    _log(
        "[Efficiency INFO] FLOPs [Vision][Sample Avg] "
        f"avg={_format_flops(avg_vision)}, total={_format_flops(model._sample_vision_flops_sum)}, "
        f"steps={model._sample_flops_steps}"
    )
    _log(
        "[Efficiency INFO] FLOPs [LLM][Sample Avg] "
        f"avg={_format_flops(avg_llm)}, total={_format_flops(model._sample_llm_flops_sum)}, "
        f"steps={model._sample_flops_steps}"
    )
    _log(
        "[Efficiency INFO] FLOPs [E2E][Sample Avg] "
        f"avg={_format_flops(avg_e2e)}, total={_format_flops(model._sample_e2e_flops_sum)}, "
        f"steps={model._sample_flops_steps} "
        f"(vision_total={_format_flops(model._sample_vision_flops_sum)}, "
        f"llm_total={_format_flops(model._sample_llm_flops_sum)}, "
        f"lm_head_total={_format_flops(model._sample_lm_head_flops_sum)})"
    )


def _finalize_sample_flops(model) -> None:
    if not getattr(model, "_sample_flops_active", False):
        return
    model._vlmeval_last_sample_flops = {
        "vision_flops": float(getattr(model, "_sample_vision_flops_sum", 0.0) or 0.0),
        "llm_flops": float(getattr(model, "_sample_llm_flops_sum", 0.0) or 0.0),
        "lm_head_flops": float(getattr(model, "_sample_lm_head_flops_sum", 0.0) or 0.0),
        "e2e_flops": float(getattr(model, "_sample_e2e_flops_sum", 0.0) or 0.0),
        "forward_steps": int(getattr(model, "_sample_flops_steps", 0) or 0),
    }
    _flush_sample_flops(model)
    _reset_sample_flops(model)
    model._sample_flops_active = False


def enable_qwen3vl_flops_profiling(model) -> bool:
    """Enable per-sample vision/LLM/E2E FLOPs profiling for Qwen3VL models.

    The patch is no-op unless `QWEN3VL_PROFILE_FLOPS=1`.
    """
    if model is None:
        return False
    if not _flops_profile_enabled():
        return False
    if getattr(model, "_vlmeval_flops_patched", False):
        return False
    model._sample_flops_active = False
    model._sample_flops_steps = 0
    model._sample_llm_flops_sum = 0.0
    model._sample_e2e_flops_sum = 0.0
    model._sample_vision_flops_sum = 0.0
    model._sample_lm_head_flops_sum = 0.0
    model._last_lm_head_flops = None
    model._vlmeval_last_sample_flops = {}
    if _flops_safe_mode_enabled():
        _log(
            "[Efficiency INFO] Qwen3VL FLOPs safe mode enabled: disable runtime FLOPs wrappers on the "
            "generate path to avoid cache-sensitive attention shape mismatches."
        )
        model._vlmeval_flops_patched = True
        return True

    core = getattr(model, "model", None)
    if core is None:
        model._vlmeval_flops_patched = True
        return True

    core._last_forward_flops = {"vision": None, "llm": None}

    if hasattr(core, "get_image_features"):
        orig_get_image_features = core.get_image_features

        def patched_get_image_features(self, *args, **kwargs):
            out, flops = _maybe_profile_flops("vision_image_encoder", lambda: orig_get_image_features(*args, **kwargs))
            if flops is not None:
                self._vlmeval_forward_vision_flops = self._vlmeval_forward_vision_flops + float(flops)
            return out

        core.get_image_features = types.MethodType(patched_get_image_features, core)

    if hasattr(core, "get_video_features"):
        orig_get_video_features = core.get_video_features

        def patched_get_video_features(self, *args, **kwargs):
            out, flops = _maybe_profile_flops("vision_video_encoder", lambda: orig_get_video_features(*args, **kwargs))
            if flops is not None:
                self._vlmeval_forward_vision_flops = self._vlmeval_forward_vision_flops + float(flops)
            return out

        core.get_video_features = types.MethodType(patched_get_video_features, core)

    if not _flops_safe_mode_enabled():
        if hasattr(core, "language_model") and hasattr(core.language_model, "forward"):
            orig_lm_forward = core.language_model.forward
            lm_module = core.language_model

            def patched_lm_forward(self, *args, **kwargs):
                out, flops = _maybe_profile_flops("llm_decoder", lambda: orig_lm_forward(*args, **kwargs))
                core._vlmeval_last_llm_flops = float(flops) if flops is not None else None
                return out

            lm_module.forward = types.MethodType(patched_lm_forward, lm_module)

        if hasattr(model, "lm_head") and hasattr(model.lm_head, "forward"):
            orig_lm_head_forward = model.lm_head.forward

            def patched_lm_head_forward(self, *args, **kwargs):
                out, flops = _maybe_profile_flops("lm_head", lambda: orig_lm_head_forward(*args, **kwargs))
                model._last_lm_head_flops = float(flops) if flops is not None else None
                return out

            model.lm_head.forward = types.MethodType(patched_lm_head_forward, model.lm_head)

    if hasattr(core, "forward"):
        orig_core_forward = core.forward

        def patched_core_forward(self, *args, **kwargs):
            self._vlmeval_forward_vision_flops = 0.0
            self._vlmeval_last_llm_flops = None
            out = orig_core_forward(*args, **kwargs)
            vision_val = self._vlmeval_forward_vision_flops
            llm_val = self._vlmeval_last_llm_flops
            self._last_forward_flops = {
                "vision": float(vision_val) if vision_val > 0 else None,
                "llm": float(llm_val) if llm_val is not None else None,
            }
            return out

        core.forward = types.MethodType(patched_core_forward, core)

    if hasattr(model, "forward"):
        orig_model_forward = model.forward

        def patched_model_forward(self, *args, **kwargs):
            input_ids = kwargs.get("input_ids", None)
            past_key_values = kwargs.get("past_key_values", None)
            inputs_embeds = kwargs.get("inputs_embeds", None)
            cache_position = kwargs.get("cache_position", None)

            if len(args) >= 1:
                input_ids = args[0]
            if len(args) >= 4:
                past_key_values = args[3]
            if len(args) >= 5:
                inputs_embeds = args[4]
            if len(args) >= 11:
                cache_position = args[10]

            is_prefill = _is_prefill_step(
                cache_position=cache_position,
                past_key_values=past_key_values,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )
            if _flops_profile_enabled() and is_prefill:
                if self._sample_flops_active and self._sample_flops_steps > 0:
                    _finalize_sample_flops(self)
                _reset_sample_flops(self)
                self._sample_flops_active = True

            out = orig_model_forward(*args, **kwargs)

            if _flops_profile_enabled() and self._sample_flops_active:
                stage_flops = getattr(self.model, "_last_forward_flops", {}) or {}
                vision_flops = stage_flops.get("vision", None)
                llm_flops = stage_flops.get("llm", None)
                lm_head_flops = getattr(self, "_last_lm_head_flops", None)
                vision_flops_value = float(vision_flops) if vision_flops is not None else 0.0
                llm_flops_value = float(llm_flops) if llm_flops is not None else 0.0
                lm_head_flops_value = float(lm_head_flops) if lm_head_flops is not None else 0.0
                e2e_flops = vision_flops_value + llm_flops_value + lm_head_flops_value
                if e2e_flops > 0.0:
                    self._sample_flops_steps += 1
                    self._sample_vision_flops_sum += vision_flops_value
                    self._sample_llm_flops_sum += llm_flops_value
                    self._sample_lm_head_flops_sum += lm_head_flops_value
                    self._sample_e2e_flops_sum += e2e_flops

            return out

        model.forward = types.MethodType(patched_model_forward, model)

    if hasattr(model, "generate"):
        orig_generate = model.generate

        def patched_generate(self, *args, **kwargs):
            out = orig_generate(*args, **kwargs)
            if _flops_profile_enabled():
                _finalize_sample_flops(self)
            return out

        model.generate = types.MethodType(patched_generate, model)

    model._vlmeval_flops_patched = True
    _log("[Efficiency INFO] Qwen3VL FLOPs profiling enabled (vision/llm/e2e).")
    return True
