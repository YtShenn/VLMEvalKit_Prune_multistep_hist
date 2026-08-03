
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Tuple
import math
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
try:
    from torch.profiler import profile as torch_profile, record_function as torch_record_function, ProfilerActivity
    _TORCH_PROFILER_AVAILABLE = True
except Exception:
    torch_profile = None
    torch_record_function = None
    ProfilerActivity = None
    _TORCH_PROFILER_AVAILABLE = False
from collections.abc import Iterable
import types
import warnings
import copy
import ast
from PIL import Image, ImageDraw
import json
from pathlib import Path

from transformers import is_torch_available
from transformers.utils import TransformersKwargs, auto_docstring, is_torchdynamo_compiling
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel,
    # Qwen3VisionTransformerPretrainedModel,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLTextRMSNorm,
    apply_rotary_pos_emb,
    GradientCheckpointingLayer,
    auto_docstring,
    Qwen3VLVisionModel,
    FlashAttentionKwargs,
    Unpack,
    Qwen3VLTextConfig,
    deprecate_kwarg,
    GenerationMixin,
    check_model_inputs,
    TransformersKwargs,
    # Qwen3RMSNorm,
    # Qwen3MLP,
    repeat_kv,
    Qwen3VLTextMLP,
    create_causal_mask,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLTextModel,
    Qwen3VLModelOutputWithPast,
    # Qwen3VLModel,
    Qwen3VLTextDecoderLayer,
    eager_attention_forward
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache, DynamicSlidingWindowLayer, DynamicLayer
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa, logger, _is_torch_npu_available
from transformers.generation.utils import GenerationMode,GenerateDecoderOnlyOutput, GenerateOutput, \
GenerateEncoderDecoderOutput,GenerateBeamDecoderOnlyOutput,GenerateBeamEncoderDecoderOutput

from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.streamers import BaseStreamer
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
from .utils import *
from .score import *

GREEN = "\033[32m"
RESET = "\033[0m"
prefix_raw = "[Efficiency INFO]"
prefix = f"{GREEN}[Efficiency INFO]{RESET}"
pad = " " * len(prefix_raw)


def _cuda_ready() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _flops_profile_enabled() -> bool:
    return _env_flag("QWEN3VL_PROFILE_FLOPS", "0")


def _format_flops(flops: float) -> str:
    abs_flops = abs(flops)
    if abs_flops >= 1e12:
        return f"{flops / 1e12:.3f} TFLOPs"
    if abs_flops >= 1e9:
        return f"{flops / 1e9:.3f} GFLOPs"
    if abs_flops >= 1e6:
        return f"{flops / 1e6:.3f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def _maybe_profile_flops(stage: str, fn: Callable[[], Any]) -> tuple[Any, Optional[float]]:
    if not _flops_profile_enabled():
        return fn(), None
    if not _TORCH_PROFILER_AVAILABLE:
        loggerinfo.warning(f"{prefix} FLOPs profiling requested but torch.profiler is unavailable.")
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
    except Exception as e:
        loggerinfo.warning(f"{prefix} FLOPs profiling failed at stage `{stage}`: {e}")
        return fn(), None


#===========roi========
class ScreenSpotROICropEnsembler:
    """ROI crop utilities for ScreenSpotPro multi-crop inference."""

    @staticmethod
    def load_roi_map(json_path: str | None) -> dict:
        if not json_path:
            return {}
        path = Path(json_path)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    

    @staticmethod
    def find_boxes(config, message, roi_map: dict, image_path: str) -> list[list[float]]:
        if not roi_map or not image_path:
            return []
        base_name = os.path.basename(image_path)
        stem = os.path.splitext(base_name)[0]
        candidates = [
            image_path,
            base_name,
            stem,
            f"{stem}.png",
            f"{stem}.jpg",
            f"{stem}.jpeg",
            f"{stem}.webp",
        ]
        sample_index: str | None = None
        question: str | None = None
        for s in message:
            if isinstance(s, dict) and s.get('type') == 'image':
                v = s.get('value')
                if sample_index is None and isinstance(s.get('sample_index'), str):
                    sample_index = s.get('sample_index')
                if question is None and isinstance(s.get('question'), str):
                    question = s.get('question')
        # sample_index = getattr(config, "_vlmeval_current_sample_index", None) or "na"
        # question = getattr(config, "_vlmeval_current_question", None) or "na"
        q_slug = ScreenSpotROICropEnsembler._vlmprune_sanitize_for_filename(str(question), max_len=80)
        save_name = f"idx{sample_index}_{q_slug}_{stem}_attn.png"
        
        # print("key (img_name): " , save_name)
        scale_ = int(os.environ.get("VLMPRUNE_CORD_SCALE",2))
        if save_name in roi_map and isinstance(roi_map[save_name], list):
            boxes = []
            for box in roi_map[save_name]:
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    # print("box: ", box)
                    boxes.append([float(v) * scale_ for v in box])   #  存储的是scale的坐标，要乘2映射回原图
            return boxes
        else:
            print(f"key {save_name} not found in roi_map or is not a list")
        return []

    @staticmethod
    def normalize_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
        if box is None or len(box) != 4:
            return None
        x0, y0, x1, y1 = [float(v) for v in box]
        max_v = max(abs(x0), abs(y0), abs(x1), abs(y1))
        if max_v <= 1.5:
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0 = int(max(0, min(width - 1, round(x0))))
        y0 = int(max(0, min(height - 1, round(y0))))
        x1 = int(max(x0 + 1, min(width, round(x1))))
        y1 = int(max(y0 + 1, min(height, round(y1))))
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def parse_point(text: str) -> tuple[float, float] | None:
        if not isinstance(text, str):
            return None
        match = re.search(r"x\s*=\s*([+-]?\d+(?:\.\d+)?)\s*,\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)", text)
        if not match:
            return None
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _vlmprune_sanitize_for_filename( s: str, max_len: int = 80) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,;:!@#%+=()-]", "", s)
        s = s.strip().replace(" ", "_")
        if len(s) > max_len:
            s = s[:max_len]
        return s or "na"

    @staticmethod
    def to_global_normalized(config, local_xy: tuple[float, float], crop_xyxy: tuple[int, int, int, int], orig_size: tuple[int, int]) -> tuple[float, float]:
        x, y = local_xy
        x0, y0, x1, y1 = crop_xyxy
        ow, oh = orig_size
        print(f"local_xy: {local_xy}, crop_xyxy: {crop_xyxy}, orig_size: {orig_size}")
        cw = max(1.0, float(x1 - x0))
        ch = max(1.0, float(y1 - y0))
        if max(abs(x), abs(y)) <= 1.5:
            abs_x = x0 + x * cw
            abs_y = y0 + y * ch
        else:
            abs_x = x0 + x / 1000 * cw
            abs_y = y0 + y / 1000 * ch

        #======可视化一下==========
        # vis_img = getattr(config, "_vlmeval_current_vis_image_pil", None)
        # grid_thw = getattr(config, "_vlmeval_current_image_grid_thw", None)
        # img_paths = getattr(config, "_vlmeval_current_image_paths", None)
        # sample_index = getattr(config, "_vlmeval_current_sample_index", None) or "na"
        # if img_paths:
        #     image_path = img_paths[0]
        #     base = os.path.splitext(os.path.basename(image_path))[0]
        # else:
        #     base = "na"
        # question = getattr(config, "_vlmeval_current_question", None) or "na"
        # q_slug = ScreenSpotROICropEnsembler._vlmprune_sanitize_for_filename(str(question), max_len=80)
        # vis_dir = os.getenv("VLMPRUNE_ATTN_VIS_DIR", None)
        # os.makedirs(vis_dir, exist_ok=True)
        # save_path = os.path.join(
        #     vis_dir,
        #     f"idx{sample_index}_{q_slug}_{base}_pred.png",
        # )
        # draw = ImageDraw.Draw(image, mode="RGBA")

                        
        gx = max(0.0, min(1.0, abs_x / max(1.0, float(ow)))) * 1000
        gy = max(0.0, min(1.0, abs_y / max(1.0, float(oh)))) * 1000
        return (gx, gy),(abs_x, abs_y)
        # return abs_x, abs_y

    @staticmethod
    def aggregate(candidates: list[dict], method: str = "max_prob", cluster_thr: float = 0.08) -> tuple[float, float] | None:
        print("in aggeragate")
        if not candidates:
            return None
        valid = []
        for c in candidates:
            p = c.get("point")
            if p is None:
                continue
            prob = c.get("prob")
            if prob is None:
                prob = 0.0
            valid.append({"point": p, "prob": float(prob)})
        if not valid:
            return None

        method = (method or "max_prob").lower()
        if method == "weighted_mean":
            weights = [max(v["prob"], 1e-6) for v in valid]
            total = sum(weights)
            x = sum(w * v["point"][0] for w, v in zip(weights, valid)) / total
            y = sum(w * v["point"][1] for w, v in zip(weights, valid)) / total
            return (x, y)

        if method == "cluster":
            print("in cluster")
            clusters = []
            for item in valid:
                px, py = item["point"]
                weight = max(item["prob"], 1e-6)
                assigned = False
                for cluster in clusters:
                    cx, cy = cluster["center"]
                    dist = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                    if dist <= cluster_thr:
                        cluster["items"].append(item)
                        cluster["sum_w"] += weight
                        cluster["sum_x"] += weight * px
                        cluster["sum_y"] += weight * py
                        cluster["center"] = (cluster["sum_x"] / cluster["sum_w"], cluster["sum_y"] / cluster["sum_w"])
                        assigned = True
                        break
                if not assigned:
                    clusters.append(
                        {
                            "items": [item],
                            "sum_w": weight,
                            "sum_x": weight * px,
                            "sum_y": weight * py,
                            "center": (px, py),
                        }
                    )
            best = max(clusters, key=lambda c: c["sum_w"])
            return best["center"]

        best = max(valid, key=lambda v: v["prob"])
        print(f"best point: {best['point']}, prob: {best['prob']:.4f}")
        return best["point"]
#=========roi end=========

'''
#新加的辅助函数
def softmax_with_policy(attn, policy, eps=1e-6):    # attn : [2, 687, 32, 32] policy : [2, 687, 1]
    B, policy_len, _ = policy.size()
    B, H, T, attn_len = attn.size()
    if policy_len != attn_len:
        if policy_len < attn_len:
            policy = F.pad(policy, (0, 0, 0, attn_len - policy_len), value=1)
        else:
            policy = policy[:, :attn_len, :]
        policy_len = policy.size(1)
    if T == 1:
        policy_bias = torch.zeros(B, 1, policy_len, 1, dtype=policy.dtype).to(device=policy.device)
        policy_bias.masked_fill_(policy.logical_not(), float("-inf"))
        policy_bias = policy_bias.permute(0, 1, 3, 2).to(policy.dtype)
        attn += policy_bias.to(device=attn.device)
        attn = torch.softmax(attn, dim=-1)
        return attn
    else:
        # If T != N (e.g., block attention), fall back to a broadcastable mask.
        if T != policy_len:
            policy_bias = torch.zeros(B, 1, 1, policy_len, dtype=policy.dtype, device=policy.device)
            policy_bias.masked_fill_(policy.logical_not().view(B, 1, 1, policy_len), float("-inf"))
            attn += policy_bias.to(device=attn.device)
            attn = torch.softmax(attn, dim=-1)
            return attn

        attn_policy = policy.reshape(B, 1, 1, policy_len)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
        eye = torch.eye(policy_len, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, policy_len, policy_len) # [1, 1, 687, 687]
        attn_policy = attn_policy + (1.0 - attn_policy) * eye   # [2, 1, 687, 687]
        policy_bias = torch.zeros(B, 1, policy_len, policy_len, dtype=attn_policy.dtype).to(device=attn_policy.device)
        policy_bias.masked_fill_(attn_policy.logical_not(), float("-inf"))
        policy_bias.to(attn_policy.dtype)
        attn += policy_bias
        attn = torch.softmax(attn, dim=-1)
        return attn
'''

#新加的辅助函数
def scaled_dot_product_attention_(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, return_attn_logits: bool = True, prune_attn_meta=None,**kwargs): 
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        temp_mask = temp_mask.to(attn_bias.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.shape, dtype=query.dtype).to(device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    if attn_bias is None:
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(device=query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if return_attn_logits and prune_attn_meta is not None:
        q_indices = prune_attn_meta.get("q_indices")
        v_start = int(prune_attn_meta.get("v_token_start", 0))
        v_num = int(prune_attn_meta.get("v_token_num", 0))
        v_end = min(v_start + v_num, S)
        if q_indices is None or q_indices.numel() == 0 or v_end <= v_start:
            attn_logits = None
        else:
            attn_logits = attn_weight[:, :, q_indices, v_start:v_end].detach()
    else:
        attn_logits = attn_weight.clone().detach() if return_attn_logits else None

    # attn_logits = attn_weight.clone().detach()

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_logits


# 新加的函数，替换原来的ALL_ATTENTION_FUNCTIONS["sdpa"]，在transformer库的基础上对标LlamaDynamicvitSdpaAttention
def sdpa_attention_forward_(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        # if not use_gqa_in_sdpa(attention_mask, key):
        #     key = repeat_kv(key, module.num_key_value_groups)
        #     value = repeat_kv(value, module.num_key_value_groups)
        # else:
        #     sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # Change
    # if not self.training:
    #     attn_output, attn_logits = scaled_dot_product_attention(
    #         query_states,
    #         key_states,
    #         value_states,
    #         attn_mask=attention_mask,
    #         dropout_p=self.attention_dropout if self.training else 0.0,
    #         # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    #         is_causal=self.is_causal and attention_mask is None and q_len > 1,
    #     )
    # else:
    return_attn_logits = kwargs.pop("return_attn_logits", True)
    prune_attn_meta = kwargs.pop("prune_attn_meta", None)
    attn_output, attn_logits = scaled_dot_product_attention_(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,#self.attention_dropout if self.training else 0.0,
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal=is_causal,#self.is_causal and attention_mask is None and q_len > 1
        return_attn_logits=return_attn_logits,
        prune_attn_meta=prune_attn_meta,
        **sdpa_kwargs,
    )
    # attn_output = torch.nn.functional.scaled_dot_product_attention(
    #     query,
    #     key,
    #     value,
    #     attn_mask=attention_mask,
    #     dropout_p=dropout,
    #     scale=scaling,
    #     is_causal=is_causal,
    #     **sdpa_kwargs,
    # )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None, attn_logits


# 替换了sdpa函数的实现，加了attn_logits返回值，修改了forward的返回值个数
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def Qwen3VLTextAttentionforward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    # print("in Qwen3VLTextAttentionforward")
    # print(f"hidden_states shape: {hidden_states.shape}")


    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # print(f"query_states shape: {query_states.shape}")
    # print(f"key_states shape: {key_states.shape}")
    # print(f"cos shape: {cos.shape}")
    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # attention_interface: Callable = eager_attention_forward
    # if self.config._attn_implementation != "eager":
    #     attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attention_interface = sdpa_attention_forward_
    if self.config._attn_implementation != "sdpa":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights, attn_logits = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, attn_logits
    # return attn_output, attn_weights, None, query_states, key_states, value_states, attn_logits  #None的那个是past_key_values

# 替换attn计算过程，加了attn_logits输出
@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def Qwen3VLTextDecoderLayerforward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    policy = None,
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # Self Attention
    hidden_states, self_attn_weights, attn_logits = self.self_attn(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        cache_position=cache_position,
        use_cache=use_cache,
        **kwargs,
    )
    # print("hidden_states: ", hidden_states)

    end_event.record()

    # 5. 关键：同步！等待 GPU 完成所有排队任务
    torch.cuda.synchronize()

    # 6. 计算时间（单位：毫秒 ms）
    elapsed_time_ms = start_event.elapsed_time(end_event)
    # print(f"[Attn Time]: {elapsed_time_ms / 1000.0:.4f} s")

    # Self Attention
    # hidden_states, _ = self.self_attn(
    #     hidden_states=hidden_states,
    #     attention_mask=attention_mask,
    #     position_ids=position_ids,
    #     past_key_values=past_key_values,
    #     use_cache=use_cache,
    #     cache_position=cache_position,
    #     position_embeddings=position_embeddings,
    #     **kwargs,
    # )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    outputs += (attn_logits, )

    return outputs,elapsed_time_ms
    # return hidden_states

# 修改的地方见'sparse'标记
class Qwen3VLTextModelPrune(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        for layer in self.layers:
            layer.forward = types.MethodType(Qwen3VLTextDecoderLayerforward,layer)
            layer.self_attn.forward = types.MethodType(Qwen3VLTextAttentionforward,layer.self_attn)
        

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        r"""
        visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
            The mask of the visual positions.
        deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
            The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
            The feature is extracted from the different visual encoder layers, and fed to the decoder
            hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ================获取一些前置信息====================
        batch_size, seq_length = inputs_embeds.shape[:2]
        v_token_start = torch.argmax(visual_pos_masks[0].to(torch.uint8)).item()  if visual_pos_masks is not None else 0
        v_token_num = visual_pos_masks[0].sum().item() if visual_pos_masks is not None else 0
        init_v_token_num = v_token_num
        visual_index_map = torch.arange(v_token_num, device=hidden_states.device) if v_token_num > 0 else None
        text_token_start = v_token_start + v_token_num
        if (visual_pos_masks is not None and seq_length > 1):
            v_t = hidden_states[:, v_token_start: text_token_start, :]
            t_t = hidden_states[:, text_token_start: , :]
            m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53] 视觉-文本相关性矩阵
            m_v_t = m_v_t.softmax(2).mean(1) # [1, 53] 平均注意力
            t_token_idx = torch.where(m_v_t > m_v_t.mean()) # 选择高于平均值的文本token
        
        
        # ================获取一些前置信息====================
        
        attn_only_exit = bool(getattr(self.config, "_vlmprune_attn_only_exit", False))
        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):

            attn_vis_enable = bool(getattr(self.config, "_vlmprune_attn_vis_enable", False))
            attn_vis_dir = getattr(self.config, "_vlmprune_attn_vis_dir", None) or os.getenv("VLMPRUNE_ATTN_VIS_DIR", None)
            attn_layers = getattr(self.config, "_vlmprune_attn_vis_layers", None) or os.getenv("VLMPRUNE_ATTN_VIS_LAYERS", None)
            # if not attn_d= True
            layer_set = self._vlmprune_parse_layer_list(attn_layers)
                    
            need_attn_logits = layer_idx in layer_set and visual_pos_masks is not None
            prune_attn_meta = None
            if need_attn_logits:
                q_indices = t_token_idx[1] + text_token_start
                prune_attn_meta = {
                    "q_indices": q_indices,
                    "v_token_start": v_token_start,
                    "v_token_num": v_token_num,
                }

            layer_outputs,attn_time = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                return_attn_logits=need_attn_logits,
                prune_attn_meta=prune_attn_meta,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if need_attn_logits:
                attn_logits = layer_outputs[1]
                q_indices = t_token_idx[1] + text_token_start
                attn_weights = self._vlmprune_extract_attn_weights(
                    attn_logits=attn_logits,
                    q_indices=q_indices,
                    v_token_start=v_token_start,
                    v_token_num=int(v_token_num),
                    hidden_len=hidden_states.shape[1],
                )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if attn_only_exit and layer_idx in layer_set:
                
                #================可视化注意力权重，调试用===========
                vis_img = getattr(self.config, "_vlmeval_current_vis_image_pil", None)
                grid_thw = getattr(self.config, "_vlmeval_current_image_grid_thw", None)
                img_paths = getattr(self.config, "_vlmeval_current_image_paths", None)
                grid_size, img = self._vlmprune_get_vis_grid(vis_img, grid_thw, int(init_v_token_num))
                sample_index = getattr(self.config, "_vlmeval_current_sample_index", None) or "na"
                if img_paths:
                    image_path = img_paths[0]
                    base = os.path.splitext(os.path.basename(image_path))[0]
                else:
                    base = "na"
                if grid_size is not None and img is not None and attn_weights is not None and img_paths and '0' in sample_index:
                        # print("in if 3")
                        question = getattr(self.config, "_vlmeval_current_question", None) or "na"
                        q_slug = self._vlmprune_sanitize_for_filename(str(question), max_len=80)
                        os.makedirs(attn_vis_dir, exist_ok=True)
                        save_path = os.path.join(
                            attn_vis_dir,
                            f"idx{sample_index}_layer{layer_idx}_{q_slug}_{base}_attn.png",
                        )
                        # print("save_path: ",save_path)
                        print("vis attn layer_idx: ",layer_idx)
                        self.visualize_attn_heatmap(
                            image=img,
                            weights=attn_weights,
                            grid_size=grid_size,
                            save_path=save_path,
                            color=(255, 0, 0),
                            dropped_local=None,
                            drop_color=(128, 128, 128, 180),
                        )



                #=================可视化注意力权重，调试用===========
                window_size=os.environ.get("VLMPRUNE_FCR_ROI_WINDOW_SIZE", 5)
                # print(window_size)
                top_k_peaks=os.environ.get("VLMPRUNE_FCR_ROI_CANDIDATE_NUM", 5)
                roi_boxes = self.get_fcr_roi_indices(attn_logits=attn_weights,top_k_peaks=int(top_k_peaks),window_size=int(window_size))
                # roi_boxes = self.get_fcr_roi_indices(attn_logits=attn_weights)
                hidden_states = self.norm(hidden_states)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        
        # print("hidden_states shape: ", hidden_states)
        # print("visual_embeds shape: ", visual_embeds.shape)
        # print("visual_pos_masks shape: ", visual_pos_masks.shape)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def _vlmprune_extract_attn_weights(self, attn_logits, q_indices, v_token_start: int, v_token_num: int, hidden_len: int):
        if attn_logits is None or attn_logits.dim() != 4:
            return None
        if attn_logits.shape[-1] == v_token_num and attn_logits.shape[-2] == q_indices.numel():
            attn_map = attn_logits.mean(1).mean(1)[0]
            return attn_map
        if attn_logits.shape[-1] == hidden_len:
            v_end = v_token_start + v_token_num
            if v_end > attn_logits.shape[-1] or q_indices.numel() == 0:
                return None
            attn_map = attn_logits[:, :, q_indices, v_token_start:v_end].mean(1).mean(1)[0]
            return attn_map
        return None

    def _vlmprune_parse_layer_list(self, layer_list):
        if layer_list is None:
            return set()
        if isinstance(layer_list, (list, tuple)):
            return set(int(x) for x in layer_list)
        if isinstance(layer_list, str):
            items = [s.strip() for s in layer_list.split(",") if s.strip()]
            return set(int(x) for x in items)
        return set()

    def _vlmprune_sanitize_for_filename(self, s: str, max_len: int = 80) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,;:!@#%+=()-]", "", s)
        s = s.strip().replace(" ", "_")
        if len(s) > max_len:
            s = s[:max_len]
        return s or "na"
    
    def _vlmprune_get_vis_grid(self, vis_img, grid_thw, v_token_num: int):
        if vis_img is None or grid_thw is None:
            return None, None
        img = vis_img.copy()
        grid_h = int(grid_thw[0][1].item())
        grid_w = int(grid_thw[0][2].item())
        if self.config.spatial_merge_size:
            grid_h = grid_h // self.config.spatial_merge_size
            grid_w = grid_w // self.config.spatial_merge_size
        if grid_h * grid_w != v_token_num:
            return None, None
        return (grid_h, grid_w), img
    
    def visualize_attn_heatmap(
        self,
        image: Image.Image,
        weights: torch.Tensor,
        grid_size: tuple[int, int],
        save_path: str,
        color=(255, 0, 0),
        dropped_local: torch.Tensor | None = None,
        drop_color=(128, 128, 128, 180),
    ):
        from PIL import ImageFont
        try:
            # 字体大小根据 patch 高度动态调整（取 patch 高度的 1/3）
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(patch_h * 0.25))
        except:
            font = ImageFont.load_default()
        h, w = grid_size
        patch_w = image.width / w
        patch_h = image.height / h
        draw = ImageDraw.Draw(image, mode="RGBA")
        w_min = float(weights.min().item())
        w_max = float(weights.max().item())
        denom = w_max - w_min if w_max > w_min else 1.0
        # print("weights:", weights)
        for idx in range(weights.numel()):
            r, c = divmod(idx, w)
            v = float((weights[idx].item() - w_min) / denom)
            alpha = int(200 * v)
            x0, y0 = c * patch_w, r * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h
            draw.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], alpha))
            t=v * 1e3
            text = f"{t:.1f}" if t >= 1 else f"{t:.2f}"
            # print("text", text, "v:", v)
            text_color = (255, 255, 255, 255) if t > 0.5 else (0, 0, 0, 255) # 根据亮度自动切黑白字
            draw.text(((x0 + x1) / 2, (y0 + y1) / 2), text, fill=text_color, font=font, anchor="mm")

        if dropped_local is not None:
            dropped_local = dropped_local.cpu().tolist()
            for idx in dropped_local:
                r, c = divmod(idx, w)
                x0, y0 = c * patch_w, r * patch_h
                x1, y1 = x0 + patch_w, y0 + patch_h
                draw.rectangle([x0, y0, x1, y1], fill=drop_color)
        # self._vlmprune_draw_overlays(image)
        image.save(save_path)

    def get_fcr_roi_indices(self, attn_logits, top_k_peaks=5, window_size=5):
        """
        attn_logits: (B, v_token_num)
        Returns:
            roi_boxes: list[list[tuple[int,int,int,int]]]  # per batch, (r0, c0, r1, c1)
        """
        if attn_logits is None:
            return []
        # if attn_logits.dim() != 2:
        #     raise ValueError(f"attn_logits must be 2D [B, v_token_num], got {attn_logits.shape}")

        v_token_num = attn_logits.shape[0]
        # print("v_token_num: ", v_token_num)
        # print("batch: ", B)
        vis_img = getattr(self.config, "_vlmeval_current_vis_image_pil", None)
        img_paths = getattr(self.config, "_vlmeval_current_image_paths", None)
        grid_thw = getattr(self.config, "_vlmeval_current_image_grid_thw", None)
        # print(f"vis_img: {vis_img}, grid_thw: {grid_thw}")
        image_path = img_paths[0] if img_paths is not None else None


        base = os.path.splitext(os.path.basename(image_path))[0]
        sample_index = getattr(self.config, "_vlmeval_current_sample_index", None) or "na"
        question = getattr(self.config, "_vlmeval_current_question", None) or "na"
        q_slug = self._vlmprune_sanitize_for_filename(str(question), max_len=80)
        save_name = f"idx{sample_index}_{q_slug}_{base}_attn.png"
        

        if vis_img is None or grid_thw is None:
            # Fallback: use original file (may be misaligned if processor resized/padded)
            img = Image.open(image_path).convert("RGB")
        else:
            # grid_thw: [num_images, 3] => (t, h, w)
            grid_h = int(grid_thw[0][1].item())
            grid_w = int(grid_thw[0][2].item())
        if self.config.spatial_merge_size:
            # print(f"spatial_merge_size: {self.config.spatial_merge_size}")
            grid_h = grid_h // self.config.spatial_merge_size
            grid_w = grid_w // self.config.spatial_merge_size
        # print(f"grid_h: {grid_h}, grid_w: {grid_w}")
        # 获取边长
        patch_w = vis_img.width / grid_w   #像素行高列宽
        patch_h = vis_img.height / grid_h
        # print(f"patch_h: {patch_h}, patch_w: {patch_w}")
        # if grid_h is None or grid_w is None:    # 这两个grid开头的变量都是长、宽的token个数，不是边长
        #     side = int(math.sqrt(v_token_num))
        #     if side * side != v_token_num:
        #         raise ValueError(
        #             f"Cannot infer square grid from v_token_num={v_token_num}. Please pass grid_h/grid_w."
        #         )
        #     grid_h, grid_w = side, side
        if grid_h * grid_w != v_token_num:
            raise ValueError(f"grid_h*grid_w must equal v_token_num, got {grid_h}*{grid_w}!={v_token_num}")

        w_min = float(attn_logits.min().item())
        w_max = float(attn_logits.max().item())
        denom = w_max - w_min if w_max > w_min else 1.0
        for idx in range(attn_logits.numel()):
            v = float((attn_logits[idx].item() - w_min) / denom)
            attn_logits[idx] = int(200 * v)
        torch.set_printoptions(profile="full")
        # print("attn_logits after normalization: ", attn_logits)
        # print("attn_shape: ", attn_logits.shape)

        print("top_k_peaks: ", top_k_peaks)
        topk_idx = torch.topk(attn_logits, k=top_k_peaks).indices

        # 1. Gaussian smoothing with a 3x3 kernel
        # kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        # kernel = kernel.view(1, 1, 3, 3).to(attn_logits.device, dtype=attn_logits.dtype)
        # smoothed = F.conv2d(attn_logits.view(1, 1, grid_h, grid_w), kernel, padding=1).squeeze(1)

        # # 2. Local maxima detection
        # pooled = F.max_pool2d(smoothed, kernel_size=3, stride=1, padding=1)
        # print("pooled: ", pooled)
        # local_max = smoothed == pooled

        # 3. Select top-k peaks among local maxima
        roi_boxes = []
        r = window_size // 2  # window_size=5 -> r=2
        # for b in range(B): # batch一直是1来着
        # coords = torch.nonzero(local_max[0], as_tuple=False)
        # if coords.numel() == 0:
        #     max_idx = torch.argmax(smoothed[0].view(-1))
        #     coords = torch.stack([max_idx // grid_w, max_idx % grid_w]).unsqueeze(0)
        # vals = smoothed[0][coords[:, 0], coords[:, 1]]
        # k = min(int(top_k_peaks), int(vals.numel()))
        # topk_idx = torch.topk(vals, k=k).indices
        # coords = coords[topk_idx]
        
        boxes = []
        for idx in topk_idx:
            r_c, c_c = idx // grid_w, idx % grid_w
        # for rc in coords:   #这里只是块索引坐标，不是实际的像素坐标，要再乘边长
            # r_c = int(rc[0].item())
            # c_c = int(rc[1].item())
            r0 = max(0, r_c - r)
            c0 = max(0, c_c - r)
            r1 = min(grid_h - 1, r_c + r)
            c1 = min(grid_w - 1, c_c + r)
            pixel_box = (
                int(c0 * patch_w), 
                int(r0 * patch_h), 
                int(c1 * patch_w + patch_w), 
                int(r1 * patch_h + patch_h)
            )
            boxes.append(pixel_box)
            roi_boxes.append(boxes)

        # 写入json文件记录
        output_dir_str = os.environ.get("OUTPUT_DIR", ".")
        json_path_str = os.environ.get("FCR_ROI_JSON_PATH", "roi_results.json")
        json_path = Path(output_dir_str) / json_path_str
        json_path.parent.mkdir(parents=True, exist_ok=True)
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    roi_data = json.load(f)
                except json.JSONDecodeError: # 防止文件损坏或为空
                    roi_data = {}
        else:
            roi_data = {}
        roi_data[save_name] = boxes

        with open(json_path, 'w') as f:
            json.dump(roi_data, f)

        return roi_boxes


# 替换language_model
class Qwen3VLModel(Qwen3VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        config.text_config.spatial_merge_size = self.config.vision_config.spatial_merge_size
        self.language_model = Qwen3VLTextModelPrune._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

        #=======timer2=====
        self.vision_encoder_cuda_time_ms_total = 0.0
        self.vision_encoder_sample_count = 0
        self.causal_inference_cuda_time = 0.0
        self.causal_inference_count = 0
        self._vision_flops_total = 0.0
        self._vision_flops_count = 0
        self._llm_flops_total = 0.0
        self._llm_flops_count = 0
        self._last_forward_flops = {"vision": None, "llm": None}
        #=======timer2=====
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""

        # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        """
        Encodes videos into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input videos.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
        """
        # Same implementation as for images
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        """
        Encodes images into continuous embeddings that can be forwarded to the language model. The deepstack visual features are also returned.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`):
                The tensors corresponding to the input images.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
        """
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # print("img_grid_thw",image_grid_thw)
        # print("spatial_merge_size", self.config.vision_config.spatial_merge_size)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        print(f"[vision] image tokens per image: {split_sizes}, total={sum(split_sizes)}")
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: Optional[torch.FloatTensor] = None,
        video_features: Optional[torch.FloatTensor] = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        # print("in get_placeholder_mask")
        # print(f"input_ids: {input_ids}")
        if input_ids is None:
            tmp=self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            # print(f"get_input_embeddings(): {tmp.shape}")
            # print(f"inputs_embeds: {inputs_embeds.shape}")
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    @auto_docstring
    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        # print("in Qwen3VLModel forward")
        # print(f"input_ids: {input_ids}")
        # print(f"inputs_embeds: {inputs_embeds}")
        # print(f"pixel_values: {pixel_values}")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        vision_flops_this_call = 0.0
        llm_flops_this_call = None

        #=======timer2=====
        vision_time_ms = 0.0
        vision_used = False
        #=======timer2=====

        if pixel_values is not None:
            #=======timer2=====
            vision_start_event = None
            vision_end_event = None
            if _cuda_ready():
                vision_start_event = torch.cuda.Event(enable_timing=True)
                vision_end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                vision_start_event.record()
            #=======timer2=====
            (image_embeds, deepstack_image_embeds), image_flops = _maybe_profile_flops(
                "vision_image_encoder", lambda: self.get_image_features(pixel_values, image_grid_thw)
            )
            if image_flops is not None:
                vision_flops_this_call += image_flops
            #=======timer2=====
            if vision_start_event is not None:
                vision_end_event.record()
                torch.cuda.synchronize()
                vision_time_ms += vision_start_event.elapsed_time(vision_end_event)
                print("vision time: ",vision_start_event.elapsed_time(vision_end_event)/1000)
                vision_used = True
            #=======timer2=====
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            #=======timer2=====
            vision_start_event = None
            vision_end_event = None
            print("_cuda_ready: ",_cuda_ready())
            if _cuda_ready():
                vision_start_event = torch.cuda.Event(enable_timing=True)
                vision_end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                vision_start_event.record()
            #=======timer2=====
            (video_embeds, deepstack_video_embeds), video_flops = _maybe_profile_flops(
                "vision_video_encoder", lambda: self.get_video_features(pixel_values_videos, video_grid_thw)
            )
            if video_flops is not None:
                vision_flops_this_call += video_flops
            #=======timer2=====
            if vision_start_event is not None:
                vision_end_event.record()
                torch.cuda.synchronize()
                vision_time_ms += vision_start_event.elapsed_time(vision_end_event)
                print("vision time: ",vision_start_event.elapsed_time(vision_end_event))
                vision_used = True
            #=======timer2=====
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        #=======timer2=====
        if vision_used:
            self.vision_encoder_cuda_time_ms_total += vision_time_ms
            self.vision_encoder_sample_count += 1
            vision_avg_ms = self.vision_encoder_cuda_time_ms_total / max(1, self.vision_encoder_sample_count)
            loggerinfo.info(
                f"{prefix} Vision Encoder Time Total (ms): {self.vision_encoder_cuda_time_ms_total:.2f}, "
                f"Vision Encoder Avg (ms): {vision_avg_ms:.2f}"
            )
        #=======timer2=====
        if _flops_profile_enabled() and vision_flops_this_call > 0:
            self._vision_flops_total += vision_flops_this_call
            self._vision_flops_count += 1
            vision_avg_flops = self._vision_flops_total / max(1, self._vision_flops_count)
            loggerinfo.info(
                f"{prefix} FLOPs [Vision] this={_format_flops(vision_flops_this_call)}, "
                f"avg={_format_flops(vision_avg_flops)}"
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs, llm_flops_this_call = _maybe_profile_flops(
            "llm_decoder",
            lambda: self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            ),
        )
        if isinstance(outputs, tuple) and len(outputs) >= 3 and hasattr(outputs[2], "last_hidden_state"):
            outputs = outputs[2]

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
        else:
            last_hidden_state = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None

        self._last_forward_flops = {
            "vision": vision_flops_this_call if vision_flops_this_call > 0 else None,
            "llm": llm_flops_this_call,
        }

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            rope_deltas=self.rope_deltas,
        )


# 后面Qwen3VLModel的textdecoder；改了forward函数的outputs的索引；修改返回值
class Qwen3VLForConditionalGeneration(Qwen3VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        #------------------------add-------------------------
        self.image_shape = 576
        self.token_length_list = []
        self.pre_prompt_length_list = []
        self._sample_flops_active = False
        self._sample_flops_steps = 0
        self._sample_llm_flops_sum = 0.0
        self._sample_e2e_flops_sum = 0.0
        self._sample_vision_flops_sum = 0.0
        self._sample_lm_head_flops_sum = 0.0
        #------------------------add-------------------------
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def _vlmprune_to_int(self, value) -> Optional[int]:
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

    def _vlmprune_is_prefill_step(
        self,
        cache_position: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.FloatTensor],
    ) -> bool:
        cache_start = self._vlmprune_to_int(cache_position)
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

    def _vlmprune_reset_sample_flops(self):
        self._sample_flops_steps = 0
        self._sample_llm_flops_sum = 0.0
        self._sample_e2e_flops_sum = 0.0
        self._sample_vision_flops_sum = 0.0
        self._sample_lm_head_flops_sum = 0.0

    def _vlmprune_flush_sample_flops(self):
        if self._sample_flops_steps <= 0:
            return
        avg_llm = self._sample_llm_flops_sum / self._sample_flops_steps
        avg_e2e = self._sample_e2e_flops_sum / self._sample_flops_steps
        loggerinfo.info(
            f"{prefix} FLOPs [LLM][Sample Avg] avg={_format_flops(avg_llm)}, "
            f"total={_format_flops(self._sample_llm_flops_sum)}, steps={self._sample_flops_steps}"
        )
        loggerinfo.info(
            f"{prefix} FLOPs [E2E][Sample Avg] avg={_format_flops(avg_e2e)}, "
            f"total={_format_flops(self._sample_e2e_flops_sum)}, steps={self._sample_flops_steps} "
            f"(vision_total={_format_flops(self._sample_vision_flops_sum)}, "
            f"llm_total={_format_flops(self._sample_llm_flops_sum)}, "
            f"lm_head_total={_format_flops(self._sample_lm_head_flops_sum)})"
        )

    def _vlmprune_finalize_sample_flops(self):
        if not self._sample_flops_active:
            return
        self._vlmprune_flush_sample_flops()
        self._vlmprune_reset_sample_flops()
        self._sample_flops_active = False

    @check_model_inputs()
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.

        Example:
            TODO: Add example
        """
        is_prefill_step = self._vlmprune_is_prefill_step(
            cache_position=cache_position,
            past_key_values=past_key_values,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        if _flops_profile_enabled() and is_prefill_step:
            if self._sample_flops_active and self._sample_flops_steps > 0:
                self._vlmprune_finalize_sample_flops()
            self._vlmprune_reset_sample_flops()
            self._sample_flops_active = True
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        #------------------------add-------------------------
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            prev_decision = outputs[0]
            out_pred_prob = outputs[1]
            outputs = outputs[2]
        else:
            prev_decision = None
            out_pred_prob = None
        #------------------------add-------------------------

        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
            if not torch.is_tensor(hidden_states) and isinstance(outputs, (tuple, list)):
                # Fall back to the first tensor that matches LM head input width.
                for item in outputs:
                    if torch.is_tensor(item) and item.shape[-1] == self.lm_head.in_features:
                        hidden_states = item
                        break
        if torch.is_tensor(hidden_states) and hidden_states.shape[-1] != self.lm_head.in_features:
            # Try to recover from unexpected tuple structures.
            if isinstance(outputs, (tuple, list)):
                for item in outputs:
                    if torch.is_tensor(item) and item.shape[-1] == self.lm_head.in_features:
                        hidden_states = item
                        break

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits, lm_head_flops = _maybe_profile_flops(
            "lm_head", lambda: self.lm_head(hidden_states[:, slice_indices, :])
        )

        if _flops_profile_enabled() and self._sample_flops_active:
            stage_flops = getattr(self.model, "_last_forward_flops", {}) or {}
            vision_flops = stage_flops.get("vision", None)
            llm_flops = stage_flops.get("llm", None)
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

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        output = Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
        )
        return_dict = kwargs.get("return_dict", self.config.return_dict)
        if return_dict:
            return output
        return (prev_decision, out_pred_prob) + output.to_tuple()

    def generate(self, *args, **kwargs):
        generated = super().generate(*args, **kwargs)
        if _flops_profile_enabled():
            self._vlmprune_finalize_sample_flops()
        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            **kwargs,
        )

        # Qwen3VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        if inputs_embeds is not None:
            vision_start_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(vision_start_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            image_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(image_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
            video_mask = (
                inputs_embeds
                == self.get_input_embeddings()(
                    torch.tensor(video_token_id, dtype=torch.long, device=inputs_embeds.device)
                )
            )[..., 0]
        else:
            vision_start_mask = input_ids == vision_start_token_id
            image_mask = input_ids == image_token_id
            video_mask = input_ids == video_token_id

        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(
                input_ids, inputs_embeds=model_kwargs.get("inputs_embeds", None)
            )

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=list(video_nums), repeat_times=expand_size
                    )
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs
