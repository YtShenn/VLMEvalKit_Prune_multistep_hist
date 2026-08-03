
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Tuple
import math
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from collections.abc import Iterable
import types
import warnings
import copy
import ast
from PIL import Image, ImageDraw

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

#新加的辅助函数
def scaled_dot_product_attention_with_policy(query, key, value, policy, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, return_attn_logits: bool = True, prune_attn_meta=None,**kwargs): 
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        temp_mask = temp_mask.to(attn_bias.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias = attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.shape, dtype=query.dtype).to(device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    if attn_bias is None:
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    block_size = int(os.environ.get("VLMPRUNE_SDPA_BLOCK", "0"))
    if block_size <= 0 and L * S >= 2048 * 2048:
        block_size = 512

    if block_size > 0 and L > block_size:
        B, H, _, D = query.shape
        attn_output = torch.empty((B, H, L, D), device=query.device, dtype=query.dtype)
        attn_logits = None
        use_compact = return_attn_logits and prune_attn_meta is not None
        q_indices = None
        v_start = None
        v_end = None
        if use_compact:
            q_indices = prune_attn_meta.get("q_indices")
            v_start = int(prune_attn_meta.get("v_token_start", 0))
            v_num = int(prune_attn_meta.get("v_token_num", 0))
            v_end = min(v_start + v_num, S)
            if q_indices is None or q_indices.numel() == 0 or v_end <= v_start:
                use_compact = False
            else:
                attn_logits = torch.empty((B, H, q_indices.numel(), v_end - v_start), device=query.device, dtype=query.dtype)
        k_t = key.transpose(-2, -1)

        for start in range(0, L, block_size):
            end = min(start + block_size, L)
            q_blk = query[:, :, start:end, :]
            attn_bias_blk = attn_bias[start:end, :]

            attn_weight = q_blk @ k_t * scale_factor
            attn_weight = attn_weight + attn_bias_blk.to(device=query.device)
            attn_weight = softmax_with_policy(attn_weight, policy)
            if use_compact:
                in_blk = (q_indices >= start) & (q_indices < end)
                if in_blk.any():
                    local = (q_indices[in_blk] - start).to(dtype=torch.long)
                    compact_pos = torch.nonzero(in_blk, as_tuple=False).squeeze(-1)
                    attn_logits[:, :, compact_pos, :] = attn_weight.index_select(2, local)[:, :, :, v_start:v_end]
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            attn_output[:, :, start:end, :] = attn_weight @ value

        if use_compact:
            attn_logits = attn_logits.detach()
        return attn_output, attn_logits

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(device=query.device)
    attn_weight = softmax_with_policy(attn_weight, policy)
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
    policy=None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

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
    attn_output, attn_logits = scaled_dot_product_attention_with_policy(
        query,
        key,
        value,
        policy,
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
    policy = None,
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
    if policy is None and self.config._attn_implementation != "sdpa":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights, attn_logits = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        policy=policy,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, attn_logits
    # return attn_output, attn_weights, None, query_states, key_states, value_states, attn_logits  #None的那个是past_key_values

# 替换attn计算过程，加了policy参数，修改了输出
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

    # ------------------------ ADD attribute "policy" -----------------------------------
    # Self Attention
    hidden_states, self_attn_weights, attn_logits = self.self_attn(
        hidden_states=hidden_states,
        policy=policy,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        cache_position=cache_position,
        use_cache=use_cache,
        **kwargs,
    )
    # ------------------------ ADD Over -----------------------------------

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

    return outputs
    # return hidden_states

# 修改的地方见'sparse'标记
class Qwen3VLTextModelPrune(Qwen3VLPreTrainedModel):
    config: Qwen3VLTextConfig
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]

    def __init__(self, config: Qwen3VLTextConfig, pruning_loc=[3, 6, 15]):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        for layer in self.layers:
            layer.forward = types.MethodType(Qwen3VLTextDecoderLayerforward,layer)
            layer.self_attn.forward = types.MethodType(Qwen3VLTextAttentionforward,layer.self_attn)
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # ------------------------------------------- Sparse ----------------------------------------------
        self.pruning_loc = pruning_loc
        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_forward = 0
        self.num_token_pool = 0

        self.init_token_total_shape = 664       
        self.generate_process_count = 0         
        self.total_cuda_time = 0
        self.causal_inference_cuda_time = 0
        self.all_FLOPs = 0
        #=======timer2=====
        self.causal_inference_count = 0
        self.token_sort_time_ms_total = 0.0
        self.token_sort_sample_count = 0
        self.token_recycle_time_ms_total = 0.0
        self.token_recycle_sample_count = 0
        #=======timer2=====
        # ------------------------------------------- Sparse ----------------------------------------------
        

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
        #====================add：===========================
        # image_shape = 576,
        # token_length_list = [],
        # pre_prompt_length_list = [],
        # logger = [],
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

        # ------------------------------------------- SparseVLM --------------------------------------------
        ## 这里pre_prompt_length_list存疑，以及v_token_start是干嘛用的
        B, L, _ = hidden_states.shape
        batch_size, seq_length = inputs_embeds.shape[:2]
        if seq_length > 1:
            self.init_token_total_shape = inputs_embeds.shape[1]
        idx_sprase_layer = 0
        out_pred_prob = None
        # init_n = self.init_token_total_shape + self.generate_process_count    # 668
        init_n = inputs_embeds.shape[1]
        prev_decision = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
        policy = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
        
        #由于逻辑不同，这里不用pre_prompt_length_list，用前面改过的qwen3部分
        # v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0 # 35
        v_token_start = torch.argmax(visual_pos_masks[0].to(torch.uint8)).item()  if visual_pos_masks is not None else 0
        v_token_num = visual_pos_masks[0].sum().item() if visual_pos_masks is not None else 0
        init_v_token_num = v_token_num
        visual_index_map = torch.arange(v_token_num, device=hidden_states.device) if v_token_num > 0 else None
        text_token_start = v_token_start + v_token_num # 611
        disable_pruning = bool(getattr(self.config, "_vlmprune_disable_pruning", False)) or os.getenv("VLMPRUNE_DISABLE_PRUNING", "0") == "1"
        

        # Select Text Raters, from SparseVLM 3.2
        # 先分别获取视觉、文本token，然后计算注意力，选取了高于均值的文本token
        if (visual_pos_masks is not None and seq_length > 1):
            v_t = hidden_states[:, v_token_start: text_token_start, :]
            t_t = hidden_states[:, text_token_start: , :]
            m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53] 视觉-文本相关性矩阵
            m_v_t = m_v_t.softmax(2).mean(1) # [1, 53] 平均注意力
            t_token_idx = torch.where(m_v_t > m_v_t.mean()) # 选择高于平均值的文本token

        num_token = []

        #=======timer2=====
        local_sort_time_ms = 0.0
        local_recycle_time_ms = 0.0
        local_sort_used = False
        local_recycle_used = False
        #=======timer2=====

        #在预填充阶段记录耗时
        total_start_event = None
        total_end_event = None
        if (visual_pos_masks is not None and seq_length > 1) and _cuda_ready():
            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            total_start_event.record()
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            if (visual_pos_masks is not None and seq_length > 1):       
                n = hidden_states.shape[1]                                  # token num
                d = hidden_states.shape[2]                                  # hidden state size 
                m = self.layers[layer_idx].mlp.up_proj.out_features         # intermediate size of the FFN
                self.all_FLOPs += 4 * n * (d**2) + 2 *(n**2) * d + 3*n*d*m 
            # Sparse Layers
            # print("disable_pruning",disable_pruning)
            if (not disable_pruning) and layer_idx in self.pruning_loc and visual_pos_masks is not None and seq_length > 1:
                
                # Training
                if self.training:
                    pass
                
                # Inference
                else:
                    q_indices = t_token_idx[1] + text_token_start
                    prune_attn_meta = {
                        "q_indices": q_indices,
                        "v_token_start": v_token_start,
                        "v_token_num": v_token_num,
                    }
                    # print("calculate layer_idx: ",layer_idx)
                    layer_outputs = decoder_layer(
                            hidden_states = hidden_states,
                            # policy=None,
                            policy=policy,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=past_key_values,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            return_attn_logits=True,
                            prune_attn_meta=prune_attn_meta,
                            **kwargs,
                    )

                    attn_logits = layer_outputs[1]

                    # Text-Visual Attention Gravity Correction, from SparseVLM+ 4.2
                    # 等于说还是在对上一轮的做处理，获得了上一轮通过layer后的attn_logits，来选token的，选完了才更新hidden_states用于下一轮
                    if V2_0 and attn_logits is not None:
                        bs, seq = hidden_states.shape[:2] # idea1: Compute Bias
                        num_heads = decoder_layer.self_attn.num_heads
                        head_dim = decoder_layer.self_attn.head_dim
                        query_states = hidden_states.new_ones(bs, seq, num_heads, head_dim).transpose(1, 2)
                        key_states = hidden_states.new_ones(bs, seq, num_heads, head_dim).transpose(1, 2)
                        cos, sin = decoder_layer.self_attn.rotary_emb(key_states, seq_len=position_ids.max().item() + 1)
                        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
                        if attn_logits.shape[-1] == hidden_states.shape[1]:
                            rope_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                            attn_logits = attn_logits / rope_weights
                        elif v_token_num > 0 and q_indices.numel() > 0:
                            v_start = v_token_start
                            v_end = v_token_start + v_token_num
                            rope_weights = torch.matmul(
                                query_states[:, :, q_indices, :],
                                key_states[:, :, v_start:v_end, :].transpose(2, 3),
                            ) / math.sqrt(head_dim)
                            attn_logits = attn_logits / rope_weights

                    cur_text_token_idx = t_token_idx[1] + text_token_start

                    # Text-Visual Priority Heads Selection, from SparseVLM+ 4.3
                    # 选择对文本-视觉交互最重要的注意力头
                    if V2_0:
                        attn_logits = select_attn_head_by_sum(attn_logits, cur_text_token_idx, v_token_start, text_token_start)
                    
                    #=======timer2=====
                    sort_start_event = None
                    sort_end_event = None
                    if _cuda_ready():
                        sort_start_event = torch.cuda.Event(enable_timing=True)
                        sort_end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        sort_start_event.record()
                    #=======timer2=====

                    # s_flag：是否启用token merge；pred这个是保留视觉token的mask，只包含图像那一部分
                    pred_score_vis, s_flag, relation_vis_text = attn_postprocess_topk(attn_logits, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx) # B, L_v
                    
                    #=======timer2=====
                    if sort_start_event is not None:
                        sort_end_event.record()
                        torch.cuda.synchronize()
                        local_sort_time_ms += sort_start_event.elapsed_time(sort_end_event)
                        print("sort time ms: ",sort_start_event.elapsed_time(sort_end_event))
                        local_sort_used = True
                    #=======timer2=====
                    
                    current_visual_index_map = visual_index_map
                    if visual_index_map is not None and pred_score_vis.shape[1] == visual_index_map.numel():
                        visual_index_map = visual_index_map[pred_score_vis[0].to(torch.bool)]
                    kept_global = None
                    dropped_local = None
                    if current_visual_index_map is not None:
                        kept_local_idx = torch.nonzero(pred_score_vis[0] > 0, as_tuple=False).squeeze(-1)
                        dropped_local_idx = torch.nonzero(pred_score_vis[0] == 0, as_tuple=False).squeeze(-1)
                        kept_global = current_visual_index_map[kept_local_idx] + v_token_start
                        dropped_local = current_visual_index_map[dropped_local_idx]

                    #==== VLMPRUNE_VISUALIZATION ====
                    vis_enable = bool(getattr(self.config, "_vlmprune_vis_enable", False))
                    vis_dir = getattr(self.config, "_vlmprune_vis_dir", None) or os.getenv("VLMPRUNE_VIS_DIR", None)
                    img_paths = getattr(self.config, "_vlmeval_current_image_paths", None)
                    vis_overlay = os.getenv("VLMPRUNE_VIS_OVERLAY", "0") == "1"
                    if vis_enable and vis_dir and img_paths:
                        image_path = img_paths[0]
                        base = os.path.splitext(os.path.basename(image_path))[0]
                        sample_index = getattr(self.config, "_vlmeval_current_sample_index", None) or "na"
                        question = getattr(self.config, "_vlmeval_current_question", None) or "na"
                        q_slug = self._vlmprune_sanitize_for_filename(str(question), max_len=80)
                        if vis_overlay:
                            save_path = os.path.join(
                                vis_dir,
                                f"idx{sample_index}_layer{layer_idx}_{q_slug}_{base}_overlay.png",
                            )
                        else:
                            save_path = os.path.join(
                                vis_dir,
                                f"idx{sample_index}_layer{layer_idx}_{q_slug}_{base}_keep{int(pred_score_vis.sum().item())}_drop{int((pred_score_vis.numel() - pred_score_vis.sum()).item())}.png",
                            )
                        os.makedirs(vis_dir, exist_ok=True)
                        vis_img = getattr(self.config, "_vlmeval_current_vis_image_pil", None)
                        grid_thw = getattr(self.config, "_vlmeval_current_image_grid_thw", None)
                        if vis_overlay:
                            grid_size, img = self._vlmprune_get_overlay_canvas(
                                "prune", vis_img, grid_thw, int(init_v_token_num), str(sample_index), base
                            )
                        else:
                            grid_size, img = self._vlmprune_get_vis_grid(vis_img, grid_thw, int(init_v_token_num))
                        if grid_size is not None and img is not None and current_visual_index_map is not None:
                            drop_color = self._vlmprune_layer_gray(layer_idx) if vis_overlay else (128, 128, 128, 180)
                            self.visualize_pruned_tokens(
                                image=img,
                                kept_global=kept_global,
                                dropped_local=dropped_local,
                                image_token_start_index=v_token_start,
                                grid_size=grid_size,
                                save_path=save_path,
                                drop_color=drop_color,
                            )

                    attn_vis_enable = bool(getattr(self.config, "_vlmprune_attn_vis_enable", False))
                    attn_vis_dir = getattr(self.config, "_vlmprune_attn_vis_dir", None) or os.getenv("VLMPRUNE_ATTN_VIS_DIR", None)
                    attn_layers = getattr(self.config, "_vlmprune_attn_vis_layers", None) or os.getenv("VLMPRUNE_ATTN_VIS_LAYERS", None)
                    # if not attn_vis_enable and attn_layers:
                    #     attn_vis_enable = True
                    if not attn_vis_dir and vis_dir:
                        attn_vis_dir = os.path.join(vis_dir, "attn")
                    layer_set = self._vlmprune_parse_layer_list(attn_layers)
                    # print("layer_set:", layer_set)
                    # print("layer_idx:", layer_idx)
                    if attn_vis_enable and attn_vis_dir and layer_idx in layer_set and attn_logits is not None:
                        # print("in if")
                        os.makedirs(attn_vis_dir, exist_ok=True)
                        vis_img = getattr(self.config, "_vlmeval_current_vis_image_pil", None)
                        grid_thw = getattr(self.config, "_vlmeval_current_image_grid_thw", None)
                        grid_size, img = self._vlmprune_get_vis_grid(vis_img, grid_thw, int(init_v_token_num))
                        attn_weights = self._vlmprune_extract_attn_weights(
                            attn_logits=attn_logits,
                            q_indices=q_indices,
                            v_token_start=v_token_start,
                            v_token_num=int(v_token_num),
                            hidden_len=hidden_states.shape[1],
                        )
                        # print("attn_weights:", attn_weights.shape)
                        # print("grid_size:", grid_size)
                        # print("img:", img)
                        if grid_size is not None and img is not None and attn_weights is not None and current_visual_index_map is not None:
                            if attn_weights.numel() == current_visual_index_map.numel():
                                full_attn = torch.zeros(int(init_v_token_num), device=attn_weights.device, dtype=attn_weights.dtype)
                                full_attn[current_visual_index_map] = attn_weights
                                attn_weights = full_attn
                            else:
                                attn_weights = None
                        if grid_size is not None and img is not None and attn_weights is not None:
                            image_path = img_paths[0]
                            base = os.path.splitext(os.path.basename(image_path))[0]
                            sample_index = getattr(self.config, "_vlmeval_current_sample_index", None) or "na"
                            question = getattr(self.config, "_vlmeval_current_question", None) or "na"
                            q_slug = self._vlmprune_sanitize_for_filename(str(question), max_len=80)
                            save_path = os.path.join(
                                attn_vis_dir,
                                f"idx{sample_index}_layer{layer_idx}_{q_slug}_{base}_attn.png",
                            )
                            # print("in vis")
                            self.visualize_attn_heatmap(
                                image=img,
                                weights=attn_weights,
                                grid_size=grid_size,
                                save_path=save_path,
                                color=(255, 0, 0),
                                dropped_local=dropped_local,
                                drop_color=(128, 128, 128, 180),
                            )
                    #==== VLMPRUNE_VISUALIZATION ====

                    # 这个是全局包含图像、文本的，剪枝后的mask，文本都是1
                    policy = torch.ones(B, hidden_states.shape[1], 1, dtype=hidden_states.dtype, device=hidden_states.device)
                    policy[:, v_token_start:text_token_start, 0] = pred_score_vis.type(dtype=hidden_states.dtype)

                    #双保险，再吧文本设1
                    # 因为qwen3这里batch都是1，所以把for省掉了
                    policy[0,:v_token_start,] = 1
                    policy[0,text_token_start:,] = 1
                    # for batch in range(len(pre_prompt_length_list)):
                    #     # keep pre prompt     
                    #     prompt_length = pre_prompt_length_list[batch]
                    #     policy[batch,:prompt_length,] = 1
                    #     # keep question
                    #     text_token_start = prompt_length + image_shape
                    #     policy[batch, text_token_start:,] = 1
                    
                    # if visual_pos_masks is not None:
                    #     print(f"[PruneDebug][pre] layer {layer_idx} v_start {v_token_start} v_num {v_token_num} text_start {text_token_start} visual_sum {visual_pos_masks.sum().item()}")

                    policy_2d = policy.squeeze(-1)
                    total_sparse_token_idx = torch.where(policy_2d == 0)[1].unsqueeze(0)
                    # merge and cluster
                    if V2_0 or deepstack_visual_embeds is not None:
                        s_flag = False
                    #用合并的
                    if s_flag and total_sparse_token_idx.shape[1]>0:
                            
                        #=======timer2=====
                        recycle_start_event = None
                        recycle_end_event = None
                        if _cuda_ready():
                            recycle_start_event = torch.cuda.Event(enable_timing=True)
                            recycle_end_event = torch.cuda.Event(enable_timing=True)
                            torch.cuda.synchronize()
                            recycle_start_event.record()
                        #=======timer2=====

                        total_sparse_token_idx = torch.where(policy_2d == 0)[1].unsqueeze(0)
                        total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx) 
                        
                        merge_token_idx_stage1 = torch.where(pred_score_vis==0)[1]
                        merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                        # ... 取 Top 30% 相对重要的落选 Token
                        merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3 ) + 1 # Top 30%
                        merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]
                       
                        merge_token_stage2 = total_sparse_token[:,merge_token_stage2_idx,:]
                        cluster_num = int(merge_token_stage2.shape[1] / 10) + 1       
                        if (cluster_num == 0) :
                            cluster_num = merge_token_stage2.shape[1]
                        
                        merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)  

                        select_token_idx = torch.where(policy_2d == 1)[1].unsqueeze(0)  # B, L_new
                        # 这里就是只保留了已选的token了
                        select_token = batch_index_select(layer_outputs[0], select_token_idx)
                        select_vis_token_num = pred_score_vis.sum()
                        select_and_merge_token = torch.cat((select_token[:,:v_token_start+select_vis_token_num,:] ,
                                merge_sparse_token,
                                select_token[:,v_token_start+select_vis_token_num:,:])
                                ,dim=1
                        )

                        layer_outputs = (select_and_merge_token, layer_outputs[1])  # B, L, C
                        # ！！有待确定，qwen3vl之前剪枝用的截取的方法，这个就直接简单粗暴取了前几个，连续的
                        # print("select token idx: ", select_token_idx)
                        keep_idx = select_token_idx[0]
                        if cluster_num > 0:
                            pad_len = keep_idx.shape[0] + cluster_num - position_ids.shape[-1]
                            if pad_len > 0:
                                tail = position_ids[..., -1:].repeat(1, 1, pad_len)
                                position_ids = torch.cat([position_ids, tail], dim=-1)
                        position_ids = position_ids[..., :keep_idx.shape[0] + cluster_num]
                        prev_decision = policy
                        # update
                        v_token_num = pred_score_vis.sum() + cluster_num # B == 1
                        # print(layer_idx, v_token_num)
                        text_token_start = v_token_start + v_token_num
                        #=======timer2=====
                        if recycle_start_event is not None:
                            recycle_end_event.record()
                            torch.cuda.synchronize()
                            local_recycle_time_ms += recycle_start_event.elapsed_time(recycle_end_event)
                            print("recycle time: ",recycle_start_event.elapsed_time(recycle_end_event))
                            local_recycle_used = True
                        #=======timer2=====
                    else:
                        select_token_idx = torch.where(policy_2d == 1)[1].unsqueeze(0)  # B, L_new
                        layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])  # B, L, C
                        position_ids = position_ids[..., select_token_idx[0]]
                        prev_decision = policy
                        if visual_pos_masks is not None:
                            keep_visual_mask = visual_pos_masks[0][select_token_idx[0]]
                            visual_pos_masks = visual_pos_masks[:, select_token_idx[0]]
                            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                                keep_visual_mask = keep_visual_mask.to(deepstack_visual_embeds[layer_idx].device)
                                deepstack_visual_embeds[layer_idx] = deepstack_visual_embeds[layer_idx][keep_visual_mask, :]
                        if attention_mask is not None:
                            keep_idx = select_token_idx[0]
                            if attention_mask.dim() == 4:
                                attention_mask = attention_mask[:, :, keep_idx, :][:, :, :, keep_idx]
                            elif attention_mask.dim() == 3:
                                attention_mask = attention_mask[:, keep_idx, :][:, :, keep_idx]
                            elif attention_mask.dim() == 2:
                                attention_mask = attention_mask[keep_idx, :][:, keep_idx]
                        
                        # update
                        v_token_num = pred_score_vis.sum() # B == 1
                        # print(layer_idx, v_token_num)
                        text_token_start = v_token_start + v_token_num
                        # if visual_pos_masks is not None:
                            # print(f"[PruneDebug][post] layer {layer_idx} v_start {v_token_start} v_num {v_token_num} text_start {text_token_start} visual_sum {visual_pos_masks.sum().item()}")
                        
                policy = torch.ones(
                    B, layer_outputs[0].shape[1], 1, dtype=hidden_states.dtype, device=hidden_states.device
                )
                prev_decision = policy
                idx_sprase_layer = idx_sprase_layer + 1 
                hidden_states = layer_outputs[0]
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
                # print("in prune layer, layer idx: ", layer_idx)
                # print("position_embeddings shape: ", position_embeddings[0].shape,position_embeddings[1].shape)
                # print("hidden_states shape: ", hidden_states.shape)
                # print("position_ids shape: ", position_ids.shape)
                # print("in prune layer, layer idx: ", layer_idx)
            # Normal Layers
            else:
                # print("in normal layer, layer idx: ", layer_idx)
                
                '''
                # if output_hidden_states:
                #     all_hidden_states += (hidden_states,)

                # if self.gradient_checkpointing and self.training:
                #     layer_outputs = self._gradient_checkpointing_func(
                #         decoder_layer.__call__,
                #         hidden_states,
                #         policy,
                #         attention_mask,
                #         position_ids,
                #         past_key_values,
                #         output_attentions,
                #         use_cache,
                #     )
                # else:
                #     layer_outputs = decoder_layer(
                #         hidden_states,
                #         policy=policy,
                #         attention_mask=attention_mask,
                #         position_ids=position_ids,
                #         past_key_value=past_key_values,
                #         output_attentions=output_attentions,
                #         use_cache=use_cache,
                #     )
                '''
                # if layer_idx == 1 and seq_length > 1:
                #     print("in first layer, layer idx: ", layer_idx)
                #     # print("position_embeddings shape: ", position_embeddings.shape)
                #     print("position_embeddings shape: ", position_embeddings[0].shape,position_embeddings[1].shape)
                #     print("hidden_states shape: ", hidden_states.shape)
                #==== VLMPRUNE_ATTN_VIS_ANY_LAYER ====
                attn_vis_enable = bool(getattr(self.config, "_vlmprune_attn_vis_enable", False))
                attn_vis_dir = getattr(self.config, "_vlmprune_attn_vis_dir", None) or os.getenv("VLMPRUNE_ATTN_VIS_DIR", None)
                attn_layers = getattr(self.config, "_vlmprune_attn_vis_layers", None) or os.getenv("VLMPRUNE_ATTN_VIS_LAYERS", None)
                # if not attn_d= True
                layer_set = self._vlmprune_parse_layer_list(attn_layers)
                single_layer_attn_only = os.environ.get("SINGLE_LAYER_ATTN_ONLY", "0") == "1"
                use1_15 = os.getenv("VLMPRUNE_USE_1_15", "0") == "1"
                if single_layer_attn_only and layer_idx not in layer_set and not use1_15:
                    continue
                attn_only_exit = bool(getattr(self.config, "_vlmprune_attn_only_exit", False)) or single_layer_attn_only
                
                attn_only_exit_layer = int(getattr(self.config, "_vlmprune_attn_only_exit_layer", 15))
                if use1_15 and layer_idx != attn_only_exit_layer:
                    attn_only_exit = False
                elif use1_15:
                    attn_only_exit = True
                need_attn_logits = attn_vis_enable and attn_vis_dir and layer_idx in layer_set and visual_pos_masks is not None
                # print("need_attn_logits: ", need_attn_logits)
                prune_attn_meta = None
                if need_attn_logits:
                    q_indices = t_token_idx[1] + text_token_start
                    prune_attn_meta = {
                         "q_indices": q_indices,
                         "v_token_start": v_token_start,
                         "v_token_num": v_token_num,
                    }
                # print("normal, layer_set: ",layer_set)
                # print("calculate layer_idx: ",layer_idx)
                layer_outputs = decoder_layer(
                    hidden_states,
                    policy=policy,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    return_attn_logits=need_attn_logits,
                    prune_attn_meta=prune_attn_meta,
                    **kwargs,
                )
                # print("need_attn_logits: ",need_attn_logits)
                current_visual_index_map = visual_index_map
                if need_attn_logits:
                    # print("need_attn_logits")
                    attn_logits = layer_outputs[1]
                    # print("attn_logits: ",attn_logits.shape)
                    vis_img = getattr(self.config, "_vlmeval_current_vis_image_pil", None)
                    grid_thw = getattr(self.config, "_vlmeval_current_image_grid_thw", None)
                    img_paths = getattr(self.config, "_vlmeval_current_image_paths", None)
                    sample_index = getattr(self.config, "_vlmeval_current_sample_index", None) or "na"
                    if img_paths:
                        image_path = img_paths[0]
                        base = os.path.splitext(os.path.basename(image_path))[0]
                    else:
                        base = "na"
                    grid_size, img = self._vlmprune_get_vis_grid(vis_img, grid_thw, int(init_v_token_num))
                    q_indices = t_token_idx[1] + text_token_start
                    attn_weights = self._vlmprune_extract_attn_weights(
                        attn_logits=attn_logits,
                        q_indices=q_indices,
                        v_token_start=v_token_start,
                        v_token_num=int(v_token_num),
                        hidden_len=hidden_states.shape[1],
                    )
                    # print("grid_size: ",grid_size)
                    # print("img: ",img)
                    # print("attn_weights: ",attn_weights)
                    # print("current_visual_index_map: ",current_visual_index_map)
                    if grid_size is not None and img is not None and attn_weights is not None and current_visual_index_map is not None:
                        # print("in if 1")
                        if attn_weights.numel() == current_visual_index_map.numel():
                            full_attn = torch.zeros(int(init_v_token_num), device=attn_weights.device, dtype=attn_weights.dtype)
                            full_attn[current_visual_index_map] = attn_weights
                            attn_weights = full_attn
                        else:
                            attn_weights = None
                    dropped_local = None
                    if current_visual_index_map is not None:
                        # print("in if 2")
                        full_idx = torch.arange(int(init_v_token_num), device=current_visual_index_map.device)
                        keep_mask = torch.zeros(int(init_v_token_num), dtype=torch.bool, device=current_visual_index_map.device)
                        keep_mask[current_visual_index_map] = True
                        dropped_local = full_idx[~keep_mask]
                    if grid_size is not None and img is not None and attn_weights is not None and img_paths:
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
                            dropped_local=dropped_local,
                            drop_color=(128, 128, 128, 180),
                        )
                        if attn_only_exit and layer_idx == attn_only_exit_layer:
                            # ------------------------------------------- SparseVLM --------------------------------------------
                            if visual_pos_masks is not None and seq_length > 1 and total_start_event is not None:
                                total_end_event.record()
                                torch.cuda.synchronize()  

                                total_cuda_time_ms = total_start_event.elapsed_time(total_end_event)
                                self.total_cuda_time += total_cuda_time_ms
                                self.num_forward += 1
                                self.num_token_pool += (sum(num_token) / self.num_layers)
                                FLOPs_avg_sample = (self.all_FLOPs / self.num_forward) * 1e-12
                                #=======timer2=====
                                prefill_avg_ms = self.total_cuda_time / max(1, self.num_forward)
                                loggerinfo.info(
                                    f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward)}, "
                                    f"Prefill Time Total (ms): {self.total_cuda_time:.2f}, "
                                    f"Prefill Avg (ms): {prefill_avg_ms:.2f}, "
                                    f"TFLOPs:{FLOPs_avg_sample:.2f}"
                                )
                                #=======timer2=====

                                # loggerinfo.info(f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward)}, Prefill Time (ms): {self.total_cuda_time:.2f}, TFLOPs:{FLOPs_avg_sample:.2f}")
                            # ------------------------------------------- SparseVLM --------------------------------------------
                            #=======timer2=====
                            if local_sort_used:
                                self.token_sort_time_ms_total += local_sort_time_ms
                                self.token_sort_sample_count += 1
                                sort_avg_ms = self.token_sort_time_ms_total / max(1, self.token_sort_sample_count)
                                loggerinfo.info(
                                    f"{prefix} Token Sort Time Total (ms): {self.token_sort_time_ms_total:.2f}, "
                                    f"Token Sort Avg (ms): {sort_avg_ms:.2f}"
                                )
                            if local_recycle_used:
                                self.token_recycle_time_ms_total += local_recycle_time_ms
                                self.token_recycle_sample_count += 1
                                recycle_avg_ms = self.token_recycle_time_ms_total / max(1, self.token_recycle_sample_count)
                                loggerinfo.info(
                                    f"{prefix} Token Recycle Time Total (ms): {self.token_recycle_time_ms_total:.2f}, "
                                    f"Token Recycle Avg (ms): {recycle_avg_ms:.2f}"
                                )
                            #=======timer2=====
                            hidden_states = layer_outputs[0]
                            hidden_states = self.norm(hidden_states)
                            return (
                                prev_decision.detach(),
                                out_pred_prob,
                                BaseModelOutputWithPast(
                                    last_hidden_state=hidden_states,
                                    past_key_values=past_key_values,
                                ),
                            )
                #==== VLMPRUNE_ATTN_VIS_ANY_LAYER ====

               
            hidden_states = layer_outputs[0]

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            # if use_cache:
            #     next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)

            num_token.append(v_token_num)
        # ------------------------------------------- SparseVLM ---------------------------
        
        '''
        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
'''
        # ------------------------------------------- SparseVLM --------------------------------------------
        if visual_pos_masks is not None and seq_length > 1 and total_start_event is not None:
            total_end_event.record()
            torch.cuda.synchronize()  

            total_cuda_time_ms = total_start_event.elapsed_time(total_end_event)
            self.total_cuda_time += total_cuda_time_ms
            self.num_forward += 1
            self.num_token_pool += (sum(num_token) / self.num_layers)
            FLOPs_avg_sample = (self.all_FLOPs / self.num_forward) * 1e-12
            #=======timer2=====
            prefill_avg_ms = self.total_cuda_time / max(1, self.num_forward)
            loggerinfo.info(
                f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward)}, "
                f"Prefill Time Total (ms): {self.total_cuda_time:.2f}, "
                f"Prefill Avg (ms): {prefill_avg_ms:.2f}, "
                f"TFLOPs:{FLOPs_avg_sample:.2f}"
            )
            #=======timer2=====
            # loggerinfo.info(f"{prefix} Equal Tokens: {int(self.num_token_pool / self.num_forward)}, Prefill Time (ms): {self.total_cuda_time:.2f}, TFLOPs:{FLOPs_avg_sample:.2f}")
        # ------------------------------------------- SparseVLM --------------------------------------------
        
        #=======timer2=====
        if local_sort_used:
            self.token_sort_time_ms_total += local_sort_time_ms
            self.token_sort_sample_count += 1
            sort_avg_ms = self.token_sort_time_ms_total / max(1, self.token_sort_sample_count)
            loggerinfo.info(
                f"{prefix} Token Sort Time Total (ms): {self.token_sort_time_ms_total:.2f}, "
                f"Token Sort Avg (ms): {sort_avg_ms:.2f}"
            )
        if local_recycle_used:
            self.token_recycle_time_ms_total += local_recycle_time_ms
            self.token_recycle_sample_count += 1
            recycle_avg_ms = self.token_recycle_time_ms_total / max(1, self.token_recycle_sample_count)
            loggerinfo.info(
                f"{prefix} Token Recycle Time Total (ms): {self.token_recycle_time_ms_total:.2f}, "
                f"Token Recycle Avg (ms): {recycle_avg_ms:.2f}"
            )
        #=======timer2=====

        hidden_states = self.norm(hidden_states)

        return (prev_decision.detach(), 
            out_pred_prob,
            BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            # hidden_states=all_hidden_states,
            # attentions=all_self_attns,
        ))
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    #==== VLMPRUNE_VIS_HELPERS ====
    def _vlmprune_parse_layer_list(self, layer_list):
        if layer_list is None:
            return set()
        if isinstance(layer_list, (list, tuple)):
            return set(int(x) for x in layer_list)
        if isinstance(layer_list, str):
            items = [s.strip() for s in layer_list.split(",") if s.strip()]
            return set(int(x) for x in items)
        return set()

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

    def _vlmprune_prepare_attn_vis_image(self, vis_img):
        if vis_img is None:
            return None
        scale = os.getenv("VLMPRUNE_ATTN_VIS_SCALE", "1")
        try:
            scale = float(scale)
        except Exception:
            scale = 1.0
        if scale <= 0:
            scale = 1.0
        if abs(scale - 1.0) < 1e-6:
            return vis_img
        w = max(1, int(vis_img.width * scale))
        h = max(1, int(vis_img.height * scale))
        return vis_img.resize((w, h), Image.BILINEAR)

    def _vlmprune_get_overlay_canvas(self, kind: str, vis_img, grid_thw, v_token_num: int, sample_index: str, base: str):
        grid_size, _ = self._vlmprune_get_vis_grid(vis_img, grid_thw, v_token_num)
        if grid_size is None or vis_img is None:
            return None, None
        cache = getattr(self.config, "_vlmprune_overlay_cache", None)
        if cache is None:
            cache = {}
            self.config._vlmprune_overlay_cache = cache
        key = f"{kind}:{sample_index}:{base}"
        if key not in cache:
            cache[key] = vis_img.copy()
        return grid_size, cache[key]

    def _vlmprune_layer_color(self, layer_idx: int):
        palette = [
            (128, 128, 128, 140),
            (255, 165, 0, 140),
            (0, 128, 255, 140),
            (255, 0, 0, 140),
            (0, 255, 0, 140),
            (160, 32, 240, 140),
        ]
        return palette[layer_idx % len(palette)]

    def _vlmprune_layer_gray(self, layer_idx: int):
        shades = [70, 100, 130, 160, 190, 220]
        v = shades[layer_idx % len(shades)]
        return (v, v, v, 170)

    def _vlmprune_sanitize_for_filename(self, s: str, max_len: int = 80) -> str:
        s = (s or "").strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff _.,;:!@#%+=()-]", "", s)
        s = s.strip().replace(" ", "_")
        if len(s) > max_len:
            s = s[:max_len]
        return s or "na"

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

    #==== VLMPRUNE_VIS_OVERLAY ====
    def _vlmprune_parse_bbox(self, bbox):
        if bbox is None:
            return None
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except Exception:
                return None
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        return [float(x) for x in bbox]

    def _vlmprune_parse_point(self, point):
        if point is None:
            return None
        if isinstance(point, str):
            try:
                point = ast.literal_eval(point)
            except Exception:
                return None
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        return [float(point[0]), float(point[1])]

    def _vlmprune_get_overlay_targets(self, image: Image.Image):
        bbox = getattr(self.config, "_vlmeval_current_bbox", None)
        img_size = getattr(self.config, "_vlmeval_current_img_size", None)
        pred_point = getattr(self.config, "_vlmeval_current_pred_point", None)
        bbox = self._vlmprune_parse_bbox(bbox)
        pred_point = self._vlmprune_parse_point(pred_point)
        if img_size is not None:
            if isinstance(img_size, str):
                try:
                    img_size = ast.literal_eval(img_size)
                except Exception:
                    img_size = None
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                img_size = (float(img_size[0]), float(img_size[1]))
            else:
                img_size = None
        vis_w, vis_h = float(image.width), float(image.height)
        base_w = img_size[0] if img_size is not None else vis_w
        base_h = img_size[1] if img_size is not None else vis_h
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + x2
                y2 = y1 + y2
            bbox_max = max(x1, y1, x2, y2)
            if bbox_max <= 1.5:
                x1, x2 = x1 * base_w, x2 * base_w
                y1, y2 = y1 * base_h, y2 * base_h
            elif bbox_max <= 1000:
                x1, x2 = x1 / 1000 * base_w, x2 / 1000 * base_w
                y1, y2 = y1 / 1000 * base_h, y2 / 1000 * base_h
            if img_size is not None:
                x1 = x1 * vis_w / img_size[0]
                x2 = x2 * vis_w / img_size[0]
                y1 = y1 * vis_h / img_size[1]
                y2 = y2 * vis_h / img_size[1]
            bbox = [x1, y1, x2, y2]
        if pred_point is not None:
            x, y = pred_point
            point_max = max(x, y)
            if point_max <= 1.5:
                x, y = x * base_w, y * base_h
            elif point_max <= 1000:
                x, y = x / 1000 * base_w, y / 1000 * base_h
            if img_size is not None:
                x = x * vis_w / img_size[0]
                y = y * vis_h / img_size[1]
            pred_point = [x, y]
        return bbox, pred_point

    def _vlmprune_draw_overlays(self, image: Image.Image):
        bbox, pred_point = self._vlmprune_get_overlay_targets(image)
        if bbox is None and pred_point is None:
            return
        draw = ImageDraw.Draw(image, mode="RGBA")
        if bbox is not None:
            draw.rectangle(bbox, outline=(0, 255, 0, 255), width=3)
        if pred_point is not None:
            x, y = pred_point
            r = 6
            draw.rectangle([x - r, y - r, x + r, y + r], outline=(255, 0, 0, 255), width=3)
    #==== VLMPRUNE_VIS_OVERLAY ====

    def visualize_pruned_tokens(
        self,
        image: Image.Image,
        kept_global: torch.Tensor,
        dropped_local: torch.Tensor,
        image_token_start_index: int,
        grid_size: tuple[int, int],
        save_path: str,
        keep_color=(255, 255, 255, 0),
        drop_color=(128, 128, 128, 180),
    ):
        h, w = grid_size
        patch_w = image.width / w
        patch_h = image.height / h
        draw = ImageDraw.Draw(image, mode="RGBA")
        kept_local = (kept_global - image_token_start_index).cpu().tolist()
        dropped_local = dropped_local.cpu().tolist()
        for idx in kept_local:
            r, c = divmod(idx, w)
            x0, y0 = c * patch_w, r * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h
            draw.rectangle([x0, y0, x1, y1], fill=keep_color)
        for idx in dropped_local:
            r, c = divmod(idx, w)
            x0, y0 = c * patch_w, r * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h
            draw.rectangle([x0, y0, x1, y1], fill=drop_color)
        # self._vlmprune_draw_overlays(image)
        image.save(save_path)

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
        h, w = grid_size
        patch_w = image.width / w
        patch_h = image.height / h
        draw = ImageDraw.Draw(image, mode="RGBA")
        w_min = float(weights.min().item())
        w_max = float(weights.max().item())
        denom = w_max - w_min if w_max > w_min else 1.0
        for idx in range(weights.numel()):
            r, c = divmod(idx, w)
            v = float((weights[idx].item() - w_min) / denom)
            alpha = int(200 * v)
            x0, y0 = c * patch_w, r * patch_h
            x1, y1 = x0 + patch_w, y0 + patch_h
            draw.rectangle([x0, y0, x1, y1], fill=(color[0], color[1], color[2], alpha))
        if dropped_local is not None:
            dropped_local = dropped_local.cpu().tolist()
            for idx in dropped_local:
                r, c = divmod(idx, w)
                x0, y0 = c * patch_w, r * patch_h
                x1, y1 = x0 + patch_w, y0 + patch_h
                draw.rectangle([x0, y0, x1, y1], fill=drop_color)
        # self._vlmprune_draw_overlays(image)
        image.save(save_path)
    #==== VLMPRUNE_VIS_HELPERS ====

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

# 替换language_model；修改language_model的传入参数，以及forward的参数
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
            if attention_mask is not None and attention_mask.ndim == 2 and total_input_ids.ndim == 2:
                if attention_mask.shape != total_input_ids.shape and attention_mask.shape == total_input_ids.transpose(0, 1).shape:
                    total_input_ids = total_input_ids.transpose(0, 1)

            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            elif attention_mask.ndim == 2 and attention_mask.shape != total_input_ids.shape:
                if attention_mask.shape[0] == total_input_ids.shape[0] and attention_mask.shape[1] > total_input_ids.shape[1]:
                    attention_mask = attention_mask[:, -total_input_ids.shape[1]:]
                else:
                    attention_mask = attention_mask[:, : total_input_ids.shape[1]]

            position_ids = torch.ones(
                3,
                total_input_ids.shape[0],
                total_input_ids.shape[1],
                dtype=total_input_ids.dtype,
                device=total_input_ids.device,
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
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
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
        #------------------------add-------------------------
        # image_shape = 576,
        # token_length_list = [],
        # pre_prompt_length_list = [],
        # logger = [],
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
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
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
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
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

        outputs = self.language_model(#sparse_llava_llama88行
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            #---------------add-----------------
            # image_shape = image_shape,
            # token_length_list = token_length_list,
            # pre_prompt_length_list = pre_prompt_length_list,
            # logger=logger
            **kwargs,
        )

        if isinstance(outputs, tuple) and len(outputs) >= 3 and hasattr(outputs[2], "last_hidden_state"):
            outputs = outputs[2]

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
        else:
            last_hidden_state = outputs[0]
            past_key_values = outputs[1] if len(outputs) > 1 else None

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            rope_deltas=self.rope_deltas,
        )


# 后面Qwen3VLModel的textdecoder；改了forward函数的参数以及outputs的索引；修改返回值；修改generate、greedy_search的参数
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
        #------------------------add-------------------------
        image_shape = 576,
        token_length_list = [],
        pre_prompt_length_list = [],
        logger = [],
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
            #------------------------add-------------------------
            # image_shape=image_shape,
            # token_length_list=token_length_list,
            # pre_prompt_length_list=pre_prompt_length_list,
            # logger=logger
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
        logits = self.lm_head(hidden_states[:, slice_indices, :])

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
