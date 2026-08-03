from dataclasses import dataclass
from typing import Any, Callable, Optional, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections.abc import Iterable
import types

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
import os

# 仅修改了forward的返回值个数
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

    attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if self.config._attn_implementation != "sdpa":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
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
    # return attn_output, attn_weights
    return attn_output, attn_weights, None, query_states, key_states, value_states  #None的那个是past_key_values


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
    **kwargs: Unpack[TransformersKwargs],
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    # Self Attention
    # hidden_states, _ = self.self_attn(
    hidden_states, self_attn_weights, present_key_value, query_states, key_states, value_states = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    outputs += (query_states, key_states, value_states)
    return outputs
    # return hidden_states

import ast
def get_env_var(name):
    var = os.environ.get(name, 'None')
    try:
        return ast.literal_eval(var.strip())
    except (ValueError, SyntaxError):
        return None

import random
def attn_sink_awared_random_pruning(hidden_states, start, total_count, ratio):
    sample_size = int(total_count * (1 - ratio))
    sample_size -= 4 # attention sink awared
    random_indices = random.sample(range(start + 4, start + total_count + 1), sample_size)
    random_indices += [start, start + 1, start + 2, start + 3]
    return torch.tensor(sorted(random_indices)), hidden_states

def is_in_bbox(x, y, x0, y0, x1, y1):
    return x >= x0 and x <= x1 and y >= y0 and y <= y1

def is_in_bbox_list(x, y, bbox:list):
    x0, y0, x1, y1 = bbox
    return is_in_bbox(x, y, x0, y0, x1, y1)

def expand_bbox(bbox:list, img_size:list, neighbor_range:float):
    x0, y0, x1, y1 = [i/32 for i in bbox] # 32 for patch size(2x2 mini-patch for 16x16 each)
    w, h = [math.ceil(i/32) for i in img_size]
    print(f"Will inference with neighbor_range={neighbor_range}")

    x0 -= neighbor_range
    y0 -= neighbor_range
    x1 += neighbor_range
    y1 += neighbor_range

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)], [w, h]

def bbox_reserve_random_pruning(hidden_states, start, total_count, ratio, bbox:str, img_size:str):
    bbox = eval(bbox)
    img_size = eval(img_size)
    bbox, img_size = expand_bbox(bbox=bbox, img_size=img_size, neighbor_range=get_env_var('NRANGE'))
    w, h = img_size

    # attn_sink_indices = [start, start + 1, start + 2, start + 3]
    attn_sink_indices = list(range(start, start+16))
    gt_indices = [i 
        for i in range(start+len(attn_sink_indices), start+total_count)
        if is_in_bbox_list(i%w, i//w, bbox=bbox)
    ]
    other_indices = [i 
        for i in range(start+len(attn_sink_indices), start+total_count)
        if not is_in_bbox_list(i%w, i//w, bbox=bbox)
    ]
    sample_size = int(total_count * (1 - ratio))
    sample_size -= len(attn_sink_indices) # attention sink awared
    sample_size -= len(gt_indices)
    random.seed(0)
    random_indices = random.sample(other_indices, sample_size)
    random_indices = attn_sink_indices+gt_indices+random_indices
    
    return torch.tensor(sorted(random_indices)), hidden_states

# 写死了neighbor range为1/4图像宽高
def expand_bbox2d(bbox:list, img_size:list):
    print(f"gt x ranges from [{bbox[0]*1000/img_size[0]},{bbox[2]*1000/img_size[0]}]")
    print(f"gt y ranges from [{bbox[1]*1000/img_size[1]},{bbox[3]*1000/img_size[1]}]")
    x0, y0, x1, y1 = [i/32 for i in bbox] # 32 for patch size(2x2 mini-patch for 16x16 each)
    w, h = [math.ceil(i/32) for i in img_size]
    neighbor_range_x = w // 4
    neighbor_range_y = h // 4
    print(f"Will inference with neighbor_range={neighbor_range_x}&{neighbor_range_y}")

    x0 -= neighbor_range_x
    y0 -= neighbor_range_x
    x1 += neighbor_range_y
    y1 += neighbor_range_y

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(w, x1)
    y1 = min(h, y1)
    return [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)], [w, h]

def bbox_reserve_rect_pruning(start, total_count, ratio, bbox:str, img_size:str):
    bbox = eval(bbox)
    img_size = eval(img_size)
    print(f"bbox={bbox} img_size={img_size}")
    bbox, img_size = expand_bbox2d(bbox=bbox, img_size=img_size)
    print(f"token level:bbox={bbox} img_size={img_size}")
    w, h = img_size

    # attn_sink_indices = list(range(start, start+16))
    attn_sink_indices = []
    low = start+len(attn_sink_indices)
    high = start+total_count
    gt_indices = [i 
        for i in range(low, high)
        if is_in_bbox_list((i-low)%w, (i-low)//w, bbox=bbox)
    ]
    other_indices = [i 
        for i in range(low, high)
        if not is_in_bbox_list((i-low)%w, (i-low)//w, bbox=bbox)
    ]
    sample_size = int(total_count * (1 - ratio))
    sample_size -= len(attn_sink_indices) # attention sink awared
    sample_size -= len(gt_indices)
    # random.seed(0)
    if sample_size < 0:
        print(f"sample_size({sample_size}) < 0, adjust to lower bound")
        sample_size = 0
    # print(f"sample_size={sample_size} prop={len(other_indices)}")
    random_indices = random.sample(other_indices, sample_size)
    random_indices = attn_sink_indices+gt_indices+random_indices
    
    return sorted(random_indices)
    # return torch.tensor(sorted(gt_indices)), hidden_states


def bbox_reserve_rect_pruning_inv(start, total_count, ratio, bbox, img_size):
    bbox = eval(bbox)
    img_size = eval(img_size)
    print(f"bbox={bbox} img_size={img_size}")
    bbox, img_size = expand_bbox2d(bbox=bbox, img_size=img_size)
    print(f"token level:bbox={bbox} img_size={img_size}")
    w, h = img_size

    low = start
    high = start + total_count
    other_indices = [i 
        for i in range(low, high)
        if not is_in_bbox_list((i-low)%w, (i-low)//w, bbox=bbox)
    ]
    remove_size = int(total_count * ratio)
    if remove_size > len(other_indices):
        print(f"remove_size({remove_size}) > len(other_indices){len(other_indices)}, adjust to upper bound")
        remove_size = len(other_indices)
    random.seed(0)
    other_indices = random.sample(other_indices, remove_size)

    return sorted(other_indices)

# PRUNE_CASE='PRUNE_BY_ATTN_MASK_REF_GT'
PRUNE_CASE='PRUNE_BY_SHRINKING_REF_GT'

# 新写的剪枝decoder，只在forward加了剪枝，增加了几个辅助函数
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
        for layer in self.layers:
            layer.forward = types.MethodType(Qwen3VLTextDecoderLayerforward,layer)
            layer.self_attn.forward = types.MethodType(Qwen3VLTextAttentionforward,layer.self_attn)
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # by jingyz1
        reduction_ratio = get_env_var('VLMPRUNER_RATIO')
        assert isinstance(reduction_ratio, float)
        print(f"Will inference with reduction_ratio={reduction_ratio}")
        # 这里是新加的prune
        self.config.VLMPruner_config = {
            "K": 3,
            "image_token_start_index": 59, 
            "image_token_length": 4160,
            # "reduction_ratio": 0.778,
            "reduction_ratio": reduction_ratio,
            "retain_token_num_for_llava_next": 320,
            "pivot_image_token": 4,
            "pivot_text_token": 4,
            "threshold": 0.8,
            "token_batch": 16,
            "max_num_truncation": 64,
            "height": 24,
            "width": 24,
        }
        # self.config.VLMPruner_config = None


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
        # print(f"in Qwen3VLTextModel forward, input_ids: {input_ids}, ")
        # print(f"inpts_embeds: {inputs_embeds}")
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

        # jingyz1 begin
        attention_mask_jyz = torch.ones_like(attention_mask)
        # jingyz1 end

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

        batch_size, seq_length = inputs_embeds.shape[:2]

        # decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            # print(f"layer_idx: {layer_idx}")
            # print(f"visual_pos_masks.shape:{visual_pos_masks.shape}" if visual_pos_masks is not None else "no visual_pos_masks")
            
            VLMPruner_config = self.config.VLMPruner_config
            if VLMPruner_config is not None:
                K = VLMPruner_config['K']  # pruned layer
                image_token_start_index = torch.argmax(visual_pos_masks[0].to(torch.uint8)).item()  if visual_pos_masks is not None else 0
                image_token_length = visual_pos_masks[0].sum().item() if visual_pos_masks is not None else 0
                VLMPruner_config['image_token_start_index'] = image_token_start_index
                VLMPruner_config['image_token_length'] = image_token_length
                # print(f"image_token_start_index: {image_token_start_index}, image_token_length: {image_token_length}")
                if decoder_layer.self_attn.layer_idx == K:
                    if seq_length > 1: # jingyz1
                        bbox = kwargs.pop('bbox', '')
                        img_size = kwargs.pop('img_size', '')

                        if PRUNE_CASE == 'PRUNE_BY_ATTN_MASK_REF_GT':
                            self.removed = bbox_reserve_rect_pruning_inv(
                                image_token_start_index, 
                                image_token_length, 
                                VLMPruner_config['reduction_ratio'],
                                bbox, img_size)

                        if PRUNE_CASE == 'PRUNE_BY_SHRINKING_REF_GT':
                            retained = bbox_reserve_rect_pruning(
                                image_token_start_index, 
                                image_token_length, 
                                VLMPruner_config['reduction_ratio'],
                                bbox, img_size)
                            retained = sorted(
                                list(range(image_token_start_index))
                                + retained
                                + list(range(image_token_start_index+image_token_length,seq_length)))
                            hidden_states = hidden_states[:,retained,:]
                            hidden_states = hidden_states.contiguous()
                            cache_position = cache_position[retained]
                            position_ids = position_ids[:, :, retained] 
                            text_position_ids = text_position_ids[:, retained]
                        # device = hidden_states.device

                        # last_layer_state = layer_outputs[0]
                        # k_states = layer_outputs[-2]

                        # keep index
                        # retained_image_tokens_index, hidden_states = attn_sink_awared_random_pruning(
                        #     last_layer_state, 
                        #     image_token_start_index, 
                        #     image_token_length, 
                        #     VLMPruner_config['reduction_ratio'])
                        # retained_image_tokens_index, hidden_states = bbox_reserve_random_pruning(
                            # last_layer_state, 
                            # image_token_start_index, 
                            # image_token_length, 
                            # VLMPruner_config['reduction_ratio'],
                            # bbox, img_size)
                        # self.removed = bbox_reserve_rect_pruning_inv(
                        # retained_image_tokens_index, hidden_states = self.get_retained_image_token(self.config, last_layer_state, k_states)
                        # retained_image_tokens_index = retained_image_tokens_index.to(device)
                        # hidden_states = hidden_states.to(device)

                        # keep_indexs = torch.cat((torch.arange(image_token_start_index,device=device), retained_image_tokens_index, torch.arange(image_token_start_index+image_token_length,seq_length,device=device)))
                        # sort index
                        # keep_indexs = keep_indexs.sort().values

                        
                        # hidden_states = hidden_states[:,keep_indexs,:]
                        # hidden_states = hidden_states.contiguous()
                        # keep_indexs = retained_image_tokens_index.sort().values
                        # hidden_states[:,keep_indexs,:] = 0

                        
                        # keep_mask_global = torch.zeros_like(visual_pos_masks[0], dtype=torch.bool)
                        # keep_mask_global[keep_indexs] = True
                        
                        # canceled by jingyz1
                        # keep_visual_mask = keep_mask_global[visual_pos_masks[0]]
                        # print("keep_visual_mask.sum(): ", keep_visual_mask.sum().item())
                        # keep_visual_mask = keep_visual_mask.to(deepstack_visual_embeds[layer_idx].device)
                        # deepstack_visual_embeds[layer_idx] = deepstack_visual_embeds[layer_idx][keep_visual_mask,:]
                        # deepstack_visual_embeds[0] = deepstack_visual_embeds[0][keep_visual_mask,:]
                        # deepstack_visual_embeds[1] = deepstack_visual_embeds[1][keep_visual_mask,:]
                        # deepstack_visual_embeds[2] = deepstack_visual_embeds[2][keep_visual_mask,:]

                        # keep_indexs = keep_indexs.to(visual_pos_masks.device)
                        # visual_pos_masks= visual_pos_masks[:,keep_indexs]
                        
                        # del keep_mask_global
                        # del keep_visual_mask
                        # new_seq_length = keep_indexs.shape[0]

                        # keep_indexs = keep_indexs.to(cache_position.device)
                        # cache_position = cache_position[keep_indexs]  # TODO: fixed?
                        # jingyz1
                        # cache_position = cache_position[:new_seq_length]
                        # VLMPruner_config['runtime_position'] = new_seq_length

                        # keep_indexs = keep_indexs.to(position_ids.device)
                        # position_ids = position_ids[:, :, keep_indexs] 

                        # jingyz1
                        # keep_indexs = keep_indexs.to(text_position_ids.device)
                        # text_position_ids = text_position_ids[:, keep_indexs]

                if PRUNE_CASE != 'PRUNE_BY_ATTN_MASK_REF_GT':
                    position_embeddings = self.rotary_emb(hidden_states, position_ids)

            if PRUNE_CASE == 'PRUNE_BY_ATTN_MASK' and layer_idx == K:
                attention_mask_jyz[0,self.removed] = 0
                attention_mask_jyz = create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask_jyz,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )

            if PRUNE_CASE != 'PRUNE_BY_ATTN_MASK_REF_GT':
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask if layer_idx < K else attention_mask_jyz,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = layer_outputs[0]

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
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
        # print("in _deepstack_process")
        # print(f"hidden_states shape: {hidden_states.shape}")
        # print(f"visual_pos_masks shape: {visual_pos_masks.shape}")
        # print(f"visual_embeds shape: {visual_embeds.shape}")
        # print(f"hidden_states: {hidden_states}")
        # print(f"visual_pos_masks: {visual_pos_masks}")
        # print(f"visual_embeds: {visual_embeds}")
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    #############添加的辅助函数###############
    def calculate_similarity(
        self, 
        sim, 
        dist_to_selected
    ):
        spatial_reward = torch.amin(dist_to_selected, dim=1, keepdim=True) * self.norm_factor 
        sim = sim * (1.0 + 0.5 * spatial_reward) 
        return sim

    def get_spatial_pos(self, indices, width):
        y = indices // width
        x = indices % width
        return torch.stack([x, y], dim=-1).float()

    def get_retained_image_token(self, config: Qwen3VLConfig, last_layer_state: torch.Tensor, any_states: torch.Tensor) -> torch.Tensor:
        VLMPruner_config = config.VLMPruner_config
        reduction_ratio = VLMPruner_config['reduction_ratio']

        VLMPruner_config = config.VLMPruner_config
        image_token_start_index = VLMPruner_config['image_token_start_index']
        image_token_length = VLMPruner_config['image_token_length']
        print("image_token_start_index: ", image_token_start_index)
        print("image_token_length: ", image_token_length)
        pivot_image_token = VLMPruner_config['pivot_image_token']

        height = VLMPruner_config['height']
        width = VLMPruner_config['width']
        self.norm_factor = 1.0 / (math.sqrt(height ** 2 + width ** 2))

        retain_token_num = int(image_token_length * (1 - reduction_ratio))
        
        any_states = any_states.permute(0, 2, 1, 3).reshape(any_states.shape[0], any_states.shape[2], -1)
        k_states_image_token = any_states[0][image_token_start_index:image_token_start_index + image_token_length, :]  # [N, D]
        k_states_image_token = k_states_image_token.detach()
        device = k_states_image_token.device
        # Step 1: 
        k_states_image_token_norm = torch.norm(k_states_image_token, p=1, dim=-1)
        pivot_indices = torch.zeros(pivot_image_token, dtype=torch.long, device=device)
        pivot_indices[0] = k_states_image_token_norm.argmax()


        print(f"k_states_image_token:{k_states_image_token.shape}")
        print(f"pivot_image_token:{pivot_image_token}")
        for i in range(1, pivot_image_token):
            print(f"k_states_image_token[pivot_indices[:i]]:{k_states_image_token[pivot_indices[:i]].shape}")
            x1 = k_states_image_token.float()
            x2 = k_states_image_token[pivot_indices[:i]].float()
            dists = torch.cdist(x1, x2)
            min_dists = dists.min(dim=1).values
            pivot_indices[i] = min_dists.argmax()

        selected = pivot_indices.clone().tolist()
        threshold = VLMPruner_config['threshold'] - 0.1 
        batch_size = VLMPruner_config['token_batch'] 

        pre_num = 0
        tokens_state = last_layer_state[0, torch.arange(image_token_length, device=device) + image_token_start_index, :]
        last_channel_var = tokens_state.var(dim=0)  # [image_token_length, 1]
        topk_indices = torch.topk(last_channel_var, k=256).indices  # [k]
        tokens_state_screen = tokens_state[:, topk_indices]  # [k, feature_dim]
        print("tokens_state_screen shape: ", tokens_state_screen.shape)

        norm_tokens = torch.nn.functional.normalize(tokens_state_screen, p=2, dim=-1)
        sim_all = torch.mm(norm_tokens, norm_tokens.t())
        # sim_all = torch.nn.functional.cosine_similarity(
        #     tokens_state_screen.unsqueeze(1),  # [k, 1, feature_dim]
        #     tokens_state_screen.unsqueeze(0),  # [1, k, feature_dim]
        #     dim=-1
        # )  # [k, k]
        spatial_all = torch.cdist(self.get_spatial_pos(torch.arange(image_token_length, device=device), width),
                                    self.get_spatial_pos(torch.arange(image_token_length, device=device), width))
        # Step 2
        while len(selected) < retain_token_num:
            if pre_num == len(selected):
                break
            pre_num = len(selected)
            threshold = threshold + 0.1
            all_indices = torch.arange(image_token_length, device=device)
            mask = torch.ones(image_token_length, dtype=torch.bool, device=device)
            mask[torch.tensor(selected, device=device)] = False
            candidate_indices = all_indices[mask]
            if len(candidate_indices) == 0:
                break  
            selected_indices_tensor = torch.tensor(selected, device=device)
            sim_matrix = self.calculate_similarity(
                sim_all[candidate_indices][:, selected_indices_tensor],
                spatial_all[candidate_indices][:, selected_indices_tensor]
            )
            max_similarities = sim_matrix.max(dim=1).values
            nondup_scores = 1.0 - max_similarities
            sorted_idx = torch.argsort(nondup_scores, descending=True)

            sorted_candidate_indices = candidate_indices[sorted_idx]
            for i in range(0, len(sorted_idx), batch_size):
                if i == batch_size and pre_num == len(selected):
                    break
                batch_indices = sorted_candidate_indices[i:i+batch_size]
                batch_sims = self.calculate_similarity(
                    sim_all[batch_indices][:, torch.tensor(selected, device=device)],
                    spatial_all[batch_indices][:, torch.tensor(selected, device=device)]
                )
                max_sims, _ = batch_sims.max(dim=1)
                mask = max_sims < threshold
                valid_indices = batch_indices[mask]
                if len(selected) + len(valid_indices) > retain_token_num:
                    mask_idx = torch.topk(1.0 - max_sims, retain_token_num - len(selected)).indices
                    valid_indices = batch_indices[mask_idx]
                if len(valid_indices) > 0:
                    selected.extend(valid_indices.tolist())
                    if len(selected) >= retain_token_num:
                        break
        # Step 3
        selected_set = set(selected)
        all_image_indices = torch.arange(image_token_length, device=device)
        discarded_mask = ~torch.isin(all_image_indices, torch.tensor(list(selected_set), device=device))
        discarded_indices = all_image_indices[discarded_mask]

        if len(discarded_indices) > 0:
            discarded_global = image_token_start_index + discarded_indices
            discarded_vecs = last_layer_state[0, discarded_global, :]
            discarded_sims = sim_all[discarded_indices][:, torch.tensor(selected, device=device)]
            max_sims, most_similar_idx = discarded_sims.max(dim=1)
            one_hot = torch.zeros_like(discarded_sims)
            one_hot.scatter_(1, most_similar_idx.unsqueeze(1), 1)
            weights = discarded_sims * one_hot
            weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-8) 
            aggregated = torch.mm(weights.T, discarded_vecs)  # [S, D]
            update_mask = weights.sum(dim=0) > 0  # [S]
            pivot_global_indices = image_token_start_index + torch.tensor(selected, device=device)[update_mask]
            last_layer_state[0, pivot_global_indices] = (
                0.3 * last_layer_state[0, pivot_global_indices] + 
                0.7 * aggregated[update_mask]
            )

        retained_image_tokens_index = torch.tensor(
            [image_token_start_index + i for i in sorted(selected)],
            device=device
        )
        return retained_image_tokens_index, last_layer_state


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
        self.language_model = Qwen3VLTextModelPrune._from_config(config.text_config)
        self.rope_deltas = None  # cache rope_deltas here

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

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

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

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


# 未做修改，仅为了替换后面Qwen3VLModel的textdecoder
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
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

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
