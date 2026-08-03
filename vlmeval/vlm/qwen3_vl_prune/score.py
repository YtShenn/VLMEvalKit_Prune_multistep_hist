import torch
import os
import math

VERSION = os.getenv("USE_VERSION", "1_0")
V2_0 = VERSION == "2_0"

RETAIN_TOKN = int(os.getenv("RETAIN_TOKN", "192"))
# qwen3vl用剪枝率比较合适
RETAIN_RATIO = float(os.getenv("RETAIN_RATIO", "0.5"))

layer_dict = {3:0,6:1,15:2}

sparse_token_list_192 = [300, 200, 110] if not V2_0 else [300, 200, 118]       # 2*576  4*300 10*200  16*110
sparse_token_list_128 = [303, 110, 36] if not V2_0 else [238, 108, 60]
sparse_token_list_96 = [238, 48, 26] if not V2_0 else [246, 54, 28]
sparse_token_list_64 = [66, 30, 17] if not V2_0 else [66, 34, 20]
sparse_ratio_dict = {
    0.5: [0.8, 0.8, 0.8],   # 每次留下已有的80%，最终是0.512
    0.2: [0.6, 0.6, 0.6],
    1.0: [1.0, 1.0, 1.0],
    # 0.25: [0.6, 0.4, 0.25], 
}

# sparse_token_dict = {
#     192: sparse_token_list_192,
#     128: sparse_token_list_128,
#     96 : sparse_token_list_96,
#     64 : sparse_token_list_64
# }

def attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start, t_token_idx, layer_idx):
    '''
    self_attn_weights: [B, H, L, L]
    '''
    if self_attn_weights is None:
        # Fallback: no attention logits available (e.g., block attention). Disable pruning.
        relation_vis_text = torch.zeros(1, v_token_num, device="cuda" if torch.cuda.is_available() else "cpu")
        mask = torch.ones_like(relation_vis_text, dtype=bool) if v_token_num != 0 else torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
        return mask, s_flag, relation_vis_text
    if self_attn_weights.shape[-1] == v_token_num and self_attn_weights.shape[-2] == t_token_idx[1].numel():
        relation_vis_text = self_attn_weights.mean(1)
        relation_vis_text = relation_vis_text.mean(1)
    else:
        self_attn_weights = self_attn_weights.mean(1)
        t_token_idx = t_token_idx[1] + text_token_start
        relation_vis_text = self_attn_weights[:, t_token_idx , v_token_start: v_token_start+v_token_num]
        relation_vis_text = relation_vis_text.mean(1)

    relation_vis = relation_vis_text
    s_flag = True       # s_flag controls whether token merge is needed.

    #Top-K 筛选，每层保留的数量是定好的
    # sparse_token_list = sparse_token_dict[RETAIN_TOKN]
    sparse_ratio_list = sparse_ratio_dict[RETAIN_RATIO]

    # 创建一个掩码，只有被选中的重要 Token 位置为 1
    if v_token_num != 0:
        mask = torch.zeros_like(relation_vis, dtype=bool)
        # print(f"layer_dict:{layer_dict}, layer_idx: {layer_idx}")
        # print(f"layer_dict[layer_idx]: {layer_dict[layer_idx]}")
        # print(f"sparse_ratio_list:{sparse_ratio_list}")
        _, indices = torch.topk(relation_vis, min(math.floor(sparse_ratio_list[layer_dict[layer_idx]]*v_token_num), v_token_num - 1), dim=1)
        mask[0][indices] = 1
    else:
        mask = torch.ones_like(relation_vis_text, dtype=bool)
        s_flag = False
    return mask, s_flag, relation_vis_text

def select_attn_head_by_sum(self_attn_weights, t_token_idx, v_token_start, text_token_start):
    if self_attn_weights.shape[-1] == (text_token_start - v_token_start) and self_attn_weights.shape[-2] == t_token_idx.numel():
        each_head_text_to_visual_attn = self_attn_weights[0]
    else:
        each_head_text_to_visual_attn = self_attn_weights[0][:, t_token_idx , v_token_start: text_token_start]
    # [28,text_token_num,visual_token_num] -> [28]
    sum_attn_per_head = each_head_text_to_visual_attn.sum((1,2))
    select_attn_head_idx = sum_attn_per_head.topk(14)[1]

    return self_attn_weights[:,select_attn_head_idx,:,:][:,:,:]

if __name__ == "__main__":

    self_attn_weights, v_token_start, v_token_num, text_token_start = torch.rand(4, 16, 1084, 1084), 36, 576, 700
    mask = attn_postprocess_topk(self_attn_weights, v_token_start, v_token_num, text_token_start)
    print(mask.shape)
