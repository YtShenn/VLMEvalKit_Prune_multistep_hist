from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped-query KV heads to attention heads."""
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@dataclass
class GUIKVConfig:
    max_capacity_prompt: int | None = None
    total_keep_ratio: float = 0.40
    window_size: int = 8
    kernel_size: int = 5
    pooling: str = "avgpool"
    alpha: float = 2.0
    temperature: float = 3.5
    merge: str | None = None


class GUIKVCluster:
    """GUI-KV cache compressor following SalesforceAIResearch/GUI-KV.

    The implementation keeps the official scoring structure: recent-window
    attention scores, spatial saliency from hidden-state L2 norms, and temporal
    redundancy from projecting previous image keys onto the current image key
    subspace. It is intentionally self-contained for the qwen3_vl_guikv backend.
    """

    def __init__(
        self,
        window_size: int = 8,
        max_capacity_prompt: int | None = None,
        total_keep_ratio: float = 0.40,
        kernel_size: int = 5,
        pooling: str = "avgpool",
        merge: str | None = None,
        alpha: float = 2.0,
        temperature: float = 3.5,
        vision_start_idx: list[int] | None = None,
        vision_end_idx: list[int] | None = None,
    ) -> None:
        self.window_size = int(window_size)
        self.max_capacity_prompt = int(max_capacity_prompt) if max_capacity_prompt is not None else None
        self.total_keep_ratio = float(total_keep_ratio)
        self.kernel_size = int(kernel_size)
        self.pooling = pooling
        self.merge = merge
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.vision_start_idx = list(vision_start_idx or [])
        self.vision_end_idx = list(vision_end_idx or [])
        self.kept_indices = None

    def reset(
        self,
        window_size: int | None = None,
        max_capacity_prompt: int | None = None,
        total_keep_ratio: float | None = None,
        kernel_size: int | None = None,
        pooling: str | None = None,
        merge: str | None = None,
        alpha: float | None = None,
        temperature: float | None = None,
        vision_start_idx: list[int] | None = None,
        vision_end_idx: list[int] | None = None,
    ) -> None:
        if window_size is not None:
            self.window_size = int(window_size)
        if max_capacity_prompt is not None:
            self.max_capacity_prompt = int(max_capacity_prompt)
        if total_keep_ratio is not None:
            self.total_keep_ratio = float(total_keep_ratio)
        if kernel_size is not None:
            self.kernel_size = int(kernel_size)
        if pooling is not None:
            self.pooling = pooling
        self.merge = merge
        if alpha is not None:
            self.alpha = float(alpha)
        if temperature is not None:
            self.temperature = float(temperature)
        if vision_start_idx is not None:
            self.vision_start_idx = list(vision_start_idx)
        if vision_end_idx is not None:
            self.vision_end_idx = list(vision_end_idx)
        self.kept_indices = None

    def _pool(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling == "avgpool":
            return F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        if self.pooling == "maxpool":
            return F.max_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)
        raise ValueError(f"Unsupported GUI-KV pooling method: {self.pooling}")

    def _valid_vision_ranges(self, q_len: int) -> tuple[list[int], list[int]]:
        starts, ends = [], []
        for s, e in zip(self.vision_start_idx, self.vision_end_idx):
            s, e = int(s), int(e)
            if 0 <= s < e <= q_len:
                starts.append(s)
                ends.append(e)
        return starts, ends

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_key_value_groups: int,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        score_key_states = key_states
        if key_states.shape[1] != query_states.shape[1]:
            score_key_states = repeat_kv(key_states, num_key_value_groups)

        window_size = min(max(1, int(self.window_size)), max(1, q_len - 1))
        if self.max_capacity_prompt is None or self.max_capacity_prompt <= 0:
            max_capacity = int(round(q_len * max(0.0, min(1.0, self.total_keep_ratio))))
        else:
            max_capacity = int(self.max_capacity_prompt)
        max_capacity = max(window_size + 2, min(q_len, max_capacity))
        if q_len <= max_capacity or max_capacity <= window_size + 1:
            return key_states, value_states

        attn_weights = torch.matmul(query_states[..., -window_size:, :], score_key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(head_dim)

        mask = torch.full(
            (window_size, window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device,
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attn_weights[:, :, -window_size:, -window_size:] += mask[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_sum = attn_weights[:, :, -window_size:, :-window_size].sum(dim=-2)
        attn_cache = self._pool(attn_weights_sum)

        starts, ends = self._valid_vision_ranges(q_len - window_size)
        if hidden_states is not None and starts and ends:
            cur_start, cur_end = starts[-1], ends[-1]
            visual_hidden_states = hidden_states[:, cur_start:cur_end, :]
            if visual_hidden_states.numel() > 0:
                importance_scores = torch.norm(visual_hidden_states, p=2, dim=-1)
                normalized_scores = torch.zeros_like(importance_scores)
                temperature = max(float(self.temperature), 1e-6)
                for batch_idx in range(importance_scores.shape[0]):
                    batch_scores = importance_scores[batch_idx]
                    standardized = (batch_scores - batch_scores.mean()) / (batch_scores.std() + 1e-8)
                    normalized_scores[batch_idx] = torch.softmax(standardized / temperature, dim=0)
                hidden_scores = normalized_scores.unsqueeze(1) * self.alpha
                width = min(cur_end - cur_start, hidden_scores.shape[-1], attn_cache.shape[-1] - cur_start)
                if width > 0:
                    attn_cache[:, :, cur_start:cur_start + width] += hidden_scores[..., :width].to(attn_cache.device)

            if len(starts) > 1:
                self._apply_temporal_redundancy(
                    attn_cache,
                    score_key_states,
                    starts,
                    ends,
                    bsz,
                    num_heads,
                    max_capacity / q_len,
                )

        if kv_heads != num_heads:
            attn_cache = attn_cache.view(bsz, kv_heads, num_key_value_groups, -1).mean(dim=2)
        topk = max_capacity - window_size
        topk = min(max(1, topk), key_states.shape[-2] - window_size)
        indices = attn_cache.topk(topk, dim=-1).indices
        self.kept_indices = indices
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
        v_past_compress = value_states[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]
        return torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)

    def _apply_temporal_redundancy(
        self,
        attn_cache: torch.Tensor,
        key_states: torch.Tensor,
        starts: list[int],
        ends: list[int],
        bsz: int,
        num_heads: int,
        budget: float,
    ) -> None:
        cur_start, cur_end = starts[-1], ends[-1]
        last_image_keys = key_states[:, :, cur_start:cur_end, :]
        if last_image_keys.shape[-2] == 0:
            return
        for batch_idx in range(bsz):
            for head_idx in range(num_heads):
                current_keys = last_image_keys[batch_idx, head_idx].float()
                rank = min(32, current_keys.shape[0], current_keys.shape[1])
                if rank <= 0:
                    continue
                try:
                    q_basis, _ = torch.linalg.qr(current_keys.T, mode="reduced")
                except RuntimeError:
                    continue
                q_basis = q_basis[:, :rank]
                all_residual_norms = []
                all_positions = []
                for img_idx in range(len(starts) - 1):
                    prev_start, prev_end = starts[img_idx], ends[img_idx]
                    if prev_end <= prev_start:
                        continue
                    prev_keys = key_states[batch_idx, head_idx, prev_start:prev_end, :].float()
                    projections = q_basis.T @ prev_keys.T
                    prev_projected = (q_basis @ projections).T
                    residual_norms = torch.norm(prev_keys - prev_projected, p=2, dim=-1)
                    all_residual_norms.append(residual_norms)
                    all_positions.append(torch.arange(prev_start, prev_end, device=residual_norms.device))
                if not all_residual_norms:
                    continue
                residual_norms = torch.cat(all_residual_norms)
                positions = torch.cat(all_positions)
                num_to_zero = int((1.0 - budget) * residual_norms.numel())
                if num_to_zero <= 0:
                    continue
                num_to_zero = min(num_to_zero, residual_norms.numel())
                _, redundant_indices = torch.topk(residual_norms, num_to_zero, largest=False)
                positions_to_zero = positions[redundant_indices]
                positions_to_zero = positions_to_zero[positions_to_zero < attn_cache.shape[-1]]
                if positions_to_zero.numel() > 0:
                    attn_cache[batch_idx, head_idx, positions_to_zero] = 0
