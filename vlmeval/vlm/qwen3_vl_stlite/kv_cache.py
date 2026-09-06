from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .config import STLiteConfig


@dataclass
class ImageTokenRange:
    start: int
    end: int
    t: int
    h: int
    w: int

    @property
    def count(self) -> int:
        return max(0, int(self.end) - int(self.start))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _normalize_01(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    lo = scores.amin(dim=-1, keepdim=True)
    hi = scores.amax(dim=-1, keepdim=True)
    return (scores - lo) / (hi - lo + 1e-6)


def _fallback_hw(count: int) -> tuple[int, int]:
    side = int(math.sqrt(max(1, count)))
    while side > 1 and count % side != 0:
        side -= 1
    return side, max(1, count // max(1, side))


def image_token_ranges_from_inputs(
    input_ids: torch.Tensor | None,
    mm_token_type_ids: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    image_token_id: int | None,
    attention_mask: torch.Tensor | None = None,
) -> list[ImageTokenRange]:
    if input_ids is None:
        return []
    if attention_mask is not None and attention_mask.ndim == 2:
        valid = attention_mask[0].bool()
    else:
        valid = torch.ones_like(input_ids[0], dtype=torch.bool)
    if mm_token_type_ids is not None:
        modality = mm_token_type_ids[0].to(input_ids.device)
        mask = (modality == 1) & valid
    elif image_token_id is not None:
        mask = (input_ids[0] == int(image_token_id)) & valid
    else:
        return []

    positions = torch.nonzero(mask, as_tuple=False).flatten()
    if positions.numel() == 0:
        return []

    starts, ends = [], []
    start = int(positions[0])
    prev = int(positions[0])
    for pos_t in positions[1:]:
        pos = int(pos_t)
        if pos != prev + 1:
            starts.append(start)
            ends.append(prev + 1)
            start = pos
        prev = pos
    starts.append(start)
    ends.append(prev + 1)

    grids: list[tuple[int, int, int]] = []
    if image_grid_thw is not None:
        raw = image_grid_thw.detach().cpu().tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw
        grids = [(max(1, int(t)), max(1, int(h)), max(1, int(w))) for t, h, w in raw]

    if grids:
        observed = sum(e - s for s, e in zip(starts, ends))
        raw_total = sum(t * h * w for t, h, w in grids)
        divisor = 1
        if observed > 0 and raw_total != observed and raw_total % observed == 0:
            divisor = max(1, int(round(math.sqrt(raw_total // observed))))
        counts = [max(1, t * max(1, h // divisor) * max(1, w // divisor)) for t, h, w in grids]
        if sum(counts) == observed and len(starts) == 1 and len(counts) > 1:
            base = starts[0]
            starts, ends = [], []
            for c in counts:
                starts.append(base)
                base += int(c)
                ends.append(base)
        if len(grids) != len(starts):
            grids = []

    ranges: list[ImageTokenRange] = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        count = max(0, int(e) - int(s))
        if count <= 0:
            continue
        if grids:
            t, h_raw, w_raw = grids[idx]
            denom = max(1, (t * h_raw * w_raw) // count)
            merge = max(1, int(round(math.sqrt(denom))))
            h, w = max(1, h_raw // merge), max(1, w_raw // merge)
            if t * h * w != count:
                h, w = _fallback_hw(count)
                t = 1
        else:
            t = 1
            h, w = _fallback_hw(count)
        ranges.append(ImageTokenRange(int(s), int(e), int(t), int(h), int(w)))
    return ranges


def compute_css_scores(hidden_states: torch.Tensor, ranges: list[ImageTokenRange], kernel_size: int = 3) -> torch.Tensor:
    bsz, seq_len, _ = hidden_states.shape
    scores = torch.zeros((bsz, seq_len), device=hidden_states.device, dtype=torch.float32)
    radius = max(1, int(kernel_size) // 2)
    for img in ranges:
        count = img.count
        if count <= 0 or img.end > seq_len:
            continue
        states = hidden_states[:, img.start:img.end, :].float()
        if img.t * img.h * img.w != count:
            t, h, w = 1, *_fallback_hw(count)
        else:
            t, h, w = img.t, img.h, img.w
        grid = F.normalize(states, p=2, dim=-1).reshape(bsz, t, h, w, -1)
        sim_sum = torch.zeros((bsz, t, h, w), device=states.device)
        sim_count = torch.zeros((bsz, t, h, w), device=states.device)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                y0, y1 = max(0, dy), h + min(0, dy)
                x0, x1 = max(0, dx), w + min(0, dx)
                if y0 >= y1 or x0 >= x1:
                    continue
                center = grid[:, :, y0:y1, x0:x1, :]
                neigh = grid[:, :, y0 - dy:y1 - dy, x0 - dx:x1 - dx, :]
                sim = (center * neigh).sum(dim=-1)
                sim_sum[:, :, y0:y1, x0:x1] += sim
                sim_count[:, :, y0:y1, x0:x1] += 1
        css = 1.0 - (sim_sum / sim_count.clamp_min(1.0))
        css = _normalize_01(css.reshape(bsz, count))
        scores[:, img.start:img.end] = css
    return scores


def compute_official_hidden_norm_scores(
    hidden_states: torch.Tensor,
    image_range: ImageTokenRange,
    kernel_size: int = 3,
) -> torch.Tensor:
    """Public ST-Lite repo CSS branch: current-frame hidden norm + 3x3 smoothing + softmax."""
    bsz, seq_len, _ = hidden_states.shape
    scores = torch.zeros((bsz, seq_len), device=hidden_states.device, dtype=torch.float32)
    if image_range.count <= 0 or image_range.end > seq_len:
        return scores

    visual_hidden_states = hidden_states[:, image_range.start:image_range.end, :].float()
    importance_scores = torch.norm(visual_hidden_states, p=2, dim=-1)
    bsz, count = importance_scores.shape
    side = int(math.sqrt(count))
    if side * side == count:
        scores_2d = importance_scores.view(bsz, 1, side, side)
        smoothed_scores = F.avg_pool2d(scores_2d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        importance_scores = importance_scores + smoothed_scores.view(bsz, count)

    normalized_scores = torch.zeros_like(importance_scores)
    for batch_idx in range(bsz):
        batch_scores = importance_scores[batch_idx]
        standardized = (batch_scores - batch_scores.mean()) / (batch_scores.std() + 1e-8)
        normalized_scores[batch_idx] = torch.softmax(standardized, dim=0)

    scores[:, image_range.start:image_range.end] = normalized_scores
    return scores


class STLiteKVCluster:
    """ST-Lite scoring and flat per-layer KV compression for Qwen3-VL."""

    def __init__(self, config: STLiteConfig | None = None, vision_ranges: list[ImageTokenRange] | None = None) -> None:
        self.config = config or STLiteConfig()
        self.vision_ranges = list(vision_ranges or [])
        self.kept_indices: torch.Tensor | None = None
        self.last_stats: dict = {}

    def reset(self, config: STLiteConfig, vision_ranges: list[ImageTokenRange] | None = None) -> None:
        self.config = config
        self.vision_ranges = list(vision_ranges or [])
        self.kept_indices = None
        self.last_stats = {}

    def _pool(self, scores: torch.Tensor) -> torch.Tensor:
        kernel = max(1, int(self.config.css_kernel_size))
        if self.config.pooling == "maxpool":
            return F.max_pool1d(scores, kernel_size=kernel, padding=kernel // 2, stride=1)
        return F.avg_pool1d(scores, kernel_size=kernel, padding=kernel // 2, stride=1)

    def _capacity(self, q_len: int, window_size: int) -> int:
        if self.config.max_capacity_prompt is not None:
            cap = int(self.config.max_capacity_prompt)
        else:
            cap = int(round(q_len * float(self.config.keep_ratio)))
        cap = max(int(self.config.min_tokens), cap)
        cap = max(window_size + 1, cap)
        return min(q_len, cap)

    def _valid_ranges(self, seq_limit: int) -> list[ImageTokenRange]:
        return [r for r in self.vision_ranges if 0 <= r.start < r.end <= seq_limit]

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_key_value_groups: int,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del attention_mask
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_query_heads, q_len, head_dim = query_states.shape
        kv_heads = key_states.shape[1]
        score_key_states = key_states
        if kv_heads != num_query_heads:
            score_key_states = repeat_kv(key_states, num_key_value_groups)

        window_size = min(max(1, int(self.config.window_size)), max(1, q_len - 1))
        max_capacity = self._capacity(q_len, window_size)
        if q_len <= max_capacity:
            self.last_stats = {"compressed_seq_len": int(q_len), "original_seq_len": int(q_len), "skipped": True}
            return key_states, value_states

        attn_weights = torch.matmul(query_states[..., -window_size:, :], score_key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(head_dim)
        causal = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(causal.size(-1), device=attn_weights.device)
        causal.masked_fill_(mask_cond < (mask_cond + 1).view(causal.size(-1), 1), 0)
        attn_weights[:, :, -window_size:, -window_size:] += causal[None, None, :, :]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_cache = attn_weights[:, :, :, :-window_size].sum(dim=-2)
        attn_cache = self._pool(attn_cache)

        ranges = self._valid_ranges(q_len - window_size)
        history_ranges = ranges[:-1] if len(ranges) > 1 else []
        current_range = ranges[-1] if ranges else None
        scoring_start = None
        if hidden_states is not None and ranges and (self.config.use_css or self.config.use_tsg):
            import time

            scoring_start = time.perf_counter()
        if hidden_states is not None and current_range is not None and self.config.use_css:
            css = compute_official_hidden_norm_scores(
                hidden_states,
                current_range,
                self.config.css_kernel_size,
            )[:, : q_len - window_size]
            attn_cache = attn_cache + float(self.config.alpha) * css.unsqueeze(1).to(attn_cache.dtype)

        if hidden_states is not None and self.config.use_tsg and history_ranges and current_range is not None:
            curr = F.normalize(hidden_states[:, current_range.start:current_range.end, :].float(), p=2, dim=-1)
            if curr.numel() > 0:
                budget = float(max_capacity / max(1, q_len))
                all_max_similarities = []
                all_positions = []
                for hist in history_ranges:
                    prev = F.normalize(hidden_states[:, hist.start:hist.end, :].float(), p=2, dim=-1)
                    if prev.numel() == 0:
                        continue
                    max_sim = torch.matmul(prev, curr.transpose(1, 2)).max(dim=-1).values
                    all_max_similarities.append(max_sim)
                    all_positions.append(torch.arange(hist.start, hist.end, device=hidden_states.device))
                if all_max_similarities:
                    for batch_idx in range(bsz):
                        batch_similarities = torch.cat([sim[batch_idx] for sim in all_max_similarities])
                        batch_positions = torch.cat(all_positions)
                        num_tokens_to_zero = int((1.0 - budget) * batch_similarities.numel())
                        if num_tokens_to_zero > 0:
                            _, redundant_indices = torch.topk(
                                batch_similarities,
                                min(num_tokens_to_zero, batch_similarities.numel()),
                                largest=True,
                            )
                            positions_to_zero = batch_positions[redundant_indices]
                            attn_cache[batch_idx, :, positions_to_zero] = 0
        scoring_overhead_s = 0.0
        if scoring_start is not None:
            import time

            scoring_overhead_s = time.perf_counter() - scoring_start

        if kv_heads != num_query_heads:
            attn_cache = attn_cache.view(bsz, kv_heads, num_key_value_groups, -1).mean(dim=2)
        topk = min(max_capacity - window_size, key_states.shape[-2] - window_size)
        topk = max(1, int(topk))
        indices = attn_cache.topk(topk, dim=-1).indices.sort(dim=-1).values
        self.kept_indices = indices
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past = key_states[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
        v_past = value_states[:, :, :-window_size, :].gather(dim=2, index=gather_indices)
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]
        compressed_k = torch.cat([k_past, k_cur], dim=2)
        compressed_v = torch.cat([v_past, v_cur], dim=2)

        # Count history visual tokens after selection, including any that fall
        # into the mandatory recent window. This is used only for ST-Lite
        # reporting; it does not change the global budget.
        history_mask = torch.zeros(q_len, dtype=torch.bool, device=key_states.device)
        for hist in history_ranges:
            history_mask[hist.start:hist.end] = True
        selected = torch.cat(
            [indices[0, 0], torch.arange(q_len - window_size, q_len, device=key_states.device)]
        )
        history_visual_after = int(history_mask[selected].sum().item())

        visual_total = sum(r.count for r in ranges)
        current_visual = current_range.count if current_range is not None else 0
        history_visual = sum(r.count for r in history_ranges)
        self.last_stats = {
            "original_seq_len": int(q_len),
            "compressed_seq_len": int(compressed_k.shape[-2]),
            "window_size": int(window_size),
            "keep_ratio": float(compressed_k.shape[-2] / max(1, q_len)),
            "layer_keep_tokens": int(compressed_k.shape[-2]),
            "history_visual_tokens": int(history_visual),
            "history_visual_tokens_after": int(history_visual_after),
            "history_visual_retention_rate": float(
                history_visual_after / max(1, history_visual)
            ),
            "current_visual_tokens": int(current_visual),
            "visual_tokens": int(visual_total),
            "text_action_tokens": int(max(0, q_len - visual_total)),
            "scoring_overhead_s": float(scoring_overhead_s),
        }
        return compressed_k, compressed_v
