from __future__ import annotations

import torch

from vlmeval.vlm.qwen3_vl_prumerge.model import truncate_history_steps
from vlmeval.vlm.qwen3_vl_prumerge.token import (
    aggregate_features,
    aggregate_visual_blocks,
    contiguous_visual_blocks,
    make_merge_plan,
    scope_block_indices,
    select_iqr,
)


def test_contiguous_blocks_keep_interleaving_boundaries():
    mask = torch.tensor([[False, True, True, False, True, True, True, False]])
    blocks = contiguous_visual_blocks(mask)
    assert [block.tolist() for block in blocks] == [[1, 2], [4, 5, 6]]


def test_iqr_selection_is_bounded_and_nonempty():
    scores = torch.tensor([0.0, 0.1, 0.2, 0.3, 10.0])
    keep = select_iqr(scores, min_keep=2, max_keep=3, multiplier=1.5)
    assert 2 <= keep.numel() <= 3
    assert keep.tolist() == sorted(keep.tolist())
    assert 4 in keep.tolist()


def test_merge_plan_preserves_selected_order_and_assigns_everything():
    features = torch.tensor([[1.0, 0.0], [0.8, 0.1], [0.0, 1.0], [0.1, 0.9]])
    plan = make_merge_plan(features, min_keep=2, max_keep=2, iqr_multiplier=1.5, variant="prumerge")
    merged = aggregate_features(features, plan)
    assert plan.keep.tolist() == sorted(plan.keep.tolist())
    assert plan.assignment.shape == (4,)
    assert merged.shape == (2, 2)
    assert torch.isfinite(merged).all()


def test_plus_adds_only_valid_uniform_supplements():
    features = torch.eye(8, dtype=torch.float32)
    regular = make_merge_plan(features, min_keep=2, max_keep=4, iqr_multiplier=1.5, variant="prumerge")
    plus = make_merge_plan(features, min_keep=2, max_keep=4, iqr_multiplier=1.5, variant="prumerge_plus")
    assert plus.keep.numel() >= regular.keep.numel()
    assert plus.keep.numel() <= 4


def test_blocks_never_mix_features_or_deepstack():
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    plans = [make_merge_plan(features[:2], min_keep=1, max_keep=1, iqr_multiplier=0.0, variant="prumerge"), None]
    merged = aggregate_visual_blocks(features, [2, 2], plans)
    deepstack = aggregate_visual_blocks(features + 10.0, [2, 2], plans)
    assert merged.shape[0] == 3
    assert torch.equal(merged[-2:], features[-2:])
    assert torch.allclose(deepstack[-2:], features[-2:] + 10.0)


def test_mock_prefill_sequence_and_deepstack_lengths_stay_aligned():
    # A model-free stand-in for the pre-layer-0 physical sequence surgery.
    sequence = torch.arange(18, dtype=torch.float32).view(1, 9, 2)
    visual_mask = torch.tensor([[False, True, True, False, True, True, False, False, False]])
    visual_positions = torch.where(visual_mask[0])[0]
    blocks = contiguous_visual_blocks(visual_mask)
    plans = [
        make_merge_plan(sequence[0, block], min_keep=1, max_keep=1, iqr_multiplier=0.0, variant="prumerge")
        for block in blocks
    ]
    lengths = [int(block.numel()) for block in blocks]
    merged_main = aggregate_visual_blocks(sequence[0, visual_positions], lengths, plans)
    merged_deepstack = aggregate_visual_blocks(sequence[0, visual_positions] + 100, lengths, plans)
    keep = torch.ones(sequence.shape[1], dtype=torch.bool)
    keep[visual_positions] = False
    keep[torch.cat([block.index_select(0, plan.keep) for block, plan in zip(blocks, plans)])] = True
    reduced_mask = visual_mask[:, keep]
    assert int(reduced_mask.sum()) == merged_main.shape[0] == merged_deepstack.shape[0]
    assert reduced_mask.shape[1] == int(keep.sum())


def test_history_is_limited_to_last_four_steps_with_attached_text():
    message = []
    for idx in range(6):
        message.extend([{"type": "text", "value": f"step-{idx}"}, {"type": "image", "value": f"{idx}.png"}])
    message.extend([{"type": "text", "value": "current"}, {"type": "image", "value": "current.png", "is_current": True}, {"type": "text", "value": "question"}])
    trimmed = truncate_history_steps(message, 4)
    values = [item["value"] for item in trimmed]
    assert "0.png" not in values and "1.png" not in values
    assert "2.png" in values and "5.png" in values and "current.png" in values
    assert "step-2" in values and "question" in values


def test_scope_isolated():
    assert scope_block_indices(3, "current") == {2}
    assert scope_block_indices(3, "history") == {0, 1}
    assert scope_block_indices(3, "all") == {0, 1, 2}
