"""CPU invariants for isolated SparseVLM bookkeeping."""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import torch
from vlmeval.vlm.qwen3_vl_sparsevlm.token import image_blocks, physical_keep_indices, retain_indices, select_raters, select_scope
from vlmeval.vlm.qwen3_vl.model import (
    _estimate_histprune_prefill_flops,
    _estimate_llm_forward_flops,
    _estimate_sparsevlm_prefill_flops,
    _estimate_vision_forward_flops,
    _vision_attention_segment_lengths,
)

class TestSparseVLM(unittest.TestCase):
    def test_non_contiguous_image_blocks_and_scopes(self):
        mask = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 1]], dtype=torch.bool)
        grid = torch.tensor([[1, 2, 2], [1, 1, 4]]) # merge=2 -> 1 and 1 is intentionally mismatched below
        # A realistic heterogeneous layout with counts 2 and 4 at merge=1.
        grid = torch.tensor([[1, 1, 2], [1, 1, 4]])
        blocks = image_blocks(mask, grid, 1)
        self.assertEqual([x.tolist() for x in blocks], [[1, 2], [4, 6, 7, 8]])
        self.assertEqual(select_scope(blocks, "history").tolist(), [1, 2])
        self.assertEqual(select_scope(blocks, "current").tolist(), [4, 6, 7, 8])

    def test_retain_one_is_identity_and_kv_indices_are_sorted(self):
        scores = torch.tensor([.1, .9, .4])
        rel = retain_indices(scores, 1.0)
        self.assertEqual(rel.tolist(), [0, 1, 2])
        keep = physical_keep_indices(8, torch.tensor([1, 4, 6]), rel)
        self.assertEqual(keep.tolist(), list(range(8)))

    def test_raters_and_four_history_blocks(self):
        hidden = torch.randn(1, 12, 4)
        visual = torch.tensor([1, 3, 5, 7, 9])
        text = torch.tensor([0, 2, 4, 6, 8, 10, 11])
        self.assertGreaterEqual(select_raters(hidden, visual, text).numel(), 1)
        mask = torch.zeros(1, 12, dtype=torch.bool); mask[0, visual] = True
        grid = torch.tensor([[1,1,1]] * 5)
        self.assertEqual(len(image_blocks(mask, grid, 1)), 5) # four history + current

    def test_segmented_flops_counts_trim_after_the_pruning_layer(self):
        cfg = SimpleNamespace(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                              num_key_value_heads=4, head_dim=16, num_hidden_layers=8, vocab_size=256)
        model = SimpleNamespace(config=SimpleNamespace(text_config=cfg))
        stats = {
            "seq_tokens_before": 100,
            "layer_prunes": [
                {"layer_idx": 1, "seq_tokens_before": 100, "seq_tokens_after": 50},
                {"layer_idx": 4, "seq_tokens_before": 50, "seq_tokens_after": 25},
            ],
        }
        expected = (
            _estimate_llm_forward_flops(model, 100, 100, num_layers=2)
            + _estimate_llm_forward_flops(model, 50, 50, num_layers=3)
            + _estimate_llm_forward_flops(model, 25, 25, num_layers=3)
        )
        self.assertEqual(_estimate_sparsevlm_prefill_flops(model, stats, 100), expected)

    def test_histprune_flops_counts_trim_before_the_drop_layer(self):
        cfg = SimpleNamespace(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                              num_key_value_heads=4, head_dim=16, num_hidden_layers=8, vocab_size=256)
        model = SimpleNamespace(config=SimpleNamespace(text_config=cfg))
        stats = {
            "prune_applied": True,
            "drop_layer": 3,
            "sequence_length_before": 100,
            "sequence_length_after": 40,
        }
        expected = (
            _estimate_llm_forward_flops(model, 100, 100, num_layers=3)
            + _estimate_llm_forward_flops(model, 40, 40, num_layers=5)
        )
        self.assertEqual(_estimate_histprune_prefill_flops(model, stats, 100), expected)

    def test_vision_attention_uses_per_grid_chunks(self):
        vision_cfg = SimpleNamespace(hidden_size=64, intermediate_size=128, num_heads=4, depth=2)
        model = SimpleNamespace(config=SimpleNamespace(vision_config=vision_cfg))
        grid = torch.tensor([[1, 2, 3], [2, 1, 4]])
        chunks = _vision_attention_segment_lengths(image_grid_thw=grid)
        self.assertEqual(chunks, [6, 4, 4])
        segmented = _estimate_vision_forward_flops(model, chunks)
        collapsed = _estimate_vision_forward_flops(model, sum(chunks))
        self.assertLess(segmented, collapsed)

if __name__ == "__main__": unittest.main()
