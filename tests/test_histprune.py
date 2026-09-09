"""CPU invariants for HistPrune-GUI reproduction/adaptation for Qwen3-VL."""
from __future__ import annotations

import unittest

import torch
from PIL import Image

from vlmeval.vlm.qwen3_vl_histprune.config import HistPruneConfig
from vlmeval.vlm.qwen3_vl_histprune.history import (
    allocate_history_budget, build_token_meta, select_from_rank_map, select_indices, uniform_indices,
)


class TestBudget(unittest.TestCase):
    def test_fixed_total_budget_and_capacity_redistribution(self):
        # Recent image has only 2 tokens, so its weighted surplus is redistributed.
        result = allocate_history_budget([2, 100, 100, 100], .4, [.9, .05, .03, .02])
        self.assertEqual(sum(result), round(.4 * 302))
        self.assertLessEqual(result[0], 2)
        self.assertTrue(all(0 <= got <= cap for got, cap in zip(result, [2, 100, 100, 100])))

    def test_zero_and_full(self):
        self.assertEqual(allocate_history_budget([4, 6], 0, [.4, .3]), [0, 0])
        self.assertEqual(allocate_history_budget([4, 6], 1, [.4, .3]), [4, 6])


class TestSelection(unittest.TestCase):
    def test_random_is_seed_reproducible(self):
        self.assertTrue(torch.equal(select_indices(30, 8, "random", 7, 0), select_indices(30, 8, "random", 7, 0)))

    def test_uniform_is_spread(self):
        chosen = uniform_indices(20, 5)
        self.assertTrue(torch.equal(chosen, torch.tensor([0, 4, 8, 12, 16])))

    def test_sobel_priority_and_fill(self):
        edge = torch.tensor([True, False, True, False, False])
        self.assertTrue(torch.equal(select_indices(5, 2, "sobel_foreground", 0, 0, edge), torch.tensor([0, 2])))
        self.assertEqual(select_indices(5, 4, "sobel_foreground", 0, 0, edge).numel(), 4)
        self.assertTrue(torch.equal(select_indices(5, 2, "sobel_background", 0, 0, edge), torch.tensor([1, 3])))

    def test_current_and_text_are_never_pruned(self):
        cfg = HistPruneConfig(history_keep_ratio=.5)
        ranks = torch.tensor([[-1, 1, 1, 0, 0, -1, -1]])
        keep, stats = select_from_rank_map(ranks, cfg)
        self.assertTrue(torch.all(keep[0, torch.tensor([0, 5, 6])]))
        self.assertEqual(sum(stats["actual"]), round(.5 * 4))


class TestRankMap(unittest.TestCase):
    def test_old_to_new_prompt_becomes_recent_first_ranks(self):
        cfg = HistPruneConfig()
        # 4 historical images of two LLM tokens plus a two-token current image.
        ids = torch.zeros((1, 14), dtype=torch.long)
        image_mask = torch.zeros((1, 14), dtype=torch.bool)
        image_mask[0, [1, 2, 4, 5, 7, 8, 10, 11, 12, 13]] = True
        grid = torch.tensor([[1, 2, 4]] * 5)
        images = [Image.new("RGB", (8, 8)) for _ in range(4)]
        meta = build_token_meta(ids, image_mask, grid, 2, cfg, images)
        self.assertEqual(meta["rank_map"][0, 1].item(), 3)
        self.assertEqual(meta["rank_map"][0, 10].item(), 0)
        self.assertEqual(meta["rank_map"][0, 12].item(), -1)  # current

    def test_current_only_has_no_meta(self):
        ids = torch.zeros((1, 2), dtype=torch.long)
        mask = torch.ones((1, 2), dtype=torch.bool)
        self.assertIsNone(build_token_meta(ids, mask, torch.tensor([[1, 2, 2]]), 2, HistPruneConfig(), []))


if __name__ == "__main__":
    unittest.main()
