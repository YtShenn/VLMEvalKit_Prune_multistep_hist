"""CPU-only invariants for GUIPruner-reproduction."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch
from PIL import Image

from vlmeval.vlm.qwen3_vl_guipruner.config import GUIPrunerConfig, resolve_current_keep_ratio
from vlmeval.vlm.qwen3_vl_guipruner.attention_patch import _trim_prefill_attention_mask
from vlmeval.vlm.qwen3_vl_guipruner.flops import correct_split_prefill_flops
from vlmeval.vlm.qwen3_vl_guipruner.ssp import select_stratified
from vlmeval.vlm.qwen3_vl_guipruner.tar import apply_tar, temporal_token_quotas
from vlmeval.vlm.qwen3_vl_guipruner.model import _history_and_current_indices


class TestTAR(unittest.TestCase):
    def test_empty_and_one_history(self):
        self.assertEqual(temporal_token_quotas(0, 100, 0.4, 0.2), [])
        self.assertEqual(temporal_token_quotas(1, 100, 0.4, 0.2), [40.0])

    def test_four_history_budget_and_gamma_one(self):
        quotas = temporal_token_quotas(4, 100, 0.4, 0.2)
        self.assertAlmostEqual(sum(quotas), math.floor(4 * 100 * 0.4))
        flat = temporal_token_quotas(4, 100, 0.4, 1.0)
        self.assertTrue(all(value == flat[0] for value in flat))
        self.assertLess(quotas[0], quotas[-1])  # oldest < newest

    def test_tar_never_touches_current_image(self):
        history = [Image.new("RGB", (320, 320), color="white") for _ in range(4)]
        current = Image.new("RGB", (640, 320), color="black")
        out, records = apply_tar(history, keep_ratio=0.4, decay=0.2, patch_size=16, spatial_merge_size=2)
        self.assertEqual(len(out), 4)
        self.assertEqual(current.size, (640, 320))
        self.assertAlmostEqual(sum(x.theoretical_token_quota for x in records), 4 * 100 * 0.4)
        self.assertTrue(all(x.actual_visual_tokens <= x.theoretical_token_quota + 1 for x in records))


class TestSSP(unittest.TestCase):
    def test_stratified_budget_order_and_disjointness(self):
        scores = torch.tensor([.1, .9, .8, .2, .7, .6, .5, .4, .3, .0])
        fg = torch.tensor([1, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool)
        result = select_stratified(scores, fg, keep_ratio=.7, rho=.3)
        self.assertEqual(result.budget, 7)
        self.assertEqual(result.final.numel(), result.budget)
        self.assertTrue(torch.all(result.final[1:] > result.final[:-1]))
        self.assertEqual(torch.unique(torch.cat((result.foreground, result.background, result.uniform))).numel(), result.budget)

    def test_history_indices_cannot_enter_ssp(self):
        # SSP accepts only the current-frame local score vector; history has no index space here.
        result = select_stratified(torch.arange(12.0), torch.zeros(12, dtype=torch.bool), .75, .3)
        self.assertTrue(torch.all(result.final < 12))

    def test_prefill_attention_mask_uses_the_same_selection_as_kv(self):
        keep = torch.tensor([0, 2, 3])
        mask_2d = torch.ones(1, 4)
        mask_4d = torch.arange(16.0).reshape(1, 1, 4, 4)
        self.assertEqual(tuple(_trim_prefill_attention_mask(mask_2d, keep, 4).shape), (1, 3))
        trimmed_4d = _trim_prefill_attention_mask(mask_4d, keep, 4)
        self.assertEqual(tuple(trimmed_4d.shape), (1, 1, 3, 3))
        self.assertTrue(torch.equal(trimmed_4d[0, 0], mask_4d[0, 0][keep][:, keep]))


class TestConfig(unittest.TestCase):
    def test_rejects_more_than_four_history_images(self):
        with self.assertRaises(ValueError):
            GUIPrunerConfig(history_steps=5)

    def test_paper_layer_numbering(self):
        self.assertEqual(GUIPrunerConfig().prune_layer, 2)

    def test_global_budget_derives_current_ratio(self):
        # H=4000, lambda=.4 -> H_tar=1600; C=2000, eta=.5 on
        # (H_tar+C) -> current ratio=(.5*3600-1600)/2000=.1.
        self.assertAlmostEqual(
            resolve_current_keep_ratio(history_original_tokens=4000, current_original_tokens=2000,
                                       history_keep_ratio=.4, overall_keep_ratio=.5),
            .1,
        )

    def test_global_prompt_budget_derives_current_ratio(self):
        # Exact endpoint accounting: baseline=1000, TAR/text fixed=100,
        # current=500, eta=.35 -> keep (350-100)/500=.5 of current.
        self.assertAlmostEqual(
            resolve_current_keep_ratio(history_original_tokens=0, current_original_tokens=500,
                                       history_keep_ratio=.4, overall_keep_ratio=.35,
                                       baseline_prompt_tokens=1000,
                                       fixed_prompt_tokens_after_tar=100),
            .5,
        )

    def test_infeasible_global_budget_fails_closed(self):
        with self.assertRaises(ValueError):
            resolve_current_keep_ratio(history_original_tokens=4000, current_original_tokens=500,
                                       history_keep_ratio=.8, overall_keep_ratio=.1)

    def test_infeasible_global_budget_can_clip_current_frame(self):
        # Requested current budget exceeds C; clip policy saturates at 100%.
        self.assertEqual(
            resolve_current_keep_ratio(history_original_tokens=4000, current_original_tokens=500,
                                       history_keep_ratio=.8, overall_keep_ratio=.9,
                                       baseline_prompt_tokens=1000,
                                       fixed_prompt_tokens_after_tar=100,
                                       infeasible_policy="clip"),
            1.0,
        )


class TestFlops(unittest.TestCase):
    def test_split_prefill_replaces_full_length_accounting(self):
        cfg = SimpleNamespace(
            hidden_size=64, intermediate_size=128, num_attention_heads=4,
            num_key_value_heads=4, head_dim=16, num_hidden_layers=8, vocab_size=256,
        )
        model = SimpleNamespace(config=SimpleNamespace(text_config=cfg))
        generic = {"vision_flops": 100.0, "llm_flops": 1e9, "lm_head_flops": 1e8, "e2e_flops": 1.1e9}
        result = correct_split_prefill_flops(
            model, generic, prompt_tokens_before_ssp=100, prompt_tokens_after_ssp=40, prune_layer_one_based=2,
        )
        self.assertLess(result["prefill_split_llm_flops"], result["prefill_full_length_llm_flops_replaced"])
        self.assertAlmostEqual(result["e2e_flops"], result["vision_flops"] + result["llm_flops"] + result["lm_head_flops"])


class TestHistoryIdentification(unittest.TestCase):
    def test_explicit_current_label_beats_order_fallback(self):
        message = [
            {"type": "image", "value": "old.png"},
            {"type": "text", "value": "Current screenshot:"},
            {"type": "image", "value": "current.png"},
            {"type": "image", "value": "later-unlabelled.png"},
        ]
        history, current, source = _history_and_current_indices(message)
        self.assertEqual((history, current, source), ([0, 3], 2, "metadata_or_prompt_label"))

    def test_order_fallback_is_explicit(self):
        message = [{"type": "image", "value": "h.png"}, {"type": "image", "value": "c.png"}]
        history, current, source = _history_and_current_indices(message)
        self.assertEqual((history, current, source), ([0], 1, "last_image_fallback"))


if __name__ == "__main__":
    unittest.main()
