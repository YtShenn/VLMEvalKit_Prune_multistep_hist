import torch
from types import SimpleNamespace

from vlmeval.vlm.qwen3_vl_divprune.token import build_token_meta, max_min_keep_relative
from vlmeval.vlm.qwen3_vl.model import _attn_prune_generate_use_cache


def test_max_min_is_deterministic_sorted_and_full_keep_is_identity():
    features = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]])
    chosen_a = max_min_keep_relative(features, 0.5)
    chosen_b = max_min_keep_relative(features, 0.5)
    assert chosen_a.tolist() == chosen_b.tolist()
    assert chosen_a.numel() == 2
    assert chosen_a.tolist() == sorted(set(chosen_a.tolist()))
    assert max_min_keep_relative(features, 1.0).tolist() == [0, 1, 2, 3]


def test_first_token_uses_most_distant_nearest_neighbor():
    # Token 3 is the unique far-away point; it has the largest nearest-neighbor
    # cosine distance and is therefore the paper Algorithm-1 initial choice.
    features = torch.tensor([[1., 0.], [0.99, .01], [0.98, .02], [-1., 0.]])
    assert max_min_keep_relative(features, 0.25).tolist() == [3]


def test_multi_image_metadata_preserves_text_and_counts_all_five_images():
    # Five image blocks have two visual tokens each, with text tokens inserted
    # between every block. Grid values encode two post-merge tokens per image.
    visual_mask = torch.zeros(1, 15, dtype=torch.bool)
    positions = torch.tensor([0, 1, 3, 4, 6, 7, 9, 10, 12, 13])
    visual_mask[0, positions] = True
    embeddings = torch.randn(1, 15, 4)
    grids = torch.tensor([[1, 1, 2]] * 5)
    meta = build_token_meta(embeddings, visual_mask, grids, merge=1, keep_ratio=.5, scope="global_all_visual")
    assert meta is not None
    assert meta["per_image_before"] == [2, 2, 2, 2, 2]
    assert sum(meta["per_image_after"]) == 5
    assert meta["kept_visual"].numel() == 5
    assert torch.isin(meta["kept_visual"], positions).all()


def test_history_only_keeps_current_image_unpruned():
    visual_mask = torch.zeros(1, 6, dtype=torch.bool)
    visual_mask[0, torch.tensor([0, 1, 3, 4])] = True
    embeddings = torch.randn(1, 6, 3)
    grids = torch.tensor([[1, 1, 2], [1, 1, 2]])
    meta = build_token_meta(embeddings, visual_mask, grids, merge=1, keep_ratio=.5, scope="history_only")
    assert meta["per_image_before"] == [2, 2]
    assert meta["per_image_after"] == [1, 2]


def test_nonfinite_features_fail_closed_at_selection_boundary():
    try:
        max_min_keep_relative(torch.tensor([[1., 0.], [float("nan"), 1.]]), .5)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite features must fail at the selection boundary")


def test_divprune_is_cache_safe_not_legacy_attnprune():
    # The custom decoder's module name includes ``attn_prune``. DivPrune must
    # nevertheless retain its physically reduced Layer-0 prefill KV cache.
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(
            _divprune_enabled=True, _divprune_config=SimpleNamespace(keep_ratio=.1),
        )),
        model=SimpleNamespace(),
    )
    assert _attn_prune_generate_use_cache(model) is True
