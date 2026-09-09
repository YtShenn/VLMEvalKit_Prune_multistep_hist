import importlib.util
import pathlib

import torch

_TOKEN_PATH = pathlib.Path(__file__).parents[1] / "vlmeval/vlm/qwen3_vl_fastv/token.py"
_SPEC = importlib.util.spec_from_file_location("fastv_token_test", _TOKEN_PATH)
_TOKEN = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TOKEN)
build_token_meta = _TOKEN.build_token_meta
select_keep_indices = _TOKEN.select_keep_indices
select_kv_sequence = _TOKEN.select_kv_sequence


def test_multi_segment_meta_and_four_history_plus_current():
    mask = torch.tensor([[0, 1, 1, 0, 1, 0, 1, 1, 0, 1]], dtype=torch.bool)
    meta = build_token_meta(mask)
    assert meta is not None
    assert meta["visual_indices"].tolist() == [1, 2, 4, 6, 7, 9]


def test_r_zero_preserves_everything():
    idx = torch.tensor([1, 3, 5])
    assert select_keep_indices(7, idx, torch.tensor([.2, .8, .4]), 0.0).tolist() == list(range(7))


def test_global_topk_preserves_nonvisual_and_original_order():
    idx = torch.tensor([1, 3, 6, 8])
    # Keep 50%, globally; tokens 3 and 8 win, and all non-visual tokens stay.
    assert select_keep_indices(10, idx, torch.tensor([.1, .9, .2, .8]), .5).tolist() == [0, 2, 3, 4, 5, 7, 8, 9]


def test_invalid_or_batched_mask_is_safe_no_metadata():
    assert build_token_meta(None) is None
    assert build_token_meta(torch.ones((2, 4), dtype=torch.bool)) is None


def test_kv_sequence_selection_stays_aligned_with_keep_indices():
    cache = torch.arange(1 * 2 * 6 * 3).reshape(1, 2, 6, 3)
    selected = select_kv_sequence(cache, torch.tensor([0, 2, 5]))
    assert selected.shape == (1, 2, 3, 3)
    assert torch.equal(selected[:, :, 1], cache[:, :, 2])


if __name__ == "__main__":
    # Keep this executable in the project inference environment, which does
    # not install pytest, while remaining pytest-discoverable as plain tests.
    test_multi_segment_meta_and_four_history_plus_current()
    test_r_zero_preserves_everything()
    test_global_topk_preserves_nonvisual_and_original_order()
    test_invalid_or_batched_mask_is_safe_no_metadata()
    test_kv_sequence_selection_stays_aligned_with_keep_indices()
    print("FastV token tests: OK")
