from __future__ import annotations

import torch
import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = "vlmeval.vlm.qwen3_vl_stlite"


def _ensure_pkg(name: str, path: Path | None = None):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)] if path is not None else []
    sys.modules[name] = mod


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg("vlmeval", ROOT / "vlmeval")
_ensure_pkg("vlmeval.vlm", ROOT / "vlmeval" / "vlm")
_ensure_pkg(PKG, ROOT / "vlmeval" / "vlm" / "qwen3_vl_stlite")
config_mod = _load(f"{PKG}.config", "vlmeval/vlm/qwen3_vl_stlite/config.py")
kv_mod = _load(f"{PKG}.kv_cache", "vlmeval/vlm/qwen3_vl_stlite/kv_cache.py")
STLiteConfig = config_mod.STLiteConfig
ImageTokenRange = kv_mod.ImageTokenRange
STLiteKVCluster = kv_mod.STLiteKVCluster
compute_css_scores = kv_mod.compute_css_scores


def _mock_cache(seq_len=48, heads=2, dim=8, layers=3):
    torch.manual_seed(7)
    query = torch.randn(1, heads, seq_len, dim)
    key = torch.randn(1, heads, seq_len, dim)
    value = torch.randn(1, heads, seq_len, dim)
    hidden = torch.randn(1, seq_len, heads * dim)
    ranges = [
        ImageTokenRange(4, 20, 1, 4, 4),
        ImageTokenRange(22, 38, 1, 4, 4),
    ]
    return [(key.clone(), query.clone(), value.clone()) for _ in range(layers)], hidden, ranges


def test_css_scores_cover_visual_ranges_only():
    _, hidden, ranges = _mock_cache()
    css = compute_css_scores(hidden, ranges, kernel_size=3)
    assert css.shape == hidden.shape[:2]
    assert torch.all(css[:, :4] == 0)
    assert torch.all(css[:, 20:22] == 0)
    assert torch.any(css[:, 4:20] > 0)
    assert torch.any(css[:, 22:38] > 0)


def test_stlite_compresses_key_value_with_recent_window_and_order():
    cfg = STLiteConfig(keep_ratio=0.5, window_size=8, min_tokens=8, alpha=1.0, use_css=True, use_tsg=True)
    layers, hidden, ranges = _mock_cache()
    cluster = STLiteKVCluster(cfg, ranges)
    key, query, value = layers[0]
    out_k, out_v = cluster.update_kv(key, query, value, None, 1, hidden_states=hidden)
    assert out_k.shape == out_v.shape
    assert out_k.shape[-2] == 24
    assert torch.equal(out_k[:, :, -8:, :], key[:, :, -8:, :])
    assert torch.equal(out_v[:, :, -8:, :], value[:, :, -8:, :])
    assert cluster.kept_indices is not None
    assert torch.all(cluster.kept_indices[..., 1:] >= cluster.kept_indices[..., :-1])
    assert cluster.last_stats["history_visual_tokens"] == 16
    assert cluster.last_stats["current_visual_tokens"] == 16


def test_flat_budget_across_layers():
    cfg = STLiteConfig(keep_ratio=0.4, window_size=8, min_tokens=8, use_css=False, use_tsg=False)
    layers, hidden, ranges = _mock_cache(layers=4)
    kept = []
    for key, query, value in layers:
        cluster = STLiteKVCluster(cfg, ranges)
        out_k, out_v = cluster.update_kv(key, query, value, None, 1, hidden_states=hidden)
        assert out_k.shape == out_v.shape
        kept.append(out_k.shape[-2])
    assert len(set(kept)) == 1
    assert kept[0] == 19


def test_tsg_gate_reduces_redundant_history_scores():
    cfg = STLiteConfig(keep_ratio=0.5, window_size=8, min_tokens=8, use_css=False, use_tsg=True, tsg_redundancy_threshold=0.5)
    layers, hidden, ranges = _mock_cache()
    hidden[:, ranges[0].start:ranges[0].end, :] = hidden[:, ranges[1].start:ranges[1].end, :]
    key, query, value = layers[0]
    cluster = STLiteKVCluster(cfg, ranges)
    out_k, out_v = cluster.update_kv(key, query, value, None, 1, hidden_states=hidden)
    assert out_k.shape == out_v.shape
    retained = cluster.kept_indices.reshape(-1)
    redundant_retained = ((retained >= ranges[0].start) & (retained < ranges[0].end)).sum().item()
    assert redundant_retained < ranges[0].count * key.shape[1]
