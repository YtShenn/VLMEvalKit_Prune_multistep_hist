#!/usr/bin/env python3
"""Free-running, read-only diagnostic of Qwen3-VL attention to history images.

Unlike ``history_attention_diagnostic.py``, this program never appends an answer
to the model input.  It first greedily generates an answer, then replays that
*generated* token prefix to measure the rows which predicted each token.  The
replay is a cache-free implementation detail: its causal hidden states are the
same as greedy decoding for a fixed generated prefix.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

try:
    from history_attention_diagnostic import (build_history_gt_lookup,
        complete_history_candidates, device_inputs, explicit_sample_indices,
        image_items, make_inputs, sampled_row_indices)
    from history_attention_utils import (bootstrap_episode, entropy, grid_rows,
        locate_visual_segments)
except ModuleNotFoundError:
    from .history_attention_diagnostic import (build_history_gt_lookup,
        complete_history_candidates, device_inputs, explicit_sample_indices,
        image_items, make_inputs, sampled_row_indices)
    from .history_attention_utils import bootstrap_episode, entropy, grid_rows, locate_visual_segments


MAIN_CATEGORIES = ("action_type", "bbox_coordinate")
ALL_CATEGORIES = MAIN_CATEGORIES + ("format", "other")
CAUSAL_NOTE = ("Free-running attention is an observation of the model read path "
               "during its own generated trajectory. It can motivate thumbnail/ROI "
               "history design, but it is not causal evidence of performance contribution. "
               "Repeated fixed spatial peaks can be positional bias or attention sinks.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--work-dir", required=True,
                   help="Must be a new OUTPUT_history_attention_freerun/... directory.")
    p.add_argument("--query-source", choices=["free_running", "gold_teacher_forced"], default="free_running")
    p.add_argument("--num-samples", type=int, default=20,
                   help="Number of eligible samples; use -1 for every eligible complete-history row.")
    p.add_argument("--sample-by", choices=["task", "sequential"], default="task")
    p.add_argument("--samples-per-task", type=int, default=1)
    p.add_argument("--sample-ids", default=None)
    p.add_argument("--history-steps", type=int, default=4)
    p.add_argument("--layers", choices=["last_third", "all"], default="last_third")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--case-count", type=int, default=6)
    p.add_argument("--num-shards", type=int, default=1,
                   help="Number of deterministic dataset shards; normally set by the multi-GPU wrapper.")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Zero-based shard index in [0, num_shards).")
    p.add_argument("--skip-summary-figures", action="store_true",
                   help="Skip aggregate figures; worker shards use this and the merger creates global figures.")
    return p.parse_args()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def setup_dirs(work_dir):
    root = Path(work_dir).resolve()
    old = (REPO_ROOT / "OUTPUT_history_attention_test").resolve()
    if root == old or old in root.parents:
        raise ValueError("Refusing to write into the legacy teacher-forced output directory.")
    for name in ("heatmaps_absolute", "heatmaps_conditional", "figures", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


class SelectiveTraceCapture:
    """Reduce selected QK rows immediately; never store an S x S attention map."""
    def __init__(self, model, query_positions, history_positions, current_positions, layers):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        candidates = [getattr(getattr(model, "model", None), "language_model", None),
                      getattr(model, "language_model", None)]
        lm = next((x for x in candidates if x is not None and hasattr(x, "layers")), None)
        if lm is None:
            raise RuntimeError("Cannot locate Qwen3-VL decoder layers.")
        start = len(lm.layers) * 2 // 3 if layers == "last_third" else 0
        self.modules, self.apply_rope = list(lm.layers[start:]), apply_rotary_pos_emb
        self.qpos = torch.tensor(query_positions, dtype=torch.long)
        self.hpos = torch.tensor(history_positions, dtype=torch.long)
        self.cpos = torch.tensor(current_positions, dtype=torch.long)
        self.conditional_sums, self.hist_sums, self.current_sums = [], [], []
        self.count, self.handles = 0, []

    def hook(self, module, args, kwargs):
        attn = module.self_attn
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        pos, mask = kwargs.get("position_embeddings"), kwargs.get("attention_mask")
        if hidden is None or pos is None:
            raise RuntimeError("Unexpected Qwen3-VL attention hook signature.")
        qp, hp, cp = self.qpos.to(hidden.device), self.hpos.to(hidden.device), self.cpos.to(hidden.device)
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        q, k = self.apply_rope(q, k, *pos)
        q = q.index_select(2, qp)
        if attn.num_key_value_groups != 1:
            b, kvh, seq, dim = k.shape
            k = k[:, :, None, :, :].expand(b, kvh, attn.num_key_value_groups, seq, dim).reshape(b, -1, seq, dim)
        score = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        if mask is not None:
            score = score + mask.index_select(-2, qp)
        prob = torch.softmax(score, dim=-1, dtype=torch.float32)
        hist = prob.index_select(-1, hp)
        hmass = hist.sum(dim=-1)
        self.conditional_sums.append((hist / (hmass.unsqueeze(-1) + 1e-12)).sum(dim=(0, 1, 2)).detach().float().cpu())
        self.hist_sums.append(hmass.sum().detach().float().cpu())
        self.current_sums.append(prob.index_select(-1, cp).sum(dim=-1).sum().detach().float().cpu() if len(cp) else torch.tensor(0.))
        self.count += int(hmass.numel())

    def __enter__(self):
        self.handles = [m.register_forward_pre_hook(self.hook, with_kwargs=True) for m in self.modules]
        return self

    def __exit__(self, *unused):
        for h in self.handles: h.remove()

    def result(self):
        if not self.count:
            raise RuntimeError("No attention rows captured")
        conditional = torch.stack(self.conditional_sums).sum(0).numpy() / self.count
        hist = float(torch.stack(self.hist_sums).sum() / self.count)
        current = float(torch.stack(self.current_sums).sum() / self.count)
        return conditional, {"history": hist, "current_visual": current,
                             "non_visual_or_other": max(0., 1. - hist - current)}


def generated_prefix_spans(tokenizer, ids):
    """Return decoded character spans for each generated token, conservatively."""
    prefixes = [""]
    for i in range(1, len(ids) + 1):
        prefixes.append(tokenizer.decode(ids[:i], skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return prefixes[-1], [(len(prefixes[i]), len(prefixes[i + 1])) for i in range(len(ids))]


def value_spans(text):
    """Find value character spans from the generated text only, never annotation."""
    action, boxes = [], []
    m = re.search(r'"action_type"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text)
    if m: action.append(m.span(1))
    b = re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', text)
    if b:
        for n in re.finditer(r'-?\d+(?:\.\d+)?', b.group(1)):
            boxes.append((b.start(1) + n.start(), b.start(1) + n.end()))
    return action, boxes


def overlaps(a, b): return a[0] < b[1] and b[0] < a[1]


def classify_generated_tokens(tokenizer, ids):
    text, spans = generated_prefix_spans(tokenizer, ids)
    action, boxes = value_spans(text)
    parsed = bool(action or boxes)
    out = []
    for i, (start, end) in enumerate(spans):
        category = "other" if not parsed else "format"
        if any(overlaps((start, end), x) for x in action): category = "action_type"
        elif any(overlaps((start, end), x) for x in boxes): category = "bbox_coordinate"
        out.append({"index": i, "id": int(ids[i]),
                    "text": tokenizer.decode([ids[i]], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    "char_span": [start, end], "category": category})
    return text, out, parsed


def parse_answer(text):
    match = re.search(r'<answer>\s*(\{.*?\})\s*</answer>', text, flags=re.S)
    if not match:
        match = re.search(r'(\{[^{}]*"action_type"[^{}]*\})', text, flags=re.S)
    if not match: return None, "no_json_answer"
    try:
        obj = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        return None, f"json_decode_error:{e.msg}"
    return obj, None


def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2-x1) * max(0., y2-y1)
    union = max(0., a[2]-a[0]) * max(0., a[3]-a[1]) + max(0., b[2]-b[0]) * max(0., b[3]-b[1]) - inter
    return inter / union if union else 0.


def posthoc_correctness(row, parsed):
    """GT is used solely after generation for labels; this is not official scoring."""
    if parsed is None: return {"status": "unparsable", "action_type_correct": None, "bbox_iou": None, "overall_success": None}
    pred_action = str(parsed.get("action_type", "")).split(":", 1)[0].lower()
    gt_action = str(row.get("gt_action", "")).split(":", 1)[0].lower()
    action_ok = bool(pred_action and gt_action and pred_action == gt_action)
    pred_box = parsed.get("bbox_2d")
    gt_box = row.get("gt_min_bbox", row.get("gt_bbox", None))
    if isinstance(gt_box, str):
        try: gt_box = ast.literal_eval(gt_box)
        except (ValueError, SyntaxError): gt_box = None
    iou = None
    if isinstance(pred_box, (list, tuple)) and isinstance(gt_box, (list, tuple)) and len(pred_box) == len(gt_box) == 4:
        try: iou = box_iou([float(x) for x in pred_box], [float(x) for x in gt_box])
        except (ValueError, TypeError): pass
    needs_box = gt_action in {"click", "long_press"}
    success = action_ok and ((iou is not None and iou >= .5) if needs_box else True)
    return {"status": "correct" if success else "incorrect", "action_type_correct": action_ok,
            "bbox_iou": iou, "overall_success": success,
            "rule": "post-hoc action-type equality plus gt_min_bbox IoU>=0.5 for click/long_press; not official evaluator"}


def make_replay(prompt, generated_ids):
    """Append generated *text* tokens to every sequence-aligned processor field.

    Qwen3-VL's processor supplies ``mm_token_type_ids`` alongside input IDs.
    Unlike pixel values/grid metadata, it has one entry per LLM token and is
    consumed with ``attention_mask`` while rebuilding M-RoPE positions.  The
    generated suffix is text modality (0); leaving this tensor at prompt length
    produces a mask/index length mismatch during the cache-free replay.
    """
    full = dict(prompt)
    prompt_len = prompt["input_ids"].shape[1]
    suffix = generated_ids[None].to(prompt["input_ids"].device)
    full["input_ids"] = torch.cat([prompt["input_ids"], suffix], dim=1)
    if "attention_mask" in prompt:
        full["attention_mask"] = torch.cat([prompt["attention_mask"], torch.ones((1, len(generated_ids)), dtype=prompt["attention_mask"].dtype, device=prompt["attention_mask"].device)], dim=1)
    # Qwen3-VL uses 0 for text, 1 for image and 2 for video.  Generated tokens
    # are text.  ``token_type_ids`` is included for processor compatibility;
    # current Qwen3-VL exposes the same tensor as ``mm_token_type_ids``.
    for key in ("mm_token_type_ids", "token_type_ids"):
        value = prompt.get(key)
        if value is not None and getattr(value, "ndim", 0) == 2 and value.shape[1] == prompt_len:
            text_suffix = torch.zeros((value.shape[0], len(generated_ids)), dtype=value.dtype, device=value.device)
            full[key] = torch.cat([value, text_suffix], dim=1)
    expected = full["input_ids"].shape[1]
    for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        value = full.get(key)
        if value is not None and value.shape[-1] != expected:
            raise ValueError(f"Replay field {key} length={value.shape[-1]} != input_ids length={expected}")
    return full


def aggregate_category(model, full, positions, hist_pos, current_pos, layers):
    with torch.no_grad(), SelectiveTraceCapture(model, positions, hist_pos, current_pos, layers) as cap:
        model(**full, output_attentions=False, use_cache=False, return_dict=True)
    return cap.result()


def frame_records(importance, segments, masses, category, episode):
    rows, offset = [], 0
    for f, seg in enumerate(segments):
        n = seg["visual_tokens"]
        local = importance[offset:offset+n].reshape(seg["grid_t"], seg["grid_height"], seg["grid_width"]).sum(0)
        offset += n
        fi = float(local.sum()); conditional = local / max(fi, 1e-12); absolute = masses["history"] * local
        flat = conditional.ravel(); top_n = max(1, int(np.ceil(.01 * len(flat)))); peak = int(flat.argmax())
        r, c = divmod(peak, seg["grid_width"])
        rows.append({"episode_id": episode, "category": category, "history_step": seg["history_step"],
                     "relative_step": f-len(segments), "F_i": fi, "A_i": masses["history"]*fi,
                     "entropy": entropy(conditional), "effective_tokens": float(np.exp(entropy(conditional))),
                     "top_1pct_mass": float(np.partition(flat, -top_n)[-top_n:].sum()),
                     "peak_grid_row_col": [int(r), int(c)], "peak_relative_xy": [(c+.5)/seg["grid_width"], (r+.5)/seg["grid_height"]],
                     "conditional_map": conditional, "absolute_map": absolute, "segment": seg})
    return rows


def draw_series(root, sample, category, frames):
    import matplotlib.pyplot as plt
    if not frames: return
    n = len(frames); vmax = max(float(np.max(x["absolute_map"])) for x in frames) or 1e-12
    for mode, key, folder, cmap, vmin, shared in (("absolute", "absolute_map", "heatmaps_absolute", "magma", 0., True),
                                                   ("conditional", "conditional_map", "heatmaps_conditional", "viridis", 0., False)):
        fig, axes = plt.subplots(1, n, figsize=(4*n, 6), squeeze=False)
        for ax, f in zip(axes[0], frames):
            seg = f["segment"]; im = Image.open(seg["path"]).convert("RGB")
            ax.imshow(im)
            map_vmax = vmax if shared else max(float(np.max(f[key])), 1e-12)
            ax.imshow(f[key], extent=(0, im.width, im.height, 0), cmap=cmap, alpha=.82,
                      vmin=vmin, vmax=map_vmax, interpolation="nearest")
            gt = sample.get("history_gt", {}).get(str(seg["path"]))
            if gt and gt.get("bbox_xyxy"):
                b = gt["bbox_xyxy"]; ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1], fill=False, ec="lime", lw=1.5))
            ax.set_title(f"t{f['relative_step']} | S={sample['categories'][category]['mass']['history']:.3f}\nF={f['F_i']:.3f}, A={f['A_i']:.4f}, Neff={f['effective_tokens']:.0f}\npeak={f['peak_grid_row_col']}")
            ax.axis("off")
        fig.suptitle(f"{mode} | sample={sample['sample_id']} | {category} | {sample['correctness']['status']}")
        fig.tight_layout(); fig.savefig(root/folder/f"{sample['sample_id']}_{category}_{sample['correctness']['status']}.png", dpi=160); plt.close(fig)


def json_safe(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating, np.integer)): return x.item()
    raise TypeError(type(x).__name__)


def write_csv(path, rows):
    fields = ["sample_id", "episode_id", "outcome", "category", "relative_step", "query_count", "S_hist", "F_i", "A_i", "entropy", "effective_tokens", "top_1pct_mass", "peak_grid_row_col", "peak_relative_xy"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k: r.get(k) for k in fields})


def make_figures(root, flat):
    import matplotlib.pyplot as plt
    for cat in MAIN_CATEGORIES:
        rows = [x for x in flat if x["category"] == cat]
        if not rows: continue
        steps = sorted(set(x["relative_step"] for x in rows))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, field in zip(axes, ("S_hist", "A_i")):
            ax.boxplot([[x[field] for x in rows if x["relative_step"] == step] for step in steps], tick_labels=steps)
            ax.set(title=f"{cat}: {field}", xlabel="history relative step")
        fig.tight_layout(); fig.savefig(root/"figures"/f"distribution_{cat}.png", dpi=160); plt.close(fig)
        for outcome in ("correct", "incorrect"):
            z = [x for x in rows if x["outcome"] == outcome]
            if z:
                fig, ax = plt.subplots(figsize=(4,4)); ax.hist2d([x["peak_relative_xy"][0] for x in z], [x["peak_relative_xy"][1] for x in z], bins=12, range=[[0,1],[0,1]])
                ax.invert_yaxis(); ax.set(title=f"{cat} peak positions: {outcome}", xlabel="relative x", ylabel="relative y")
                fig.tight_layout(); fig.savefig(root/"figures"/f"peak_{cat}_{outcome}.png", dpi=160); plt.close(fig)


def make_case_gallery(root, samples, count):
    """A compact low/median/high history-mass gallery, separated by outcome."""
    import matplotlib.pyplot as plt
    for outcome in ("correct", "incorrect"):
        candidates = []
        for s in samples:
            payload = s.get("categories", {}).get("action_type")
            if s.get("status") == "ok" and s.get("correctness", {}).get("status") == outcome and payload:
                candidates.append((payload["mass"]["history"], s))
        if not candidates: continue
        candidates.sort(key=lambda x: x[0])
        picks = []
        for i in sorted(set([0, len(candidates)//2, len(candidates)-1])):
            if candidates[i][1] not in picks: picks.append(candidates[i][1])
        fig, axes = plt.subplots(1, len(picks), figsize=(5*len(picks), 5), squeeze=False)
        for ax, sample in zip(axes[0], picks):
            path = root / "heatmaps_absolute" / f"{sample['sample_id']}_action_type_{outcome}.png"
            if path.exists(): ax.imshow(plt.imread(path))
            ax.set_title(f"{outcome}: {sample['sample_id']}")
            ax.axis("off")
        fig.suptitle(f"action_type absolute attention: low / median / high S_hist ({outcome})")
        fig.tight_layout(); fig.savefig(root/"figures"/f"case_gallery_{outcome}.png", dpi=160); plt.close(fig)


def main():
    a = parse_args()
    if a.query_source != "free_running":
        raise NotImplementedError("Use the legacy history_attention_diagnostic.py for gold teacher-forced queries; it remains unchanged.")
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        raise ValueError("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards.")
    root = setup_dirs(a.work_dir)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    os.environ.update({"GUI_ODYSSEY_STATE_PACKET_ENABLE":"0", "GUI_ODYSSEY_MAX_HISTORY_IMAGES":str(a.history_steps),
                       "ANDROID_CONTROL_STATE_PACKET_ENABLE":"0", "ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS":"1", "ANDROID_CONTROL_MAX_HISTORY_IMAGES":str(a.history_steps)})
    from vlmeval.dataset import build_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dataset = build_dataset(a.data)
    if dataset is None: raise RuntimeError(f"Cannot build dataset {a.data}")
    processor = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype="auto", device_map="auto", attn_implementation="sdpa").eval()
    data = dataset.data; image_id = model.config.image_token_id; merge = model.config.vision_config.spatial_merge_size
    candidates = complete_history_candidates(data, a.history_steps)
    requested_count = len(candidates) if a.num_samples == -1 else a.num_samples
    if requested_count < 1:
        raise ValueError("--num-samples must be positive or -1 (all eligible complete-history rows).")
    selected_all = (explicit_sample_indices(data, a.sample_ids) if a.sample_ids else
                    sampled_row_indices(data, requested_count, a.sample_by, a.samples_per_task, a.seed, candidates))
    # Split after deterministic selection so workers are disjoint and the union
    # is exactly the requested sample set, independent of GPU scheduling.
    selected = [row_idx for ordinal, row_idx in enumerate(selected_all) if ordinal % a.num_shards == a.shard_index]
    metadata = {"query_source":"free_running", "teacher_forcing":False, "data":a.data, "model":a.model,
                "seed":a.seed, "history_steps":a.history_steps, "layers":a.layers, "git_commit":git_commit(),
                "sharding":{"num_shards":a.num_shards, "shard_index":a.shard_index,
                            "selected_before_sharding":len(selected_all), "selected_in_this_shard":len(selected)},
                "generation":{"do_sample":False, "temperature":a.temperature, "max_new_tokens":a.max_new_tokens},
                "replay":"Generated prefixes are replayed without KV cache to capture exact selected causal rows; no gold text enters a forward pass.",
                "correctness_note":"GT is only used post-hoc. This diagnostic uses action equality and gt_min_bbox IoU>=0.5, not the official evaluator.",
                "interpretation_boundary":CAUSAL_NOTE}
    (root/"metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    history_gt = build_history_gt_lookup(data); samples = []; out = root/"per_sample.jsonl"; out.write_text("")
    for idx in selected:
        row = data.iloc[idx]; sid = str(row.get("index", idx)); episode = str(row.get("episode_id", row.get("task_id", sid)))
        rec = {"sample_id":sid, "episode_id":episode, "status":"skipped", "categories":{}, "history_gt":{}}
        try:
            prompt_obj = dataset.build_prompt(row); messages = prompt_obj[0] if isinstance(prompt_obj, tuple) else prompt_obj
            records = image_items(messages); hist = [x for x in records if x["kind"] == "history"]
            if len(hist) < a.history_steps: raise ValueError(f"incomplete_history:{len(hist)}")
            # make_inputs is used only for its normal prompt construction. Its gold argument is deliberately empty.
            prompt, _, _ = make_inputs(processor, messages, "")
            prompt = device_inputs(prompt, model); plen = int(prompt["input_ids"].shape[1])
            with torch.no_grad():
                generated = model.generate(**prompt, do_sample=False, max_new_tokens=a.max_new_tokens,
                                           return_dict_in_generate=True, output_scores=True, use_cache=True)
            ids = generated.sequences[0, plen:].detach().cpu().tolist()
            if not ids: raise ValueError("empty_generation")
            text, tokens, semantically_parsed = classify_generated_tokens(processor.tokenizer, ids)
            parsed, parse_error = parse_answer(text); correctness = posthoc_correctness(row, parsed)
            stop = "eos" if ids[-1] in set(getattr(model.generation_config, "eos_token_id", []) if isinstance(getattr(model.generation_config, "eos_token_id", []), list) else [getattr(model.generation_config, "eos_token_id", -1)]) else "max_new_tokens_or_other"
            scores = generated.scores or []
            for j, tok in enumerate(tokens):
                if j < len(scores):
                    lp = torch.log_softmax(scores[j][0].float(), -1)[ids[j]].item(); tok["logprob"] = float(lp)
            replay = make_replay(prompt, torch.tensor(ids, dtype=prompt["input_ids"].dtype, device=prompt["input_ids"].device))
            grids = grid_rows(replay.get("image_grid_thw"), merge); segments = locate_visual_segments(replay["input_ids"][0], image_id, grids, records)
            hist_segments = [x for x in segments if x["kind"] == "history"]; current_segments = [x for x in segments if x["kind"] == "current"]
            hpos = [p for s in hist_segments for p in range(s["sequence_start"], s["sequence_end"])]
            cpos = [p for s in current_segments for p in range(s["sequence_start"], s["sequence_end"])]
            rec.update(status="ok", task_description=str(row.get("instruction", row.get("question", ""))), prompt_length=plen,
                       generated_text=text, generated_token_ids=ids, generated_tokens=tokens, stop_reason=stop,
                       parse_error=parse_error, parsed_answer=parsed, semantic_parse_available=semantically_parsed,
                       correctness=correctness, history_image_paths=[x["path"] for x in hist_segments], visual_tokens=segments)
            rec["history_gt"] = {str(s["path"]): history_gt.get(os.path.abspath(s["path"])) for s in hist_segments}
            for category in ALL_CATEGORIES:
                token_ids = [t["index"] for t in tokens if t["category"] == category]
                if not token_ids: continue
                qpos = [plen - 1 + j for j in token_ids]
                importance, mass = aggregate_category(model, replay, qpos, hpos, cpos, a.layers)
                frames = frame_records(importance, hist_segments, mass, category, episode)
                fi_sum, ai_sum = sum(x["F_i"] for x in frames), sum(x["A_i"] for x in frames)
                rec["categories"][category] = {"query_count":len(qpos), "predicted_token_indices":token_ids,
                                                 "query_positions":qpos,
                                                 "attention_queries":[{"predicted_token_index":j,
                                                                       "predicted_token_id":ids[j],
                                                                       "query_position":plen-1+j,
                                                                       "query_source":"prompt_last_token" if j == 0 else "previous_generated_token"}
                                                                      for j in token_ids],
                                                 "mass":mass, "frames":frames,
                                                 "validation":{"sum_F_i":fi_sum, "sum_A_i":ai_sum,
                                                               "sum_F_i_close_to_1":bool(np.isclose(fi_sum, 1., atol=1e-5)),
                                                               "sum_A_i_close_to_S_hist":bool(np.isclose(ai_sum, mass["history"], atol=1e-5))}}
            for category in MAIN_CATEGORIES:
                if category in rec["categories"]: draw_series(root, rec, category, rec["categories"][category]["frames"])
        except Exception as e:
            logging.exception("sample %s failed", sid); rec["skip_reason"] = f"{type(e).__name__}:{e}"
        samples.append(rec)
        with out.open("a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False, default=json_safe)+"\n")
    flat = []
    for s in samples:
        if s["status"] != "ok": continue
        for cat, payload in s["categories"].items():
            for fr in payload["frames"]:
                flat.append({"sample_id":s["sample_id"], "episode_id":s["episode_id"], "outcome":s["correctness"]["status"],
                             "category":cat, "query_count":payload["query_count"], "S_hist":payload["mass"]["history"], **fr})
    write_csv(root/"frame_statistics.csv", flat)
    if not a.skip_summary_figures:
        make_figures(root, flat); make_case_gallery(root, samples, a.case_count)
    summaries = {}
    warnings = []
    for cat in ALL_CATEGORIES:
        for outcome in ("all", "correct", "incorrect"):
            for step in sorted(set(x["relative_step"] for x in flat)):
                z = [x for x in flat if x["category"] == cat and x["relative_step"] == step and (outcome == "all" or x["outcome"] == outcome)]
                if z: summaries[f"{cat}/{outcome}/t{step}"] = {k:bootstrap_episode(z, k, seed=a.seed) for k in ("S_hist","F_i","A_i","entropy","effective_tokens","top_1pct_mass")}
        z = [x for x in flat if x["category"] == cat]
        if z:
            edge = np.mean([min(x["peak_relative_xy"][0], 1-x["peak_relative_xy"][0], x["peak_relative_xy"][1], 1-x["peak_relative_xy"][1]) < .08 for x in z])
            common, count = Counter(tuple(x["peak_grid_row_col"]) for x in z).most_common(1)[0]
            if edge > .5 or count / len(z) > .4:
                warnings.append(f"{cat}: possible positional artifact (edge_peak_fraction={edge:.3f}, most_common_peak={common}, fraction={count/len(z):.3f})")
    outcomes = Counter(s.get("correctness", {}).get("status", "skipped") for s in samples)
    ok_samples = [s for s in samples if s["status"] == "ok"]
    action_labels = [s["correctness"]["action_type_correct"] for s in ok_samples if s["correctness"].get("action_type_correct") is not None]
    overall_labels = [s["correctness"]["overall_success"] for s in ok_samples if s["correctness"].get("overall_success") is not None]
    summary = {"samples_requested":a.num_samples, "eligible_complete_history_rows":len(candidates), "samples_selected":len(selected),
               "samples_selected_before_sharding":len(selected_all), "sharding":{"num_shards":a.num_shards, "shard_index":a.shard_index},
               "samples_ok":len(ok_samples), "outcomes":dict(outcomes),
               "structured_valid":sum(s.get("parsed_answer") is not None for s in ok_samples),
               "action_type_accuracy":(float(np.mean(action_labels)) if action_labels else None),
               "overall_success_rate":(float(np.mean(overall_labels)) if overall_labels else None),
               "per_category_step":summaries, "artifact_warnings":warnings, "interpretation_boundary":CAUSAL_NOTE}
    (root/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=json_safe))


if __name__ == "__main__": main()
