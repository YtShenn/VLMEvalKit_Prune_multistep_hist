#!/usr/bin/env python3
"""Teacher-forced diagnostic of Qwen3-VL attention to full-resolution history.

This is intentionally an offline observation tool.  It neither changes VLMEval's
inference code path nor performs cropping, pruning, deletion, or causal tests.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# A number of experiment environments export another VLMEvalKit checkout in
# PYTHONPATH.  This diagnostic must use the checkout containing this script,
# otherwise importing `vlmeval` can silently load incompatible custom models.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image

try:  # supports both `python utils/script.py` and `python -m utils.script`
    from history_attention_utils import (bootstrap_episode, entropy, grid_rows,
        load_compatible_roi_json, locate_visual_segments, normalized_importance,
        roi_boxes_for_path, roi_mask_xyxy, semantic_token_indices)
except ModuleNotFoundError:
    from .history_attention_utils import (bootstrap_episode, entropy, grid_rows,
        load_compatible_roi_json, locate_visual_segments, normalized_importance,
        roi_boxes_for_path, roi_mask_xyxy, semantic_token_indices)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='VLMEval dataset name (currently direct gold `answer` required).')
    p.add_argument('--model', required=True, help='Local/HuggingFace Qwen3-VL model path.')
    p.add_argument('--work-dir', required=True)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--sample-by', choices=['task','sequential'], default='task',
                   help='`task`: deterministic round-robin by task_id/question; `sequential`: dataset order.')
    p.add_argument('--samples-per-task', type=int, default=1,
                   help='Maximum samples per task when --sample-by task.')
    p.add_argument('--sample-ids', default=None,
                   help='Comma-separated dataset `index` values to diagnose exactly (overrides task sampling).')
    p.add_argument('--history-steps', type=int, default=3)
    p.add_argument('--roi-json', default=None)
    p.add_argument('--layers', choices=['last_third','all'], default='last_third')
    p.add_argument('--action-token-mode', choices=['semantic','all'], default='all')
    p.add_argument('--random-roi-samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-attention-tensors', choices=['0','1'], default='0')
    return p


def setup_dirs(root):
    root = Path(root)
    for x in ('heatmaps','figures','logs'): (root/x).mkdir(parents=True, exist_ok=True)
    return root


def image_items(message):
    """Prompt order is processor order; last image is current, preceding are history."""
    items = [x for x in message if isinstance(x, dict) and x.get('type') == 'image']
    records=[]
    for i, x in enumerate(items):
        path=str(x.get('value','')).removeprefix('file://')
        records.append({'image_index': i, 'path': path,
                        'kind': 'current' if i == len(items)-1 else 'history',
                        'history_step': None if i == len(items)-1 else i})
    return records


def make_inputs(processor, messages, gold):
    """Exactly the normal chat template + teacher-forced gold continuation."""
    from qwen_vl_utils import process_vision_info
    content=[]
    for s in messages:
        if s['type']=='image': content.append({'type':'image','image':'file://' + str(s['value']).removeprefix('file://')})
        elif s['type']=='text': content.append({'type':'text','text':s['value']})
        else: raise ValueError('Only image/text prompts are supported by this diagnostic')
    chat=[{'role':'user','content':content}]
    prompt_text=processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs=process_vision_info(chat, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    common=dict(images=images, videos=videos, video_metadata=None, do_resize=False, return_tensors='pt', **(video_kwargs or {}))
    prompt=processor(text=prompt_text, **common)
    full=processor(text=prompt_text + gold, **common)
    return prompt, full, prompt_text


def device_inputs(batch, model):
    # BatchFeature.to(model.dtype) would corrupt token IDs; move device only.
    try: return batch.to(model.device)
    except Exception:
        dev=next(model.parameters()).device
        return {k: v.to(dev) if hasattr(v,'to') else v for k,v in batch.items()}


class SelectiveAttentionCapture:
    """Exact attention statistics without retaining decoder ``S x S`` tensors.

    The normal model forward remains SDPA.  At selected decoder layers this hook
    recomputes only gold-query rows of QK^T, including the model-created causal
    mask, and immediately reduces them to the two requested history statistics.
    This is mathematically the same eager attention row but costs O(M*S), not
    O(S*S), where M is the (small) number of selected gold action tokens.
    """
    def __init__(self, model, query_positions, history_positions, layers):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        candidates = [
            getattr(getattr(model, 'model', None), 'language_model', None),
            getattr(model, 'language_model', None),
        ]
        language_model = next((x for x in candidates if x is not None and hasattr(x, 'layers')), None)
        if language_model is None:
            raise RuntimeError('Cannot locate Qwen3-VL decoder layers for selective attention capture.')
        total = len(language_model.layers)
        start = total * 2 // 3 if layers == 'last_third' else 0
        self.modules = list(language_model.layers[start:])
        self.query_positions = torch.tensor(query_positions, dtype=torch.long)
        self.history_positions = torch.tensor(history_positions, dtype=torch.long)
        self.apply_rope = apply_rotary_pos_emb
        self.conditional_sums, self.mass_sums, self.count = [], [], 0
        self.handles = []

    def _hook(self, module, args, kwargs):
        attn = module.self_attn
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        pos = kwargs.get('position_embeddings')
        mask = kwargs.get('attention_mask')
        if hidden is None or pos is None:
            raise RuntimeError('Unexpected Qwen3-VL attention call signature.')
        qpos, hpos = self.query_positions.to(hidden.device), self.history_positions.to(hidden.device)
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(hidden).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        q, k = self.apply_rope(q, k, *pos)
        q = q.index_select(2, qpos)
        # Qwen3 uses grouped KV heads; repeat exactly as eager_attention_forward.
        if attn.num_key_value_groups != 1:
            b, kvh, seq, dim = k.shape
            k = k[:, :, None, :, :].expand(b, kvh, attn.num_key_value_groups, seq, dim).reshape(b, -1, seq, dim)
        scores = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        if mask is not None:
            scores = scores + mask.index_select(-2, qpos)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        hist = probs.index_select(-1, hpos)
        hist_mass = hist.sum(dim=-1)
        conditional = hist / (hist_mass.unsqueeze(-1) + 1e-12)
        self.conditional_sums.append(conditional.sum(dim=(0, 1, 2)).detach().float().cpu())
        self.mass_sums.append(hist_mass.sum().detach().float().cpu())
        self.count += int(hist_mass.numel())

    def __enter__(self):
        for layer in self.modules:
            self.handles.append(layer.register_forward_pre_hook(self._hook, with_kwargs=True))
        return self

    def __exit__(self, *unused):
        for h in self.handles: h.remove()
        self.handles = []

    def statistics(self):
        if not self.conditional_sums or not self.count:
            raise RuntimeError('No decoder attention rows were captured.')
        denom = float(self.count)
        return (torch.stack(self.conditional_sums).sum(0).numpy() / denom,
                float(torch.stack(self.mass_sums).sum().item() / denom))


def gold_from_row(row):
    answer=row.get('answer', None)
    if isinstance(answer, str) and answer.strip():
        return answer, None
    # AndroidControl-Curated has structured annotation rather than an `answer`
    # string. Convert only its documented gold fields; never use model output.
    action = str(row.get('gt_action', '') or '').strip()
    if action:
        input_text = row.get('gt_input_text', None)
        if input_text is not None and str(input_text).strip().lower() not in {'', 'none', 'nan', 'no input text'}:
            action = f'{action}:{str(input_text).strip().lower()}'
        payload = {'action_type': action}
        if action.split(':', 1)[0].lower() in {'click', 'long_press'}:
            bbox = row.get('gt_max_bbox', None)
            if bbox is None or (isinstance(bbox, float) and np.isnan(bbox)):
                bbox = row.get('gt_min_bbox', None)
            if isinstance(bbox, str):
                try: bbox = json.loads(bbox)
                except json.JSONDecodeError: bbox = None
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                return None, 'android_gold_has_no_valid_bbox_for_click_or_long_press'
            payload = {'bbox_2d': [int(round(float(x))) for x in bbox], 'action_type': action}
        return '<answer>' + json.dumps(payload, ensure_ascii=False) + '</answer>', None
    return None, 'no_direct_gold_action: requires answer or AndroidControl gt_action annotation'


def sampled_row_indices(data, num_samples, sample_by, samples_per_task, seed, candidate_indices=None):
    """Deterministic task-balanced selection, preserving within-task examples.

    GUI-Odyssey annotations do not consistently expose one task-id column across
    releases, so the explicit ids are preferred and task text is the safe fallback.
    """
    candidates = list(range(len(data))) if candidate_indices is None else list(candidate_indices)
    if sample_by == 'sequential':
        return candidates[:num_samples]
    groups = defaultdict(list)
    for i in candidates:
        row = data.iloc[i]
        key = row.get('task_id', row.get('episode_id', row.get('task', row.get('instruction', row.get('question', i)))))
        groups[str(key)].append(i)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    # Do not always choose the first complete-history step (usually step_4).
    # That would confound t-K with a trajectory's initial screenshot and its
    # first image position in the prompt.  Shuffle within each task deterministically.
    for key in keys:
        rng.shuffle(groups[key])
    selected = []
    # A round yields one sample per task before any task gets its second sample.
    for round_idx in range(max(1, samples_per_task)):
        for key in keys:
            candidates = groups[key]
            if round_idx < len(candidates):
                selected.append(candidates[round_idx])
                if len(selected) >= num_samples: return selected
    return selected


def complete_history_candidates(data, history_steps):
    """Pre-filter known trajectory metadata before task-balanced sampling."""
    if '_prev_image_paths' not in data.columns:
        return list(range(len(data)))
    return [i for i, value in enumerate(data['_prev_image_paths'])
            if isinstance(value, list) and len(value) >= history_steps]


def explicit_sample_indices(data, sample_ids):
    requested = [x.strip() for x in str(sample_ids).split(',') if x.strip()]
    by_id = {str(row.get('index', i)): i for i, (_, row) in enumerate(data.iterrows())}
    missing = [x for x in requested if x not in by_id]
    if missing:
        raise ValueError(f'Unknown --sample-ids: {missing}; these are dataset `index` values.')
    return [by_id[x] for x in requested]


def matched_random_boxes(box, size, n, rng):
    x1,y1,x2,y2=map(float,box); w,h=min(size[0],max(1,x2-x1)),min(size[1],max(1,y2-y1))
    # Draw displacement around true center then clip: same frame, exact size/aspect,
    # and center distribution remains centred on the true ROI rather than global.
    cx,cy=(x1+x2)/2,(y1+y2)/2; out=[]
    for _ in range(n):
        nx=np.clip(cx+rng.normal(0,size[0]/4),w/2,size[0]-w/2)
        ny=np.clip(cy+rng.normal(0,size[1]/4),h/2,size[1]-h/2)
        out.append([nx-w/2,ny-h/2,nx+w/2,ny+h/2])
    return out


def history_gt_from_row(row):
    """Extract the action target belonging to one historical screenshot.

    AndroidControl-Curated's gt_min_bbox is the tight UI-element target, while
    gt_max_bbox is a permissive official-evaluation region that may cover most
    of the screen.  Attention overlays must use the former and must never fall
    back to the latter.  The fallback names make the overlay useful for
    compatible GUI datasets too.
    """
    action = str(row.get('gt_action', row.get('action_type', '')) or '').strip()
    box = None
    for key in ('gt_min_bbox', 'gt_bbox', 'bbox'):
        value = row.get(key, None)
        if isinstance(value, str):
            try: value = ast.literal_eval(value)
            except (ValueError, SyntaxError): value = None
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                box = [float(x) for x in value]
                break
            except (TypeError, ValueError):
                pass
    return {'action_type': action, 'bbox_xyxy': box} if action or box is not None else None


def build_history_gt_lookup(data):
    lookup = {}
    for _, row in data.iterrows():
        path = str(row.get('image_path', row.get('image', '')) or '')
        gt = history_gt_from_row(row)
        if path and gt is not None:
            lookup[os.path.abspath(path)] = gt
    return lookup


def save_heatmap(path, image_path, heat, roi_boxes=None, gt=None, title=''):
    import matplotlib.pyplot as plt
    im=Image.open(image_path).convert('RGB')
    # Portrait Android screenshots were previously drawn in a 6x4 landscape
    # canvas. A single 32x32 visual cell became ~8x10 output pixels and an
    # alpha=.55 overlay blended even the maximum magma color into dark UI.
    # Keep a portrait canvas, crop its whitespace, and make relative maxima
    # visibly bright. Values are still frame-normalized I(k)/F_i.
    fig_w=4.5; fig_h=max(5.5, fig_w * im.height / im.width)
    fig,ax=plt.subplots(figsize=(fig_w,fig_h))
    ax.imshow(im)
    vmax=max(float(np.max(heat)), 1e-12)
    ax.imshow(heat, extent=(0,im.width,im.height,0), cmap='magma', alpha=.90,
              vmin=0.0, vmax=vmax, interpolation='nearest')
    for b in roi_boxes or []:
        ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='cyan',lw=1.5))
    # Green is the historical step's annotation/retained action ROI; cyan is an
    # optional external ROI JSON candidate. They must not be conflated.
    if gt and gt.get('bbox_xyxy') is not None:
        b = gt['bbox_xyxy']
        ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,ec='lime',lw=2.0))
        if gt.get('action_type'):
            ax.text(b[0], max(0, b[1]-4), f"GT: {gt['action_type']}", color='lime', fontsize=8,
                    bbox={'facecolor':'black','alpha':.55,'pad':1,'edgecolor':'none'})
    ax.set_title(title); ax.axis('off'); fig.tight_layout()
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.03); plt.close(fig)


def figures(root, samples, seed):
    import matplotlib.pyplot as plt
    frames=[f for s in samples for f in s.get('frames',[])]; steps=sorted(set(f['relative_step'] for f in frames))
    if not frames:
        (root/'figures'/'NO_SUCCESSFUL_HISTORY_FRAMES.txt').write_text(
            'No successful samples with history frames; inspect per_sample.jsonl skip_reason fields.\n'
        )
        return
    # Relative step is t-K...t-1, preserving actual prompt ordering.
    vals={z:[f['F_i'] for f in frames if f['relative_step']==z] for z in steps}
    means=[np.mean(vals[z]) for z in steps]
    cis=[]
    for z in steps:
        rec=[{'episode_id':f['episode_id'],'v':f['F_i']} for f in frames if f['relative_step']==z]
        cis.append(bootstrap_episode(rec,'v',seed=seed)['ci95'])
    fig,ax=plt.subplots(); ax.errorbar(steps,means,yerr=np.array([[m-c[0],c[1]-m] for m,c in zip(means,cis)]).T,fmt='o-'); ax.set(xlabel='history relative step',ylabel='frame attention F_i'); fig.tight_layout(); fig.savefig(root/'figures/frame_importance_ci.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(); ax.boxplot([[f['entropy'] for f in frames if f['relative_step']==z] for z in steps],tick_labels=steps); ax.set(xlabel='history relative step',ylabel='within-frame attention entropy'); fig.tight_layout(); fig.savefig(root/'figures/history_entropy.png',dpi=180); plt.close(fig)
    masses=[s['S_hist'] for s in samples if s.get('S_hist') is not None]
    if masses:
        fig,ax=plt.subplots(); ax.hist(masses,bins=min(30,len(masses))); ax.set(xlabel='S_hist',ylabel='samples'); fig.tight_layout(); fig.savefig(root/'figures/history_total_mass.png',dpi=180); plt.close(fig)
    roi=[r for s in samples for r in s.get('roi',[])]
    if roi:
        real=[r['enrichment'] for r in roi]; rnd=[x for r in roi for x in r['random_enrichment']]
        fig,ax=plt.subplots(); ax.violinplot([real,rnd],showmeans=True); ax.set_xticks([1,2],['real ROI','matched random']); ax.set_ylabel('ROI enrichment'); fig.tight_layout(); fig.savefig(root/'figures/roi_enrichment.png',dpi=180); plt.close(fig)
    # Six to twelve readable paper-case panels: entropy extremes (distributed vs
    # concentrated) plus largest last-frame share.  Each saved heatmap already
    # includes original screenshot, shared normalized overlay, ROI box, and F_i.
    scored=[]
    for s in samples:
        fs=s.get('frames',[])
        if fs:
            recent=max(fs,key=lambda x:x['relative_step'])['F_i'] / max(sum(x['F_i'] for x in fs),1e-12)
            scored.append((s, np.mean([x['entropy'] for x in fs]), recent))
    chosen=[]
    for ordered in (sorted(scored,key=lambda x:x[1]), sorted(scored,key=lambda x:-x[1]), sorted(scored,key=lambda x:-x[2])):
        for x in ordered:
            if x[0] not in chosen: chosen.append(x[0])
            if len(chosen)>=12: break
        if len(chosen)>=12: break
    for s in chosen:
        fs=s['frames']; fig,axes=plt.subplots(1,len(fs),figsize=(5*len(fs),4)); axes=np.atleast_1d(axes)
        for ax,f in zip(axes,fs): ax.imshow(plt.imread(f['heatmap_path'])); ax.axis('off')
        fig.suptitle(f"sample {s['sample_id']} | S_hist={s['S_hist']:.4f}"); fig.tight_layout(); fig.savefig(root/'figures'/f"case_{s['sample_id']}.png",dpi=180); plt.close(fig)
    # Single-step gallery, useful for inspecting the distribution of spatial maps.
    recent=[max(s['frames'],key=lambda x:x['relative_step']) for s in chosen if s.get('frames')]
    if recent:
        fig,axes=plt.subplots(1,len(recent),figsize=(4*len(recent),4)); axes=np.atleast_1d(axes)
        for ax,f in zip(axes,recent): ax.imshow(plt.imread(f['heatmap_path'])); ax.axis('off')
        fig.tight_layout(); fig.savefig(root/'figures/single_step_history_heatmap_gallery.png',dpi=180); plt.close(fig)


def main():
    a=args_parser().parse_args(); root=setup_dirs(a.work_dir)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    # Full-resolution observation: do not allow optional state-packet thumbnails/crops.
    os.environ['GUI_ODYSSEY_STATE_PACKET_ENABLE']='0'; os.environ['GUI_ODYSSEY_MAX_HISTORY_IMAGES']=str(a.history_steps)
    os.environ['ANDROID_CONTROL_STATE_PACKET_ENABLE']='0'; os.environ['ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS']='1'; os.environ['ANDROID_CONTROL_MAX_HISTORY_IMAGES']=str(a.history_steps)
    from vlmeval.dataset import build_dataset
    import vlmeval
    imported_root = Path(vlmeval.__file__).resolve().parents[1]
    if imported_root != REPO_ROOT:
        raise RuntimeError(
            f'Imported vlmeval from {imported_root}, expected {REPO_ROOT}. '
            'Remove the conflicting checkout from PYTHONPATH.'
        )
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dataset=build_dataset(a.data)
    if dataset is None: raise RuntimeError(f'Cannot build dataset {a.data}')
    processor=AutoProcessor.from_pretrained(a.model)
    # SDPA avoids materializing full SxS maps.  SelectiveAttentionCapture below
    # extracts exact gold-query rows from selected layers, so eager is unnecessary.
    model=AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype='auto',device_map='auto',attn_implementation='sdpa').eval()
    roi_map, roi_scale=(None,None) if not a.roi_json else load_compatible_roi_json(a.roi_json)
    metadata={'data':a.data,'model':a.model,'seed':a.seed,'layers':a.layers,'action_token_mode':a.action_token_mode,
              'sample_by':a.sample_by,'samples_per_task':a.samples_per_task,
              'attention_backend':'sdpa + exact selective gold-query attention rows','teacher_forcing':True,'use_cache':False,
              'roi_coordinate_note':'existing roi_results convention uses coordinate_scale=4 unless explicitly specified',
              'random_roi_sampling':'same-frame exact area/aspect boxes; Gaussian displacement around real ROI center then boundary clipping'}
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    output=[]; out_file=root/'per_sample.jsonl'; out_file.write_text(''); image_id=getattr(model.config,'image_token_id',None)
    if image_id is None: raise RuntimeError('Model config lacks image_token_id; Qwen3-VL mapping cannot be validated.')
    merge=getattr(getattr(model.config,'vision_config',None),'spatial_merge_size',1)
    data=getattr(dataset,'data',None)
    history_gt_lookup=build_history_gt_lookup(data)
    eligible_indices=complete_history_candidates(data,a.history_steps)
    selected_indices=(explicit_sample_indices(data,a.sample_ids) if a.sample_ids
                      else sampled_row_indices(data,a.num_samples,a.sample_by,a.samples_per_task,a.seed,eligible_indices))
    metadata['complete_history_candidates']=len(eligible_indices)
    metadata['selected_rows']=len(selected_indices)
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    for idx in selected_indices:
        row=data.iloc[idx]; sid=str(row.get('index',idx)); episode=str(row.get('episode_id',row.get('task_id',row.get('_trajectory_key',sid))))
        rec={'sample_id':sid,'episode_id':episode,'task_description':str(row.get('instruction',row.get('question',row.get('description','')))),'status':'skipped'}
        gold,why=gold_from_row(row)
        if why:
            rec['skip_reason']=why; output.append(rec)
            with out_file.open('a') as fh: fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
            continue
        try:
            prompt_obj=dataset.build_prompt(row); messages=prompt_obj[0] if isinstance(prompt_obj,tuple) else prompt_obj
            records=image_items(messages); hist=[x for x in records if x['kind']=='history']
            rec.update(gold_action=gold,history_image_paths=[x['path'] for x in hist])
            if len(hist)<a.history_steps: raise ValueError(f'incomplete_history: got {len(hist)}, need {a.history_steps}')
            prompt, full, _=make_inputs(processor,messages,gold)
            prompt_len=int(prompt['input_ids'].shape[1]); full_ids=full['input_ids'][0]
            tail=full_ids[prompt_len:].tolist()
            if not tail: raise ValueError('gold action produced no continuation tokens')
            token_texts=[processor.tokenizer.decode([x],clean_up_tokenization_spaces=False) for x in tail]
            selected=list(range(len(tail))) if a.action_token_mode=='all' else semantic_token_indices(token_texts)
            if not selected: raise ValueError('semantic filter selected no tokens; rerun --action-token-mode all and inspect token text')
            grids=grid_rows(full.get('image_grid_thw'),merge); segments=locate_visual_segments(full_ids,image_id,grids,records)
            hist_segments=[x for x in segments if x['kind']=='history']; hist_pos=[p for x in hist_segments for p in range(x['sequence_start'],x['sequence_end'])]
            queries=[prompt_len+j-1 for j in selected] # j zero based => |x|+j-1
            full=device_inputs(full,model)
            with torch.no_grad(), SelectiveAttentionCapture(model=model, query_positions=queries, history_positions=hist_pos, layers=a.layers) as capture:
                model(**full,output_attentions=False,use_cache=False,return_dict=True)
            importance,s_hist=capture.statistics()
            rec.update(status='ok',prompt_length=prompt_len,selected_query_tokens=[{'text':token_texts[j],'action_token_index':j,'query_position':queries[n]} for n,j in enumerate(selected)],S_hist=s_hist,visual_tokens=segments,frames=[])
            rng=np.random.default_rng(a.seed+idx)
            offset=0; rois=[]
            for f, seg in enumerate(hist_segments):
                n=seg['visual_tokens']; local=importance[offset:offset+n].reshape(seg['grid_t'],seg['grid_height'],seg['grid_width']).sum(0); offset+=n
                fi=float(local.sum()); norm=local/max(fi,1e-12); rel=f-len(hist_segments)
                flat_norm=norm.reshape(-1)
                top_count=max(1, int(np.ceil(0.01 * len(flat_norm))))
                top_idx=int(np.argmax(flat_norm)); peak_row, peak_col=divmod(top_idx, seg['grid_width'])
                spatial_stats={
                    'attention_peak': float(flat_norm[top_idx]),
                    'attention_top_1pct_mass': float(np.partition(flat_norm, -top_count)[-top_count:].sum()),
                    'attention_effective_tokens_exp_entropy': float(np.exp(entropy(norm))),
                    'attention_peak_grid_row_col': [int(peak_row), int(peak_col)],
                }
                path=root/'heatmaps'/f'{sid}_t{rel}.png'; boxes=[]
                if roi_map is not None: boxes=roi_boxes_for_path(roi_map,seg['path'],roi_scale)
                history_gt=history_gt_lookup.get(os.path.abspath(seg['path']))
                save_heatmap(path,seg['path'],norm,boxes,history_gt,f't{rel}: F={fi:.4f}')
                frame={'episode_id':episode,'history_step':seg['history_step'],'relative_step':rel,'F_i':fi,'entropy':entropy(norm),'heatmap_path':str(path),'image_path':seg['path'],'history_gt':history_gt,**spatial_stats}
                rec['frames'].append(frame)
                if a.save_attention_tensors=='1':
                    np.save(root/'heatmaps'/f'{sid}_t{rel}_spatial_attention.npy', norm.astype(np.float32))
                for box in boxes:
                    iw,ih=Image.open(seg['path']).size; mask=roi_mask_xyxy(box,seg['grid_height'],seg['grid_width'],(iw,ih)); count=int(mask.sum())
                    if not count: continue
                    mass=float(local[mask].sum()); enrich=mass/(count/len(hist_pos)); random_e=[]
                    for rb in matched_random_boxes(box,(iw,ih),a.random_roi_samples,rng):
                        rm=roi_mask_xyxy(rb,seg['grid_height'],seg['grid_width'],(iw,ih)); rc=max(1,int(rm.sum())); random_e.append(float(local[rm].sum()/(rc/len(hist_pos))) )
                    rois.append({'history_step':seg['history_step'],'box_xyxy':box,
                                 'ROI_mass':mass,'ROI_enrichment':enrich,
                                 'mass':mass,'enrichment':enrich, # backwards-readable aliases
                                 'random_enrichment':random_e,'random_percentile':float(np.mean(np.asarray(random_e)<=enrich))})
            rec['roi']=rois
            if a.save_attention_tensors=='1': torch.save({'I_history':torch.tensor(importance),'segments':segments},root/'heatmaps'/f'{sid}_attention.pt')
        except Exception as e:
            logging.exception('sample %s failed',sid); rec['skip_reason']=f'{type(e).__name__}: {e}'
        output.append(rec)
        with out_file.open('a') as fh: fh.write(json.dumps(rec,ensure_ascii=False)+'\n')
    ok=[x for x in output if x['status']=='ok']; figures(root,ok,a.seed)
    roi_records=[]
    for s in ok:
        for r in s.get('roi',[]):
            roi_records.append({'episode_id':s['episode_id'],'real':r['enrichment'],'random_mean':float(np.mean(r['random_enrichment']))})
    roi_delta=[{'episode_id':r['episode_id'],'delta':r['real']-r['random_mean']} for r in roi_records]
    roi_summary=bootstrap_episode(roi_delta,'delta',seed=a.seed)
    if roi_delta:
        # Sign-flip permutation test at episode level avoids treating patches as IID.
        ep=defaultdict(list)
        for r in roi_delta: ep[r['episode_id']].append(r['delta'])
        d=np.asarray([np.mean(v) for v in ep.values()]); rng=np.random.default_rng(a.seed)
        null=np.asarray([np.mean(d*rng.choice([-1,1],len(d))) for _ in range(10000)])
        roi_summary['paired_episode_signflip_pvalue']=float((np.abs(null)>=abs(d.mean())).mean())
    summary={'samples_requested':a.num_samples,'samples_ok':len(ok),'samples_skipped':len(output)-len(ok),
             'S_hist':bootstrap_episode(ok,'S_hist',seed=a.seed),
             'roi_real_minus_matched_random':roi_summary,
             'note':'CI uses episode bootstrap. Attention diagnoses read paths, not causality; ROI enrichment >1 is not performance contribution.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
