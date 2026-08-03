import torch
import time
import statistics
import re
import torch.distributed as dist
import os.path as osp
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'
SKIP_MISSING_IMAGE_MSG = '__SKIP_MISSING_IMAGE__'


def _extract_first_image_path(struct):
    if not isinstance(struct, list):
        return None
    for item in struct:
        if isinstance(item, dict) and item.get('type') == 'image':
            value = item.get('value')
            if value is not None and str(value).strip():
                return str(value)
    return None


def _extract_all_image_paths(struct):
    if not isinstance(struct, list):
        return []
    out = []
    for item in struct:
        if not isinstance(item, dict) or item.get('type') != 'image':
            continue
        value = item.get('value')
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _build_sample_meta(dataset_name, row, struct):
    image_path = str(row.get('image_path', '') or '') if row.get('image_path', None) is not None else _extract_first_image_path(struct)
    image_paths = _extract_all_image_paths(struct)
    if image_path and image_path not in image_paths:
        image_paths = list(image_paths) + [image_path]
    meta = {
        'dataset_name': str(dataset_name),
        'sample_index': str(row.get('index', '')),
        'image_path': image_path,
        'image_paths': image_paths,
    }
    for key in ('gt_action', 'answer', 'instruction', 'question', 'history', 'task_id', 'task_filename'):
        if key in row and row.get(key) is not None:
            meta[key] = row.get(key)
    return meta


def _androidcontrol_image_value(row):
    value = row.get('image_path', None)
    if value is None or str(value).strip() == '':
        value = row.get('image', '')
    return str(value or '')


def _androidcontrol_trajectory_key(row):
    image_value = _androidcontrol_image_value(row).replace('\\', '/').rstrip('/')
    if '/' not in image_value:
        return image_value
    return image_value.rsplit('/', 1)[0]


def _androidcontrol_step_idx(row):
    image_value = _androidcontrol_image_value(row)
    m = re.search(r'step_(\d+)\.[^.]+$', image_value)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _sort_androidcontrol_sequential(data):
    if len(data) == 0:
        return data
    ordered = data.copy()
    ordered['_tmp_traj_key'] = [_androidcontrol_trajectory_key(row) for _, row in ordered.iterrows()]
    ordered['_tmp_step_idx'] = [_androidcontrol_step_idx(row) for _, row in ordered.iterrows()]
    ordered['_tmp_row_order'] = list(range(len(ordered)))
    ordered = ordered.sort_values(
        by=['_tmp_traj_key', '_tmp_step_idx', '_tmp_row_order'],
        kind='stable',
    ).reset_index(drop=True)
    return ordered.drop(columns=['_tmp_traj_key', '_tmp_step_idx', '_tmp_row_order'])


def _task_key_for_sampling(dataset_name, row):
    dataset_name = str(dataset_name or '')
    if 'AndroidControl_Curated' in dataset_name:
        traj = _androidcontrol_trajectory_key(row)
        if traj:
            return f"trajectory:{traj}"
        for key in ('task_filename', 'task_id', 'episode', 'revised_task', 'instruction'):
            value = row.get(key, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return f'{key}:{text}'
        return f"index:{row.get('index', '')}"

    if 'GUIOdyssey' in dataset_name:
        for key in ('task_id', 'episode'):
            value = row.get(key, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return f'{key}:{text}'
        image_value = row.get('image', row.get('image_path', ''))
        image_name = osp.basename(str(image_value))
        stem = osp.splitext(image_name)[0]
        if '_' in stem:
            return f"image_prefix:{stem.rsplit('_', 1)[0]}"
        for key in ('instruction', 'question', 'category'):
            value = row.get(key, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return f'{key}:{text}'
        return f"image:{stem}"

    for key in ('task_id', 'episode', 'task_filename', 'instruction', 'question'):
        value = row.get(key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f'{key}:{text}'
    return f"index:{row.get('index', '')}"


def _maybe_apply_eval_sampling(data, dataset_name, rank=0):
    mode = os.getenv('VLM_EVAL_SAMPLE_MODE', 'off').strip().lower()
    if mode in ('', '0', 'off', 'false', 'none'):
        if 'AndroidControl_Curated' in str(dataset_name or ''):
            return _sort_androidcontrol_sequential(data)
        return data

    seed = int(os.getenv('VLM_EVAL_SAMPLE_SEED', os.getenv('SEED', '42')))

    if mode == 'task':
        task_keys = [_task_key_for_sampling(dataset_name, row) for _, row in data.iterrows()]
        unique_keys = sorted(set(task_keys))
        target = int(os.getenv('VLM_EVAL_SAMPLE_TASKS', os.getenv('VLM_EVAL_SAMPLE_COUNT', '200')))
        if target <= 0 or target >= len(unique_keys):
            if rank == 0:
                print(
                    f'[EvalSample] mode=task disabled_effectively target={target} '
                    f'available_tasks={len(unique_keys)} kept_steps={len(data)} seed={seed}',
                    flush=True,
                )
            return data

        rng = np.random.default_rng(seed)
        selected_keys = set(rng.choice(unique_keys, size=target, replace=False).tolist())
        keep_mask = [key in selected_keys for key in task_keys]
        filtered = data.loc[keep_mask].copy()
        if 'AndroidControl_Curated' in str(dataset_name or ''):
            filtered = _sort_androidcontrol_sequential(filtered)
        if rank == 0:
            print(
                f'[EvalSample] mode=task seed={seed} selected_tasks={target}/{len(unique_keys)} '
                f'kept_steps={len(filtered)}/{len(data)}',
                flush=True,
            )
        return filtered

    if mode == 'sample':
        target = int(os.getenv('VLM_EVAL_SAMPLE_COUNT', '200'))
        if target <= 0 or target >= len(data):
            if rank == 0:
                print(
                    f'[EvalSample] mode=sample disabled_effectively target={target} '
                    f'available_steps={len(data)} seed={seed}',
                    flush=True,
                )
            return data
        rng = np.random.default_rng(seed)
        picked = np.sort(rng.choice(len(data), size=target, replace=False))
        filtered = data.iloc[picked].copy()
        if 'AndroidControl_Curated' in str(dataset_name or ''):
            filtered = _sort_androidcontrol_sequential(filtered)
        if rank == 0:
            print(
                f'[EvalSample] mode=sample seed={seed} selected_steps={len(filtered)}/{len(data)}',
                flush=True,
            )
        return filtered

    if rank == 0:
        print(f'[EvalSample] unknown mode={mode}, fallback to full dataset', flush=True)
    return data


def _write_experiment_summary(work_dir, summary: dict) -> None:
    try:
        dump(summary, osp.join(work_dir, 'summary.json'))
    except Exception as err:
        warnings.warn(f'[SummaryWriteFailed] work_dir={work_dir} error={err}')


def build_sheet_indices(dataset_len, rank, world_size):
    shuffle_before_split = os.getenv('VLM_SHUFFLE_BEFORE_SPLIT', '1') == '1'
    if not shuffle_before_split or world_size <= 1:
        return list(range(rank, dataset_len, world_size))

    split_seed = int(os.getenv('VLM_SPLIT_SEED', os.getenv('SEED', '42')))
    rng = np.random.default_rng(split_seed)
    all_indices = rng.permutation(dataset_len).tolist()

    if rank == 0:
        print(f'[Shard] Using deterministic shuffled split with seed={split_seed}, world_size={world_size}')

    return all_indices[rank::world_size]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(model, work_dir, model_name, dataset, index_set=None, api_nproc=4, ignore_failed=False):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    data = dataset.data
    data = _maybe_apply_eval_sampling(data, dataset_name, rank=rank)
    if index_set is not None:
        data = data[data['index'].isin(index_set)]

    model = supported_VLM[model_name]() if isinstance(model, str) else model
    assert getattr(model, 'is_api', False)
    if hasattr(model, 'set_dump_image'):
        model.set_dump_image(dataset.dump_image)

    lt, indices = len(data), list(data['index'])

    structs = []
    for i in range(lt):
        item = data.iloc[i]
        if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
            assert hasattr(model, 'build_prompt')
            struct = model.build_prompt(item, dataset=dataset_name)
        else:
            struct = dataset.build_prompt(item)
        structs.append(struct)

    out_file = f'{work_dir}/{model_name}_{dataset_name}_supp.pkl'

    # To reuse records in MMBench_V11
    if dataset_name in ['MMBench', 'MMBench_CN']:
        pred_format = get_pred_file_format()
        v11_pred = f'{work_dir}/{model_name}_{dataset_name}_V11.{pred_format}'
        if osp.exists(v11_pred):
            try:
                reuse_inds = load('http://opencompass.openxlab.space/utils/mmb_reuse.pkl')
                data = load(v11_pred)
                ans_map = {x: y for x, y in zip(data['index'], data['prediction']) if x in reuse_inds}
                dump(ans_map, out_file)
            except Exception as err:
                print(type(err), err)

    res = {}
    if osp.exists(out_file):
        res = load(out_file)
        if ignore_failed:
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    if index_set is not None:
        res = {k: v for k, v in res.items() if k in index_set}
    os.remove(out_file)
    return res


def infer_data(model, model_name, work_dir, dataset, out_file, verbose=False, api_nproc=4, use_vllm=False):
    dataset_name = dataset.dataset_name
    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    res = load(prev_file) if osp.exists(prev_file) else {}
    if osp.exists(out_file):
        res.update(load(out_file))

    rank, world_size = get_rank_and_world_size()
    base_data = _maybe_apply_eval_sampling(dataset.data, dataset_name, rank=rank)
    sheet_indices = build_sheet_indices(len(base_data), rank, world_size)
    lt = len(sheet_indices)
    data = base_data.iloc[sheet_indices]
    data_indices = [i for i in data['index']]

    # If finished, will exit without building the model
    all_finished = True
    for i in range(lt):
        idx = data.iloc[i]['index']
        if idx not in res:
            all_finished = False
    if all_finished:
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model

    # Data need to be inferred
    data = data[~data['index'].isin(res)]
    lt = len(data)

    kwargs = {}
    if model_name is not None and (
        'Llama-4' in model_name
        or 'Qwen2-VL' in model_name
        or 'Qwen2.5-VL' in model_name
    ):
        kwargs = {'use_vllm': use_vllm}

    # (25.06.05) In newer version of transformers (after 4.50), with device_map='auto' and torchrun launcher,
    # Transformers automatically adopt TP parallelism, which leads to compatibility problems with VLMEvalKit
    # (In VLMEvalKit, we use torchrun to launch multiple model instances on a single node).
    # To bypass this problem, we unset `WORLD_SIZE` before building the model to not use TP parallel.
    ws_bak = os.environ.pop('WORLD_SIZE', None)
    model = supported_VLM[model_name](**kwargs) if isinstance(model, str) else model
    if ws_bak:
        os.environ['WORLD_SIZE'] = ws_bak

    is_api = getattr(model, 'is_api', False)
    if is_api:
        lt, indices = len(data), list(data['index'])
        supp = infer_data_api(
            model=model,
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            index_set=set(indices),
            api_nproc=api_nproc)
        for idx in indices:
            assert idx in supp
        res.update(supp)
        res = {k: res[k] for k in data_indices}
        dump(res, out_file)
        return model
    else:
        model.set_dump_image(dataset.dump_image)

    stop_after_index = None
    timing_enabled = os.getenv('VLM_TIMING', '0') == '1'
    timing_verbose = os.getenv('VLM_TIMING_VERBOSE', '0') == '1'
    timing_sync = os.getenv('VLM_TIMING_SYNC', '1') == '1'
    progress_interval = int(os.getenv('VLM_PROGRESS_INTERVAL', '100'))
    progress_acc = os.getenv('VLM_PROGRESS_ACC', '1') == '1'
    timing_records = [] if timing_enabled else None
    infer_records = [] if timing_enabled else None
    processed_count = 0
    online_correct = 0
    online_total = 0
    summary = {
        'dataset': str(dataset_name),
        'model': str(model_name),
        'work_dir': str(work_dir),
        'world_size': int(world_size),
        'timing_enabled': bool(timing_enabled),
        'stage_timing_enabled': bool(os.getenv('VLM_STAGE_TIMING', '0') == '1'),
        'prune_timing_enabled': bool(os.getenv('VLM_PRUNE_TIMING', '0') == '1'),
        'roi_prune_enabled': bool(os.getenv('QWEN3VL_ENABLE_ROI_PRUNE', '0') == '1'),
        'roi_prune_json': str(os.getenv('QWEN3VL_ROI_PRUNE_JSON', '') or ''),
        'roi_prune_layer_order': int(os.getenv('QWEN3VL_ROI_PRUNE_LAYER_ORDER', '16') or 16),
        'roi_prune_topk_keep': int(os.getenv('QWEN3VL_ROI_PRUNE_TOPK_KEEP', '4') or 4),
        'roi_prune_uniform_keep_every': int(os.getenv('QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_EVERY', '0') or 0),
        'roi_prune_uniform_keep_offset': int(os.getenv('QWEN3VL_ROI_PRUNE_UNIFORM_KEEP_OFFSET', '0') or 0),
        'roi_prune_use_cache': bool(os.getenv('QWEN3VL_ROI_PRUNE_USE_CACHE', '0') == '1'),
        'eval_sample_mode': str(os.getenv('VLM_EVAL_SAMPLE_MODE', 'off') or 'off'),
        'eval_sample_tasks': int(os.getenv('VLM_EVAL_SAMPLE_TASKS', os.getenv('VLM_EVAL_SAMPLE_COUNT', '200')) or 200),
        'eval_sample_seed': int(os.getenv('VLM_EVAL_SAMPLE_SEED', os.getenv('SEED', '42')) or 42),
        'processed_samples': 0,
    }

    pbar = tqdm(range(lt), desc=f'Infer {model_name}/{dataset_name}, Rank {rank}/{world_size}')
    for i in pbar:
        idx = data.iloc[i]['index']
        if idx in res:
            continue

        prompt_time = 0.0
        if timing_enabled:
            if timing_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

        try:
            if hasattr(model, 'use_custom_prompt') and model.use_custom_prompt(dataset_name):
                struct = model.build_prompt(data.iloc[i], dataset=dataset_name)
            else:
                struct = dataset.build_prompt(data.iloc[i])
        except AssertionError as e:
            # Optional: skip broken samples (e.g., missing image files) instead of aborting the whole run.
            if os.getenv('VLM_SKIP_MISSING_IMAGE', '1') == '1' and 'screenshot not found' in str(e):
                warnings.warn(f'[SkipMissingImage] index={idx} reason={e}')
                res[idx] = SKIP_MISSING_IMAGE_MSG
                if (i + 1) % 10 == 0:
                    dump(res, out_file)
                continue
            raise
        if timing_enabled:
            prompt_time = time.perf_counter() - start_time
        extra_kwargs = {}

        if isinstance(struct, tuple): # by jingyz1
            extra_kwargs = struct[1]
            struct = struct[0]
        extra_kwargs = dict(extra_kwargs or {})
        extra_kwargs['sample_meta'] = _build_sample_meta(dataset_name, data.iloc[i], struct)

        # If `SKIP_ERR` flag is set, the model will skip the generation if error is encountered
        if os.environ.get('SKIP_ERR', False) == '1':
            FAIL_MSG = 'Failed to obtain answer'
            try:
                response = model.generate(message=struct, dataset=dataset_name)
            except RuntimeError as err:
                torch.cuda.synchronize()
                warnings.warn(f'{type(err)} {str(err)}')
                response = f'{FAIL_MSG}: {type(err)} {str(err)}'
        else:
            response = model.generate(message=struct, dataset=dataset_name, **extra_kwargs) # jingyz1
        if timing_enabled:
            if timing_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            timing_records.append((idx, elapsed))
            infer_time = elapsed - prompt_time
            infer_records.append((idx, infer_time))
            if timing_verbose:
                print(f'[Timing] index={idx} seconds={elapsed:.6f} prompt_time={prompt_time:.6f} infer_time={infer_time:.6f}', flush=True)
            else:
                pbar.set_postfix({'sec': f'{elapsed:.3f}'})
        processed_count += 1

        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        if progress_acc and hasattr(dataset, 'eval_single'):
            try:
                one = dataset.eval_single(data.iloc[i], response)
                online_total += 1
                if one.get('is_correct', False):
                    online_correct += 1
            except Exception:
                pass

        res[idx] = response
        if (i + 1) % 10 == 0:
            dump(res, out_file)
        if progress_interval > 0 and processed_count % progress_interval == 0:
            acc_msg = ''
            if progress_acc and online_total > 0:
                cur_acc = 100.0 * online_correct / online_total
                acc_msg = f' online_acc={cur_acc:.2f}% ({online_correct}/{online_total})'
            if timing_enabled and timing_records:
                recent = timing_records[-progress_interval:] if len(timing_records) >= progress_interval else timing_records
                recent_avg = sum(x[1] for x in recent) / len(recent)
                total_avg = sum(x[1] for x in timing_records) / len(timing_records)
                print(
                    f'[Progress] processed={processed_count}/{lt} recent_avg_s={recent_avg:.4f} total_avg_s={total_avg:.4f}{acc_msg}',
                    flush=True
                )
            else:
                print(f'[Progress] processed={processed_count}/{lt}{acc_msg}', flush=True)
        if stop_after_index is not None and str(idx) == str(stop_after_index):
            dump(res, out_file)
            raise SystemExit(0)

    res = {k: res[k] for k in data_indices}
    dump(res, out_file)
    if timing_enabled and timing_records:
        durations = [x[1] for x in timing_records]
        avg = sum(durations) / len(durations)
        med = statistics.median(durations)
        total = sum(durations)
        summary.update(
            {
                'processed_samples': int(len(durations)),
                'avg_total_wall_s': float(avg),
                'median_total_wall_s': float(med),
                'total_wall_s': float(total),
            }
        )
        print(
            f'[Timing] samples={len(durations)} avg_s={avg:.6f} median_s={med:.6f} total_s={total:.3f}',
            flush=True
        )

        durations = [x[1] for x in infer_records]
        avg = sum(durations) / len(durations)
        med = statistics.median(durations)
        total = sum(durations)
        summary.update(
            {
                'avg_infer_wall_s': float(avg),
                'median_infer_wall_s': float(med),
                'total_infer_wall_s': float(total),
            }
        )
        print(
            f'[Timing] samples={len(durations)} infer avg_s={avg:.6f} median_s={med:.6f} total_s={total:.3f}',
            flush=True
        )
    if os.getenv('VLM_STAGE_TIMING', '0') == '1' and hasattr(model, '_vlmeval_stage_records'):
        recs = getattr(model, '_vlmeval_stage_records', None) or []
        vision_vals = [r.get('vision_s') for r in recs if isinstance(r, dict) and r.get('vision_s') is not None]
        llm_vals = [r.get('llm_s') for r in recs if isinstance(r, dict) and r.get('llm_s') is not None]
        local_count = min(len(vision_vals), len(llm_vals))
        local_vision_sum = float(sum(vision_vals[:local_count]))
        local_llm_sum = float(sum(llm_vals[:local_count]))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        stats = torch.tensor([local_vision_sum, local_llm_sum, float(local_count)], device=device, dtype=torch.float64)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_vision_sum, total_llm_sum, total_count = stats.tolist()
        if total_count > 0:
            summary.update(
                {
                    'stage_timing_samples': int(total_count),
                    'avg_visual_encode_s': float(total_vision_sum / total_count),
                    'avg_llm_stage_s': float(total_llm_sum / total_count),
                    'total_visual_encode_s': float(total_vision_sum),
                    'total_llm_stage_s': float(total_llm_sum),
                }
            )
            print(
                f'[StageTimingSummary] samples={int(total_count)} vision_avg_s={total_vision_sum/total_count:.6f} llm_avg_s={total_llm_sum/total_count:.6f}',
                flush=True,
            )
    if hasattr(model, '_vlmeval_generate_timing_records'):
        recs = getattr(model, '_vlmeval_generate_timing_records', None) or []
        keys = [
            'vision_s',
            'prefill_s',
            'decode_s',
            'prefill_before_prune_layer_s',
            'prefill_split_to_prune_start_s',
            'prune_layer_to_prefill_end_s',
            'split_layer_to_prefill_end_without_prune_s',
            'prune_selection_s',
            'prune_op_s',
            'prune_layer_to_finish_s',
            'total_s',
        ]
        local = {k: [float(r.get(k, 0.0) or 0.0) for r in recs if isinstance(r, dict)] for k in keys}
        local_count = len(recs)
        if local_count > 0:
            values = [float(sum(local[k])) for k in keys] + [float(local_count)]
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            stats = torch.tensor(values, device=device, dtype=torch.float64)
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            vals = stats.tolist()
            total_count = vals[-1]
            if total_count > 0:
                avgs = {k: vals[i] / total_count for i, k in enumerate(keys)}
                summary.update(
                    {
                        'generate_timing_samples': int(total_count),
                        'avg_encode_s': float(avgs['vision_s']),
                        'avg_prefill_s': float(avgs['prefill_s']),
                        'avg_decode_s': float(avgs['decode_s']),
                        'avg_prefill_before_prune_layer_s': float(avgs['prefill_before_prune_layer_s']),
                        'avg_prefill_split_to_prune_start_s': float(avgs['prefill_split_to_prune_start_s']),
                        'avg_prune_layer_to_prefill_end_s': float(avgs['prune_layer_to_prefill_end_s']),
                        'avg_split_layer_to_prefill_end_without_prune_s': float(avgs['split_layer_to_prefill_end_without_prune_s']),
                        'avg_prune_selection_s': float(avgs['prune_selection_s']),
                        'avg_prune_op_s': float(avgs['prune_op_s']),
                        'avg_prune_layer_to_finish_s': float(avgs['prune_layer_to_finish_s']),
                        'avg_total_generate_s': float(avgs['total_s']),
                        'total_encode_s': float(vals[0]),
                        'total_prefill_s': float(vals[1]),
                        'total_decode_s': float(vals[2]),
                        'total_prefill_before_prune_layer_s': float(vals[3]),
                        'total_prefill_split_to_prune_start_s': float(vals[4]),
                        'total_prune_layer_to_prefill_end_s': float(vals[5]),
                        'total_split_layer_to_prefill_end_without_prune_s': float(vals[6]),
                        'total_prune_selection_s': float(vals[7]),
                        'total_prune_op_s': float(vals[8]),
                        'total_prune_layer_to_finish_s': float(vals[9]),
                        'total_generate_s': float(vals[10]),
                    }
                )
                local_decode_tokens_sum = 0.0
                local_decode_steps_sum = 0.0
                local_prompt_seq_tokens_sum = 0.0
                local_template_enabled_count = 0.0
                local_template_static_token_count_sum = 0.0
                local_template_static_decode_steps_sum = 0.0
                local_template_unknown_decode_steps_sum = 0.0
                local_template_total_decode_tokens_sum = 0.0
                for rec in recs:
                    if not isinstance(rec, dict):
                        continue
                    local_decode_tokens_sum += float(rec.get('decode_tokens', 0.0) or 0.0)
                    local_decode_steps_sum += float(rec.get('decode_steps', 0.0) or 0.0)
                    local_prompt_seq_tokens_sum += float(rec.get('prompt_seq_tokens', 0.0) or 0.0)
                    enabled = float(bool(rec.get('template_prefill_enabled', False)))
                    local_template_enabled_count += enabled
                    local_template_static_token_count_sum += float(rec.get('template_static_token_count', 0.0) or 0.0)
                    local_template_static_decode_steps_sum += float(rec.get('template_static_decode_steps', 0.0) or 0.0)
                    local_template_unknown_decode_steps_sum += float(rec.get('template_unknown_decode_steps', 0.0) or 0.0)
                    local_template_total_decode_tokens_sum += float(rec.get('template_decode_tokens', 0.0) or 0.0)

                extra = torch.tensor(
                    [
                        local_decode_tokens_sum,
                        local_decode_steps_sum,
                        local_prompt_seq_tokens_sum,
                        local_template_enabled_count,
                        local_template_static_token_count_sum,
                        local_template_static_decode_steps_sum,
                        local_template_unknown_decode_steps_sum,
                        local_template_total_decode_tokens_sum,
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1 and dist.is_available() and dist.is_initialized():
                    dist.all_reduce(extra, op=dist.ReduceOp.SUM)
                extra_vals = extra.tolist()
                total_template_enabled = extra_vals[3]
                summary.update(
                    {
                        'avg_decode_tokens': float(extra_vals[0] / total_count),
                        'total_decode_tokens': float(extra_vals[0]),
                        'avg_decode_steps': float(extra_vals[1] / total_count),
                        'total_decode_steps': float(extra_vals[1]),
                        'avg_prompt_seq_tokens': float(extra_vals[2] / total_count),
                        'total_prompt_seq_tokens': float(extra_vals[2]),
                        'template_prefill_enabled_count': int(total_template_enabled),
                        'avg_template_static_token_count': float(extra_vals[4] / total_count),
                        'total_template_static_token_count': float(extra_vals[4]),
                        'avg_template_static_decode_steps': float(extra_vals[5] / total_count),
                        'total_template_static_decode_steps': float(extra_vals[5]),
                        'avg_template_unknown_decode_steps': float(extra_vals[6] / total_count),
                        'total_template_unknown_decode_steps': float(extra_vals[6]),
                        'avg_template_total_decode_tokens': float(extra_vals[7] / total_count),
                        'total_template_total_decode_tokens': float(extra_vals[7]),
                    }
                )
                if total_template_enabled > 0:
                    summary.update(
                        {
                            'avg_template_static_token_count_on_enabled': float(extra_vals[4] / total_template_enabled),
                            'avg_template_static_decode_steps_on_enabled': float(extra_vals[5] / total_template_enabled),
                            'avg_template_unknown_decode_steps_on_enabled': float(extra_vals[6] / total_template_enabled),
                            'avg_template_total_decode_tokens_on_enabled': float(extra_vals[7] / total_template_enabled),
                        }
                    )
                print(
                    '[GenerateTimingSummary] '
                    f"samples={int(total_count)} "
                    f"encode_avg_s={avgs['vision_s']:.6f} "
                    f"prefill_avg_s={avgs['prefill_s']:.6f} "
                    f"decode_avg_s={avgs['decode_s']:.6f} "
                    f"decode_tokens_avg={extra_vals[0] / total_count:.6f} "
                    f"decode_steps_avg={extra_vals[1] / total_count:.6f} "
                    f"template_static_decode_steps_avg={extra_vals[5] / total_count:.6f} "
                    f"template_unknown_decode_steps_avg={extra_vals[6] / total_count:.6f} "
                    f"prefill_before_prune_layer_avg_s={avgs['prefill_before_prune_layer_s']:.6f} "
                    f"prefill_split_to_prune_start_avg_s={avgs['prefill_split_to_prune_start_s']:.6f} "
                    f"prune_layer_to_prefill_end_avg_s={avgs['prune_layer_to_prefill_end_s']:.6f} "
                    f"split_layer_to_prefill_end_without_prune_avg_s={avgs['split_layer_to_prefill_end_without_prune_s']:.6f} "
                    f"prune_selection_avg_s={avgs['prune_selection_s']:.6f} "
                    f"prune_op_avg_s={avgs['prune_op_s']:.6f} "
                    f"prune_layer_to_finish_avg_s={avgs['prune_layer_to_finish_s']:.6f} "
                    f"total_generate_avg_s={avgs['total_s']:.6f}",
                    flush=True,
                )
    if os.getenv('VLM_PRUNE_TIMING', '0') == '1' and hasattr(model, '_vlmeval_prune_records'):
        recs = getattr(model, '_vlmeval_prune_records', None) or []
        sort_vals = [r.get('sort_s') for r in recs if isinstance(r, dict) and r.get('sort_s') is not None]
        recycle_vals = [r.get('recycle_s') for r in recs if isinstance(r, dict) and r.get('recycle_s') is not None]
        local_count = min(len(sort_vals), len(recycle_vals))
        local_sort_sum = float(sum(sort_vals[:local_count]))
        local_recycle_sum = float(sum(recycle_vals[:local_count]))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        stats = torch.tensor([local_sort_sum, local_recycle_sum, float(local_count)], device=device, dtype=torch.float64)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_sort_sum, total_recycle_sum, total_count = stats.tolist()
        if total_count > 0:
            local_applied = 0.0
            local_top4_empty = 0.0
            local_seq_before_sum = 0.0
            local_seq_after_sum = 0.0
            local_visual_before_sum = 0.0
            local_visual_after_sum = 0.0
            for rec in recs:
                if not isinstance(rec, dict):
                    continue
                st = rec.get('stats', {}) or {}
                local_applied += float(bool(st.get('prune_applied', False)))
                local_top4_empty += float(len(st.get('selected_top4_grids', []) or []) == 0)
                local_seq_before_sum += float(st.get('seq_tokens_before', 0.0) or 0.0)
                local_seq_after_sum += float(st.get('seq_tokens_after', 0.0) or 0.0)
                local_visual_before_sum += float(st.get('visual_tokens_before', 0.0) or 0.0)
                local_visual_after_sum += float(st.get('visual_tokens_after', 0.0) or 0.0)

            local_decode_tokens_sum = 0.0
            local_prompt_tokens_sum = 0.0
            local_decode_steps_sum = 0.0
            gen_recs = getattr(model, '_vlmeval_generate_timing_records', None) or []
            for rec in gen_recs:
                if not isinstance(rec, dict):
                    continue
                local_decode_tokens_sum += float(rec.get('decode_tokens', 0.0) or 0.0)
                local_prompt_tokens_sum += float(rec.get('prompt_seq_tokens', 0.0) or 0.0)
                local_decode_steps_sum += float(rec.get('decode_steps', 0.0) or 0.0)

            extra = torch.tensor(
                [
                    local_applied,
                    local_top4_empty,
                    local_seq_before_sum,
                    local_seq_after_sum,
                    local_visual_before_sum,
                    local_visual_after_sum,
                    local_decode_tokens_sum,
                    local_prompt_tokens_sum,
                    local_decode_steps_sum,
                ],
                device=device,
                dtype=torch.float64,
            )
            if world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.all_reduce(extra, op=dist.ReduceOp.SUM)
            extra_vals = extra.tolist()
            summary.update(
                {
                    'prune_timing_samples': int(total_count),
                    'avg_prune_sort_s': float(total_sort_sum / total_count),
                    'avg_prune_recycle_s': float(total_recycle_sum / total_count),
                    'total_prune_sort_s': float(total_sort_sum),
                    'total_prune_recycle_s': float(total_recycle_sum),
                    'prune_applied_count': int(extra_vals[0]),
                    'top4_empty_count': int(extra_vals[1]),
                    'avg_seq_tokens_before': float(extra_vals[2] / total_count),
                    'avg_seq_tokens_after': float(extra_vals[3] / total_count),
                    'avg_visual_tokens_before': float(extra_vals[4] / total_count),
                    'avg_visual_tokens_after': float(extra_vals[5] / total_count),
                    'avg_decode_tokens': float(extra_vals[6] / total_count),
                    'avg_prompt_seq_tokens': float(extra_vals[7] / total_count),
                    'avg_decode_steps': float(extra_vals[8] / total_count),
                    'total_decode_tokens': float(extra_vals[6]),
                    'total_prompt_seq_tokens': float(extra_vals[7]),
                    'total_decode_steps': float(extra_vals[8]),
                }
            )
            print(
                f'[PruneTimingSummary] samples={int(total_count)} sort_avg_s={total_sort_sum/total_count:.6f} recycle_avg_s={total_recycle_sum/total_count:.6f}',
                flush=True,
            )
    if rank == 0:
        _write_experiment_summary(work_dir, summary)
    return model


# Add for agent evaluation
def _is_structured_record(v):
    return isinstance(v, dict) and 'prediction' in v and 'extra_records' in v


# A wrapper for infer_data, do the pre & post processing
def infer_data_job(
    model, work_dir, model_name, dataset, verbose=False, api_nproc=4, ignore_failed=False, use_vllm=False
):
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name
    # 使用环境变量控制的文件格式
    result_file = get_pred_file_path(work_dir, model_name, dataset_name, use_env_format=True)

    prev_file = f'{work_dir}/{model_name}_{dataset_name}_PREV.pkl'
    if osp.exists(result_file):
        if rank == 0:
            data = load(result_file)
            # breakpoint()
            results = {k: v for k, v in zip(data['index'], data['prediction'])}
            if not ignore_failed:
                results = {k: v for k, v in results.items() if FAIL_MSG not in str(v)}
            dump(results, prev_file)
        if world_size > 1:
            dist.barrier()

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model=model, work_dir=work_dir, model_name=model_name, dataset=dataset,
        out_file=out_file, verbose=verbose, api_nproc=api_nproc, use_vllm=use_vllm)
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        data = _maybe_apply_eval_sampling(dataset.data, dataset_name, rank=rank).copy()
        for x in data['index']:
            assert x in data_all
        if os.getenv('SPLIT_THINK', False):
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                prediction = [data_all[x]['prediction'] for x in data['index']]
                extra_records = [data_all[x]['extra_records'] for x in data['index']]
                data['extra_records'] = extra_records
            else:
                prediction = [str(data_all[x]) for x in data['index']]

            def split_thinking(s):
                if '</think>' in s:
                    splits = s.split('</think>')
                    prediction = splits[-1].strip()
                    if len(splits) == 2 and '<think>' in splits[0]:
                        thinking = splits[0].split('<think>')[1].strip()
                    else:
                        thinking = '</think>'.join(splits[:-1])
                        thinking += '</think>'
                        warnings.warn('Failed to parse thinking, multiple </think> tags or missing <think> tag.')
                else:
                    thinking = ''
                    prediction = s
                return (prediction, thinking)
            split_func = model.split_thinking if hasattr(model, 'split_thinking') else split_thinking
            print(f'Prediction format: {os.getenv("SPLIT_THINK")},splitting func: {split_func}')
            tups = [split_func(x) for x in prediction]
            data['prediction'] = [x[0] for x in tups]
            data['thinking'] = [x[1] for x in tups]
        else:
            # data['prediction'] = [str(data_all[x]) for x in data['index']]
            # Add for agent evaluation
            if all(_is_structured_record(data_all[x]) for x in data['index']):
                data['prediction'] = [data_all[x]['prediction'] for x in data['index']]
                data['extra_records'] = [data_all[x]['extra_records'] for x in data['index']]
            else:
                data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')

        dump(data, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    if world_size > 1:
        dist.barrier()
    return model
