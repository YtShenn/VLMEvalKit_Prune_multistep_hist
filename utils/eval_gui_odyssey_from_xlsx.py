import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SKIP_MISSING_IMAGE_MSG = '__SKIP_MISSING_IMAGE__'
METRIC_MAP = {
    'GUIOdyssey_high_app_split': 'macro',
    'GUIOdyssey_high_device_split': 'macro',
    'GUIOdyssey_high_random_split': 'micro',
    'GUIOdyssey_high_task_split': 'macro',
    'GUIOdyssey_low_app_split': 'macro',
    'GUIOdyssey_low_device_split': 'macro',
    'GUIOdyssey_low_random_split': 'micro',
    'GUIOdyssey_low_task_split': 'macro',
}


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def text_matching(gt: str, pred: str, threshold: float = 0.5) -> bool:
    gt = str(gt).strip()
    pred = str(pred).strip()
    if gt in pred or pred in gt:
        return True
    dist = levenshtein_distance(gt, pred)
    length = max(len(gt), len(pred))
    score = 0.0 if length == 0 else 1 - float(dist) / float(length)
    return score >= threshold


def _safe_eval_tuple_like(val):
    if isinstance(val, (list, tuple)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return val


def click_matching(gt_info, pred_info, sam2_bbox=None, coord_threshold=0.14):
    pred_info = _safe_eval_tuple_like(pred_info)
    gt_info = _safe_eval_tuple_like(gt_info)

    if sam2_bbox is not None and isinstance(sam2_bbox, list) and len(sam2_bbox) == 4:
        try:
            x1, y1, x2, y2 = sam2_bbox
            px, py = float(pred_info[0]), float(pred_info[1])
            if x1 <= px <= x2 and y1 <= py <= y2:
                return True
        except Exception:
            pass

    try:
        pred = np.asarray(pred_info, dtype=float) / 1000
        gt = np.asarray(gt_info, dtype=float) / 1000
        return np.linalg.norm(pred - gt) <= coord_threshold
    except Exception:
        return False


def action_matching(pred_action, pred_info, gt_action, gt_info, sam2_bbox=None):
    pred_action = str(pred_action).strip()
    gt_action = str(gt_action).strip()
    if isinstance(pred_info, str):
        pred_info = pred_info.strip()
    if isinstance(gt_info, str):
        gt_info = gt_info.strip()

    if pred_action != gt_action:
        return {'is_correct': 'no', 'info': 'action_fail'}

    if gt_action not in ['SCROLL', 'CLICK', 'TYPE', 'LONG_PRESS']:
        return {'is_correct': 'yes', 'info': 'action_correct'}
    if gt_action == 'TYPE':
        return {'is_correct': 'yes', 'info': 'type_correct'} if text_matching(gt_info, pred_info) else {'is_correct': 'no', 'info': 'type_fail'}
    if gt_action == 'SCROLL':
        return {'is_correct': 'yes', 'info': 'scroll_correct'} if str(gt_info).lower() == str(pred_info).lower() else {'is_correct': 'no', 'info': 'scroll_fail'}
    if gt_action in ['CLICK', 'LONG_PRESS']:
        return {'is_correct': 'yes', 'info': 'click_correct'} if click_matching(gt_info, pred_info, sam2_bbox) else {'is_correct': 'no', 'info': 'click_fail'}
    return {'is_correct': 'no', 'info': 'invalid'}


def simple_decode(text: str):
    raw = str(text).strip()
    m = re.search(r'(CLICK|LONG_PRESS|SCROLL|TYPE|PRESS_HOME|PRESS_BACK|PRESS_RECENT|COMPLETE|IMPOSSIBLE)\s*:\s*(.+)', raw, re.IGNORECASE | re.DOTALL)
    if m:
        action = m.group(1).upper()
        info = m.group(2).strip()
        if action in ['CLICK', 'LONG_PRESS']:
            info = _safe_eval_tuple_like(info)
        return {'action': action, 'info': info}

    m2 = re.search(r'\b(COMPLETE|IMPOSSIBLE|PRESS_HOME|PRESS_BACK|PRESS_RECENT)\b', raw, re.IGNORECASE)
    if m2:
        return {'action': m2.group(1).upper(), 'info': ''}

    raise ValueError(f'Cannot parse action from: {raw}')


def check_sr(eval_dict: List[Dict[str, Any]]):
    episode_dict = {}
    steps_map = {}
    for data in eval_dict:
        img = os.path.basename(str(data.get('image', '')))
        if '_' not in img:
            continue
        tail = img.split('_')[-1]
        episode = img.replace(f'_{tail}', '')
        if episode not in episode_dict:
            episode_dict[episode] = []
        else:
            assert steps_map[episode] == data['more_info']['step_length']

        episode_dict[episode].append(data['is_correct'])
        steps_map[episode] = data['more_info']['step_length']

    cnt, tot = 0, 0
    for k, v in episode_dict.items():
        if len(v) != steps_map[k]:
            continue
        tot += 1
        uniq = list(set(v))
        if len(uniq) == 1 and uniq[0] == 'yes':
            cnt += 1
    sr = 0 if tot == 0 else round(cnt / tot * 100, 2)
    return cnt, tot, sr


def stat_result(eval_dict: List[Dict[str, Any]], metric: str):
    if len(eval_dict) == 0:
        return {
            'AMS': 0.0,
            'SR': 0.0,
            'total': 0,
            'action_type': '0 / 0 = 0.00',
            'text': '0 / 0 = 0.00',
        }

    text_correct = sum(1 for x in eval_dict if x['info'] == 'type_correct')
    type_correct = sum(1 for x in eval_dict if x['info'] != 'action_fail')
    text_total = sum(1 for x in eval_dict if str(x['info']).startswith('type_'))

    if metric == 'macro':
        action_correct = sum(1 for x in eval_dict if x['is_correct'] == 'yes')
        ams = round(action_correct / len(eval_dict) * 100, 2)
        _, _, sr = check_sr(eval_dict)
    elif metric == 'micro':
        task_cate_dict = {}
        for sample in eval_dict:
            cat = sample.get('more_info', {}).get('category', 'unknown')
            task_cate_dict.setdefault(cat, []).append(sample)
        acc_list, sr_list = [], []
        for _, v in task_cate_dict.items():
            _, _, sr = check_sr(v)
            sr_list.append(sr)
            acc = round(sum(1 for x in v if x['is_correct'] == 'yes') / len(v) * 100, 2) if v else 0.0
            acc_list.append(acc)
        ams = float(np.round(np.mean(acc_list), 2)) if acc_list else 0.0
        sr = float(np.round(np.mean(sr_list), 2)) if sr_list else 0.0
    else:
        raise ValueError(f'No metric {metric} found.')

    return {
        'AMS': ams,
        'SR': sr,
        'total': len(eval_dict),
        'action_type': f'{type_correct} / {len(eval_dict)} = {(type_correct / len(eval_dict) * 100):.2f}',
        'text': f'{text_correct} / {text_total} = {(text_correct / text_total * 100) if text_total else 0:.2f}',
    }


def infer_dataset_name(path: str, fallback: str = 'GUIOdyssey_high_task_split') -> str:
    name = os.path.basename(path)
    m = re.search(r'(GUIOdyssey_(?:high|low)_(?:app|device|random|task)_split)', name)
    if m:
        return m.group(1)
    m = re.search(r'(GUIOdyssey_(?:high|low)_(?:app|device|random|task)_split)', path)
    if m:
        return m.group(1)
    return fallback


def metric_type_for_dataset(dataset_name: str) -> str:
    return METRIC_MAP.get(dataset_name, 'macro')


def summarize_yes_no(eval_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(eval_list)
    yes = sum(1 for x in eval_list if str(x.get('is_correct', '')).lower() == 'yes')
    no = sum(1 for x in eval_list if str(x.get('is_correct', '')).lower() == 'no')
    invalid = total - yes - no
    acc = round((yes / total * 100), 2) if total else 0.0
    return {'yes': yes, 'no': no, 'invalid': invalid, 'accuracy': acc}


def evaluate_rows(rows: List[Dict[str, Any]], metric: str):
    eval_dict = []
    skipped_missing_image = 0

    for row in rows:
        pred = str(row.get('prediction', ''))
        if pred == SKIP_MISSING_IMAGE_MSG:
            skipped_missing_image += 1
            continue

        gt = str(row.get('answer', ''))
        more_info = {
            'category': row.get('category', 'unknown'),
            'step_length': int(row.get('step_length', 1)),
            'sam2_bbox': _safe_eval_tuple_like(row.get('sam2_bbox', [])),
        }
        sample_eval = {
            'question': str(row.get('question', '')),
            'pred': pred,
            'gt': gt,
            'more_info': more_info,
            'image': str(row.get('image', row.get('image_path', ''))),
        }

        try:
            gt_simple = simple_decode(gt)
            pred_simple = simple_decode(pred)
            match = action_matching(
                pred_simple['action'], pred_simple['info'],
                gt_simple['action'], gt_simple['info'],
                more_info['sam2_bbox'],
            )
        except Exception:
            match = {'is_correct': 'no', 'info': 'invalid'}

        sample_eval.update(match)
        eval_dict.append(sample_eval)

    info = stat_result(eval_dict, metric)
    info['skipped_missing_image'] = skipped_missing_image
    info['evaluated'] = len(eval_dict)
    info['total_with_skipped'] = len(rows)
    return {'info': info, 'pred': eval_dict}


def evaluate_from_score_json(score_json: str, dataset_name: str) -> Dict[str, Any]:
    payload = json.load(open(score_json, 'r', encoding='utf-8'))
    eval_list = payload.get('pred', [])
    if not isinstance(eval_list, list):
        raise ValueError(f'Invalid score json format: `pred` should be list, got {type(eval_list)}')

    metric = metric_type_for_dataset(dataset_name)
    info_recomputed = stat_result(eval_list, metric)
    info_original = payload.get('info', {}) if isinstance(payload.get('info', {}), dict) else {}
    yes_no = summarize_yes_no(eval_list)

    return {
        'dataset': dataset_name,
        'metric': metric,
        'source': score_json,
        'total_samples': len(eval_list),
        'accuracy_yes_no': yes_no,
        'official_info_in_file': info_original,
        'recomputed_info': info_recomputed,
    }


def evaluate_from_xlsx(xlsx_path: str, dataset_name: str) -> Dict[str, Any]:
    df = pd.read_excel(xlsx_path)
    rows = df.to_dict(orient='records')
    metric = metric_type_for_dataset(dataset_name)

    metrics = evaluate_rows(rows, metric)
    score_json = xlsx_path.replace('.xlsx', '_score.json')
    with open(score_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    merged = evaluate_from_score_json(score_json, dataset_name)
    merged['official_info_from_evaluate'] = metrics['info']
    merged['generated_score_json'] = score_json
    return merged


def pretty_print(result: Dict[str, Any]) -> None:
    print('=' * 80)
    print(f"Dataset: {result['dataset']}")
    print(f"Metric:  {result['metric']}")
    print(f"Source:  {result['source']}")
    print(f"Total:   {result['total_samples']}")
    yes_no = result['accuracy_yes_no']
    print('-' * 80)
    print(f"Accuracy (yes/no): {yes_no['accuracy']}% ({yes_no['yes']}/{result['total_samples']})")
    print(f"Count no: {yes_no['no']}, invalid: {yes_no['invalid']}")
    print('-' * 80)
    print('Official info in file:')
    print(json.dumps(result.get('official_info_in_file', {}), ensure_ascii=False, indent=2))
    print('-' * 80)
    print('Recomputed info:')
    print(json.dumps(result.get('recomputed_info', {}), ensure_ascii=False, indent=2))
    if result.get('generated_score_json'):
        print('-' * 80)
        print(f"Generated score json: {result['generated_score_json']}")
    print('=' * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate GUIOdyssey result from either *_score.json or .xlsx.')
    parser.add_argument('--input', required=True, help='Path to GUIOdyssey score json or xlsx file')
    parser.add_argument('--dataset', default=None, help='Dataset name, e.g. GUIOdyssey_high_task_split')
    parser.add_argument('--save_json', default=None, help='Optional path to save merged summary json')
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input not found: {input_path}')

    dataset_name = args.dataset or infer_dataset_name(input_path)
    if input_path.endswith('.json'):
        result = evaluate_from_score_json(input_path, dataset_name)
    elif input_path.endswith('.xlsx'):
        result = evaluate_from_xlsx(input_path, dataset_name)
    else:
        raise ValueError('`--input` must be .json or .xlsx')

    pretty_print(result)

    out_json = args.save_json
    if out_json is None:
        stem, _ = os.path.splitext(input_path)
        out_json = f'{stem}_eval_summary.json'
    out_json = os.path.abspath(out_json)

    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Saved summary to: {out_json}')


if __name__ == '__main__':
    main()
