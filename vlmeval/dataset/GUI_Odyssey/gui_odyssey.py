import os
import re
import json
import ast
import numpy as np
import pandas as pd
from PIL import Image

from ..image_base import ImageBaseDataset
from ...smp import *

SKIP_MISSING_IMAGE_MSG = '__SKIP_MISSING_IMAGE__'


def _guiodyssey_debug_prompt_enabled():
    return os.environ.get('GUI_ODYSSEY_DEBUG_HISTORY_PROMPT', '0') == '1'

def levenshtein_distance(s1, s2):
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


def text_matching(gt, pred, threshold=0.5):
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


def _normalize_pred_point_to_1k(pred_info, image_path=None):
    """Optionally normalize predicted point into [0, 1000] coordinate space.

    Controlled by env vars:
      - VLM_GUIODYSSEY_COORD_NORM: "0" (off, default) or "1" (on)
      - VLM_GUIODYSSEY_COORD_NORM_MODE: "auto" (default) | "pixel" | "unit"
    """
    if os.getenv('VLM_GUIODYSSEY_COORD_NORM', '0') != '1':
        return pred_info

    mode = os.getenv('VLM_GUIODYSSEY_COORD_NORM_MODE', 'auto').strip().lower()
    p = _safe_eval_tuple_like(pred_info)
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return pred_info

    try:
        x, y = float(p[0]), float(p[1])
    except Exception:
        return pred_info

    def _pixel_to_1k(px, py, img_path):
        if img_path is None or not osp.exists(img_path):
            return [px, py]
        try:
            w, h = Image.open(img_path).size
            if w <= 0 or h <= 0:
                return [px, py]
            return [px / w * 1000.0, py / h * 1000.0]
        except Exception:
            return [px, py]

    if mode == 'pixel':
        return _pixel_to_1k(x, y, image_path)
    if mode == 'unit':
        return [x * 1000.0, y * 1000.0]

    # auto mode
    m = max(abs(x), abs(y))
    if m <= 1.5:
        return [x * 1000.0, y * 1000.0]
    if m <= 1000.0:
        return [x, y]
    return _pixel_to_1k(x, y, image_path)


def click_matching(gt_info, pred_info, sam2_bbox=None, coord_threshold=0.14, image_path=None):
    pred_info = _safe_eval_tuple_like(pred_info)
    gt_info = _safe_eval_tuple_like(gt_info)
    pred_info = _normalize_pred_point_to_1k(pred_info, image_path=image_path)

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


def action_matching(pred_action, pred_info, gt_action, gt_info, sam2_bbox=None, image_path=None):
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
        return {'is_correct': 'yes', 'info': 'click_correct'} if click_matching(
            gt_info, pred_info, sam2_bbox, image_path=image_path
        ) else {'is_correct': 'no', 'info': 'click_fail'}
    return {'is_correct': 'no', 'info': 'invalid'}


def simple_decode(text):
    """Decode one command-like action from model text.

    Expected style: ACTION: INFO, e.g. CLICK: (123, 456)
    """
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


def check_sr(eval_dict):
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


def stat_result(eval_dict, metric):
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
        ams = round(action_correct / len(eval_dict) * 100, 2) if eval_dict else 0.0
        _, _, sr = check_sr(eval_dict)
    elif metric == 'micro':
        task_cate_dict = {}
        for sample in eval_dict:
            cat = sample['more_info']['category']
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
        'action_type': f'{type_correct} / {len(eval_dict)} = {(type_correct / len(eval_dict) * 100) if eval_dict else 0:.2f}',
        'text': f'{text_correct} / {text_total} = {(text_correct / text_total * 100) if text_total else 0:.2f}',
    }


class GUIOdyssey(ImageBaseDataset):
    MODALITY = 'IMAGE'
    TYPE = 'GUI'
    DATASET_URL = {
        'GUIOdyssey_high_app_split': '',
        'GUIOdyssey_high_device_split': '',
        'GUIOdyssey_high_random_split': '',
        'GUIOdyssey_high_task_split': '',
        'GUIOdyssey_low_app_split': '',
        'GUIOdyssey_low_device_split': '',
        'GUIOdyssey_low_random_split': '',
        'GUIOdyssey_low_task_split': '',
    }

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

    SYSTEM_PROMPT = (
        'You are a GUI mobile agent. Given the current screenshot, task instruction, and previous actions, '
        'output exactly one next action command.\\n'
        'Allowed actions: CLICK: (x, y), LONG_PRESS: (x, y), SCROLL: UP/DOWN/LEFT/RIGHT, TYPE: <text>, '
        'PRESS_HOME, PRESS_BACK, PRESS_RECENT, COMPLETE, IMPOSSIBLE.\\n'
        'Coordinates must be in [0,1000]. Return command only.'
    )
    HISTORY_PROMPT_PREFIX = (
        'Please generate the next move according to the instruction, previous actions, '
        'previous ui screenshot and current ui screenshot. '
    )

    def __init__(self, dataset='GUIOdyssey_high_random_split', skip_noimg=True, skeleton=False):
        self.dataset_name = dataset
        self.img_root = ''
        self.include_history_screenshots = os.environ.get('GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS', '1') == '1'
        self.max_history_images = max(0, int(os.environ.get('GUI_ODYSSEY_MAX_HISTORY_IMAGES', '4')))
        self.history_keep_system_prompt = os.environ.get('GUI_ODYSSEY_HISTORY_KEEP_SYSTEM_PROMPT', '0') == '1'
        if skeleton:
            return

        data = self.load_data(dataset)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(list(data))

        if 'image' not in data.columns and 'image_path' in data.columns:
            data['image'] = data['image_path']
        if 'image_path' not in data.columns and 'image' in data.columns:
            data['image_path'] = data['image']

        required_cols = ['image', 'question', 'answer']
        for col in required_cols:
            assert col in data.columns, f'Missing required column `{col}` in GUIOdyssey data'

        data = data.reset_index(drop=True)
        data['index'] = [str(i + 1) for i in range(len(data))]
        self.data = data
        self.meta_only = True

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def _resolve_data_root(self):
        env_root = os.environ.get('GUI_ODYSSEY_ROOT', '').strip()
        if env_root:
            return env_root
        return '/mnt/storage2/users/ytshen_data/GUIOdyssey'

    def _dataset_json_path(self, dataset):
        root = self._resolve_data_root()
        short_name = dataset.replace('GUIOdyssey_', '')
        return osp.join(root, 'test_anno', f'{short_name}.json')

    def _resolve_image_path(self, image_value):
        p = str(image_value)
        if osp.exists(p):
            return p
        root = self._resolve_data_root()
        cand = osp.join(root, 'screenshots', osp.basename(p))
        if osp.exists(cand):
            return cand
        return p

    def _parse_history_image_list(self, value, return_invalid=False):
        parsed = _safe_eval_tuple_like(value)
        if not isinstance(parsed, (list, tuple)):
            return ([], []) if return_invalid else []
        out = []
        invalid = []
        for item in parsed:
            text = str(item).strip()
            if not text:
                continue
            resolved = self._resolve_image_path(text)
            if osp.isfile(resolved) and resolved not in out:
                out.append(resolved)
            elif return_invalid:
                invalid.append(resolved)
        if return_invalid:
            return out, invalid
        return out

    def _resolve_history_pairs(self, line, current_image_path, history_actions):
        if not self.include_history_screenshots or self.max_history_images <= 0:
            return [], dict(
                raw_history_count=0,
                valid_history_count=0,
                invalid_history_paths=[],
                invalid_history_pairs=[],
                truncated_history_count=0,
            )
        history_actions = list(history_actions) if isinstance(history_actions, list) else []
        for key in ('history_screenshot', 'history_screenshots', 'history_image_paths', 'history_images'):
            if key not in line or line.get(key) is None:
                continue
            raw_images = _safe_eval_tuple_like(line.get(key))
            if not isinstance(raw_images, (list, tuple)):
                continue
            pairs = []
            invalid = []
            for idx, item in enumerate(raw_images):
                text = str(item).strip()
                if not text:
                    continue
                resolved = self._resolve_image_path(text)
                action_text = history_actions[idx] if idx < len(history_actions) else None
                if resolved == current_image_path or not osp.isfile(resolved):
                    invalid.append(dict(image=resolved, action=action_text, index=idx))
                    continue
                pairs.append(dict(image=resolved, action=action_text, index=idx))
            truncated = max(0, len(pairs) - self.max_history_images)
            if self.max_history_images > 0 and len(pairs) > self.max_history_images:
                pairs = pairs[-self.max_history_images:]
            debug = dict(
                raw_history_count=len(raw_images),
                valid_history_count=len(pairs),
                invalid_history_paths=[x['image'] for x in invalid],
                invalid_history_pairs=invalid,
                truncated_history_count=truncated,
            )
            return pairs, debug
        return [], dict(
            raw_history_count=0,
            valid_history_count=0,
            invalid_history_paths=[],
            invalid_history_pairs=[],
            truncated_history_count=0,
        )

    def _resolve_history_screenshots(self, line, current_image_path, return_debug=False):
        if not self.include_history_screenshots or self.max_history_images <= 0:
            debug = dict(
                raw_history_count=0,
                valid_history_count=0,
                invalid_history_paths=[],
                invalid_history_pairs=[],
                truncated_history_count=0,
            )
            return ([], debug) if return_debug else []
        for key in ('history_screenshot', 'history_screenshots', 'history_image_paths', 'history_images'):
            if key not in line or line.get(key) is None:
                continue
            parsed, invalid = self._parse_history_image_list(line.get(key), return_invalid=True)
            if parsed:
                deduped = [p for p in parsed if p != current_image_path and osp.isfile(p)]
                truncated = max(0, len(deduped) - self.max_history_images)
                if len(deduped) > self.max_history_images:
                    deduped = deduped[-self.max_history_images:]
                debug = dict(
                    raw_history_count=len(parsed) + len(invalid),
                    valid_history_count=len(deduped),
                    invalid_history_paths=invalid,
                    invalid_history_pairs=[],
                    truncated_history_count=truncated,
                )
                return (deduped, debug) if return_debug else deduped
            if invalid:
                debug = dict(
                    raw_history_count=len(invalid),
                    valid_history_count=0,
                    invalid_history_paths=invalid,
                    invalid_history_pairs=[],
                    truncated_history_count=0,
                )
                return ([], debug) if return_debug else []
        debug = dict(
            raw_history_count=0,
            valid_history_count=0,
            invalid_history_paths=[],
            invalid_history_pairs=[],
            truncated_history_count=0,
        )
        return ([], debug) if return_debug else []

    def _format_history_actions(self, history_actions):
        if len(history_actions) == 0:
            return 'None'
        return '\n'.join([f'{i + 1}. {a}' for i, a in enumerate(history_actions)])

    def _maybe_debug_print_prompt(
        self,
        line,
        current_image_path,
        history_image_paths,
        history_actions,
        prompt_text,
        history_debug=None,
    ):
        if not _guiodyssey_debug_prompt_enabled():
            return
        sample_index = str(line.get('index', ''))
        image_value = str(line.get('image', line.get('image_path', '')))
        print(
            f'[GUIOdysseyDebug] sample_index={sample_index} image={image_value} '
            f'current_image={current_image_path} history_image_count={len(history_image_paths)}',
            flush=True,
        )
        print(f'[GUIOdysseyDebug] history_images={history_image_paths}', flush=True)
        print(f'[GUIOdysseyDebug] history_actions={history_actions}', flush=True)
        if isinstance(history_debug, dict):
            print(
                '[GUIOdysseyDebug] '
                f'raw_history_count={history_debug.get("raw_history_count", 0)} '
                f'valid_history_count={history_debug.get("valid_history_count", 0)} '
                f'truncated_history_count={history_debug.get("truncated_history_count", 0)}',
                flush=True,
            )
            print(
                f'[GUIOdysseyDebug] invalid_history_paths={history_debug.get("invalid_history_paths", [])}',
                flush=True,
            )
            print(
                f'[GUIOdysseyDebug] invalid_history_pairs={history_debug.get("invalid_history_pairs", [])}',
                flush=True,
            )
        print('[GUIOdysseyDebug] prompt_begin', flush=True)
        print(prompt_text, flush=True)
        print('[GUIOdysseyDebug] prompt_end', flush=True)

    def load_data(self, dataset):
        jp = self._dataset_json_path(dataset)
        assert osp.exists(jp), (
            f'GUI-Odyssey annotation file not found: {jp}. '\
            'Please run `python vlmeval/dataset/GUI_Odyssey/format_converter.py --data-root <GUI_ODYSSEY_ROOT>/data --his_len 4 --level high --type standard` first.'
        )
        data = load(jp)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        image_path = self._resolve_image_path(line['image'])
        assert osp.exists(image_path), (
            f'GUI-Odyssey screenshot not found: {image_path}. '
            'Please make sure screenshots are downloaded under <GUI_ODYSSEY_ROOT>/screenshots.'
        )
        instruction = str(line['question'])
        his_actions = _safe_eval_tuple_like(line.get('history_action', '[]'))
        if not isinstance(his_actions, list):
            his_actions = []
        hist_pairs, history_debug = self._resolve_history_pairs(line, image_path, his_actions)
        hist_images = [x['image'] for x in hist_pairs]
        if hist_pairs:
            his_actions = [x['action'] for x in hist_pairs if str(x.get('action', '')).strip()]
        else:
            hist_images, history_debug = self._resolve_history_screenshots(
                line,
                image_path,
                return_debug=True,
            )
            if hist_images:
                his_actions = his_actions[-len(hist_images):]
        hist = self._format_history_actions(his_actions)

        if not hist_images:
            user_prompt = (
                f'Task: {instruction}\\n'
                f'Previous Actions:\\n{hist}\\n'
                'Provide the next action command only.'
            )
            debug_prompt = f'{self.SYSTEM_PROMPT}\n[IMAGE]{image_path}\n{user_prompt}'
            self._maybe_debug_print_prompt(
                line=line,
                current_image_path=image_path,
                history_image_paths=[],
                history_actions=his_actions,
                prompt_text=debug_prompt,
                history_debug=history_debug,
            )
            msgs = [
                dict(type='text', value=self.SYSTEM_PROMPT),
                dict(type='image', value=image_path),
                dict(type='text', value=user_prompt),
            ]
            return msgs

        if self.history_keep_system_prompt:
            intro = f'Task: {instruction}\\nPrevious Actions and Screenshots:\\n'
            outro = (
                'Provide the next action command only. '
                'The final image is the current screenshot. '
                'Use coordinates in [0,1000] on the current screenshot only.'
            )
            debug_parts = [self.SYSTEM_PROMPT, intro]
            msgs = [dict(type='text', value=self.SYSTEM_PROMPT), dict(type='text', value=intro)]
            for i, (hist_image_path, action_text) in enumerate(zip(hist_images, his_actions)):
                debug_parts.append(f'Image_{i}: [HISTORY_IMAGE] {hist_image_path}')
                debug_parts.append(f'{i + 1}. {action_text}')
                msgs.append(dict(type='text', value=f'Image_{i}:'))
                msgs.append(dict(type='image', value=hist_image_path))
                msgs.append(dict(type='text', value=f'{i + 1}. {action_text}\\n'))
            debug_parts.append(f'Current Screenshot: [CURRENT_IMAGE] {image_path}')
            debug_parts.append(outro)
            self._maybe_debug_print_prompt(
                line=line,
                current_image_path=image_path,
                history_image_paths=hist_images,
                history_actions=his_actions,
                prompt_text='\n'.join(debug_parts),
                history_debug=history_debug,
            )
            msgs.append(dict(type='text', value='Current Screenshot:'))
            msgs.append(dict(type='image', value=image_path))
            msgs.append(dict(type='text', value=outro))
            return msgs

        intro = f'{self.HISTORY_PROMPT_PREFIX}Instruction: {instruction}\n'
        outro = (
            'Provide the command-style action directly. '
            'The final image is the current screenshot.'
        )
        debug_parts = [intro]
        for i, (hist_image_path, action_text) in enumerate(zip(hist_images, his_actions)):
            debug_parts.append(f'Image_{i}: [HISTORY_IMAGE] {hist_image_path}')
            debug_parts.append(f'Step_{i}: {action_text}.')
        debug_parts.append(f'Image_{len(hist_images)}: [CURRENT_IMAGE] {image_path}')
        debug_parts.append(outro)
        self._maybe_debug_print_prompt(
            line=line,
            current_image_path=image_path,
            history_image_paths=hist_images,
            history_actions=his_actions,
            prompt_text='\n'.join(debug_parts),
            history_debug=history_debug,
        )
        msgs = [dict(type='text', value=intro)]
        for i, (hist_image_path, action_text) in enumerate(zip(hist_images, his_actions)):
            msgs.append(dict(type='text', value=f'Image_{i}:'))
            msgs.append(dict(type='image', value=hist_image_path))
            msgs.append(dict(type='text', value=f'Step_{i}: {action_text}.\n'))
        msgs.append(dict(type='text', value=f'Image_{len(hist_images)}:'))
        msgs.append(dict(type='image', value=image_path))
        msgs.append(dict(type='text', value=outro))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'prediction' in data and 'answer' in data

        metric = self.METRIC_MAP[self.dataset_name]
        eval_dict = []
        skipped_missing_image = 0
        for i in range(len(data)):
            row = data.iloc[i]
            pred = str(row['prediction'])
            if pred == SKIP_MISSING_IMAGE_MSG:
                skipped_missing_image += 1
                continue
            gt = str(row['answer'])
            more_info = {
                'category': row.get('category', 'unknown'),
                'step_length': int(row.get('step_length', 1)),
                'sam2_bbox': row.get('sam2_bbox', []),
            }

            sample_eval = {
                'question': str(row.get('question', '')),
                'pred': pred,
                'gt': gt,
                'more_info': more_info,
                'image': str(row.get('image', row.get('image_path', ''))),
            }
            image_path = self._resolve_image_path(sample_eval['image'])
            try:
                gt_simple = simple_decode(gt)
                pred_simple = simple_decode(pred)
                match = action_matching(
                    pred_simple['action'], pred_simple['info'],
                    gt_simple['action'], gt_simple['info'],
                    more_info['sam2_bbox'],
                    image_path=image_path,
                )
            except Exception:
                match = {'is_correct': 'no', 'info': 'invalid'}

            sample_eval.update(match)
            eval_dict.append(sample_eval)

        info = stat_result(eval_dict, metric)
        info['skipped_missing_image'] = skipped_missing_image
        info['evaluated'] = len(eval_dict)
        info['total_with_skipped'] = len(data)
        metrics = {'info': info, 'pred': eval_dict}

        score_pth = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(metrics, score_pth)
        return info

    def eval_single(self, line, prediction):
        """Online single-sample evaluation for progress logging.

        Returns:
            dict: {'is_correct': bool, 'info': str}
        """
        gt = str(line.get('answer', ''))
        more_info = {
            'category': line.get('category', 'unknown'),
            'step_length': int(line.get('step_length', 1)),
            'sam2_bbox': line.get('sam2_bbox', []),
        }
        try:
            gt_simple = simple_decode(gt)
            pred_simple = simple_decode(str(prediction))
            image_path = self._resolve_image_path(line.get('image', line.get('image_path', '')))
            match = action_matching(
                pred_simple['action'], pred_simple['info'],
                gt_simple['action'], gt_simple['info'],
                more_info['sam2_bbox'],
                image_path=image_path,
            )
            return {'is_correct': match.get('is_correct') == 'yes', 'info': match.get('info', '')}
        except Exception:
            return {'is_correct': False, 'info': 'invalid'}
