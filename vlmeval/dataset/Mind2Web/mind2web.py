import ast
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..image_base import ImageBaseDataset
from ...smp import dump, load, osp
from .state_packet import build_state_packet, state_packet_debug_enabled, state_packet_enabled


MIND2WEB_DATASETS = [
    "Mind2Web_test_task",
    "Mind2Web_test_website",
    "Mind2Web_test_domain",
    "Mind2Web_task",
    "Mind2Web_website",
    "Mind2Web_domain",
]

PROMPT_ORIGIN = (
    "Please generate the next move according to the instruction, previous actions, "
    "previous ui screenshot and current ui screenshot. Instruction: {}\n"
)

STRICT_OUTPUT_INSTRUCTION = (
    "\nOutput exactly one Python dict only. Do not explain.\n"
    "Action schema:\n"
    '- CLICK/HOVER/ENTER: {"action_type": 4, "click_point": (x,y)}\n'
    '- TYPE: {"action_type": 3, "click_point": (x,y), "value": "text"}\n'
    '- SELECT: {"action_type": 2, "click_point": (x,y), "value": "option"}\n'
    "Coordinates are normalized integers from 0 to 1000 on the current screenshot.\n"
)


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)


def _nonnull(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _safe_literal_eval(value):
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return value


def process_string(text: str) -> str:
    """Match HistPrune-GUI: convert integer 0-1000 points in strings to 0-1 floats."""
    import re

    pattern = r"\((\d+),(\d+)\)"

    def replace(match):
        x = round(float(match.group(1)) / 1000, 2)
        y = round(float(match.group(2)) / 1000, 2)
        return f"({x:.2f},{y:.2f})"

    return re.sub(pattern, replace, str(text or "").strip())


def action2step(action: dict, image_size, return_bbox: bool = False):
    action_type = action["operation"]["original_op"]
    assert action_type in ["CLICK", "TYPE", "SELECT", "HOVER", "ENTER"]

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{int(1000 * item)}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [
            action["bbox"]["x"],
            action["bbox"]["y"],
            action["bbox"]["x"] + action["bbox"]["width"],
            action["bbox"]["y"] + action["bbox"]["height"],
        ]
        bbox = [
            bbox[0] / image_size[0],
            bbox[1] / image_size[1],
            bbox[2] / image_size[0],
            bbox[3] / image_size[1],
        ]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ["CLICK", "HOVER", "ENTER"]:
        action_step = '{{"action_type": {}, "click_point": {}}}'.format(4, click_point)
    elif action_type == "SELECT":
        select_value = action["operation"]["value"]
        action_step = '{{"action_type": {}, "click_point": {}, "value": "{}"}}'.format(2, click_point, select_value)
    elif action_type == "TYPE":
        typed_text = action["operation"]["value"]
        action_step = '{{"action_type": {}, "click_point": {}, "value": "{}"}}'.format(3, click_point, typed_text)

    if return_bbox:
        return action_step, bbox
    return action_step


def calculate_f1(pred: str, label: str) -> float:
    pred = set(str(pred).strip().split())
    label = set(str(label).strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0
    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def _split_name(dataset: str) -> str:
    if dataset.startswith("Mind2Web_test_"):
        return dataset[len("Mind2Web_test_"):]
    if dataset.startswith("Mind2Web_"):
        return dataset[len("Mind2Web_"):]
    return dataset


class Mind2Web(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {name: "" for name in MIND2WEB_DATASETS}

    def __init__(self, dataset: str = "Mind2Web_test_task", skip_noimg: bool = True, skeleton: bool = False):
        self.dataset_name = dataset
        self.data_root = os.getenv("MIND2WEB_DATA_ROOT", "/mnt/storage2/Datasets/Mind2Web").strip()
        self.ann_root = os.getenv("MIND2WEB_ANN_ROOT", osp.join(self.data_root, "mind2web_annots")).strip()
        default_img_root = osp.join(self.data_root, "mind2web_images")
        nested_img_root = osp.join(default_img_root, "ming2web_images")
        if osp.isdir(nested_img_root):
            default_img_root = nested_img_root
        self.img_root = os.getenv("MIND2WEB_IMAGE_ROOT", default_img_root).strip()
        self.his_num = max(0, _env_int("MIND2WEB_HIS_NUM", 2))
        self.top_k = max(0, _env_int("MIND2WEB_TOP_K", 0))
        self.with_no_history = _env_flag("MIND2WEB_WITH_NO_HISTORY", os.getenv("MIND2WEB_WITH_NO_HISTROY", "0"))
        self.strict_output_prompt = _env_flag("MIND2WEB_STRICT_OUTPUT_PROMPT", "1")
        self.use_history_state_packet = state_packet_enabled()
        self.prompt_origin = os.getenv("MIND2WEB_PROMPT_ORIGIN", PROMPT_ORIGIN)
        self.skip_noimg = skip_noimg
        self._state_packet_records = []
        self.meta_only = True
        if skeleton:
            return
        data = self.load_data(dataset)
        data = data.reset_index(drop=True)
        data["index"] = [str(i + 1) for i in range(len(data))]
        self.data = data

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def _dataset_json_path(self, dataset: str) -> str:
        split = _split_name(dataset)
        explicit = os.getenv("MIND2WEB_ANN_FILE", "").strip()
        if explicit:
            return explicit
        candidates = [
            osp.join(self.ann_root, f"mind2web_data_test_{split}.json"),
            osp.join(self.ann_root, "data", f"test_{split}", f"test_{split}.json"),
            osp.join(self.ann_root, "data", f"test_{split}.json"),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return candidates[0]

    def _resolve_image_path(self, annot_id: str, action_uid: str, value: Optional[str] = None) -> str:
        if value:
            text = str(value)
            if osp.exists(text):
                return text
            cand = osp.join(self.img_root, osp.basename(text))
            if osp.exists(cand):
                return cand
        filename = f"{annot_id}-{action_uid}.jpg"
        candidates = [
            osp.join(self.img_root, filename),
            osp.join(self.img_root, "ming2web_images", filename),
            osp.join(self.data_root, "mind2web_images", filename),
            osp.join(self.data_root, "mind2web_images", "ming2web_images", filename),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        return candidates[0]

    def _image_size(self, image_path: str):
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    def _flatten_episodes(self, raw) -> pd.DataFrame:
        rows = []
        running_index = 0
        for episode_order, episode in enumerate(raw):
            goal = episode.get("confirmed_task", episode.get("task", ""))
            annot_id = str(episode.get("annotation_id", episode.get("annot_id", episode_order)))
            previous_actions: List[str] = []
            previous_imgs: List[str] = []
            results_actions: List[dict] = []
            for step_order, step in enumerate(episode.get("actions", [])):
                if "bbox" not in step:
                    continue
                action_uid = str(step.get("action_uid", step_order))
                image_path = self._resolve_image_path(annot_id, action_uid, step.get("image", step.get("image_path")))
                if self.skip_noimg and not osp.exists(image_path):
                    continue
                image_size = self._image_size(image_path)
                if image_size is None:
                    continue
                try:
                    answer, bbox_ref = action2step(step, image_size, return_bbox=True)
                    answer_dict = ast.literal_eval(answer)
                except Exception:
                    continue
                row = {
                    "index": running_index,
                    "annotation_id": annot_id,
                    "annot_id": annot_id,
                    "episode_id": annot_id,
                    "episode_order": episode_order,
                    "step_id": step_order,
                    "step_order": step_order,
                    "action_uid": action_uid,
                    "instruction": goal,
                    "question": goal,
                    "confirmed_task": goal,
                    "image": image_path,
                    "image_path": image_path,
                    "answer": answer,
                    "gt_action": answer,
                    "gt_action_dict": answer_dict,
                    "gt_bbox": bbox_ref,
                    "operation": step.get("operation", {}),
                    "bbox": step.get("bbox", {}),
                    "_step": step,
                    "_prev_action_texts": list(previous_actions),
                    "_prev_image_paths": list(previous_imgs),
                    "_episode_result_index": len(results_actions),
                }
                rows.append(row)
                previous_actions.append(answer)
                previous_imgs.append(image_path)
                running_index += 1
        return pd.DataFrame(rows)

    def load_data(self, dataset):
        json_path = self._dataset_json_path(dataset)
        assert osp.exists(json_path), (
            f"Mind2Web annotation file not found: {json_path}. "
            "Set MIND2WEB_ANN_ROOT to the directory containing mind2web_data_test_{task,website,domain}.json "
            "or set MIND2WEB_ANN_FILE to an explicit JSON file."
        )
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self._flatten_episodes(raw)

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = str(line.get("image_path", line.get("image", "")))
        assert osp.exists(image_path), (
            f"Mind2Web screenshot not found: {image_path}. "
            "Set MIND2WEB_IMAGE_ROOT to the directory containing {annotation_id}-{action_uid}.jpg."
        )
        return [image_path]

    def _history_actions(self, line) -> List[str]:
        actions = line.get("_prev_action_texts", [])
        if not isinstance(actions, list):
            actions = []
        return [str(x) for x in actions][-self.his_num:]

    def _history_images(self, line, current_image_path: str) -> List[str]:
        if self.with_no_history or self.his_num <= 0:
            return []
        images = line.get("_prev_image_paths", [])
        if not isinstance(images, list):
            images = []
        out = []
        for p in images[-self.his_num:]:
            p = str(p)
            if p and p != current_image_path and osp.exists(p):
                out.append(p)
        return out

    def _action_packet_for_step(self, step: dict, image_path: str, answer: str) -> dict:
        image_size = self._image_size(image_path)
        gt_coordinate = None
        gt_bbox = None
        if image_size is not None and isinstance(step.get("bbox"), dict):
            w, h = image_size
            bbox = step["bbox"]
            x1 = float(bbox.get("x", 0.0))
            y1 = float(bbox.get("y", 0.0))
            x2 = x1 + float(bbox.get("width", 0.0))
            y2 = y1 + float(bbox.get("height", 0.0))
            gt_bbox = [x1, y1, x2, y2]
            gt_coordinate = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        return {
            # Mind2Web CLICK/TYPE/SELECT all point to an interacted element.
            # The shared AndroidControl packet builder uses the bbox only for
            # click-like actions, so expose the packet action as click while
            # keeping the original action dict in step_instruction.
            "gt_action": "click",
            "gt_coordinate": gt_coordinate,
            "gt_bbox": gt_bbox,
            "step_instruction": answer,
        }

    def _previous_rows_for_history(self, line, history_image_paths: List[str]):
        if not history_image_paths:
            return []
        ep_id = str(line.get("episode_id", line.get("annotation_id", "")))
        step_order = int(line.get("step_order", line.get("step_id", 0)))
        prev = self.data[
            (self.data["episode_id"].astype(str) == ep_id)
            & (self.data["step_order"].astype(int) < step_order)
        ].sort_values("step_order", kind="stable")
        return list(prev.tail(len(history_image_paths)).to_dict("records"))

    def _build_history_visual_entries(self, sample_index: str, history_image_paths: List[str], history_rows: List[dict]):
        if not self.use_history_state_packet:
            return [
                {
                    "history_index": i,
                    "action_text": str(row.get("answer", "")),
                    "images": [dict(type="image", value=hist_image_path)],
                    "debug_items": [dict(kind="original", path=hist_image_path, crop_xyxy=None, estimated_tokens=None)],
                }
                for i, (hist_image_path, row) in enumerate(zip(history_image_paths, history_rows))
            ]

        entries = []
        for i, (hist_image_path, row) in enumerate(zip(history_image_paths, history_rows)):
            action_packet = self._action_packet_for_step(row.get("_step", {}), hist_image_path, str(row.get("answer", "")))
            packet_images, packet_meta = build_state_packet(
                image_path=hist_image_path,
                action_packet=action_packet,
                sample_index=str(sample_index),
                history_index=i,
            )
            packet_meta["dataset_name"] = str(self.dataset_name)
            self._state_packet_records.append(packet_meta)
            entries.append(
                {
                    "history_index": i,
                    "action_text": action_packet["step_instruction"],
                    "images": [item.to_message_item() for item in packet_images],
                    "debug_items": [
                        {
                            "kind": item.kind,
                            "path": item.path,
                            "crop_xyxy": item.crop_xyxy,
                            "estimated_tokens": item.estimated_tokens,
                        }
                        for item in packet_images
                    ],
                    "packet_meta": packet_meta,
                }
            )
        return entries

    def _build_prompt_text(self, instruction: str, history_images: List[str], history_actions: List[str]) -> str:
        prompt = self.prompt_origin.format(instruction)
        if self.with_no_history:
            for i, action in enumerate(history_actions):
                prompt += f"Step_{i}: {action} .\n"
            prompt += "Image_0:<image>\n"
            return prompt
        for i, action in enumerate(history_actions):
            prompt += f"Image_{i}:<image>\n"
            prompt += f"Step_{i}: {action} .\n"
        prompt += f"Image_{len(history_images)}:<image>\n"
        return prompt

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = self.dump_image(line)[0]
        history_actions = self._history_actions(line)
        history_images = self._history_images(line, image_path)
        if not self.with_no_history and len(history_images) < len(history_actions):
            history_actions = history_actions[-len(history_images):] if history_images else []
        prompt = self.prompt_origin.format(str(line.get("instruction", "")))

        if self.with_no_history:
            msgs = [dict(type="text", value=prompt)]
            for i, action in enumerate(history_actions):
                msgs.append(dict(type="text", value=f"Step_{i}: {action} .\n"))
            msgs.append(dict(type="text", value="Image_0:"))
            msgs.append(dict(type="image", value=image_path))
            if self.strict_output_prompt:
                msgs.append(dict(type="text", value=STRICT_OUTPUT_INSTRUCTION))
            return msgs

        history_rows = self._previous_rows_for_history(line, history_images)
        if len(history_rows) < len(history_images):
            history_images = history_images[-len(history_rows):] if history_rows else []
            history_actions = history_actions[-len(history_rows):] if history_rows else []
        history_entries = self._build_history_visual_entries(str(line.get("index", "")), history_images, history_rows)

        msgs = [dict(type="text", value=prompt)]
        for entry in history_entries:
            i = int(entry["history_index"])
            if not self.use_history_state_packet:
                msgs.append(dict(type="text", value=f"Image_{i}:"))
                for image_item in entry.get("images", []):
                    msgs.append(dict(image_item))
            else:
                for image_item, debug_item in zip(entry.get("images", []), entry.get("debug_items", [])):
                    kind = str(debug_item.get("kind", "history_image"))
                    crop_xyxy = debug_item.get("crop_xyxy", None)
                    label = f"Image_{i} {kind}"
                    if crop_xyxy is not None:
                        label += f" crop_xyxy={list(crop_xyxy)}"
                    msgs.append(dict(type="text", value=f"{label}:"))
                    msgs.append(dict(image_item))
            msgs.append(dict(type="text", value=f"Step_{i}: {entry.get('action_text', '')} .\n"))
        msgs.append(dict(type="text", value=f"Image_{len(history_images)}:"))
        msgs.append(dict(type="image", value=image_path))
        if self.strict_output_prompt:
            msgs.append(dict(type="text", value=STRICT_OUTPUT_INSTRUCTION))
        if state_packet_debug_enabled():
            print(
                "[Mind2WebPrompt] "
                f"sample_index={line.get('index', '')} history_images={len(history_images)} "
                f"state_packet={int(self.use_history_state_packet)} "
                f"message_images={sum(1 for x in msgs if x.get('type') == 'image')}",
                flush=True,
            )
        return msgs

    def summarize_state_packet_records(self):
        recs = list(getattr(self, "_state_packet_records", []) or [])
        if not recs:
            return {}
        count = float(len(recs))
        orig_tokens = float(sum(float(r.get("original_estimated_tokens", 0.0) or 0.0) for r in recs))
        packet_tokens = float(sum(float(r.get("packet_estimated_tokens", 0.0) or 0.0) for r in recs))
        total_s = float(sum(float(r.get("state_packet_total_s", 0.0) or 0.0) for r in recs))
        return {
            "state_packet_enabled": bool(self.use_history_state_packet),
            "state_packet_history_image_count": int(count),
            "avg_state_packet_original_estimated_tokens": float(orig_tokens / count),
            "avg_state_packet_packet_estimated_tokens": float(packet_tokens / count),
            "state_packet_total_original_estimated_tokens": float(orig_tokens),
            "state_packet_total_packet_estimated_tokens": float(packet_tokens),
            "state_packet_avg_compression_ratio": float(packet_tokens / max(1.0, orig_tokens)),
            "avg_state_packet_total_s": float(total_s / count),
            "total_state_packet_total_s": float(total_s),
        }

    @staticmethod
    def _parse_prediction(text: str):
        text = process_string(str(text or ""))
        parsed = _safe_literal_eval(text)
        if not isinstance(parsed, dict):
            return None
        click_point = parsed.get("click_point", None)
        if isinstance(click_point, (list, tuple)) and len(click_point) >= 2:
            try:
                x, y = float(click_point[0]), float(click_point[1])
                if max(abs(x), abs(y)) > 1.5:
                    x /= 1000.0
                    y /= 1000.0
                parsed["click_point"] = [x, y]
            except Exception:
                pass
        return parsed

    @classmethod
    def _eval_record(cls, row: Dict, prediction: str) -> Dict:
        action_step_ref = row.get("gt_action_dict", None)
        if not isinstance(action_step_ref, dict):
            action_step_ref = _safe_literal_eval(row.get("answer", ""))
        bbox_ref = _safe_literal_eval(row.get("gt_bbox", []))
        step_result = {
            "annot_id": str(row.get("annotation_id", row.get("annot_id", ""))),
            "img_path": str(row.get("image_path", row.get("image", ""))),
            "instruction": str(row.get("instruction", "")),
            "sentence": str(prediction),
            "Op_match": False,
            "Ele_match": False,
            "Op_F1": [0, action_step_ref.get("action_type") if isinstance(action_step_ref, dict) else None],
            "wrong_format": False,
        }
        action_pred = cls._parse_prediction(prediction)
        if action_pred is None or not isinstance(action_step_ref, dict):
            step_result["wrong_format"] = True
            return step_result
        try:
            if action_pred["action_type"] == action_step_ref["action_type"]:
                step_result["Op_match"] = True
            click_point = action_pred["click_point"]
            if len(click_point) >= 2 and len(bbox_ref) >= 4:
                if bbox_ref[0] <= click_point[0] <= bbox_ref[2] and bbox_ref[1] <= click_point[1] <= bbox_ref[3]:
                    step_result["Ele_match"] = True
            pred_str = str(action_pred["action_type"])
            if action_pred["action_type"] in [3, 2]:
                pred_str += " " + str(action_pred.get("value", "")).lower()
            ref_str = str(action_step_ref["action_type"])
            if action_step_ref["action_type"] in [3, 2]:
                ref_str += " " + str(action_step_ref.get("value", "")).lower()
            step_result["Op_F1"][0] = calculate_f1(pred_str, ref_str)
        except Exception:
            logging.info("format wrong")
            step_result["wrong_format"] = True
        return step_result

    def eval_single(self, row, prediction):
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        return self._eval_record(row, prediction)

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert "prediction" in data, f"Missing `prediction` column in {eval_file}"
        records = []
        for i in range(len(data)):
            row = data.iloc[i].to_dict()
            records.append(self._eval_record(row, _nonnull(row.get("prediction", ""))))

        episode_map: Dict[str, List[Dict]] = {}
        for rec in records:
            episode_map.setdefault(str(rec.get("annot_id", "")), []).append(rec)

        num_step = len(records)
        num_episode = len(episode_map)
        num_op = sum(1 for x in records if x["Op_match"])
        num_ele = sum(1 for x in records if x["Ele_match"])
        num_step_success = sum(1 for x in records if x["Op_F1"][0] == 1.0 and x["Ele_match"])
        num_episode_success = sum(
            1 for xs in episode_map.values() if xs and all(x["Op_F1"][0] == 1.0 and x["Ele_match"] for x in xs)
        )
        op_f1 = {4: [], 2: [], 3: []}
        macro_ele_acc = {}
        macro_step_acc = {}
        macro_action_f1 = {}
        for ep_idx, (_, xs) in enumerate(episode_map.items()):
            macro_ele_acc[ep_idx] = [1 if x["Ele_match"] else 0 for x in xs]
            macro_step_acc[ep_idx] = [1 if x["Op_F1"][0] == 1.0 and x["Ele_match"] else 0 for x in xs]
            macro_action_f1[ep_idx] = [x["Op_F1"][0] for x in xs]
            for x in xs:
                if x["Op_F1"][1] in op_f1:
                    op_f1[x["Op_F1"][1]].append(x["Op_F1"][0])

        op_category_means = [float(np.mean(x)) if x else 0.0 for x in op_f1.values()]
        marco_op_f1 = float(np.mean(op_category_means)) if op_category_means else 0.0
        macro_ele = float(np.mean([np.mean(x) for x in macro_ele_acc.values() if x])) if macro_ele_acc else 0.0
        macro_step = float(np.mean([np.mean(x) for x in macro_step_acc.values() if x])) if macro_step_acc else 0.0
        macro_action = float(np.mean([np.mean(x) for x in macro_action_f1.values() if x])) if macro_action_f1 else 0.0
        results = {
            "Operation_F1": marco_op_f1,
            "Element_Accuracy": float(num_ele / num_step) if num_step else 0.0,
            "Operation_Accuracy": float(num_op / num_step) if num_step else 0.0,
            "Step_Success_Rate": float(num_step_success / num_step) if num_step else 0.0,
            "Episode_Success_Rate": float(num_episode_success / num_episode) if num_episode else 0.0,
            "Operation_F1_Categories": op_category_means,
            "Macro_Element_Accuracy": macro_ele,
            "Macro_Operation_F1": macro_action,
            "Macro_Step_Success_Rate": macro_step,
            "Num_wrong_format": int(sum(1 for x in records if x.get("wrong_format", False))),
            "Total": int(num_step),
            "Episode_Total": int(num_episode),
            "Episode_Success": int(num_episode_success),
        }
        result_file = eval_file.replace(".xlsx", "_mind2web_eval.json").replace(".csv", "_mind2web_eval.json")
        dump(results, result_file)
        return results
