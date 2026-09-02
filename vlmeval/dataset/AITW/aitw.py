import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from ..image_base import ImageBaseDataset
from ...smp import dump, get_intermediate_file_path, load, osp
from .action_matching import action2semantic_step, action2step, action_2_format, check_actions_match, pred_2_format
from ..AndroidControl_Curated.state_packet import (
    build_state_packet,
    state_packet_debug_enabled as android_state_packet_debug_enabled,
)


AITW_TASKS = ["general", "single", "webshopping", "install", "googleapps"]
AITW_DATASETS = [f"AITW_{task}" for task in AITW_TASKS] + ["AITW_all"]

DEFAULT_PROMPT_ORIGIN = (
    "Please generate the next move according to the instruction, previous actions, "
    "previous ui screenshot and current ui screenshot. Instruction: {}.\n"
)

OUTPUT_INSTRUCTION = (
    'Output one action as a Python/JSON dict only. Use these forms:\n'
    '- click: {"action_type": 4, "click_point": (x,y)}\n'
    '- input text: {"action_type": 3, "typed_text": "xxx"}\n'
    '- scroll down: {"action_type": 0}\n'
    '- scroll up: {"action_type": 1}\n'
    '- scroll left: {"action_type": 8}\n'
    '- scroll right: {"action_type": 9}\n'
    "For other global actions, keep the AITW action_type_id. "
    "click_point must be integer coordinates normalized to 0-1000 on the current screenshot."
)

SEMANTIC_OUTPUT_INSTRUCTION = (
    'Output one action as a JSON dict only. Use semantic action text, not numeric action ids:\n'
    '- click: {"action_type": "click", "bbox_2d": [x, y]}\n'
    '- input text: {"action_type": "input_text: xxx"}\n'
    '- scroll down: {"action_type": "scroll down"}\n'
    '- scroll up: {"action_type": "scroll up"}\n'
    '- scroll left: {"action_type": "scroll left"}\n'
    '- scroll right: {"action_type": "scroll right"}\n'
    '- back: {"action_type": "navigate_back"}\n'
    '- home: {"action_type": "navigate_home"}\n'
    '- complete: {"action_type": "complete"}\n'
    "For click, bbox_2d is a single [x, y] point normalized to 0-1000 on the current screenshot."
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


def _env_with_fallback(name: str, fallback_name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return os.getenv(fallback_name, default).strip()


def _state_packet_enabled() -> bool:
    return _env_flag("AITW_STATE_PACKET_ENABLE", os.getenv("ANDROID_CONTROL_STATE_PACKET_ENABLE", "0"))


def _state_packet_debug_enabled() -> bool:
    return _env_flag("AITW_STATE_PACKET_DEBUG", os.getenv("ANDROID_CONTROL_STATE_PACKET_DEBUG", "0"))


def _structured_fast_decode_enabled() -> bool:
    return _env_flag("QWEN3VL_ENABLE_STRUCTURED_FAST_DECODE", "0")


def _semantic_action_prompt_enabled() -> bool:
    value = os.getenv("AITW_SEMANTIC_ACTION_PROMPT", "auto").strip().lower()
    if value in {"auto", ""}:
        return _structured_fast_decode_enabled()
    return value in {"1", "true", "yes", "on", "semantic", "text"}


def _sync_android_state_packet_env_from_aitw() -> None:
    # The packet builder is shared with AndroidControl and reads the
    # ANDROID_CONTROL_* knobs internally. Mirror AITW_* values into those knobs
    # right before building packets, while leaving explicit Android values usable.
    mapping = {
        "STATE_PACKET_ENABLE": "0",
        "STATE_PACKET_DEBUG": "0",
        "STATE_PACKET_CACHE_DIR": "/tmp/aitw_state_packet_cache",
        "STATE_PACKET_PATCH_SIZE": "16",
        "STATE_PACKET_MERGE_SIZE": "2",
        "STATE_PACKET_THUMB_LONG_EDGE": "192",
        "STATE_PACKET_ROI_LONG_EDGE": "224",
        "STATE_PACKET_ROI_SHORT_SIDE_RATIO": "0.22",
        "STATE_PACKET_ROI_MIN_SIDE_PX": "160",
    }
    for suffix, default in mapping.items():
        aitw_name = f"AITW_{suffix}"
        android_name = f"ANDROID_CONTROL_{suffix}"
        os.environ[android_name] = _env_with_fallback(aitw_name, android_name, default)


class AndroidInTheWild(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {name: "" for name in AITW_DATASETS}
    TASKS = AITW_TASKS

    def __init__(
        self,
        dataset: str = "AITW_general",
        skip_noimg: bool = True,
        split: Optional[str] = None,
        skeleton: bool = False,
    ):
        self.dataset_name = dataset
        self.split = split or os.getenv("AITW_SPLIT", "test")
        self.ann_root = os.getenv("AITW_ANN_ROOT", "/mnt/storage2/Datasets/aitw_data/aitw_annots")
        self.img_root = os.getenv("AITW_IMAGE_ROOT", "/mnt/storage2/Datasets/aitw_data/aitw_images")
        self.his_num = max(0, _env_int("AITW_HIS_NUM", 4))
        self.with_no_history = _env_flag("AITW_WITH_NO_HISTORY", "0")
        self.use_history_state_packet = _state_packet_enabled()
        self._state_packet_records = []
        self.prompt_origin = os.getenv("AITW_PROMPT_ORIGIN", DEFAULT_PROMPT_ORIGIN)
        self.skip_noimg = skip_noimg
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

    def _json_path(self, split: str) -> str:
        return osp.join(self.ann_root, f"aitw_data_{split}.json")

    def _selected_tasks(self, dataset: str) -> List[str]:
        if dataset == "AITW_all":
            return list(self.TASKS)
        if not dataset.startswith("AITW_"):
            raise ValueError(f"Unsupported AITW dataset name: {dataset}")
        task = dataset[len("AITW_"):]
        if task not in self.TASKS:
            raise ValueError(f"Unsupported AITW task: {task}")
        return [task]

    def _resolve_image_path(self, img_filename: str) -> str:
        text = str(img_filename)
        if text.endswith(".png"):
            rel = text
        else:
            rel = f"{text}.png"
        if osp.isabs(rel):
            return rel
        return osp.join(self.img_root, rel)

    def _flatten_split(self, raw: Dict, dataset: str) -> pd.DataFrame:
        rows = []
        running_index = 0
        for task in self._selected_tasks(dataset):
            episodes = raw.get(task, [])
            for episode_order, episode in enumerate(episodes):
                prev_actions: List[str] = []
                prev_actions_semantic: List[str] = []
                prev_images: List[str] = []
                for step_order, step in enumerate(episode):
                    step = dict(step)
                    image_path = self._resolve_image_path(step.get("img_filename", ""))
                    ep_id = str(step.get("ep_id", f"{task}_{episode_order}"))
                    step_id = int(step.get("step", step_order))
                    row = {
                        "index": running_index,
                        "task": task,
                        "episode_id": ep_id,
                        "ep_id": ep_id,
                        "episode_order": episode_order,
                        "step_id": step_id,
                        "step_order": step_order,
                        "image_path": image_path,
                        "image": image_path,
                        "img_filename": step.get("img_filename", ""),
                        "goal": step.get("goal", ""),
                        "instruction": step.get("goal", ""),
                        "question": step.get("goal", ""),
                        "action_type_id": step.get("action_type_id", ""),
                        "action_type_text": step.get("action_type_text", ""),
                        "touch": step.get("touch", []),
                        "lift": step.get("lift", []),
                        "type_text": step.get("type_text", ""),
                        "annot_position": step.get("annot_position", []),
                        "action_addition": step.get("action_addition", ""),
                        "_prev_action_texts": list(prev_actions),
                        "_prev_action_texts_semantic": list(prev_actions_semantic),
                        "_prev_image_paths": list(prev_images),
                    }
                    row["answer"] = action2step(row)
                    row["answer_semantic"] = action2semantic_step(row)
                    rows.append(row)
                    prev_actions.append(row["answer"])
                    prev_actions_semantic.append(row["answer_semantic"])
                    prev_images.append(image_path)
                    running_index += 1
        return pd.DataFrame(rows)

    def load_data(self, dataset):
        json_path = self._json_path(self.split)
        assert osp.exists(json_path), (
            f"AITW annotation file not found: {json_path}. "
            "Set AITW_ANN_ROOT to the directory containing aitw_data_{train,val,test}.json."
        )
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return self._flatten_split(raw, dataset)

    def dump_image(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = str(line.get("image_path", line.get("image", "")))
        assert osp.exists(image_path), (
            f"AITW screenshot not found: {image_path}. "
            "Set AITW_IMAGE_ROOT to the directory containing task subfolders."
        )
        return [image_path]

    def _history_actions(self, line) -> List[str]:
        key = "_prev_action_texts_semantic" if _semantic_action_prompt_enabled() else "_prev_action_texts"
        actions = line.get(key, [])
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

    def _image_size(self, image_path: str):
        try:
            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    def _bbox_norm_yxhw_to_xyxy_px(self, bbox, image_size):
        if bbox is None or image_size is None:
            return None
        try:
            y, x, h, w = [float(v) for v in bbox[:4]]
            img_w, img_h = image_size
            return [
                x * img_w,
                y * img_h,
                (x + w) * img_w,
                (y + h) * img_h,
            ]
        except Exception:
            return None

    def _action_packet_for_row(self, row, image_path: str) -> dict:
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        action = action_2_format(row_dict)
        image_size = self._image_size(image_path)
        gt_coordinate = None
        gt_bbox = None
        if image_size is not None and action.get("touch_yx") is not None:
            img_w, img_h = image_size
            y, x = action["touch_yx"]
            gt_coordinate = [float(x) * img_w, float(y) * img_h]
        if image_size is not None and action.get("candidate_bboxes_yxhw"):
            gt_point = action.get("touch_yx")
            chosen = None
            for bbox in action.get("candidate_bboxes_yxhw", []) or []:
                if gt_point is not None:
                    by, bx, bh, bw = bbox
                    if by <= gt_point[0] <= by + bh and bx <= gt_point[1] <= bx + bw:
                        chosen = bbox
                        break
            # AITW annot_position is not guaranteed to be the clicked target.
            # Only pass a bbox when it actually contains the click point; otherwise
            # let the shared state-packet builder crop around gt_coordinate.
            if chosen is not None:
                gt_bbox = self._bbox_norm_yxhw_to_xyxy_px(chosen, image_size)

        kind = str(action.get("kind", "global"))
        if kind == "tap":
            gt_action = "click"
        elif kind == "type":
            gt_action = f"input_text: {action.get('typed_text', '')}"
        elif kind in ("scroll", "drag"):
            gt_action = str(action.get("text", row_dict.get("action_type_text", "scroll")))
        else:
            gt_action = str(action.get("text", row_dict.get("action_type_text", "")))

        return {
            "gt_action": gt_action,
            "gt_coordinate": gt_coordinate,
            "gt_bbox": gt_bbox,
            "step_instruction": action2semantic_step(row_dict) if _semantic_action_prompt_enabled() else action2step(row_dict),
            "instruction": row_dict.get("instruction", row_dict.get("goal", "")),
        }

    def _previous_rows_for_history(self, line, history_image_paths: List[str]):
        if not history_image_paths:
            return []
        task = str(line.get("task", ""))
        ep_id = str(line.get("episode_id", line.get("ep_id", "")))
        step_order = int(line.get("step_order", line.get("step_id", 0)))
        prev = self.data[
            (self.data["task"].astype(str) == task)
            & (self.data["episode_id"].astype(str) == ep_id)
            & (self.data["step_order"].astype(int) < step_order)
        ].sort_values("step_order", kind="stable")
        rows = list(prev.tail(len(history_image_paths)).to_dict("records"))
        return rows

    def _build_history_visual_entries(self, sample_index: str, history_image_paths: List[str], history_rows: List[dict]):
        if not self.use_history_state_packet:
            return [
                {
                    "history_index": i,
                    "action_text": action2semantic_step(row) if _semantic_action_prompt_enabled() else action2step(row),
                    "images": [dict(type="image", value=hist_image_path)],
                    "debug_items": [
                        {
                            "kind": "original",
                            "path": hist_image_path,
                            "crop_xyxy": None,
                            "estimated_tokens": None,
                        }
                    ],
                }
                for i, (hist_image_path, row) in enumerate(zip(history_image_paths, history_rows))
            ]

        _sync_android_state_packet_env_from_aitw()
        entries = []
        for i, (hist_image_path, row) in enumerate(zip(history_image_paths, history_rows)):
            action_packet = self._action_packet_for_row(row, hist_image_path)
            packet_images, packet_meta = build_state_packet(
                image_path=hist_image_path,
                action_packet=action_packet,
                sample_index=str(sample_index),
                history_index=i,
            )
            packet_meta["dataset_name"] = str(self.dataset_name)
            self._state_packet_records.append(packet_meta)
            debug_items = []
            for item in packet_images:
                debug_items.append(
                    {
                        "kind": item.kind,
                        "path": item.path,
                        "crop_xyxy": item.crop_xyxy,
                        "estimated_tokens": item.estimated_tokens,
                    }
                )
            entries.append(
                {
                    "history_index": i,
                    "action_text": action_packet["step_instruction"],
                    "images": [item.to_message_item() for item in packet_images],
                    "debug_items": debug_items,
                    "packet_meta": packet_meta,
                }
            )
        return entries

    def _build_prompt_text(self, instruction: str, history_images: List[str], history_actions: List[str]) -> str:
        prompt = self.prompt_origin.format(instruction)
        output_instruction = SEMANTIC_OUTPUT_INSTRUCTION if _semantic_action_prompt_enabled() else OUTPUT_INSTRUCTION
        if self.with_no_history:
            for i, action_text in enumerate(history_actions):
                prompt += f"Step_{i}: {action_text} .\n"
            prompt += "Image_0:<image>\n"
            prompt += output_instruction
            return prompt

        for i, action_text in enumerate(history_actions):
            prompt += f"Image_{i}:<image>\n"
            prompt += f"Step_{i}: {action_text} .\n"
        prompt += f"Image_{len(history_images)}:<image>\n"
        prompt += output_instruction
        return prompt

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = self.dump_image(line)[0]
        instruction = str(line.get("instruction", line.get("goal", "")))
        history_actions = self._history_actions(line)
        history_images = self._history_images(line, image_path)
        if not self.with_no_history and len(history_images) < len(history_actions):
            history_actions = history_actions[-len(history_images):] if history_images else []
        prompt = self._build_prompt_text(instruction, history_images, history_actions)

        if self.with_no_history:
            msgs = [dict(type="image", value=image_path)]
            msgs.append(dict(type="text", value=prompt))
            return msgs

        history_rows = self._previous_rows_for_history(line, history_images)
        if len(history_rows) < len(history_images):
            history_images = history_images[-len(history_rows):] if history_rows else []
            history_actions = history_actions[-len(history_rows):] if history_rows else []
            prompt = self._build_prompt_text(instruction, history_images, history_actions)
        history_entries = self._build_history_visual_entries(
            sample_index=str(line.get("index", "")),
            history_image_paths=history_images,
            history_rows=history_rows,
        )

        msgs = []
        for entry in history_entries:
            i = int(entry["history_index"])
            for image_item, debug_item in zip(entry.get("images", []), entry.get("debug_items", [])):
                kind = str(debug_item.get("kind", "history_image"))
                crop_xyxy = debug_item.get("crop_xyxy", None)
                if self.use_history_state_packet:
                    label = f"Image_{i} {kind}"
                    if crop_xyxy is not None:
                        label += f" crop_xyxy={list(crop_xyxy)}"
                    msgs.append(dict(type="text", value=f"{label}:"))
                msgs.append(dict(image_item))
        msgs.append(dict(type="image", value=image_path))
        msgs.append(dict(type="text", value=prompt))
        if _state_packet_debug_enabled() or android_state_packet_debug_enabled():
            print(
                "[AITWPrompt] "
                f"sample_index={line.get('index', '')} "
                f"history_images={len(history_images)} "
                f"state_packet={int(self.use_history_state_packet)} "
                f"message_images={sum(1 for x in msgs if x.get('type') == 'image')}",
                flush=True,
            )
        return msgs

    def summarize_state_packet_records(self):
        summary = {}
        recs = list(getattr(self, "_state_packet_records", []) or [])
        if recs:
            count = float(len(recs))
            orig_tokens = float(sum(float(r.get("original_estimated_tokens", 0.0) or 0.0) for r in recs))
            packet_tokens = float(sum(float(r.get("packet_estimated_tokens", 0.0) or 0.0) for r in recs))
            total_s = float(sum(float(r.get("state_packet_total_s", 0.0) or 0.0) for r in recs))
            summary.update(
                {
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
            )
        return summary

    @staticmethod
    def _ratio(num: int, den: int) -> float:
        return round(float(num) / float(den) * 100.0, 2) if den else 0.0

    @classmethod
    def _summarize_task(cls, records: List[Dict]) -> Dict[str, object]:
        total = len(records)
        wrong_format = sum(1 for x in records if bool(x.get("wrong_format", False)))
        action_acc = cls._ratio(sum(1 for x in records if x["action_match"]), total)
        type_acc = cls._ratio(sum(1 for x in records if x["type_match"]), total)
        text_records = [x for x in records if x["gt_kind"] == "type"]
        click_records = [x for x in records if x["gt_kind"] == "tap"]
        scroll_records = [x for x in records if x["gt_kind"] in ("scroll", "drag")]
        episode_map = {}
        for rec in records:
            ep_key = str(rec.get("episode_id", ""))
            if not ep_key:
                ep_key = str(rec.get("index", ""))
            episode_map.setdefault(ep_key, []).append(rec)
        episode_total = len(episode_map)
        episode_success = sum(
            1 for ep_records in episode_map.values()
            if ep_records and all(bool(x.get("action_match", False)) for x in ep_records)
        )
        return {
            "Action Acc": action_acc,
            "Step SR": cls._ratio(episode_success, episode_total),
            "Type Acc": type_acc,
            "Text Acc": cls._ratio(sum(1 for x in text_records if x["text_match"]), len(text_records)),
            "Click Acc": cls._ratio(sum(1 for x in click_records if x["click_match"]), len(click_records)),
            "Scroll Acc": cls._ratio(sum(1 for x in scroll_records if x["scroll_match"]), len(scroll_records)),
            "Both Click Acc": cls._ratio(
                sum(1 for x in click_records if x["both_click_match"]),
                len(click_records),
            ),
            "Num wrong format": int(wrong_format),
            "Total": int(total),
            "Episode Total": int(episode_total),
            "Episode Success": int(episode_success),
        }

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert "prediction" in data, f"Missing `prediction` column in {eval_file}"
        records = []
        for i in range(len(data)):
            row = data.iloc[i]
            step_data = row.to_dict()
            gt_action = action_2_format(step_data)
            pred_action = pred_2_format(_nonnull(row.get("prediction", "")))
            match = check_actions_match(gt_action, pred_action)
            rec = {
                "index": str(row.get("index", i)),
                "task": str(row.get("task", "unknown")),
                "episode_id": str(row.get("episode_id", row.get("ep_id", ""))),
                "step_id": int(row.get("step_id", row.get("step_order", -1))),
                "prediction": str(row.get("prediction", "")),
                "gt_action": gt_action,
                "pred_action": pred_action,
                "wrong_format": pred_action is None,
                "gt_kind": gt_action.get("kind", ""),
            }
            rec.update(match)
            records.append(rec)

        tasks = self._selected_tasks(self.dataset_name)
        summary = {}
        for task in tasks:
            summary[task] = self._summarize_task([x for x in records if x["task"] == task])
        if self.dataset_name == "AITW_all":
            action_accs = [summary[t]["Action Acc"] for t in tasks if summary[t]["Total"] > 0]
            step_srs = [summary[t]["Step SR"] for t in tasks if summary[t]["Episode Total"] > 0]
            summary["Avg Action Acc"] = round(float(np.mean(action_accs)), 2) if action_accs else 0.0
            summary["Avg Step SR"] = round(float(np.mean(step_srs)), 2) if step_srs else 0.0
        summary["evaluated"] = len(records)

        score_file = get_intermediate_file_path(eval_file, "_score", "json")
        detail_file = get_intermediate_file_path(eval_file, "_details", "json")
        dump(summary, score_file)
        dump(records, detail_file)
        return summary

    def eval_single(self, line, prediction):
        gt_action = action_2_format(line.to_dict() if hasattr(line, "to_dict") else dict(line))
        pred_action = pred_2_format(prediction)
        match = check_actions_match(gt_action, pred_action)
        return {"is_correct": bool(match.get("action_match", False)), "info": gt_action.get("kind", "")}


AITWDataset = AndroidInTheWild
