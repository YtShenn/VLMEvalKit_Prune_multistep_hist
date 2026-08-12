import ast
import json
import os
import re
from typing import List, Optional

import pandas as pd

from ..image_base import ImageBaseDataset
from ...smp import dump, load, osp
from .official_eval import (
    evaluate_android_control_records_official,
    use_official_android_control_eval,
)
from .state_packet import build_state_packet, state_packet_debug_enabled, state_packet_enabled


PROMPT_TEMPLATE = """
Instruction:
Observe screenshot(s) carefully and propose the most possible element in screenshot1 with bbox_2d[x1, y1, x2, y2] and action_type that can make screenshot1 finish the following Task.

Task: {Question}. Past_Actions: {past_actions}.

Output_format:
<answer>{{"bbox_2d": [x1, y1, x2, y2], "action_type": ACTION_TYPE}}</answer>

NOTES:
- Output inside <answer> tag should always in JSON FORMAT.
- [x1, y1, x2, y2] should be absolute screen coordinates.
- ACTION_TYPE includes: "click", "long_press", "swipe:up", "swipe:down", "swipe:left", "swipe:right", "input_text: some text", "wait", "navigate_back", "navigate_home", "open_app:app_name".
- ONLY "click" and "long_press" need bbox_2d; others only need ACTION_TYPE.
""".strip()


SYSTEM_PROMPT = """
Instruction:
Observe screenshot(s) carefully and propose the most possible element in screenshot1 with bbox_2d[x1, y1, x2, y2] and action_type.

Output_format:
<answer>{{"bbox_2d": [x1, y1, x2, y2], "action_type": ACTION_TYPE}}</answer>

NOTES:
- Output inside <answer> tag should always in JSON FORMAT.
- [x1, y1, x2, y2] should be absolute screen coordinates on screenshot1, the current screenshot only.
- ACTION_TYPE includes: "click", "long_press", "swipe:up", "swipe:down", "swipe:left", "swipe:right", "input_text: some text", "wait", "navigate_back", "navigate_home", "open_app:app_name".
- ONLY "click" and "long_press" need bbox_2d; others only need ACTION_TYPE.
""".strip()


def _safe_literal_eval(val):
    if isinstance(val, (list, tuple, dict)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return val


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return int(default)


def _debug_print_enabled() -> bool:
    return _env_flag("ANDROID_CONTROL_DEBUG_HISTORY_PROMPT", "0")


def _sequential_order_enabled() -> bool:
    return _env_flag("ANDROID_CONTROL_SEQUENTIAL_ORDER", "1")


def _keep_prompt_template_enabled() -> bool:
    return _env_flag("ANDROID_CONTROL_HISTORY_KEEP_SYSTEM_PROMPT", "0")


def _normalize_candidate_actions(val):
    if val is None:
        return []
    if isinstance(val, float) and pd.isna(val):
        return []
    if isinstance(val, str):
        parsed = _safe_literal_eval(val)
        if parsed == val:
            return []
        val = parsed
    if isinstance(val, dict):
        return [val]
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, list):
        return val
    return []


def _correct_type(action_type):
    t = str(action_type).strip().lower()
    if "tap" in t:
        t = t.replace("tap", "click")
    if "scroll" in t:
        t = t.replace("scroll", "swipe")
        if "up" in t:
            t = t.replace("up", "down")
        elif "down" in t:
            t = t.replace("down", "up")
        elif "left" in t:
            t = t.replace("left", "right")
        elif "right" in t:
            t = t.replace("right", "left")
    if "press_back" in t:
        t = t.replace("press_back", "navigate_back")
    if "type" in t:
        t = t.replace("type", "input_text")
    return t


def _f1_like_match(gt_text, pred_text):
    gt = str(gt_text).replace("[", "").replace("]", "").strip().lower()
    pred = str(pred_text).replace("[", "").replace("]", "").strip().lower()
    if gt in pred or pred in gt:
        return True
    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())
    if len(gt_tokens) == 1 and len(pred_tokens) == 1:
        g = list(gt_tokens)[0]
        p = list(pred_tokens)[0]
        return g in p or p in g
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return False
    common = gt_tokens.intersection(pred_tokens)
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return False
    return (2 * precision * recall / (precision + recall)) >= 0.5


def _type_acc(gt_type, pred_type):
    if pred_type is None:
        return False
    gt = _correct_type(gt_type)
    pred = _correct_type(pred_type)
    if ":" in gt and ":" in pred:
        gt_main, gt_text = gt.split(":", 1)
        pred_main, pred_text = pred.split(":", 1)
        return gt_main == pred_main and _f1_like_match(gt_text, pred_text)
    return gt == pred


def _inside_box(gt_box, pred_box):
    if pred_box is None:
        return False
    try:
        gx1, gy1, gx2, gy2 = [float(x) for x in gt_box]
        if len(pred_box) == 4:
            px1, py1, px2, py2 = [float(x) for x in pred_box]
            cx = (px1 + px2) / 2.0
            cy = (py1 + py2) / 2.0
        elif len(pred_box) == 2:
            cx, cy = [float(x) for x in pred_box]
        else:
            return False
        return gx1 <= cx <= gx2 and gy1 <= cy <= gy2
    except Exception:
        return False


def _parse_response(text):
    raw = str(text)
    answer_matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    action_matches = re.findall(r"<action>(.*?)</action>", raw, re.DOTALL)
    candidate = (answer_matches + action_matches)[-1].strip() if (answer_matches or action_matches) else raw.strip()
    bbox_pred, type_pred = None, None

    try:
        pred_dict = _safe_literal_eval(candidate)
        if isinstance(pred_dict, dict):
            if "bbox_2d" in pred_dict:
                bbox_pred = pred_dict["bbox_2d"]
            elif "bbox" in pred_dict:
                bbox_pred = pred_dict["bbox"]
            elif "point" in pred_dict:
                bbox_pred = pred_dict["point"]
            elif "coordinate" in pred_dict:
                bbox_pred = pred_dict["coordinate"]
            if isinstance(bbox_pred, str):
                bbox_pred = _safe_literal_eval(bbox_pred)
            type_pred = pred_dict.get("type", None) or pred_dict.get("action_type", None)
    except Exception:
        bbox_pred = None

    if bbox_pred is None:
        bbox_pat = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        bbox_match = re.search(bbox_pat, raw)
        if bbox_match:
            bbox_pred = [
                float(bbox_match.group(1)),
                float(bbox_match.group(2)),
                float(bbox_match.group(3)),
                float(bbox_match.group(4)),
            ]

    if type_pred is None:
        type_pat = r"[\"']?(?:action_)?type[\"']?\s*:\s*[\"']([^\"']+)[\"']"
        type_match = re.search(type_pat, raw)
        if type_match:
            type_pred = type_match.group(1)

    return bbox_pred, type_pred


def _calc_single(gt_action, pred):
    gt_type = gt_action[0]
    gt_bbox = gt_action[1] if len(gt_action) > 1 else None
    bbox_pred, type_pred = _parse_response(pred)
    if gt_type not in ["click", "long_press"]:
        bbox_flag = None
    else:
        bbox_flag = _inside_box(gt_bbox, bbox_pred)
    type_flag = _type_acc(gt_type, type_pred)
    type_bbox_flag = type_flag if bbox_flag is None else (bbox_flag and type_flag)
    return bbox_flag, type_flag, type_bbox_flag


def _calc_multi(gt_action_list, pred):
    bbox_flags, type_flags, both_flags = [], [], []
    for gt_action in gt_action_list:
        b, t, bt = _calc_single(gt_action, pred)
        bbox_flags.append(b)
        type_flags.append(t)
        both_flags.append(bt)
    bbox_flag = any(x for x in bbox_flags if x is not None) if any(x is not None for x in bbox_flags) else None
    return bbox_flag, any(type_flags), any(both_flags)

def _task_key(item):
    """Best-effort task identifier for task-level SR aggregation."""
    for k in ["task_filename", "task_id", "episode", "revised_task", "instruction"]:
        v = item.get(k, None)
        if v is not None:
            s = str(v).strip()
            if s != "":
                return f"{k}:{s}"
    return f"index:{item.get('index', '')}"


class AndroidControlCurated(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI"
    DATASET_URL = {
        "AndroidControl_Curated_Low_Point": "",
        "AndroidControl_Curated_High_Point": "",
        "AndroidControl_Curated_Low_BBox": "",
        "AndroidControl_Curated_High_BBox": "",
        "AndroidControl_Curated_High_Task_Improved": "",
    }
    BENCHMARK_MAP = {
        "AndroidControl_Curated_Low_Point": "android_control_low_point.json",
        "AndroidControl_Curated_High_Point": "android_control_high_point.json",
        "AndroidControl_Curated_Low_BBox": "android_control_low_bbox.json",
        "AndroidControl_Curated_High_BBox": "android_control_high_bbox.json",
        "AndroidControl_Curated_High_Task_Improved": "android_control_high_task-improved.json",
    }

    def __init__(
        self,
        dataset="AndroidControl_Curated_High_BBox",
        skip_noimg=True,
        skeleton=False,
        include_history_screenshots: Optional[bool] = None,
        max_history_images: Optional[int] = None,
    ):
        self.dataset_name = dataset
        self.meta_only = True
        self.skip_noimg = skip_noimg
        self.img_root = self._resolve_image_root()
        self.include_history_screenshots = (
            _env_flag("ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS", "0")
            if include_history_screenshots is None
            else bool(include_history_screenshots)
        )
        self.sequential_order = _sequential_order_enabled()
        self.history_keep_prompt_template = _keep_prompt_template_enabled()
        self.use_history_state_packet = state_packet_enabled()
        self._state_packet_records = []
        self.max_history_images = max(
            0,
            _env_int("ANDROID_CONTROL_MAX_HISTORY_IMAGES", 0)
            if max_history_images is None
            else int(max_history_images),
        )
        if skeleton:
            return
        data = self.load_data(dataset)
        data = data.reset_index(drop=True)
        data["image_path"] = [self._resolve_image_path(x) for x in data["image"]]
        data = self._prepare_sequential_metadata(data)
        data["index"] = [str(i + 1) for i in range(len(data))]
        self.data = data

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def _resolve_src_root(self):
        return os.environ.get("ANDROID_CONTROL_CURATED_ROOT", "/home/ytshen/storage_net2/AndroidControl_Curated-main/src")

    def _resolve_image_root(self):
        return os.environ.get("ANDROID_CONTROL_CURATED_IMAGE_ROOT", self._resolve_src_root())

    def _resolve_json_path(self, dataset_name):
        file_name = self.BENCHMARK_MAP[dataset_name]
        return osp.join(self._resolve_src_root(), "benchmark_resource", file_name)

    def _resolve_image_path(self, image_rel_or_abs):
        p = str(image_rel_or_abs)
        if osp.isabs(p) and osp.exists(p):
            return p
        cands = [
            osp.join(self._resolve_image_root(), p),
            osp.join(self._resolve_src_root(), p),
        ]
        for c in cands:
            if osp.exists(c):
                return c
        return cands[0]

    def _parse_history_image_list(self, value) -> List[str]:
        parsed = _safe_literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            out = []
            for item in parsed:
                text = str(item).strip()
                if text:
                    out.append(self._resolve_image_path(text))
            return out
        return []

    def _parse_step_index(self, image_path: str) -> int:
        match = re.search(r"step_(\d+)\.[^.]+$", str(image_path or ""))
        if not match:
            return -1
        try:
            return int(match.group(1))
        except Exception:
            return -1

    def _has_step_trajectory_path(self, image_path: str) -> bool:
        return self._parse_step_index(image_path) >= 0

    def _trajectory_key_from_path(self, image_path: str) -> str:
        normalized = str(image_path or "").replace("\\", "/").rstrip("/")
        if "/" not in normalized:
            return normalized
        return normalized.rsplit("/", 1)[0]

    def _build_previous_action_texts(self, group: pd.DataFrame) -> List[List[str]]:
        prev_texts = []
        running = []
        for _, row in group.iterrows():
            prev_texts.append(list(running))
            step_inst = str(row.get("step_instruction", "")).strip()
            if step_inst:
                running.append(step_inst)
        return prev_texts

    def _build_previous_action_packets(self, group: pd.DataFrame) -> List[List[dict]]:
        prev_packets = []
        running = []
        for _, row in group.iterrows():
            prev_packets.append([dict(x) for x in running])
            gt_bbox = row.get("gt_max_bbox", None)
            if gt_bbox is None:
                gt_bbox = row.get("gt_min_bbox", None)
            packet = {
                "gt_action": row.get("gt_action", ""),
                "gt_coordinate": row.get("gt_coordinate", None),
                "gt_bbox": gt_bbox,
                "step_instruction": row.get("step_instruction", ""),
            }
            running.append(packet)
        return prev_packets

    def _prepare_sequential_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["_orig_order"] = list(range(len(data)))
        data["_trajectory_key"] = [self._trajectory_key_from_path(x) for x in data["image_path"]]
        data["_step_idx"] = [self._parse_step_index(x) for x in data["image_path"]]
        if self.sequential_order:
            data = data.sort_values(
                by=["_trajectory_key", "_step_idx", "_orig_order"],
                kind="stable",
            ).reset_index(drop=True)

        prev_action_texts_all = [[] for _ in range(len(data))]
        prev_image_paths_all = [[] for _ in range(len(data))]
        prev_action_packets_all = [[] for _ in range(len(data))]

        for _, group in data.groupby("_trajectory_key", sort=False):
            sorted_group = group.sort_values(by=["_step_idx", "_orig_order"], kind="stable")
            if not any(int(x) >= 0 for x in sorted_group["_step_idx"].tolist()):
                continue
            group_indices = list(sorted_group.index)
            prev_action_texts = self._build_previous_action_texts(sorted_group)
            prev_action_packets = self._build_previous_action_packets(sorted_group)
            running_images: List[str] = []
            for local_pos, data_idx in enumerate(group_indices):
                prev_action_texts_all[data_idx] = list(prev_action_texts[local_pos])
                prev_image_paths_all[data_idx] = list(running_images)
                prev_action_packets_all[data_idx] = list(prev_action_packets[local_pos])
                running_images.append(str(sorted_group.iloc[local_pos]["image_path"]))

        data["_prev_action_texts"] = prev_action_texts_all
        data["_prev_image_paths"] = prev_image_paths_all
        data["_prev_action_packets"] = prev_action_packets_all
        return data

    def _infer_history_from_current_image(self, current_image_path: str, limit: int) -> List[str]:
        if limit <= 0:
            return []
        current_path = str(current_image_path or "").strip()
        match = re.search(r"(.*[/\\\\]step_)(\d+)(\.[^.]+)$", current_path)
        if not match:
            return []
        prefix, step_text, suffix = match.groups()
        try:
            step_idx = int(step_text)
        except Exception:
            return []
        hist_paths = []
        for prev_step in range(step_idx - 1, 0, -1):
            cand = f"{prefix}{prev_step}{suffix}"
            if osp.exists(cand):
                hist_paths.append(cand)
            if len(hist_paths) >= limit:
                break
        return hist_paths

    def _resolve_history_screenshots(self, line, current_image_path: str) -> List[str]:
        limit = max(0, int(self.max_history_images))
        if not self.include_history_screenshots or limit <= 0:
            return []
        precomputed = line.get("_prev_image_paths", None)
        if isinstance(precomputed, list) and precomputed:
            deduped = []
            for p in precomputed:
                text = str(p).strip()
                if text and text != current_image_path and text not in deduped:
                    deduped.append(text)
            return deduped[-limit:]
        for key in (
            "history_screenshot",
            "history_screenshots",
            "history_image_paths",
            "history_images",
            "previous_screenshots",
        ):
            if key not in line or line.get(key) is None:
                continue
            parsed = self._parse_history_image_list(line.get(key))
            if parsed:
                deduped = []
                for p in parsed:
                    if p != current_image_path and p not in deduped:
                        deduped.append(p)
                return deduped[-limit:]
        if not self._has_step_trajectory_path(current_image_path):
            return []
        return self._infer_history_from_current_image(current_image_path, limit=limit)

    def _resolve_history_action_texts(self, line) -> List[str]:
        texts = line.get("_prev_action_texts", None)
        if isinstance(texts, list):
            return [str(x).strip() for x in texts if str(x).strip()]
        return []

    def _resolve_history_action_packets(self, line) -> List[dict]:
        packets = line.get("_prev_action_packets", None)
        if not isinstance(packets, list):
            return []
        return [dict(x) for x in packets if isinstance(x, dict)]

    def _format_history_actions(self, action_texts: List[str]) -> str:
        if not action_texts:
            return "None"
        return "".join([f"Step{i + 1}: {text};" for i, text in enumerate(action_texts)])

    def _build_history_visual_entries(self, sample_index: str, history_image_paths: List[str], history_action_packets: List[dict]):
        if not self.use_history_state_packet:
            entries = []
            for i, (hist_image_path, action_packet) in enumerate(zip(history_image_paths, history_action_packets)):
                entries.append(
                    dict(
                        history_index=i,
                        action_text=str(action_packet.get("step_instruction", "") or ""),
                        images=[dict(type="image", value=hist_image_path)],
                        debug_items=[
                            dict(
                                kind="original",
                                path=hist_image_path,
                                crop_xyxy=None,
                                estimated_tokens=None,
                            )
                        ],
                    )
                )
            return entries

        entries = []
        for i, (hist_image_path, action_packet) in enumerate(zip(history_image_paths, history_action_packets)):
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
                    dict(
                        kind=item.kind,
                        path=item.path,
                        crop_xyxy=item.crop_xyxy,
                        estimated_tokens=item.estimated_tokens,
                    )
                )
            entries.append(
                dict(
                    history_index=i,
                    action_text=str(action_packet.get("step_instruction", "") or ""),
                    images=[item.to_message_item() for item in packet_images],
                    debug_items=debug_items,
                    packet_meta=packet_meta,
                )
            )
        return entries

    def summarize_state_packet_records(self):
        recs = list(getattr(self, "_state_packet_records", []) or [])
        if not recs:
            return {}
        count = float(len(recs))
        orig_tokens = float(sum(float(r.get("original_estimated_tokens", 0.0) or 0.0) for r in recs))
        packet_tokens = float(sum(float(r.get("packet_estimated_tokens", 0.0) or 0.0) for r in recs))
        thumb_tokens = float(sum(float(r.get("thumbnail_estimated_tokens", 0.0) or 0.0) for r in recs))
        roi_tokens = float(sum(float(r.get("roi_estimated_tokens", 0.0) or 0.0) for r in recs))
        open_s = float(sum(float(r.get("open_image_s", 0.0) or 0.0) for r in recs))
        thumb_s = float(sum(float(r.get("thumbnail_build_s", 0.0) or 0.0) for r in recs))
        roi_s = float(sum(float(r.get("roi_build_s", 0.0) or 0.0) for r in recs))
        total_s = float(sum(float(r.get("state_packet_total_s", 0.0) or 0.0) for r in recs))
        return {
            "state_packet_enabled": bool(self.use_history_state_packet),
            "state_packet_history_image_count": int(count),
            "avg_state_packet_original_estimated_tokens": float(orig_tokens / count),
            "avg_state_packet_packet_estimated_tokens": float(packet_tokens / count),
            "avg_state_packet_thumbnail_estimated_tokens": float(thumb_tokens / count),
            "avg_state_packet_roi_estimated_tokens": float(roi_tokens / count),
            "state_packet_total_original_estimated_tokens": float(orig_tokens),
            "state_packet_total_packet_estimated_tokens": float(packet_tokens),
            "state_packet_avg_compression_ratio": float(packet_tokens / max(1.0, orig_tokens)),
            "avg_state_packet_open_image_s": float(open_s / count),
            "avg_state_packet_thumbnail_build_s": float(thumb_s / count),
            "avg_state_packet_roi_build_s": float(roi_s / count),
            "avg_state_packet_total_s": float(total_s / count),
            "total_state_packet_open_image_s": float(open_s),
            "total_state_packet_thumbnail_build_s": float(thumb_s),
            "total_state_packet_roi_build_s": float(roi_s),
            "total_state_packet_total_s": float(total_s),
        }

    def _build_reference_style_intro(self, instruction: str) -> str:
        return (
            "Please generate the next move according to the instruction, previous actions, "
            "previous ui screenshot and current ui screenshot. "
            f"Instruction: {instruction}\n"
        )

    def _build_reference_style_outro(self) -> str:
        return (
            "Output exactly in the following format:\n"
            "<answer>{\"bbox_2d\": [x1, y1, x2, y2], \"action_type\": ACTION_TYPE}</answer>\n"
            "The final image is the current screenshot. Predict bbox_2d on the current screenshot only. "
            "Use absolute screen coordinates. "
            "ACTION_TYPE includes: click, long_press, swipe:up, swipe:down, swipe:left, swipe:right, "
            "input_text: some text, wait, navigate_back, navigate_home, open_app:app_name.\n"
        )

    def _build_keep_prompt_intro(self, instruction: str, history_text: str) -> str:
        return (
            f"Task: {instruction}\n"
            "Previous Actions and Screenshots:\n"
            f"{history_text}\n"
        )

    def _build_keep_prompt_outro(self) -> str:
        return (
            "Provide the next action only using the required JSON format. "
            "The final image is the current screenshot. "
            "Predict bbox_2d on the current screenshot only."
        )

    def _maybe_debug_print_prompt(
        self,
        line,
        current_image_path: str,
        history_image_paths: List[str],
        history_text: str,
        prompt: str,
    ) -> None:
        if not _debug_print_enabled():
            return
        sample_index = str(line.get("index", ""))
        step_idx = line.get("_step_idx", None)
        print(
            f"[AndroidControlDebug] sample_index={sample_index} "
            f"current_image={current_image_path} "
            f"history_image_count={len(history_image_paths)} "
            f"step_idx={step_idx}",
            flush=True,
        )
        print(f"[AndroidControlDebug] history_images={history_image_paths}", flush=True)
        print(f"[AndroidControlDebug] history_text={history_text}", flush=True)
        if self.use_history_state_packet:
            print("[AndroidControlDebug] history_state_packet_enabled=1", flush=True)
        print("[AndroidControlDebug] prompt_begin", flush=True)
        print(prompt, flush=True)
        print("[AndroidControlDebug] prompt_end", flush=True)

    def load_data(self, dataset):
        json_path = self._resolve_json_path(dataset)
        assert osp.exists(json_path), f"AndroidControl benchmark file not found: {json_path}"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        image_path = self.dump_image(line)
        current_image_path = image_path[0]
        history_image_paths = self._resolve_history_screenshots(line, current_image_path)
        instruction = str(line.get("instruction", ""))
        history_action_texts = self._resolve_history_action_texts(line)
        history_action_packets = self._resolve_history_action_packets(line)
        history = self._format_history_actions(history_action_texts)
        if not self.include_history_screenshots:
            prompt = PROMPT_TEMPLATE.format(Question=instruction, past_actions=str(line.get("history", "")))
            self._maybe_debug_print_prompt(
                line=line,
                current_image_path=current_image_path,
                history_image_paths=[],
                history_text=str(line.get("history", "")),
                prompt=prompt,
            )
            return [dict(type="image", value=current_image_path), dict(type="text", value=prompt)]
        hist_actions_kept = history_action_texts[-len(history_image_paths):] if history_image_paths else []
        hist_action_packets_kept = history_action_packets[-len(history_image_paths):] if history_image_paths else []
        history_entries = self._build_history_visual_entries(
            sample_index=str(line.get("index", "")),
            history_image_paths=history_image_paths,
            history_action_packets=hist_action_packets_kept,
        )
        if self.history_keep_prompt_template:
            intro = self._build_keep_prompt_intro(instruction, history)
            outro = self._build_keep_prompt_outro()
            debug_parts = [SYSTEM_PROMPT, intro]
            msgs = [dict(type="text", value=SYSTEM_PROMPT), dict(type="text", value=intro)]
            if not self.use_history_state_packet:
                for i, (hist_image_path, action_text) in enumerate(zip(history_image_paths, hist_actions_kept)):
                    debug_parts.append(f"Image_{i}: [HISTORY_IMAGE] {hist_image_path}")
                    debug_parts.append(f"{i + 1}. {action_text}")
                    msgs.append(dict(type="text", value=f"Image_{i}:"))
                    msgs.append(dict(type="image", value=hist_image_path))
                    msgs.append(dict(type="text", value=f"{i + 1}. {action_text}\n"))
            else:
                for entry in history_entries:
                    i = int(entry["history_index"])
                    action_text = str(entry["action_text"])
                    debug_parts.append(f"HistoryStep_{i}: action={action_text}")
                    for image_item, debug_item in zip(entry["images"], entry.get("debug_items", [])):
                        label = str(debug_item.get("kind", "history_image"))
                        debug_parts.append(
                            f"HistoryStep_{i} {label}: {debug_item.get('path')} crop_xyxy={debug_item.get('crop_xyxy')} "
                            f"est_tokens={debug_item.get('estimated_tokens')}"
                        )
                        msgs.append(dict(type="text", value=f"HistoryStep_{i} {label}:"))
                        msgs.append(dict(image_item))
                    msgs.append(dict(type="text", value=f"{i + 1}. {action_text}\n"))
            debug_parts.append(f"Current Screenshot: [CURRENT_IMAGE] {current_image_path}")
            debug_parts.append(outro)
            self._maybe_debug_print_prompt(
                line=line,
                current_image_path=current_image_path,
                history_image_paths=history_image_paths,
                history_text=history,
                prompt="\n".join(debug_parts),
            )
            msgs.append(dict(type="text", value="Current Screenshot:"))
            msgs.append(dict(type="image", value=current_image_path))
            msgs.append(dict(type="text", value=outro))
            if state_packet_debug_enabled() and history_entries:
                for entry in history_entries:
                    packet_meta = entry.get("packet_meta", None)
                    if not isinstance(packet_meta, dict):
                        continue
                    print(
                        "[AndroidControlStatePacketPrompt] "
                        f"sample_index={packet_meta.get('sample_index')} "
                        f"hist_index={packet_meta.get('history_index')} "
                        f"action_type={packet_meta.get('action_type')} "
                        f"orig_tokens_est={packet_meta.get('original_estimated_tokens')} "
                        f"packet_tokens_est={packet_meta.get('packet_estimated_tokens')} "
                        f"roi_crop_xyxy={packet_meta.get('roi_crop_xyxy')} "
                        f"thumbnail_size=({packet_meta.get('thumbnail_width')},{packet_meta.get('thumbnail_height')}) "
                        f"roi_size=({packet_meta.get('roi_width')},{packet_meta.get('roi_height')}) "
                        f"open_s={float(packet_meta.get('open_image_s', 0.0)):.6f} "
                        f"thumb_s={float(packet_meta.get('thumbnail_build_s', 0.0)):.6f} "
                        f"roi_s={float(packet_meta.get('roi_build_s', 0.0)):.6f} "
                        f"packet_total_s={float(packet_meta.get('state_packet_total_s', 0.0)):.6f}",
                        flush=True,
                    )
            return msgs
        prompt = self._build_reference_style_intro(instruction)
        if not self.use_history_state_packet:
            for i, action in enumerate(hist_actions_kept):
                prompt += f"Image_{i}: [historical screenshot]\nStep_{i}: {action}.\n"
        else:
            for entry in history_entries:
                i = int(entry["history_index"])
                action_text = str(entry["action_text"])
                prompt += f"HistoryStep_{i}: [history packet images]\nStep_{i}: {action_text}.\n"
        prompt += f"Image_{len(hist_actions_kept)}: [current screenshot]\n"
        prompt += self._build_reference_style_outro()
        self._maybe_debug_print_prompt(
            line=line,
            current_image_path=current_image_path,
            history_image_paths=history_image_paths,
            history_text=history,
            prompt=prompt,
        )
        msgs = [dict(type="text", value=self._build_reference_style_intro(instruction))]
        if not self.use_history_state_packet:
            for i, (img_path, action) in enumerate(zip(history_image_paths, hist_actions_kept)):
                msgs.append(dict(type="text", value=f"Image_{i}:"))
                msgs.append(dict(type="image", value=img_path))
                msgs.append(dict(type="text", value=f"Step_{i}: {action}.\n"))
        else:
            for entry in history_entries:
                i = int(entry["history_index"])
                action_text = str(entry["action_text"])
                for image_item, debug_item in zip(entry["images"], entry.get("debug_items", [])):
                    label = str(debug_item.get("kind", "history_image"))
                    msgs.append(dict(type="text", value=f"HistoryStep_{i} {label}:"))
                    msgs.append(dict(image_item))
                msgs.append(dict(type="text", value=f"Step_{i}: {action_text}.\n"))
        msgs.append(dict(type="text", value=f"Image_{len(hist_actions_kept)}:"))
        msgs.append(dict(type="image", value=current_image_path))
        msgs.append(dict(type="text", value=self._build_reference_style_outro()))
        if state_packet_debug_enabled() and history_entries:
            for entry in history_entries:
                packet_meta = entry.get("packet_meta", None)
                if not isinstance(packet_meta, dict):
                    continue
                print(
                    "[AndroidControlStatePacketPrompt] "
                    f"sample_index={packet_meta.get('sample_index')} "
                    f"hist_index={packet_meta.get('history_index')} "
                    f"action_type={packet_meta.get('action_type')} "
                    f"orig_tokens_est={packet_meta.get('original_estimated_tokens')} "
                    f"packet_tokens_est={packet_meta.get('packet_estimated_tokens')} "
                    f"roi_crop_xyxy={packet_meta.get('roi_crop_xyxy')} "
                    f"thumbnail_size=({packet_meta.get('thumbnail_width')},{packet_meta.get('thumbnail_height')}) "
                    f"roi_size=({packet_meta.get('roi_width')},{packet_meta.get('roi_height')}) "
                    f"open_s={float(packet_meta.get('open_image_s', 0.0)):.6f} "
                    f"thumb_s={float(packet_meta.get('thumbnail_build_s', 0.0)):.6f} "
                    f"roi_s={float(packet_meta.get('roi_build_s', 0.0)):.6f} "
                    f"packet_total_s={float(packet_meta.get('state_packet_total_s', 0.0)):.6f}",
                    flush=True,
                )
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        records = data.to_dict("records") if isinstance(data, pd.DataFrame) else list(data)
        if use_official_android_control_eval():
            return evaluate_android_control_records_official(
                records,
                dataset_name=self.dataset_name,
                eval_file=eval_file,
                image_resolver=self._resolve_image_path,
            )
        use_improved = self.dataset_name == "AndroidControl_Curated_High_Task_Improved"

        total = len(records)
        no_bbox = 0
        bbox_hit = 0
        type_hit = 0
        both_hit = 0

        type_stat = {}
        eval_dump = []
        task_stat = {}

        for item in records:
            response = item.get("prediction", "")
            gt_action_type = str(item.get("gt_action", ""))
            gt_input_text = item.get("gt_input_text", None)
            if gt_input_text is not None and str(gt_input_text).strip().lower() != "no input text":
                gt_action_type = f"{gt_action_type}:{str(gt_input_text).lower()}"

            gt_bbox = item.get("gt_bbox", None)
            if gt_bbox is None:
                gt_bbox = item.get("gt_max_bbox", None)
            if isinstance(gt_bbox, str):
                gt_bbox = _safe_literal_eval(gt_bbox)

            gt_action = [gt_action_type, gt_bbox]
            if use_improved:
                gt_all = [gt_action]
                for cand in _normalize_candidate_actions(item.get("candidate_actions", [])):
                    gt_all.append([cand.get("action_type", ""), cand.get("action_bounds", [])])
                bbox_flag, type_flag, both_flag = _calc_multi(gt_all, response)
            else:
                bbox_flag, type_flag, both_flag = _calc_single(gt_action, response)

            gt_main_type = _correct_type(gt_action_type).split(":")[0]
            if gt_main_type not in ["click", "long_press"]:
                no_bbox += 1
            if bbox_flag:
                bbox_hit += 1
            if type_flag:
                type_hit += 1
            if both_flag:
                both_hit += 1

            if gt_main_type not in type_stat:
                type_stat[gt_main_type] = [0, 0]
            type_stat[gt_main_type][0] += 1
            if type_flag:
                type_stat[gt_main_type][1] += 1

            eval_row = dict(item)
            eval_row["bbox_flag"] = bbox_flag
            eval_row["type_flag"] = type_flag
            eval_row["type_bbox_flag"] = both_flag
            eval_dump.append(eval_row)

            tk = _task_key(item)
            if tk not in task_stat:
                task_stat[tk] = {"total": 0, "all_correct": True}
            task_stat[tk]["total"] += 1
            task_stat[tk]["all_correct"] = task_stat[tk]["all_correct"] and bool(both_flag)

        bbox_den = max(total - no_bbox, 1)
        box_acc = round(bbox_hit / bbox_den * 100, 1)
        type_acc = round(type_hit / max(total, 1) * 100, 1)
        type_bbox_acc = round(both_hit / max(total, 1) * 100, 1)
        total_tasks = len(task_stat)
        task_success = sum(1 for _, v in task_stat.items() if v["all_correct"])
        task_sr = round(task_success / max(total_tasks, 1) * 100, 1)
        type_breakdown = {
            k: {
                "count": v[0],
                "type_acc": round(v[1] / max(v[0], 1) * 100, 1),
            }
            for k, v in type_stat.items()
        }

        detail_file = eval_file.replace(".xlsx", "_android_control_detail.json")
        dump(eval_dump, detail_file)
        return {
            "Total": total,
            "NoBBoxTypeCount": no_bbox,
            "Box_Accuracy": box_acc,
            "Type_Accuracy": type_acc,
            "Type_and_Box_Accuracy": type_bbox_acc,
            "Step_Success_Rate": type_bbox_acc,
            "Task_Total": total_tasks,
            "Task_Success_Count": task_success,
            "Task_Success_Rate": task_sr,
            "Type_Breakdown": type_breakdown,
            "Detail_File": detail_file,
        }
