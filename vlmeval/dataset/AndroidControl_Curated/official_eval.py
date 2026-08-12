import ast
import json
import os
import re
from typing import Any

import pandas as pd
from PIL import Image

from ...smp import dump


OFFICIAL_EVAL_ENV = "ANDROID_CONTROL_CURATED_EVAL_MODE"
OFFICIAL_EVAL_VALUE = "official"

OFFICIAL_GROUPS = {
    "AndroidControl-Curated-Easy": [
        "AndroidControl_Curated_Low_Point",
        "AndroidControl_Curated_Low_BBox",
    ],
    "AndroidControl-Curated-Hard": [
        "AndroidControl_Curated_High_Point",
        "AndroidControl_Curated_High_Task_Improved",
    ],
    "AndroidControl-Curated-Box-Hard": [
        "AndroidControl_Curated_High_Point",
        "AndroidControl_Curated_High_BBox",
    ],
}


def use_official_android_control_eval() -> bool:
    return str(os.getenv(OFFICIAL_EVAL_ENV, "")).strip().lower() == OFFICIAL_EVAL_VALUE


def infer_android_control_dataset_name(path: str) -> str:
    full_path = str(path)
    name = os.path.basename(full_path)
    candidates = [
        "AndroidControl_Curated_Low_Point",
        "AndroidControl_Curated_High_Point",
        "AndroidControl_Curated_Low_BBox",
        "AndroidControl_Curated_High_BBox",
        "AndroidControl_Curated_High_Task_Improved",
    ]
    for candidate in candidates:
        if candidate in name or candidate in full_path:
            return candidate
    raise ValueError(
        "Cannot infer AndroidControl dataset name from path. "
        "Please pass the dataset name explicitly."
    )


def _safe_literal_eval(val: Any):
    if isinstance(val, (list, tuple, dict)):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return val


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


def correct_type(action_type):
    action_type = str(action_type)
    if "tap" in action_type:
        action_type = action_type.replace("tap", "click")
    if "scroll" in action_type:
        action_type = action_type.replace("scroll", "swipe")
        if "up" in action_type:
            action_type = action_type.replace("up", "down")
        elif "down" in action_type:
            action_type = action_type.replace("down", "up")
        elif "left" in action_type:
            action_type = action_type.replace("left", "right")
        elif "right" in action_type:
            action_type = action_type.replace("right", "left")
    if "press_back" in action_type:
        action_type = action_type.replace("press_back", "navigate_back")
    if "type" in action_type:
        action_type = action_type.replace("type", "input_text")
    return action_type


def _f1_like_match(ground_truth_str, predicted_str):
    predicted_str = str(predicted_str).replace("[", "").replace("]", "")
    ground_truth_str = str(ground_truth_str).replace("[", "").replace("]", "")
    if ground_truth_str in predicted_str or predicted_str in ground_truth_str:
        return True
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())
    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return True
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0 or len(ground_truth_tokens) == 0:
        return False
    precision = len(common_tokens) / len(predicted_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    if precision + recall == 0:
        return False
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score >= 0.5


def type_acc(type_gt, type_pred):
    if type_pred is None:
        return False
    type_gt = correct_type(type_gt)
    type_pred = correct_type(type_pred)
    if ":" in type_gt and ":" in type_pred:
        gt_main, gt_text = type_gt.split(":", 1)
        pred_main, pred_text = type_pred.split(":", 1)
        return gt_main == pred_main and _f1_like_match(gt_text, pred_text)
    return type_gt.lower() == type_pred.lower()


def inside_box(gt_box, pred_box):
    if pred_box is None:
        return False
    try:
        gt_x1, gt_y1, gt_x2, gt_y2 = [float(x) for x in gt_box]
        if len(pred_box) == 4:
            pred_x1, pred_y1, pred_x2, pred_y2 = [float(x) for x in pred_box]
            pred_center_x = (pred_x1 + pred_x2) / 2
            pred_center_y = (pred_y1 + pred_y2) / 2
        elif len(pred_box) == 2:
            pred_center_x, pred_center_y = [float(x) for x in pred_box]
        else:
            return False
    except Exception:
        return False
    return gt_x1 <= pred_center_x <= gt_x2 and gt_y1 <= pred_center_y <= gt_y2


def distance_similarity(gt_box, pred_box, w, h, pred=None):
    if pred_box is None:
        return False
    try:
        if isinstance(gt_box, list) and len(gt_box) == 2:
            gt_x, gt_y = float(gt_box[0]), float(gt_box[1])
        else:
            return False
        if len(pred_box) == 4:
            pred_x = (float(pred_box[0]) + float(pred_box[2])) / 2
            pred_y = (float(pred_box[1]) + float(pred_box[3])) / 2
        elif len(pred_box) == 2:
            pred_x = float(pred_box[0])
            pred_y = float(pred_box[1])
        else:
            return False
        if w <= 0 or h <= 0:
            return False
        distance = ((gt_x - pred_x) / float(w)) ** 2 + ((gt_y - pred_y) / float(h)) ** 2
        return distance < 0.14 ** 2
    except Exception:
        _ = pred
        return False


def parse_response(response_text):
    raw = str(response_text)
    answer_matches = re.findall(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    action_matches = re.findall(r"<action>(.*?)</action>", raw, re.DOTALL)
    candidate = (answer_matches + action_matches)[-1].strip() if (answer_matches or action_matches) else raw.strip()

    bbox_pred = None
    type_value = None
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
            type_value = pred_dict.get("type") or pred_dict.get("action_type")
    except Exception:
        bbox_pred = None

    if bbox_pred is None:
        bbox_pattern = r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
        bbox_match = re.search(bbox_pattern, raw)
        if bbox_match:
            bbox_pred = [
                float(bbox_match.group(1)),
                float(bbox_match.group(2)),
                float(bbox_match.group(3)),
                float(bbox_match.group(4)),
            ]

    if type_value is None:
        type_pattern = r"[\"']?(?:action_)?type[\"']?\s*:\s*[\"']([^\"']+)[\"']"
        type_match = re.search(type_pattern, raw)
        if type_match:
            type_value = type_match.group(1)
    return bbox_pred, type_value


def calculate_single_android(gt_action, pred, w, h, use_distance=False):
    gt_type = gt_action[0]
    gt_bbox = gt_action[1] if len(gt_action) >= 2 else None
    bbox_pred, type_pred = parse_response(pred)
    if gt_type not in ["click", "long_press"]:
        bbox_flag = None
        type_flag = type_acc(gt_type, type_pred)
    else:
        bbox_flag = distance_similarity(gt_bbox, bbox_pred, w, h, pred=pred) if use_distance else inside_box(gt_bbox, bbox_pred)
        type_flag = type_acc(gt_type, type_pred)
    type_bbox_flag = type_flag if bbox_flag is None else (bbox_flag and type_flag)
    return bbox_flag, type_flag, type_bbox_flag


def calculate_multi_android(gt_action_list, pred, w, h, use_distance=False):
    bbox_flag_list = []
    type_flag_list = []
    type_bbox_flag_list = []
    for gt_action in gt_action_list:
        bbox_flag, type_flag, type_bbox_flag = calculate_single_android(gt_action, pred, w, h, use_distance=use_distance)
        bbox_flag_list.append(bbox_flag)
        type_flag_list.append(type_flag)
        type_bbox_flag_list.append(type_bbox_flag)
    bbox_non_none = [x for x in bbox_flag_list if x is not None]
    bbox_flag = any(bbox_non_none) if bbox_non_none else None
    type_flag = any(type_flag_list)
    type_bbox_flag = any(type_bbox_flag_list)
    return bbox_flag, type_flag, type_bbox_flag


def _dataset_uses_distance(dataset_name: str) -> bool:
    return dataset_name in {
        "AndroidControl_Curated_Low_Point",
        "AndroidControl_Curated_High_Point",
    }


def _task_key(item: dict) -> str:
    for key in ["task_filename", "task_id", "episode", "revised_task", "instruction"]:
        value = item.get(key, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return f"{key}:{text}"
    return f"index:{item.get('index', '')}"


def _resolve_image_path_for_record(item, image_resolver=None):
    if image_resolver is not None:
        image_value = item.get("image", item.get("image_path", ""))
        return image_resolver(image_value)
    return item.get("image_path", item.get("image", ""))


def _read_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception:
        return None


def evaluate_android_control_records_official(records, dataset_name: str, eval_file: str | None = None, image_resolver=None):
    use_improved = dataset_name == "AndroidControl_Curated_High_Task_Improved"
    use_distance = _dataset_uses_distance(dataset_name)

    total_count = len(records)
    missing_image_count = 0
    no_bbox_count = 0
    bbox_flag_count = 0
    type_flag_count = 0
    type_bbox_flag_count = 0
    type_stat = {}
    eval_dump = []
    task_stat = {}

    for item in records:
        image_path = _resolve_image_path_for_record(item, image_resolver=image_resolver)
        image_size = _read_image_size(image_path)
        if image_size is None:
            missing_image_count += 1
            continue
        w, h = image_size
        response = item.get("prediction", item.get("response", ""))
        gt_action_type = str(item.get("gt_action", ""))
        gt_input_text = item.get("gt_input_text", None)
        if gt_input_text is not None and str(gt_input_text).strip().lower() != "no input text":
            gt_action_type = f"{gt_action_type}:{str(gt_input_text).lower()}"
        if "scroll" in gt_action_type:
            gt_action_type = correct_type(gt_action_type)

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
            bbox_flag, type_flag, type_bbox_flag = calculate_multi_android(gt_all, response, w, h, use_distance=False)
        else:
            bbox_flag, type_flag, type_bbox_flag = calculate_single_android(gt_action, response, w, h, use_distance=use_distance)

        gt_main_type = correct_type(gt_action_type).split(":")[0]
        if gt_main_type not in ["click", "long_press"]:
            no_bbox_count += 1
        if bbox_flag:
            bbox_flag_count += 1
        if type_flag:
            type_flag_count += 1
        if type_bbox_flag:
            type_bbox_flag_count += 1

        if gt_main_type not in type_stat:
            type_stat[gt_main_type] = [0, 0]
        type_stat[gt_main_type][0] += 1
        if type_flag:
            type_stat[gt_main_type][1] += 1

        eval_row = dict(item)
        eval_row["bbox_flag"] = bbox_flag
        eval_row["type_flag"] = type_flag
        eval_row["type_bbox_flag"] = type_bbox_flag
        eval_dump.append(eval_row)

        tk = _task_key(item)
        if tk not in task_stat:
            task_stat[tk] = {"all_correct": True}
        task_stat[tk]["all_correct"] = task_stat[tk]["all_correct"] and bool(type_bbox_flag)

    effective_total = max(total_count - missing_image_count, 0)
    bbox_total = max(effective_total - no_bbox_count, 0)
    grounding = round(bbox_flag_count / max(bbox_total, 1) * 100, 1)
    type_pct = round(type_flag_count / max(effective_total, 1) * 100, 1)
    sr_pct = round(type_bbox_flag_count / max(effective_total, 1) * 100, 1)
    task_total = len(task_stat)
    task_success = sum(1 for v in task_stat.values() if v["all_correct"])
    task_sr = round(task_success / max(task_total, 1) * 100, 1)

    detail_file = None
    if eval_file:
        detail_file = eval_file.replace(".xlsx", "_android_control_detail_official.json")
        dump(eval_dump, detail_file)

    metrics = {
        "Eval_Logic": "official_android_control_curated",
        "Dataset": dataset_name,
        "Type (%)": type_pct,
        "Grounding (%)": grounding,
        "SR (%)": sr_pct,
        "Type_Accuracy": type_pct,
        "Box_Accuracy": grounding,
        "Step_Success_Rate": sr_pct,
        "Type_and_Box_Accuracy": sr_pct,
        "Total": total_count,
        "Effective_Total": effective_total,
        "BBox_Total": bbox_total,
        "Type_Correct": type_flag_count,
        "Grounding_Correct": bbox_flag_count,
        "SR_Correct": type_bbox_flag_count,
        "NoBBoxTypeCount": no_bbox_count,
        "Missing_Image_Count": missing_image_count,
        "Task_Total": task_total,
        "Task_Success_Count": task_success,
        "Task_Success_Rate": task_sr,
        "Type_Breakdown": {
            key: {
                "count": value[0],
                "type_acc": round(value[1] / max(value[0], 1) * 100, 1),
            }
            for key, value in type_stat.items()
        },
    }
    if detail_file:
        metrics["Detail_File"] = detail_file
    return metrics


def evaluate_android_control_eval_file_official(eval_file: str, dataset_name: str, image_resolver=None):
    if str(eval_file).lower().endswith(".xlsx"):
        records = pd.read_excel(eval_file).to_dict("records")
    else:
        with open(eval_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    return evaluate_android_control_records_official(
        records,
        dataset_name=dataset_name,
        eval_file=eval_file,
        image_resolver=image_resolver,
    )


def aggregate_android_control_group_metrics(metrics_by_dataset: dict[str, dict], group_name: str):
    dataset_names = OFFICIAL_GROUPS[group_name]
    missing = [name for name in dataset_names if name not in metrics_by_dataset]
    if missing:
        raise ValueError(f"Missing datasets for {group_name}: {missing}")
    total = sum(metrics_by_dataset[name]["Effective_Total"] for name in dataset_names)
    bbox_total = sum(metrics_by_dataset[name]["BBox_Total"] for name in dataset_names)
    type_correct = sum(metrics_by_dataset[name]["Type_Correct"] for name in dataset_names)
    grounding_correct = sum(metrics_by_dataset[name]["Grounding_Correct"] for name in dataset_names)
    sr_correct = sum(metrics_by_dataset[name]["SR_Correct"] for name in dataset_names)
    return {
        "Group": group_name,
        "Datasets": dataset_names,
        "Type (%)": round(type_correct / max(total, 1) * 100, 1),
        "Grounding (%)": round(grounding_correct / max(bbox_total, 1) * 100, 1),
        "SR (%)": round(sr_correct / max(total, 1) * 100, 1),
        "Total": total,
        "BBox_Total": bbox_total,
        "Type_Correct": type_correct,
        "Grounding_Correct": grounding_correct,
        "SR_Correct": sr_correct,
    }
