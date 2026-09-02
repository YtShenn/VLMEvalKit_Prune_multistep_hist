import ast
import json
import math
import re
from typing import Any, Dict, Optional, Tuple

from .action_type import (
    ACTION_ID_TO_NAME,
    COMPLETE,
    DUAL_POINT,
    GLOBAL_ACTIONS,
    PRESS_BACK,
    PRESS_ENTER,
    PRESS_HOME,
    SCROLL_ACTIONS,
    SCROLL_DOWN,
    SCROLL_LEFT,
    SCROLL_NAME_TO_ID,
    SCROLL_RIGHT,
    SCROLL_UP,
    TYPE,
    UNKNOWN,
)


FormatAction = Dict[str, Any]


def _to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _to_float_pair(value: Any, scale: float = 1.0) -> Optional[Tuple[float, float]]:
    if isinstance(value, str):
        value = value.strip()
        try:
            value = ast.literal_eval(value)
        except Exception:
            nums = re.findall(r"-?\d+(?:\.\d+)?", value)
            if len(nums) >= 2:
                value = [nums[0], nums[1]]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]) / scale, float(value[1]) / scale
    except Exception:
        return None


def _clean_action_text(text: Any) -> str:
    return str(text or "").strip().lower().replace("_", " ").replace(":", " ")


def _step_text(step_data: Dict[str, Any]) -> str:
    return " ".join(
        _clean_action_text(step_data.get(k, ""))
        for k in ("action_type_text", "action_addition")
    )


def _scroll_id_from_text(text: str) -> Optional[int]:
    text = _clean_action_text(text)
    for key, value in SCROLL_NAME_TO_ID.items():
        if key in text:
            return value
    return None


def _is_scroll_step(step_data: Dict[str, Any]) -> bool:
    action_id = _to_int(step_data.get("action_type_id"))
    if action_id in SCROLL_ACTIONS:
        return True
    return _scroll_id_from_text(_step_text(step_data)) is not None


def _point_xy_to_yx(point: Any, scale: float = 1.0) -> Optional[Tuple[float, float]]:
    xy = _to_float_pair(point, scale=scale)
    if xy is None:
        return None
    x, y = xy
    return y, x


def _pred_point_xy_to_yx(point: Any, scale: float = 1.0) -> Optional[Tuple[float, float]]:
    if isinstance(point, str):
        try:
            point = ast.literal_eval(point.strip())
        except Exception:
            nums = re.findall(r"-?\d+(?:\.\d+)?", point)
            point = nums
    if not isinstance(point, (list, tuple)):
        return None
    if len(point) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in point[:4]]
            return ((y1 + y2) / 2.0 / scale, (x1 + x2) / 2.0 / scale)
        except Exception:
            return None
    return _point_xy_to_yx(point, scale=scale)


def _tap_bbox_xywh_to_yxhw(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(bbox, str):
        try:
            bbox = ast.literal_eval(bbox)
        except Exception:
            nums = re.findall(r"-?\d+(?:\.\d+)?", bbox)
            bbox = nums
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    return y, x, h, w


def _candidate_bboxes(step_data: Dict[str, Any]):
    raw = step_data.get("annot_position", [])
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except Exception:
            raw = []
    if not isinstance(raw, (list, tuple)):
        return []
    boxes = []
    for i in range(0, len(raw) - 3, 4):
        box = _tap_bbox_xywh_to_yxhw(raw[i:i + 4])
        if box is not None:
            boxes.append(box)
    return boxes


def _fixed_scroll(action_id: int) -> FormatAction:
    if action_id == SCROLL_DOWN:
        touch, lift = (0.8, 0.5), (0.2, 0.5)
    elif action_id == SCROLL_UP:
        touch, lift = (0.2, 0.5), (0.8, 0.5)
    elif action_id == SCROLL_LEFT:
        touch, lift = (0.5, 0.2), (0.5, 0.8)
    else:
        touch, lift = (0.5, 0.8), (0.5, 0.2)
    return {
        "action_type": DUAL_POINT,
        "touch_yx": touch,
        "lift_yx": lift,
        "kind": "scroll",
        "scroll_id": action_id,
        "text": ACTION_ID_TO_NAME.get(action_id, str(action_id)),
    }


def action2step(step_data: Dict[str, Any]) -> str:
    action = action_2_format(step_data)
    kind = action.get("kind")
    if kind == "tap":
        x = int(round(float(action["touch_yx"][1]) * 1000))
        y = int(round(float(action["touch_yx"][0]) * 1000))
        return f'{{"action_type": 4, "click_point": ({x},{y})}}'
    if kind == "type":
        text = str(action.get("typed_text", ""))
        return json.dumps({"action_type": TYPE, "typed_text": text}, ensure_ascii=False)
    if kind == "scroll":
        return json.dumps({"action_type": int(action.get("scroll_id", SCROLL_DOWN))}, ensure_ascii=False)
    return json.dumps({"action_type": int(action.get("action_type", UNKNOWN))}, ensure_ascii=False)


def action2semantic_step(step_data: Dict[str, Any]) -> str:
    action = action_2_format(step_data)
    kind = action.get("kind")
    if kind == "tap":
        x = int(round(float(action["touch_yx"][1]) * 1000))
        y = int(round(float(action["touch_yx"][0]) * 1000))
        return json.dumps({"action_type": "click", "bbox_2d": [x, y]}, ensure_ascii=False)
    if kind == "type":
        text = str(action.get("typed_text", ""))
        return json.dumps({"action_type": f"input_text: {text}"}, ensure_ascii=False)
    if kind == "scroll":
        return json.dumps({"action_type": str(action.get("text", ACTION_ID_TO_NAME.get(SCROLL_DOWN, "scroll down")))}, ensure_ascii=False)
    action_id = int(action.get("action_type", UNKNOWN))
    if action_id == PRESS_BACK:
        text = "navigate_back"
    elif action_id == PRESS_HOME:
        text = "navigate_home"
    elif action_id == PRESS_ENTER:
        text = "enter"
    elif action_id == COMPLETE:
        text = "complete"
    else:
        text = ACTION_ID_TO_NAME.get(action_id, str(action_id))
    return json.dumps({"action_type": text}, ensure_ascii=False)


def action_2_format(step_data: Dict[str, Any]) -> FormatAction:
    action_id = _to_int(step_data.get("action_type_id"), UNKNOWN)
    text = _step_text(step_data)
    scroll_id = _scroll_id_from_text(text)
    if action_id in SCROLL_ACTIONS:
        scroll_id = action_id
    if scroll_id is not None:
        action = _fixed_scroll(scroll_id)
        gt_touch = _point_xy_to_yx(step_data.get("touch"))
        gt_lift = _point_xy_to_yx(step_data.get("lift"))
        if gt_touch is not None and gt_lift is not None:
            action["touch_yx"] = gt_touch
            action["lift_yx"] = gt_lift
        return action

    if action_id == TYPE:
        return {
            "action_type": TYPE,
            "kind": "type",
            "typed_text": str(step_data.get("type_text", "") or ""),
        }

    if action_id == DUAL_POINT:
        touch = _point_xy_to_yx(step_data.get("touch"))
        lift = _point_xy_to_yx(step_data.get("lift"))
        if touch is not None and lift is not None:
            dist = math.dist(touch, lift)
            if dist <= 0.04 and "long" not in text:
                return {
                    "action_type": DUAL_POINT,
                    "kind": "tap",
                    "touch_yx": touch,
                    "lift_yx": lift,
                    "candidate_bboxes_yxhw": _candidate_bboxes(step_data),
                }
            return {
                "action_type": DUAL_POINT,
                "kind": "drag",
                "touch_yx": touch,
                "lift_yx": lift,
            }

    return {
        "action_type": int(action_id if action_id is not None else UNKNOWN),
        "kind": "global",
        "text": ACTION_ID_TO_NAME.get(int(action_id or UNKNOWN), str(action_id)),
    }


def _strip_code_fence(text: str) -> str:
    text = str(text).strip()
    fence = re.search(r"```(?:json|python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _extract_dict_text(response: Any) -> Optional[str]:
    raw = _strip_code_fence(str(response))
    tag = re.findall(r"<(?:answer|action)>(.*?)</(?:answer|action)>", raw, flags=re.DOTALL | re.IGNORECASE)
    if tag:
        raw = tag[-1].strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start:end + 1]
    return None


def _parse_payload(response: Any) -> Optional[Dict[str, Any]]:
    if isinstance(response, dict):
        return response
    text = _extract_dict_text(response)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        payload = ast.literal_eval(text)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def pred_2_format(response: Any) -> Optional[FormatAction]:
    payload = _parse_payload(response)
    if payload is None:
        return None

    raw_action = payload.get("action_type", payload.get("type"))
    action_id = _to_int(raw_action, None)
    if action_id is None:
        action_text = _clean_action_text(raw_action or payload.get("action", ""))
        action_id = _scroll_id_from_text(action_text)
        if action_id is None and "click" in action_text:
            action_id = DUAL_POINT
        elif action_id is None and ("type" in action_text or "input" in action_text):
            action_id = TYPE
        elif action_id is None and "back" in action_text:
            action_id = PRESS_BACK
        elif action_id is None and "home" in action_text:
            action_id = PRESS_HOME
        elif action_id is None and "enter" in action_text:
            action_id = PRESS_ENTER
        elif action_id is None and "complete" in action_text:
            action_id = COMPLETE
    if action_id is None:
        return None

    if action_id in SCROLL_ACTIONS:
        return _fixed_scroll(action_id)
    if action_id == TYPE:
        typed_text = payload.get("typed_text", payload.get("text", ""))
        if not typed_text and isinstance(raw_action, str) and ":" in raw_action:
            typed_text = raw_action.split(":", 1)[1].strip()
        return {
            "action_type": TYPE,
            "kind": "type",
            "typed_text": str(typed_text or ""),
        }
    if action_id == DUAL_POINT:
        point = (
            payload.get("click_point", None)
            if "click_point" in payload
            else payload.get("point", payload.get("coordinate", payload.get("bbox_2d", payload.get("bbox"))))
        )
        scale = 1000.0
        touch = _pred_point_xy_to_yx(point, scale=scale)
        if touch is None:
            return None
        return {
            "action_type": DUAL_POINT,
            "kind": "tap",
            "touch_yx": touch,
            "lift_yx": touch,
        }
    if action_id in GLOBAL_ACTIONS:
        return {
            "action_type": int(action_id),
            "kind": "global",
            "text": ACTION_ID_TO_NAME.get(int(action_id), str(action_id)),
        }
    return {
        "action_type": int(action_id),
        "kind": "global",
        "text": ACTION_ID_TO_NAME.get(int(action_id), str(action_id)),
    }


def is_tap_action(action: Optional[FormatAction]) -> bool:
    return bool(action and action.get("kind") == "tap")


def _point_in_bbox(point_yx, bbox_yxhw, extra_width=0.0, extra_height=0.0) -> bool:
    y, x = point_yx
    by, bx, bh, bw = bbox_yxhw
    by -= extra_height
    bx -= extra_width
    bh += extra_height * 2
    bw += extra_width * 2
    return by <= y <= by + bh and bx <= x <= bx + bw


def _tap_actions_match(gt: FormatAction, pred: FormatAction) -> bool:
    gt_point = gt.get("touch_yx")
    pred_point = pred.get("touch_yx")
    if gt_point is None or pred_point is None:
        return False
    for bbox in gt.get("candidate_bboxes_yxhw", []) or []:
        if _point_in_bbox(pred_point, bbox, extra_width=0.1, extra_height=0.1):
            return True
    return math.dist(gt_point, pred_point) <= 0.14


def _drag_direction(action: FormatAction) -> Optional[str]:
    touch = action.get("touch_yx")
    lift = action.get("lift_yx")
    if touch is None or lift is None:
        return None
    dy = float(lift[0]) - float(touch[0])
    dx = float(lift[1]) - float(touch[1])
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def _text_match(gt: Any, pred: Any) -> bool:
    gt = str(gt or "").strip().lower()
    pred = str(pred or "").strip().lower()
    if gt == pred or gt in pred or pred in gt:
        return True
    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())
    if not gt_tokens or not pred_tokens:
        return False
    common = gt_tokens & pred_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return (2 * precision * recall / max(1e-8, precision + recall)) >= 0.5


def check_actions_match(gt_action: FormatAction, pred_action: Optional[FormatAction]) -> Dict[str, bool]:
    result = {
        "action_match": False,
        "type_match": False,
        "text_match": False,
        "click_match": False,
        "scroll_match": False,
        "both_click_match": False,
    }
    if pred_action is None:
        return result

    gt_kind = gt_action.get("kind")
    pred_kind = pred_action.get("kind")
    if gt_kind == "tap":
        result["type_match"] = pred_kind == "tap"
        result["click_match"] = result["type_match"] and _tap_actions_match(gt_action, pred_action)
        result["both_click_match"] = result["click_match"]
        result["action_match"] = result["both_click_match"]
        return result
    if gt_kind == "type":
        result["type_match"] = pred_kind == "type"
        result["text_match"] = result["type_match"] and _text_match(
            gt_action.get("typed_text", ""),
            pred_action.get("typed_text", ""),
        )
        result["action_match"] = result["text_match"]
        return result
    if gt_kind in ("scroll", "drag"):
        result["type_match"] = pred_kind in ("scroll", "drag")
        if gt_action.get("scroll_id") is not None and pred_action.get("scroll_id") is not None:
            result["scroll_match"] = result["type_match"] and int(gt_action["scroll_id"]) == int(pred_action["scroll_id"])
        else:
            result["scroll_match"] = result["type_match"] and _drag_direction(gt_action) == _drag_direction(pred_action)
        result["action_match"] = result["scroll_match"]
        return result

    result["type_match"] = int(gt_action.get("action_type", UNKNOWN)) == int(pred_action.get("action_type", -1))
    result["action_match"] = result["type_match"]
    return result
