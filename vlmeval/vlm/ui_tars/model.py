from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

from ..qwen2_vl import Qwen2VLChat


def _safe_parse_call(text: str) -> tuple[str | None, dict[str, Any]]:
    raw = str(text).strip()
    if "Action:" in raw:
        raw = raw.split("Action:", 1)[1].strip()
    raw = raw.splitlines()[0].strip()
    if not raw.endswith(")"):
        return None, {}

    try:
        node = ast.parse(raw, mode="eval")
    except Exception:
        return None, {}
    if not isinstance(node, ast.Expression) or not isinstance(node.body, ast.Call):
        return None, {}
    call = node.body

    func_name = None
    if isinstance(call.func, ast.Name):
        func_name = call.func.id
    elif isinstance(call.func, ast.Attribute):
        func_name = call.func.attr
    if func_name is None:
        return None, {}

    kwargs: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is None:
            continue
        try:
            kwargs[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            try:
                kwargs[kw.arg] = ast.unparse(kw.value)
            except Exception:
                kwargs[kw.arg] = None
    return str(func_name).lower(), kwargs


def _extract_xy(v) -> tuple[float, float] | None:
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return None
    if isinstance(v, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", v)
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
    return None


def _first_image_wh_from_message(message: list[dict]) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except Exception:
        return None
    for item in message:
        if not isinstance(item, dict) or item.get("type") != "image":
            continue
        p = str(item.get("value", ""))
        if p.startswith("file://"):
            p = p[len("file://"):]
        if not p:
            continue
        try:
            with Image.open(p) as img:
                return img.size
        except Exception:
            continue
    return None


def _extract_action_block(text: str) -> str:
    raw = str(text).strip()
    if "Action:" in raw:
        return raw.split("Action:", 1)[1].strip()
    return raw


def _parse_uitars_first_action(text: str) -> tuple[str | None, dict[str, str]]:
    action_block = _extract_action_block(text)
    lines = [ln.strip() for ln in action_block.splitlines() if ln.strip()]
    if not lines:
        return None, {}
    call_line = lines[0]
    fn, kwargs = _safe_parse_call(call_line)
    if fn is None:
        return None, {}
    action_inputs: dict[str, str] = {}
    for k, v in kwargs.items():
        action_inputs[str(k)] = str(v)
    return fn, action_inputs


def _parse_coord_like(v: str | Any) -> list[float] | None:
    if isinstance(v, (list, tuple)):
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    s = str(v).strip()
    try:
        vv = ast.literal_eval(s)
        if isinstance(vv, (list, tuple)):
            return [float(x) for x in vv]
    except Exception:
        pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        return [float(x) for x in nums[:4]]
    return None


def _coord_to_1k_xy(v: str | Any, img_wh: tuple[int, int] | None) -> tuple[int, int] | None:
    vals = _parse_coord_like(v)
    if vals is None or len(vals) < 2:
        return None
    x, y = vals[0], vals[1]
    if img_wh is not None and max(abs(x), abs(y)) <= 1.5:
        return int(round(x * 1000)), int(round(y * 1000))
    if max(abs(x), abs(y)) <= 1.5:
        return int(round(x * 1000)), int(round(y * 1000))
    if max(abs(x), abs(y)) <= 1000:
        return int(round(x)), int(round(y))
    if img_wh is not None:
        w, h = img_wh
        if w > 0 and h > 0:
            return int(round(x / w * 1000)), int(round(y / h * 1000))
    return int(round(x)), int(round(y))


def _coord_to_pixel_bbox(v: str | Any, img_wh: tuple[int, int] | None) -> list[int] | None:
    vals = _parse_coord_like(v)
    if vals is None or len(vals) < 2:
        return None
    if len(vals) == 2:
        vals = [vals[0], vals[1], vals[0], vals[1]]
    if len(vals) < 4:
        return None
    x1, y1, x2, y2 = vals[:4]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5 and img_wh is not None:
        w, h = img_wh
        return [
            int(round(x1 * w)),
            int(round(y1 * h)),
            int(round(x2 * w)),
            int(round(y2 * h)),
        ]
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _to_gui_odyssey_command(response: str, img_wh: tuple[int, int] | None) -> str:
    normalize_coords = os.getenv("UITARS_GUIODYSSEY_COORD_NORM", "1") == "1"
    action_type, action_inputs = _parse_uitars_first_action(response)
    if action_type is None:
        # Also normalize already command-style outputs, e.g. "CLICK: (1209, 2762)".
        raw = str(response).strip()
        m = re.search(r'^(CLICK|LONG_PRESS)\s*:\s*\(([-\d.]+)\s*,\s*([-\d.]+)\)', raw, re.IGNORECASE)
        if m:
            act = m.group(1).upper()
            x = float(m.group(2))
            y = float(m.group(3))
            if normalize_coords:
                pt = _coord_to_1k_xy([x, y], img_wh)
                if pt:
                    return f"{act}: ({pt[0]}, {pt[1]})"
            return f"{act}: ({int(round(x))}, {int(round(y))})"
        return response

    if action_type in ["click", "tap"]:
        if normalize_coords:
            pt = _coord_to_1k_xy(action_inputs.get("start_box", action_inputs.get("point", "")), img_wh)
        else:
            vv = _parse_coord_like(action_inputs.get("start_box", action_inputs.get("point", "")))
            pt = None if vv is None or len(vv) < 2 else (int(round(vv[0])), int(round(vv[1])))
        if pt:
            return f"CLICK: ({pt[0]}, {pt[1]})"
    if action_type in ["long_press", "longpress"]:
        if normalize_coords:
            pt = _coord_to_1k_xy(action_inputs.get("start_box", action_inputs.get("point", "")), img_wh)
        else:
            vv = _parse_coord_like(action_inputs.get("start_box", action_inputs.get("point", "")))
            pt = None if vv is None or len(vv) < 2 else (int(round(vv[0])), int(round(vv[1])))
        if pt:
            return f"LONG_PRESS: ({pt[0]}, {pt[1]})"
    if action_type in ["scroll", "swipe"]:
        direct = str(action_inputs.get("direction", action_inputs.get("dir", ""))).strip().upper()
        if direct not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            spt = _coord_to_1k_xy(action_inputs.get("start_box", ""), img_wh)
            ept = _coord_to_1k_xy(action_inputs.get("end_box", ""), img_wh)
            if spt and ept:
                dx = ept[0] - spt[0]
                dy = ept[1] - spt[1]
                if abs(dx) >= abs(dy):
                    direct = "RIGHT" if dx > 0 else "LEFT"
                else:
                    direct = "DOWN" if dy > 0 else "UP"
        if direct in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return f"SCROLL: {direct}"
    if action_type in ["type", "input_text"]:
        txt = action_inputs.get("content", action_inputs.get("text", ""))
        return f"TYPE: {txt}"
    if action_type in ["press_home", "navigate_home", "home"]:
        return "PRESS_HOME"
    if action_type in ["press_back", "navigate_back", "back"]:
        return "PRESS_BACK"
    if action_type in ["press_recent", "navigate_recent", "recent"]:
        return "PRESS_RECENT"
    if action_type in ["finished", "complete", "done", "finish"]:
        content = str(action_inputs.get("content", "")).lower()
        if any(w in content for w in ["unsuccessful", "infeasible", "impossible", "cannot", "can't", "unable", "fail", "error"]):
            return "IMPOSSIBLE"
        return "COMPLETE"
    if action_type in ["impossible", "incomplete", "fail"]:
        return "IMPOSSIBLE"
    return response


def _to_android_control_answer(response: str, img_wh: tuple[int, int] | None) -> str:
    action_type, action_inputs = _parse_uitars_first_action(response)
    if action_type is None:
        return response

    payload = {}
    if action_type in ["click", "tap", "long_press", "longpress"]:
        bbox = _coord_to_pixel_bbox(action_inputs.get("start_box", action_inputs.get("point", "")), img_wh)
        if bbox:
            payload["bbox_2d"] = bbox
        payload["action_type"] = "long_press" if "long" in action_type else "click"
    elif action_type in ["scroll", "swipe"]:
        direct = str(action_inputs.get("direction", action_inputs.get("dir", ""))).strip().lower()
        if direct not in ["up", "down", "left", "right"]:
            spt = _coord_to_pixel_bbox(action_inputs.get("start_box", ""), img_wh)
            ept = _coord_to_pixel_bbox(action_inputs.get("end_box", ""), img_wh)
            if spt and ept:
                dx = ept[2] - spt[0]
                dy = ept[3] - spt[1]
                if abs(dx) >= abs(dy):
                    direct = "right" if dx > 0 else "left"
                else:
                    direct = "down" if dy > 0 else "up"
            else:
                direct = "down"
        payload["action_type"] = f"swipe:{direct}"
    elif action_type in ["type", "input_text"]:
        txt = action_inputs.get("content", action_inputs.get("text", ""))
        payload["action_type"] = f"input_text:{txt}"
    elif action_type in ["press_home", "navigate_home", "home"]:
        payload["action_type"] = "navigate_home"
    elif action_type in ["press_back", "navigate_back", "back"]:
        payload["action_type"] = "navigate_back"
    elif action_type in ["wait"]:
        payload["action_type"] = "wait"
    elif action_type in ["finished", "complete", "done", "finish"]:
        payload["action_type"] = "wait"
    else:
        return response

    return f"<answer>{json.dumps(payload, ensure_ascii=False)}</answer>"


class UITars15Chat(Qwen2VLChat):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("force_qwen25_vl", True)
        super().__init__(*args, **kwargs)

    def generate_inner(self, message, dataset=None):
        response = super().generate_inner(message=message, dataset=dataset)
        if not isinstance(dataset, str):
            return response
        img_wh = _first_image_wh_from_message(message)
        if dataset.startswith("GUIOdyssey"):
            return _to_gui_odyssey_command(response, img_wh=img_wh)
        if dataset.startswith("AndroidControl"):
            return _to_android_control_answer(response, img_wh=img_wh)
        return response
