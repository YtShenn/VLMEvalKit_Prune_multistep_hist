import hashlib
import os
import os.path as osp
import re
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() == "1"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)).strip())
    except Exception:
        return float(default)


def state_packet_enabled() -> bool:
    return _env_flag("ANDROID_CONTROL_STATE_PACKET_ENABLE", "0")


def state_packet_debug_enabled() -> bool:
    return _env_flag("ANDROID_CONTROL_STATE_PACKET_DEBUG", "0")


def _packet_cache_dir() -> str:
    root = os.environ.get("ANDROID_CONTROL_STATE_PACKET_CACHE_DIR", "").strip()
    if root:
        return root
    return "/tmp/android_control_state_packet_cache"


def _patch_size() -> int:
    return max(1, _env_int("ANDROID_CONTROL_STATE_PACKET_PATCH_SIZE", 16))


def _merge_size() -> int:
    return max(1, _env_int("ANDROID_CONTROL_STATE_PACKET_MERGE_SIZE", 2))


def _thumb_long_edge() -> int:
    return max(32, _env_int("ANDROID_CONTROL_STATE_PACKET_THUMB_LONG_EDGE", 256))


def _roi_long_edge() -> int:
    return max(32, _env_int("ANDROID_CONTROL_STATE_PACKET_ROI_LONG_EDGE", 256))


def _roi_short_side_ratio() -> float:
    return max(0.05, _env_float("ANDROID_CONTROL_STATE_PACKET_ROI_SHORT_SIDE_RATIO", 0.28))


def _roi_min_side_px() -> int:
    return max(32, _env_int("ANDROID_CONTROL_STATE_PACKET_ROI_MIN_SIDE_PX", 224))


def _align_to_multiple(val: int, multiple: int) -> int:
    val = max(1, int(val))
    multiple = max(1, int(multiple))
    return max(multiple, int(round(val / multiple)) * multiple)


def _resize_with_long_edge(img: Image.Image, long_edge: int, align_base: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = float(long_edge) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_w = _align_to_multiple(new_w, align_base)
    new_h = _align_to_multiple(new_h, align_base)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _estimate_visual_tokens(width: int, height: int) -> int:
    patch = _patch_size()
    merge = _merge_size()
    if width <= 0 or height <= 0:
        return 0
    grid_w = max(1, width // patch)
    grid_h = max(1, height // patch)
    tokens = (grid_w * grid_h) // (merge * merge)
    return max(1, int(tokens))


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def _centered_box(w: int, h: int, side: int) -> tuple[int, int, int, int]:
    side = max(1, min(side, w, h))
    cx = w / 2.0
    cy = h / 2.0
    x1 = max(0, int(round(cx - side / 2.0)))
    y1 = max(0, int(round(cy - side / 2.0)))
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return x1, y1, x2, y2


def _box_from_point(point, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None
    try:
        px = float(point[0])
        py = float(point[1])
    except Exception:
        return None
    side = max(_roi_min_side_px(), int(round(min(w, h) * _roi_short_side_ratio())))
    x1 = int(round(px - side / 2.0))
    y1 = int(round(py - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side
    return _clip_box(x1, y1, x2, y2, w, h)


def _box_from_bbox(bbox, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except Exception:
        return None
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(_roi_min_side_px() * 0.15, min(bw, bh) * 0.2)
    side = max(_roi_min_side_px(), int(round(max(bw, bh) + 2.0 * pad)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return _clip_box(
        int(round(cx - side / 2.0)),
        int(round(cy - side / 2.0)),
        int(round(cx + side / 2.0)),
        int(round(cy + side / 2.0)),
        w,
        h,
    )


def _scroll_box(direction: str, w: int, h: int) -> tuple[int, int, int, int]:
    direction = str(direction or "").strip().lower()
    if "up" in direction or "down" in direction:
        box_w = max(_roi_min_side_px(), int(round(w * 0.7)))
        box_h = max(_roi_min_side_px(), int(round(h * 0.45)))
        x1 = max(0, int(round((w - box_w) / 2.0)))
        y1 = max(0, int(round(h * (0.1 if "up" in direction else 0.45))))
        return _clip_box(x1, y1, x1 + box_w, y1 + box_h, w, h)
    if "left" in direction or "right" in direction:
        box_w = max(_roi_min_side_px(), int(round(w * 0.45)))
        box_h = max(_roi_min_side_px(), int(round(h * 0.7)))
        y1 = max(0, int(round((h - box_h) / 2.0)))
        x1 = max(0, int(round(w * (0.1 if "left" in direction else 0.45))))
        return _clip_box(x1, y1, x1 + box_w, y1 + box_h, w, h)
    side = max(_roi_min_side_px(), int(round(min(w, h) * 0.45)))
    return _centered_box(w, h, side)


def _roi_box_for_action(action_packet: dict, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = image_size
    action_type = str(action_packet.get("gt_action", "") or "").strip().lower()
    gt_bbox = action_packet.get("gt_bbox", None)
    gt_point = action_packet.get("gt_coordinate", None)

    if action_type in {"click", "tap", "long_press"}:
        box = _box_from_bbox(gt_bbox, w, h)
        if box is not None:
            return box
        box = _box_from_point(gt_point, w, h)
        if box is not None:
            return box
    if "scroll" in action_type or "swipe" in action_type:
        return _scroll_box(action_type, w, h)
    side = max(_roi_min_side_px(), int(round(min(w, h) * 0.35)))
    return _centered_box(w, h, side)


def _save_packet_image(img: Image.Image, source_path: str, suffix: str) -> str:
    os.makedirs(_packet_cache_dir(), exist_ok=True)
    key = hashlib.md5(
        f"{source_path}|{suffix}|{img.size[0]}x{img.size[1]}".encode("utf-8")
    ).hexdigest()[:16]
    out_path = osp.join(_packet_cache_dir(), f"{key}_{suffix}.png")
    if not osp.exists(out_path):
        img.save(out_path, format="PNG")
    return out_path


@dataclass
class PacketImage:
    kind: str
    path: str
    width: int
    height: int
    estimated_tokens: int
    crop_xyxy: Optional[tuple[int, int, int, int]] = None

    def to_message_item(self) -> dict:
        return {
            "type": "image",
            "value": self.path,
            "resized_width": int(self.width),
            "resized_height": int(self.height),
        }


def build_state_packet(
    *,
    image_path: str,
    action_packet: dict,
    sample_index: str,
    history_index: int,
) -> tuple[list[PacketImage], dict]:
    t0 = time.perf_counter()
    img = Image.open(image_path).convert("RGB")
    open_s = time.perf_counter() - t0
    orig_w, orig_h = img.size
    orig_tokens = _estimate_visual_tokens(orig_w, orig_h)
    align_base = _patch_size() * _merge_size()

    t1 = time.perf_counter()
    thumb = _resize_with_long_edge(img, _thumb_long_edge(), align_base)
    thumb_path = _save_packet_image(thumb, image_path, f"s{sample_index}_h{history_index}_thumb")
    thumb_s = time.perf_counter() - t1
    thumb_img = PacketImage(
        kind="thumbnail",
        path=thumb_path,
        width=int(thumb.size[0]),
        height=int(thumb.size[1]),
        estimated_tokens=_estimate_visual_tokens(int(thumb.size[0]), int(thumb.size[1])),
    )

    t2 = time.perf_counter()
    crop_xyxy = _roi_box_for_action(action_packet, img.size)
    roi = img.crop(crop_xyxy)
    roi = _resize_with_long_edge(roi, _roi_long_edge(), align_base)
    roi_path = _save_packet_image(roi, image_path, f"s{sample_index}_h{history_index}_roi")
    roi_s = time.perf_counter() - t2
    roi_img = PacketImage(
        kind="action_roi",
        path=roi_path,
        width=int(roi.size[0]),
        height=int(roi.size[1]),
        estimated_tokens=_estimate_visual_tokens(int(roi.size[0]), int(roi.size[1])),
        crop_xyxy=tuple(int(v) for v in crop_xyxy),
    )

    total_s = time.perf_counter() - t0
    packet_tokens = int(thumb_img.estimated_tokens + roi_img.estimated_tokens)
    debug_enabled = state_packet_debug_enabled()
    meta = {
        "sample_index": str(sample_index),
        "history_index": int(history_index),
        "source_image_path": str(image_path),
        "action_text": str(action_packet.get("step_instruction", "") or ""),
        "action_type": str(action_packet.get("gt_action", "") or ""),
        "original_estimated_tokens": int(orig_tokens),
        "packet_estimated_tokens": int(packet_tokens),
        "thumbnail_estimated_tokens": int(thumb_img.estimated_tokens),
        "roi_estimated_tokens": int(roi_img.estimated_tokens),
        "open_image_s": float(open_s),
        "thumbnail_build_s": float(thumb_s),
        "roi_build_s": float(roi_s),
        "state_packet_total_s": float(total_s),
        "compression_ratio_vs_original": float(packet_tokens / max(1, orig_tokens)),
    }
    if debug_enabled:
        meta.update(
            {
                "original_width": int(orig_w),
                "original_height": int(orig_h),
                "thumbnail_width": int(thumb_img.width),
                "thumbnail_height": int(thumb_img.height),
                "roi_width": int(roi_img.width),
                "roi_height": int(roi_img.height),
                "roi_crop_xyxy": list(roi_img.crop_xyxy) if roi_img.crop_xyxy is not None else None,
                "gt_coordinate": action_packet.get("gt_coordinate", None),
                "gt_bbox": action_packet.get("gt_bbox", None),
            }
        )
        print(
            "[AndroidControlStatePacket] "
            f"sample_index={sample_index} hist_index={history_index} "
            f"action_type={meta['action_type']} "
            f"orig_size=({orig_w},{orig_h}) orig_tokens_est={orig_tokens} "
            f"thumb_size=({thumb_img.width},{thumb_img.height}) thumb_tokens_est={thumb_img.estimated_tokens} "
            f"roi_crop_xyxy={meta['roi_crop_xyxy']} "
            f"roi_size=({roi_img.width},{roi_img.height}) roi_tokens_est={roi_img.estimated_tokens} "
            f"packet_tokens_est={packet_tokens} "
            f"open_s={open_s:.6f} thumb_s={thumb_s:.6f} roi_s={roi_s:.6f} total_s={total_s:.6f}",
            flush=True,
        )
    return [thumb_img, roi_img], meta
