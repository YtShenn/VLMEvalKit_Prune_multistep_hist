import hashlib
import os
import os.path as osp
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageStat


@dataclass
class DeltaPacketConfig:
    # Dataset adapters only fill this config; the algorithm below is dataset-agnostic.
    cache_dir: str
    patch_size: int = 16
    merge_size: int = 2
    thumb_long_edge: int = 192
    roi_long_edge: int = 224
    max_delta_rois: int = 2
    min_roi_side_px: int = 160
    roi_pad_ratio: float = 0.18
    diff_threshold: float = 26.0
    large_change_ratio: float = 0.42
    small_change_ratio: float = 0.015
    min_component_area_ratio: float = 0.004
    nms_iou_threshold: float = 0.45
    max_roi_area_ratio: float = 0.45
    align_long_edge: int = 192
    align_max_shift_ratio: float = 0.35
    align_step_px: int = 8
    diff_mode: str = "illumination_invariant"
    local_contrast_radius: int = 9
    edge_weight: float = 0.45
    contrast_weight: float = 0.35
    rank_weight: float = 0.20
    action_focus_boost: float = 1.35
    action_outside_decay: float = 0.55
    text_top_band_ratio: float = 0.16
    text_bottom_band_ratio: float = 0.42
    tap_context_side_ratio: float = 0.42
    persistent_prior_on_small_change: bool = True
    persistent_prior_on_medium_change: bool = True
    full_on_large_change: bool = True
    full_on_large_roi: bool = True
    visualize: bool = False
    visualize_dir: Optional[str] = None


@dataclass
class DeltaPacketImage:
    # A packet image mirrors the history-state PacketImage interface so prompt
    # builders can swap full screenshots for compact packets with little glue.
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


def _align_to_multiple(val: int, multiple: int) -> int:
    val = max(1, int(val))
    multiple = max(1, int(multiple))
    return max(multiple, int(round(val / multiple)) * multiple)


def _resize_with_long_edge(img: Image.Image, long_edge: int, align_base: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = float(long_edge) / float(max(w, h))
    new_w = _align_to_multiple(max(1, int(round(w * scale))), align_base)
    new_h = _align_to_multiple(max(1, int(round(h * scale))), align_base)
    return img.resize((new_w, new_h), Image.BILINEAR)


def _estimate_visual_tokens(width: int, height: int, cfg: DeltaPacketConfig) -> int:
    if width <= 0 or height <= 0:
        return 0
    grid_w = max(1, int(width) // max(1, int(cfg.patch_size)))
    grid_h = max(1, int(height) // max(1, int(cfg.patch_size)))
    tokens = (grid_w * grid_h) // max(1, int(cfg.merge_size) * int(cfg.merge_size))
    return max(1, int(tokens))


def _save_packet_image(img: Image.Image, source_key: str, suffix: str, cfg: DeltaPacketConfig) -> str:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    key = hashlib.md5(
        f"{source_key}|{suffix}|{img.size[0]}x{img.size[1]}".encode("utf-8")
    ).hexdigest()[:16]
    out_path = osp.join(cfg.cache_dir, f"{key}_{suffix}.png")
    if not osp.exists(out_path):
        img.save(out_path, format="PNG")
    return out_path


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def _resize_box(box: tuple[int, int, int, int], src_size: tuple[int, int], dst_size: tuple[int, int]):
    sw, sh = src_size
    dw, dh = dst_size
    if min(sw, sh, dw, dh) <= 0:
        return None
    sx = float(dw) / float(sw)
    sy = float(dh) / float(sh)
    x1, y1, x2, y2 = box
    return _clip_box(x1 * sx, y1 * sy, x2 * sx, y2 * sy, dw, dh)


def _box_from_action_packet(action_packet: Optional[dict], image_size: tuple[int, int], cfg: DeltaPacketConfig):
    if not isinstance(action_packet, dict):
        return None
    w, h = image_size
    bbox = action_packet.get("gt_bbox", None)
    if bbox is None:
        bbox = action_packet.get("bbox_2d", None) or action_packet.get("bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad = max(float(cfg.min_roi_side_px) * 0.15, max(bw, bh) * float(cfg.roi_pad_ratio))
            return _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)
        except Exception:
            pass
    point = action_packet.get("gt_coordinate", None) or action_packet.get("point", None)
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            px, py = [float(v) for v in point[:2]]
            side = max(int(cfg.min_roi_side_px), int(round(min(w, h) * 0.22)))
            return _clip_box(px - side / 2.0, py - side / 2.0, px + side / 2.0, py + side / 2.0, w, h)
        except Exception:
            pass
    return None


def _action_direction(action_packet: Optional[dict]) -> str:
    if not isinstance(action_packet, dict):
        return ""
    text = str(action_packet.get("gt_action", "") or action_packet.get("action_type", "") or "").lower()
    for key in ("up", "down", "left", "right"):
        if key in text:
            return key
    return ""


def _action_kind(action_packet: Optional[dict]) -> str:
    if not isinstance(action_packet, dict):
        return "unknown"
    text = str(action_packet.get("gt_action", "") or action_packet.get("action_type", "") or "").lower()
    if any(x in text for x in ("navigate_back", "back", "navigate_home", "home", "open_app")):
        return "navigation"
    if any(x in text for x in ("input_text", "type", "text")):
        return "text_input"
    if "long" in text:
        return "long_press"
    if any(x in text for x in ("swipe", "scroll", "fling")):
        return "scroll"
    if any(x in text for x in ("click", "tap", "press")):
        return "tap"
    if "wait" in text:
        return "wait"
    return "unknown"


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _edge_array(img: Image.Image) -> np.ndarray:
    edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
    return np.asarray(edges, dtype=np.float32)


def _local_contrast_array(img: Image.Image, radius: int) -> np.ndarray:
    gray = img.convert("L")
    radius = max(1, int(radius))
    mean = gray.filter(ImageFilter.GaussianBlur(radius=radius))
    sq = Image.fromarray(np.uint8(np.clip(np.asarray(gray, dtype=np.float32) ** 2 / 255.0, 0, 255)))
    mean_sq = sq.filter(ImageFilter.GaussianBlur(radius=radius))
    gray_arr = np.asarray(gray, dtype=np.float32)
    mean_arr = np.asarray(mean, dtype=np.float32)
    mean_sq_arr = np.asarray(mean_sq, dtype=np.float32) * 255.0
    var = np.maximum(mean_sq_arr - mean_arr ** 2, 16.0)
    norm = (gray_arr - mean_arr) / np.sqrt(var)
    return np.clip((norm + 3.0) * (255.0 / 6.0), 0.0, 255.0).astype(np.float32)


def _rank_bits(img: Image.Image) -> np.ndarray:
    arr = _to_gray_array(img)
    padded = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    center = padded[1:-1, 1:-1]
    bits = np.zeros(arr.shape, dtype=np.uint8)
    offsets = [
        (-1, -1), (0, -1), (1, -1), (-1, 0),
        (1, 0), (-1, 1), (0, 1), (1, 1),
    ]
    for bit, (dx, dy) in enumerate(offsets):
        neigh = padded[1 + dy:1 + dy + arr.shape[0], 1 + dx:1 + dx + arr.shape[1]]
        bits |= ((neigh >= center).astype(np.uint8) << bit)
    return bits


def _rank_hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    xor = np.bitwise_xor(a, b)
    out = np.zeros(xor.shape, dtype=np.float32)
    for bit in range(8):
        out += ((xor >> bit) & 1).astype(np.float32)
    return out * (255.0 / 8.0)


def _action_focus_mask(size: tuple[int, int], action_packet: Optional[dict], cfg: DeltaPacketConfig) -> Optional[np.ndarray]:
    # Action priors are deliberately soft: they boost likely changed GUI zones
    # without preventing unexpected visual changes from being detected.
    kind = _action_kind(action_packet)
    w, h = size
    if kind == "unknown":
        return None
    mask = np.zeros((h, w), dtype=bool)
    if kind == "text_input":
        top_h = int(round(h * float(cfg.text_top_band_ratio)))
        bottom_y = int(round(h * (1.0 - float(cfg.text_bottom_band_ratio))))
        mask[:max(1, top_h), :] = True
        mask[max(0, bottom_y):, :] = True
    elif kind == "scroll":
        return None
    elif kind in ("tap", "long_press"):
        point_box = _box_from_action_packet(action_packet, size, cfg)
        if point_box is not None:
            x1, y1, x2, y2 = point_box
            side = int(round(max(w, h) * float(cfg.tap_context_side_ratio)))
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            x1, y1, x2, y2 = _clip_box(cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0, w, h)
            mask[y1:y2, x1:x2] = True
        bottom_y = int(round(h * 0.55))
        mask[max(0, bottom_y):, :] = True
    elif kind in ("navigation", "wait"):
        return None
    return mask


def _action_candidate_boxes(size: tuple[int, int], action_packet: Optional[dict], cfg: DeltaPacketConfig) -> list[tuple[int, int, int, int]]:
    # Deterministic action-conditioned proposals keep important GUI regions
    # even when structural residuals are too small or fragmented.
    kind = _action_kind(action_packet)
    w, h = size
    boxes: list[tuple[int, int, int, int]] = []
    if kind == "text_input":
        boxes.append(_clip_box(0, 0, w, h * float(cfg.text_top_band_ratio), w, h))
        boxes.append(_clip_box(0, h * (1.0 - float(cfg.text_bottom_band_ratio)), w, h, w, h))
    elif kind in ("tap", "long_press"):
        point_box = _box_from_action_packet(action_packet, size, cfg)
        if point_box is not None:
            x1, y1, x2, y2 = point_box
            side = int(round(max(w, h) * float(cfg.tap_context_side_ratio)))
            cx = int(round((x1 + x2) / 2.0))
            cy = int(round((y1 + y2) / 2.0))
            boxes.append(_clip_box(cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0, w, h))
        boxes.append(_clip_box(0, h * 0.55, w, h, w, h))
    elif kind == "scroll":
        direction = _action_direction(action_packet)
        band = int(round(h * 0.22))
        if direction == "up":
            boxes.append(_clip_box(0, h - band, w, h, w, h))
        elif direction == "down":
            boxes.append(_clip_box(0, 0, w, band, w, h))
        elif direction == "left":
            boxes.append(_clip_box(w - int(round(w * 0.22)), 0, w, h, w, h))
        elif direction == "right":
            boxes.append(_clip_box(0, 0, int(round(w * 0.22)), h, w, h))
    return boxes


def _prepare_align_pair(ref: Image.Image, cur: Image.Image, cfg: DeltaPacketConfig):
    long_edge = max(32, int(cfg.align_long_edge))
    ref_small = _resize_with_long_edge(ref, long_edge, 1)
    cur_small = _resize_with_long_edge(cur, long_edge, 1)
    if ref_small.size != cur_small.size:
        ref_small = ref_small.resize(cur_small.size, Image.BILINEAR)
    return _to_gray_array(ref_small), _to_gray_array(cur_small), cur_small.size


def _shift_score(ref_arr: np.ndarray, cur_arr: np.ndarray, dx: int, dy: int) -> float:
    h, w = cur_arr.shape
    x1_cur = max(0, dx)
    y1_cur = max(0, dy)
    x2_cur = min(w, w + dx)
    y2_cur = min(h, h + dy)
    x1_ref = max(0, -dx)
    y1_ref = max(0, -dy)
    x2_ref = x1_ref + max(0, x2_cur - x1_cur)
    y2_ref = y1_ref + max(0, y2_cur - y1_cur)
    if x2_cur <= x1_cur or y2_cur <= y1_cur:
        return float("inf")
    overlap = float((x2_cur - x1_cur) * (y2_cur - y1_cur)) / float(max(1, h * w))
    if overlap < 0.35:
        return float("inf")
    diff = np.abs(cur_arr[y1_cur:y2_cur, x1_cur:x2_cur] - ref_arr[y1_ref:y2_ref, x1_ref:x2_ref])
    return float(diff.mean() / max(0.05, overlap))


def _estimate_shift(ref: Image.Image, cur: Image.Image, action_packet: Optional[dict], cfg: DeltaPacketConfig):
    # Coarse shift-only alignment handles common GUI scroll/swipe transitions
    # before computing the delta map. Action direction constrains the search.
    action_kind = _action_kind(action_packet)
    if action_kind in ("text_input", "tap", "long_press", "navigation"):
        return 0, 0, 0.0
    ref_arr, cur_arr, small_size = _prepare_align_pair(ref, cur, cfg)
    sw, sh = small_size
    max_shift = max(1, int(round(max(sw, sh) * float(cfg.align_max_shift_ratio))))
    step = max(1, int(cfg.align_step_px))
    direction = _action_direction(action_packet)
    if direction in ("up", "down"):
        dx_values = [0]
        dy_values = list(range(-max_shift, max_shift + 1, step))
    elif direction in ("left", "right"):
        dx_values = list(range(-max_shift, max_shift + 1, step))
        dy_values = [0]
    else:
        dx_values = list(range(-max_shift, max_shift + 1, step))
        dy_values = list(range(-max_shift, max_shift + 1, step))

    best = (0, 0, float("inf"))
    for dy in dy_values:
        for dx in dx_values:
            score = _shift_score(ref_arr, cur_arr, dx, dy)
            if score < best[2]:
                best = (dx, dy, score)

    sx = float(cur.size[0]) / float(max(1, sw))
    sy = float(cur.size[1]) / float(max(1, sh))
    return int(round(best[0] * sx)), int(round(best[1] * sy)), float(best[2])


def _shift_image(ref: Image.Image, dx: int, dy: int, size: tuple[int, int]) -> Image.Image:
    if ref.size != size:
        ref = ref.resize(size, Image.BILINEAR)
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return ref.copy()
    # Use padded translation instead of wrap-around offset; GUI screenshots do
    # not have toroidal boundaries, and wrapping creates false full-screen deltas.
    out = Image.new("RGB", size, tuple(int(x) for x in ImageStat.Stat(ref).median))
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(size[0], size[0] - dx)
    src_y2 = min(size[1], size[1] - dy)
    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    if src_x2 > src_x1 and src_y2 > src_y1:
        out.paste(ref.crop((src_x1, src_y1, src_x2, src_y2)), (dst_x1, dst_y1))
    return out


def _exposed_mask(size: tuple[int, int], dx: int, dy: int) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    if dx > 0:
        mask[:, :dx] = True
    elif dx < 0:
        mask[:, w + dx:] = True
    if dy > 0:
        mask[:dy, :] = True
    elif dy < 0:
        mask[h + dy:, :] = True
    return mask


def _build_saliency(
    ref_aligned: Image.Image,
    cur: Image.Image,
    dx: int,
    dy: int,
    action_packet: Optional[dict],
    cfg: DeltaPacketConfig,
):
    # Illumination-invariant mode favors structure over raw color, making
    # dimming overlays and keyboard theme redraws less likely to dominate.
    if str(cfg.diff_mode).lower() in ("rgb", "pixel", "legacy"):
        rgb_diff = np.asarray(ImageChops.difference(cur, ref_aligned).convert("L"), dtype=np.float32)
        edge_diff = np.abs(_edge_array(cur) - _edge_array(ref_aligned))
        sal = 0.7 * rgb_diff + 0.3 * edge_diff
    else:
        contrast_diff = np.abs(
            _local_contrast_array(cur, cfg.local_contrast_radius)
            - _local_contrast_array(ref_aligned, cfg.local_contrast_radius)
        )
        edge_diff = np.abs(_edge_array(cur) - _edge_array(ref_aligned))
        rank_diff = _rank_hamming(_rank_bits(cur), _rank_bits(ref_aligned))
        total_w = max(1e-6, float(cfg.contrast_weight) + float(cfg.edge_weight) + float(cfg.rank_weight))
        sal = (
            float(cfg.contrast_weight) * contrast_diff
            + float(cfg.edge_weight) * edge_diff
            + float(cfg.rank_weight) * rank_diff
        ) / total_w
    focus_mask = _action_focus_mask(cur.size, action_packet, cfg)
    if focus_mask is not None:
        sal = np.where(
            focus_mask,
            sal * float(cfg.action_focus_boost),
            sal * float(cfg.action_outside_decay),
        )
    sal[_exposed_mask(cur.size, dx, dy)] = np.maximum(sal[_exposed_mask(cur.size, dx, dy)], 255.0)
    changed = sal >= float(cfg.diff_threshold)
    ratio = float(changed.mean()) if changed.size else 0.0
    return sal, changed, ratio


def _component_boxes(mask: np.ndarray, cfg: DeltaPacketConfig) -> list[tuple[int, int, int, int]]:
    # Connected components turn a dense change mask into candidate GUI regions.
    h, w = mask.shape
    min_area = max(8, int(round(h * w * float(cfg.min_component_area_ratio))))
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for yy in range(h):
        xs = np.flatnonzero(mask[yy] & ~seen[yy])
        for xx in xs:
            if seen[yy, xx] or not mask[yy, xx]:
                continue
            stack = [(int(xx), int(yy))]
            seen[yy, xx] = True
            min_x = max_x = int(xx)
            min_y = max_y = int(yy)
            area = 0
            while stack:
                x, y = stack.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if seen[ny, nx] or not mask[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((nx, ny))
            if area >= min_area:
                boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


def _pad_box(box: tuple[int, int, int, int], size: tuple[int, int], cfg: DeltaPacketConfig):
    w, h = size
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad = max(8, int(round(max(bw, bh) * float(cfg.roi_pad_ratio))))
    if bw < cfg.min_roi_side_px:
        extra = int(round((cfg.min_roi_side_px - bw) / 2.0))
        x1 -= extra
        x2 += extra
    if bh < cfg.min_roi_side_px:
        extra = int(round((cfg.min_roi_side_px - bh) / 2.0))
        y1 -= extra
        y2 += extra
    return _clip_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)


def _rank_delta_boxes(saliency: np.ndarray, boxes: list[tuple[int, int, int, int]], cfg: DeltaPacketConfig):
    # Rank compact high-saliency regions, then suppress heavily overlapping boxes.
    scored = []
    h, w = saliency.shape
    for box in boxes:
        x1, y1, x2, y2 = _pad_box(box, (w, h), cfg)
        area = max(1, (x2 - x1) * (y2 - y1))
        score = float(saliency[y1:y2, x1:x2].sum()) / float(area ** 0.35)
        scored.append((score, (x1, y1, x2, y2)))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    for _, box in scored:
        if all(_iou(box, kept) < float(cfg.nms_iou_threshold) for kept in selected):
            selected.append(box)
        if len(selected) >= max(0, int(cfg.max_delta_rois)):
            break
    return selected


def _box_area_ratio(box: tuple[int, int, int, int], size: tuple[int, int]) -> float:
    w, h = size
    x1, y1, x2, y2 = box
    return float(max(0, x2 - x1) * max(0, y2 - y1)) / float(max(1, w * h))


def _make_packet_image(
    *,
    img: Image.Image,
    source_key: str,
    suffix: str,
    kind: str,
    crop_xyxy: Optional[tuple[int, int, int, int]],
    cfg: DeltaPacketConfig,
    long_edge: int,
) -> DeltaPacketImage:
    align_base = max(1, int(cfg.patch_size) * int(cfg.merge_size))
    resized = _resize_with_long_edge(img, int(long_edge), align_base)
    path = _save_packet_image(resized, source_key, suffix, cfg)
    return DeltaPacketImage(
        kind=kind,
        path=path,
        width=int(resized.size[0]),
        height=int(resized.size[1]),
        estimated_tokens=_estimate_visual_tokens(int(resized.size[0]), int(resized.size[1]), cfg),
        crop_xyxy=crop_xyxy,
    )


def _safe_name(text: str, max_len: int = 120) -> str:
    cleaned = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    name = "".join(cleaned).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return (name or "image")[:max_len]


def _visualization_path(current_image_path: str, source_key: str, sample_index: str, cfg: DeltaPacketConfig) -> str:
    root = cfg.visualize_dir or osp.join(cfg.cache_dir, "visualizations")
    os.makedirs(root, exist_ok=True)
    normalized = str(current_image_path or "").replace("\\", "/").rstrip("/")
    parent = osp.basename(osp.dirname(normalized))
    grandparent = osp.basename(osp.dirname(osp.dirname(normalized)))
    stem = osp.splitext(osp.basename(normalized))[0]
    readable = _safe_name("_".join(x for x in (grandparent, parent, stem) if x))
    key = hashlib.md5(f"{source_key}|visualize".encode("utf-8")).hexdigest()[:8]
    return osp.join(root, f"{readable}_s{sample_index}_{key}_current_delta_vis.png")


def _packet_visualization_path(current_image_path: str, source_key: str, sample_index: str, cfg: DeltaPacketConfig) -> str:
    return _visualization_path(current_image_path, source_key, sample_index, cfg)


def _draw_labeled_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    for offset in range(3):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
    text_pos = (max(0, x1), max(0, y1 - 16))
    try:
        bbox = draw.textbbox(text_pos, label)
        draw.rectangle(bbox, fill=color)
        draw.text(text_pos, label, fill=(255, 255, 255))
    except Exception:
        draw.text(text_pos, label, fill=color)


def _wrap_text(text: str, width: int) -> list[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []
    width = max(16, int(width))
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        nxt = word if not cur else f"{cur} {word}"
        if len(nxt) <= width:
            cur = nxt
            continue
        if cur:
            lines.append(cur)
        cur = word
    if cur:
        lines.append(cur)
    return lines


def _draw_text_panel(draw: ImageDraw.ImageDraw, lines: list[str], image_size: tuple[int, int]) -> None:
    if not lines:
        return
    w, h = image_size
    lines = lines[:4]
    text = "\n".join(lines)
    font = None
    line_height = 34
    margin_x = 10
    margin_y = 12
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        line_height = 40
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 28)
            line_height = 38
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
    pos = (margin_x, max(0, h - line_height * len(lines) - margin_y))
    try:
        bbox = draw.multiline_textbbox(pos, text, spacing=4, font=font)
        panel = (
            max(0, bbox[0] - margin_x),
            max(0, bbox[1] - 8),
            min(w, bbox[2] + margin_x),
            min(h, bbox[3] + 8),
        )
        draw.rectangle(panel, fill=(0, 0, 0))
        draw.multiline_text(pos, text, fill=(255, 230, 64), spacing=6, font=font)
    except Exception:
        draw.text(pos, text, fill=(255, 230, 64), font=font)


def _gt_box_from_packet(gt_packet: Optional[dict], image_size: tuple[int, int]):
    if not isinstance(gt_packet, dict):
        return None
    bbox = gt_packet.get("gt_bbox", None)
    if bbox is None:
        bbox = gt_packet.get("bbox_2d", None) or gt_packet.get("bbox", None)
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            w, h = image_size
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            return _clip_box(x1, y1, x2, y2, w, h)
        except Exception:
            return None
    return None


def _save_delta_visualization(
    *,
    cur: Image.Image,
    packet_images: list[DeltaPacketImage],
    current_image_path: str,
    source_key: str,
    sample_index: str,
    route: str,
    change_ratio: Optional[float],
    dx: int,
    dy: int,
    cfg: DeltaPacketConfig,
    current_gt_packet: Optional[dict] = None,
) -> Optional[str]:
    if not bool(cfg.visualize):
        return None
    vis = cur.copy()
    draw = ImageDraw.Draw(vis)
    colors = {
        "current_delta_roi": (255, 64, 64),
        "current_persistent_prior_roi": (56, 128, 255),
        "current_full": (40, 180, 90),
    }
    for idx, item in enumerate(packet_images, 1):
        if item.kind == "current_full":
            box = (0, 0, vis.size[0] - 1, vis.size[1] - 1)
            _draw_labeled_box(draw, box, "full fallback", colors["current_full"])
            continue
        if item.crop_xyxy is None:
            continue
        if item.kind.startswith("current_delta_roi"):
            label = item.kind.replace("current_", "")
            color = colors["current_delta_roi"]
        elif item.kind == "current_persistent_prior_roi":
            label = "persistent_prior"
            color = colors["current_persistent_prior_roi"]
        else:
            label = item.kind
            color = (255, 180, 32)
        _draw_labeled_box(draw, item.crop_xyxy, label, color)
    gt_box = _gt_box_from_packet(current_gt_packet, vis.size)
    if gt_box is not None:
        _draw_labeled_box(draw, gt_box, "GT", (255, 230, 64))
    header = f"route={route} change={change_ratio} shift=({dx},{dy}) images={len(packet_images)}"
    try:
        bbox = draw.textbbox((4, 4), header)
        draw.rectangle(bbox, fill=(0, 0, 0))
    except Exception:
        pass
    draw.text((4, 4), header, fill=(255, 255, 255))
    if isinstance(current_gt_packet, dict):
        gt_action = str(current_gt_packet.get("gt_action", "") or current_gt_packet.get("action_type", "") or "")
        instruction = str(
            current_gt_packet.get("step_instruction", "")
            or current_gt_packet.get("instruction", "")
            or ""
        )
        panel_lines = []
        if gt_action:
            panel_lines.extend(_wrap_text(f"GT action: {gt_action}", 72))
        if instruction:
            panel_lines.extend(_wrap_text(f"Instruction: {instruction}", 72))
        _draw_text_panel(draw, panel_lines, vis.size)
    out_path = _packet_visualization_path(current_image_path, source_key, sample_index, cfg)
    vis.save(out_path, format="PNG")
    return out_path


def build_current_delta_packet(
    *,
    current_image_path: str,
    reference_image_path: Optional[str],
    previous_action_packet: Optional[dict],
    sample_index: str,
    cfg: DeltaPacketConfig,
    current_gt_packet: Optional[dict] = None,
) -> tuple[list[DeltaPacketImage], dict]:
    # Main zero-training current-frame routing: full screenshot for no/invalid
    # references or large changes; otherwise thumbnail + delta ROIs + optional
    # persistent prior from the previous interaction region.
    t0 = time.perf_counter()
    cur = Image.open(current_image_path).convert("RGB")
    open_current_s = time.perf_counter() - t0
    cur_w, cur_h = cur.size
    orig_tokens = _estimate_visual_tokens(cur_w, cur_h, cfg)

    if not reference_image_path or not osp.exists(str(reference_image_path)):
        source_key = f"{current_image_path}|ref=none"
        images = [
            DeltaPacketImage(
                kind="current_full",
                path=current_image_path,
                width=cur_w,
                height=cur_h,
                estimated_tokens=orig_tokens,
                crop_xyxy=None,
            )
        ]
        vis_path = _save_delta_visualization(
            cur=cur,
            packet_images=images,
            current_image_path=current_image_path,
            source_key=source_key,
            sample_index=sample_index,
            route="full_no_reference",
            change_ratio=None,
            dx=0,
            dy=0,
            cfg=cfg,
            current_gt_packet=current_gt_packet,
        )
        gt_box = _gt_box_from_packet(current_gt_packet, cur.size)
        meta = {
            "sample_index": str(sample_index),
            "source_image_path": str(current_image_path),
            "reference_image_path": str(reference_image_path or ""),
            "route": "full_no_reference",
            "route_reason": "missing_reference",
            "original_estimated_tokens": int(orig_tokens),
            "packet_estimated_tokens": int(orig_tokens),
            "delta_change_ratio": None,
            "delta_component_count": 0,
            "current_delta_image_count": int(len(images)),
            "current_delta_roi_count": 0,
            "alignment_shift_dx": 0,
            "alignment_shift_dy": 0,
            "original_width": int(cur_w),
            "original_height": int(cur_h),
            "large_change_ratio_threshold": float(cfg.large_change_ratio),
            "small_change_ratio_threshold": float(cfg.small_change_ratio),
            "diff_threshold": float(cfg.diff_threshold),
            "max_delta_rois": int(cfg.max_delta_rois),
            "gt_action": str(current_gt_packet.get("gt_action", "")) if isinstance(current_gt_packet, dict) else "",
            "gt_bbox_crop_xyxy": list(gt_box) if gt_box is not None else None,
            "gt_instruction": str(current_gt_packet.get("step_instruction", "") or current_gt_packet.get("instruction", "")) if isinstance(current_gt_packet, dict) else "",
            "visualization_path": vis_path,
            "current_delta_total_s": float(time.perf_counter() - t0),
            "open_current_image_s": float(open_current_s),
        }
        return images, meta

    ref = Image.open(reference_image_path).convert("RGB")
    ref_orig_size = ref.size
    if ref.size != cur.size:
        ref = ref.resize(cur.size, Image.BILINEAR)

    action_kind = _action_kind(previous_action_packet)
    align_t = time.perf_counter()
    dx, dy, align_score = _estimate_shift(ref, cur, previous_action_packet, cfg)
    ref_aligned = _shift_image(ref, dx, dy, cur.size)
    align_s = time.perf_counter() - align_t

    diff_t = time.perf_counter()
    saliency, changed_mask, change_ratio = _build_saliency(ref_aligned, cur, dx, dy, previous_action_packet, cfg)
    candidate_boxes = _action_candidate_boxes(cur.size, previous_action_packet, cfg)
    full_due_to_action = action_kind == "navigation"
    full_due_to_change = bool(cfg.full_on_large_change) and change_ratio >= float(cfg.large_change_ratio)
    if full_due_to_change:
        components = []
        delta_boxes = []
    elif full_due_to_action:
        components = []
        delta_boxes = []
    else:
        components = _component_boxes(changed_mask, cfg)
        delta_boxes = _rank_delta_boxes(saliency, components + candidate_boxes, cfg)
    large_roi_boxes = [
        box for box in delta_boxes
        if _box_area_ratio(box, cur.size) >= float(cfg.max_roi_area_ratio)
    ]
    full_due_to_large_roi = bool(cfg.full_on_large_roi) and bool(large_roi_boxes)
    skip_delta_rois = full_due_to_change or full_due_to_large_roi or full_due_to_action
    if full_due_to_large_roi:
        delta_boxes = []
    diff_s = time.perf_counter() - diff_t

    source_key = f"{current_image_path}|ref={reference_image_path}|shift={dx},{dy}"
    route = "delta"
    images: list[DeltaPacketImage] = []
    packet_tokens = 0

    if skip_delta_rois:
        if full_due_to_action:
            route = "full_action_navigation"
        else:
            route = "full_large_roi" if full_due_to_large_roi else "full_large_change"
        images.append(
            DeltaPacketImage(
                kind="current_full",
                path=current_image_path,
                width=cur_w,
                height=cur_h,
                estimated_tokens=orig_tokens,
                crop_xyxy=None,
            )
        )
        packet_tokens = int(orig_tokens)
    else:
        thumb_t = time.perf_counter()
        thumb_img = _make_packet_image(
            img=cur,
            source_key=source_key,
            suffix=f"s{sample_index}_current_thumb",
            kind="current_thumbnail",
            crop_xyxy=None,
            cfg=cfg,
            long_edge=cfg.thumb_long_edge,
        )
        images.append(thumb_img)
        packet_tokens += int(thumb_img.estimated_tokens)
        thumb_s = time.perf_counter() - thumb_t

        roi_s_total = 0.0
        for idx, box in enumerate(delta_boxes):
            roi_t = time.perf_counter()
            roi = cur.crop(box)
            roi_img = _make_packet_image(
                img=roi,
                source_key=source_key,
                suffix=f"s{sample_index}_current_delta{idx + 1}",
                kind=f"current_delta_roi_{idx + 1}",
                crop_xyxy=tuple(int(v) for v in box),
                cfg=cfg,
                long_edge=cfg.roi_long_edge,
            )
            images.append(roi_img)
            packet_tokens += int(roi_img.estimated_tokens)
            roi_s_total += time.perf_counter() - roi_t

        prior_box = None
        should_add_prior = (
            bool(cfg.persistent_prior_on_medium_change)
            or (change_ratio <= float(cfg.small_change_ratio) and bool(cfg.persistent_prior_on_small_change))
        )
        if should_add_prior:
            raw_prior = _box_from_action_packet(previous_action_packet, ref_orig_size, cfg)
            if raw_prior is not None:
                prior_box = _resize_box(raw_prior, ref_orig_size, cur.size)
                if prior_box is not None:
                    prior_box = _clip_box(
                        prior_box[0] + dx,
                        prior_box[1] + dy,
                        prior_box[2] + dx,
                        prior_box[3] + dy,
                        cur_w,
                        cur_h,
                    )
                    if all(_iou(prior_box, box) < float(cfg.nms_iou_threshold) for box in delta_boxes):
                        roi_t = time.perf_counter()
                        prior = cur.crop(prior_box)
                        prior_img = _make_packet_image(
                            img=prior,
                            source_key=source_key,
                            suffix=f"s{sample_index}_current_prior",
                            kind="current_persistent_prior_roi",
                            crop_xyxy=tuple(int(v) for v in prior_box),
                            cfg=cfg,
                            long_edge=cfg.roi_long_edge,
                        )
                        images.append(prior_img)
                        packet_tokens += int(prior_img.estimated_tokens)
                        roi_s_total += time.perf_counter() - roi_t
                        if change_ratio <= float(cfg.small_change_ratio):
                            route = "delta_small_with_prior"
                        else:
                            route = "delta_with_prior"
        else:
            prior_box = None
        if route == "delta" and change_ratio <= float(cfg.small_change_ratio):
            route = "delta_small"
    total_s = time.perf_counter() - t0
    if route == "full_action_navigation":
        route_reason = "previous_action_implies_scene_navigation"
    elif route == "full_large_change":
        route_reason = "change_ratio_above_large_threshold"
    elif route == "full_large_roi":
        route_reason = "delta_roi_area_above_threshold"
    elif route == "delta_with_prior":
        route_reason = "medium_change_delta_packet_with_prior"
    elif route == "delta_small_with_prior":
        route_reason = "change_ratio_below_small_threshold_with_prior"
    elif route == "delta_small":
        route_reason = "change_ratio_below_small_threshold_without_prior"
    else:
        route_reason = "medium_change_delta_packet"
    vis_path = _save_delta_visualization(
        cur=cur,
        packet_images=images,
        current_image_path=current_image_path,
        source_key=source_key,
        sample_index=sample_index,
        route=route,
        change_ratio=change_ratio,
        dx=dx,
        dy=dy,
        cfg=cfg,
        current_gt_packet=current_gt_packet,
    )
    gt_box = _gt_box_from_packet(current_gt_packet, cur.size)

    meta = {
        "sample_index": str(sample_index),
        "source_image_path": str(current_image_path),
        "reference_image_path": str(reference_image_path),
        "route": str(route),
        "route_reason": route_reason,
        "original_estimated_tokens": int(orig_tokens),
        "packet_estimated_tokens": int(packet_tokens),
        "thumbnail_estimated_tokens": int(sum(x.estimated_tokens for x in images if x.kind == "current_thumbnail")),
        "delta_roi_estimated_tokens": int(sum(x.estimated_tokens for x in images if x.kind.startswith("current_delta_roi"))),
        "persistent_prior_estimated_tokens": int(sum(x.estimated_tokens for x in images if x.kind == "current_persistent_prior_roi")),
        "current_delta_image_count": int(len(images)),
        "current_delta_roi_count": int(sum(1 for x in images if x.kind.startswith("current_delta_roi"))),
        "delta_change_ratio": float(change_ratio),
        "delta_component_count": int(len(components)),
        "raw_delta_component_count": int(len(components)),
        "alignment_shift_dx": int(dx),
        "alignment_shift_dy": int(dy),
        "alignment_score": float(align_score),
        "action_kind": str(action_kind),
        "diff_mode": str(cfg.diff_mode),
        "action_candidate_crop_xyxy": [list(box) for box in candidate_boxes],
        "action_candidate_count": int(len(candidate_boxes)),
        "open_current_image_s": float(open_current_s),
        "alignment_s": float(align_s),
        "delta_diff_s": float(diff_s),
        "current_delta_total_s": float(total_s),
        "compression_ratio_vs_original": float(packet_tokens / max(1, orig_tokens)),
        "original_width": int(cur_w),
        "original_height": int(cur_h),
        "large_change_ratio_threshold": float(cfg.large_change_ratio),
        "small_change_ratio_threshold": float(cfg.small_change_ratio),
        "diff_threshold": float(cfg.diff_threshold),
        "max_delta_rois": int(cfg.max_delta_rois),
        "max_roi_area_ratio_threshold": float(cfg.max_roi_area_ratio),
        "large_roi_crop_xyxy": [list(box) for box in large_roi_boxes],
        "large_roi_area_ratios": [float(_box_area_ratio(box, cur.size)) for box in large_roi_boxes],
        "persistent_prior_on_medium_change": bool(cfg.persistent_prior_on_medium_change),
        "gt_action": str(current_gt_packet.get("gt_action", "")) if isinstance(current_gt_packet, dict) else "",
        "gt_bbox_crop_xyxy": list(gt_box) if gt_box is not None else None,
        "gt_instruction": str(current_gt_packet.get("step_instruction", "") or current_gt_packet.get("instruction", "")) if isinstance(current_gt_packet, dict) else "",
        "visualization_path": vis_path,
        "delta_roi_crop_xyxy": [list(x.crop_xyxy) for x in images if x.kind.startswith("current_delta_roi")],
        "persistent_prior_crop_xyxy": [
            list(x.crop_xyxy) for x in images if x.kind == "current_persistent_prior_roi" and x.crop_xyxy is not None
        ],
    }
    if route.startswith("delta"):
        meta["thumbnail_build_s"] = float(locals().get("thumb_s", 0.0))
        meta["roi_build_s"] = float(locals().get("roi_s_total", 0.0))
    else:
        meta["thumbnail_build_s"] = 0.0
        meta["roi_build_s"] = 0.0
    return images, meta
