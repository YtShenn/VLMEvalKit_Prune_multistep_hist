import argparse
import ast
import json
import os
import random
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample N multi-step tasks from GUIOdyssey / AndroidControl datasets, "
            "render GT annotations on each step screenshot, and save one folder per task."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Dataset names, e.g. GUIOdyssey_high_task_split "
            "AndroidControl_Curated_High_Task_Improved"
        ),
    )
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks to sample per dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for task sampling.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="OUTPUT/task_visualizations",
        help="Root output directory.",
    )
    return parser.parse_args()


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


FONT_TITLE = load_font(24)
FONT_TEXT = load_font(18)
FONT_SMALL = load_font(16)


ANDROID_BENCHMARK_MAP = {
    "AndroidControl_Curated_Low_Point": "android_control_low_point.json",
    "AndroidControl_Curated_High_Point": "android_control_high_point.json",
    "AndroidControl_Curated_Low_BBox": "android_control_low_bbox.json",
    "AndroidControl_Curated_High_BBox": "android_control_high_bbox.json",
    "AndroidControl_Curated_High_Task_Improved": "android_control_high_task-improved.json",
}


def safe_literal_eval(value):
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return value


def sanitize_name(text):
    text = str(text).strip()
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "task"


def wrap_text(draw, text, font, max_width):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def measure_multiline_height(draw, lines, font, spacing):
    if not lines:
        return 0
    total = 0
    for idx, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        total += bbox[3] - bbox[1]
        if idx != len(lines) - 1:
            total += spacing
    return total


def draw_multiline(draw, xy, lines, font, fill, spacing):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + spacing


def parse_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def step_index_from_path(image_path):
    match = re.search(r"(?:^|[/\\\\])(?:step_|.*_)(\d+)(?:\.[^.]+)?$", str(image_path))
    if match:
        return parse_int(match.group(1), default=10**9)
    return 10**9


def gui_task_key(row):
    image_path = str(row.get("image_path", row.get("image", "")))
    stem = Path(image_path).stem
    return re.sub(r"_\d+$", "", stem)


def android_task_key(row):
    image_path = str(row.get("image_path", row.get("image", ""))).replace("\\", "/").rstrip("/")
    if "/" in image_path:
        return image_path.rsplit("/", 1)[0]
    return image_path


def ensure_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def scale_gui_coord(coord, image_size):
    width, height = image_size
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        x = float(coord[0])
        y = float(coord[1])
    except Exception:
        return None
    if max(abs(x), abs(y)) <= 1.5:
        return [x * width, y * height]
    if max(abs(x), abs(y)) <= 1000.0:
        return [x / 1000.0 * width, y / 1000.0 * height]
    return [x, y]


def scale_gui_bbox(bbox, image_size):
    width, height = image_size
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    max_abs = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_abs <= 1.5:
        return [x1 * width, y1 * height, x2 * width, y2 * height]
    if max_abs <= 1000.0:
        return [x1 / 1000.0 * width, y1 / 1000.0 * height, x2 / 1000.0 * width, y2 / 1000.0 * height]
    return [x1, y1, x2, y2]


def draw_point(draw, point, color, radius=10, width=4):
    if point is None:
        return
    x, y = point
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=width)
    draw.line([x - radius - 4, y, x + radius + 4, y], fill=color, width=width)
    draw.line([x, y - radius - 4, x, y + radius + 4], fill=color, width=width)


def draw_bbox(draw, bbox, color, width=4):
    if bbox is None:
        return
    draw.rectangle(bbox, outline=color, width=width)


def draw_badge(draw, text, image_size):
    width, height = image_size
    label = str(text).strip()
    if not label:
        return
    padding_x = 12
    padding_y = 8
    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    box_w = bbox[2] - bbox[0] + padding_x * 2
    box_h = bbox[3] - bbox[1] + padding_y * 2
    x1 = 16
    y1 = height - box_h - 16
    x2 = x1 + box_w
    y2 = y1 + box_h
    draw.rounded_rectangle([x1, y1, x2, y2], radius=10, fill=(0, 0, 0, 180))
    draw.text((x1 + padding_x, y1 + padding_y), label, font=FONT_SMALL, fill=(255, 255, 255))


def build_canvas(image, task_instruction, step_instruction, gt_summary):
    image = ensure_rgb(image)
    width, height = image.size
    dummy = Image.new("RGB", (width, 200), "white")
    dummy_draw = ImageDraw.Draw(dummy)

    sections = [
        ("Task", str(task_instruction or "N/A"), FONT_TITLE),
        ("Step", str(step_instruction or "N/A"), FONT_TEXT),
        ("GT", str(gt_summary or "N/A"), FONT_TEXT),
    ]

    wrapped_sections = []
    text_width = width - 40
    total_height = 24
    for title, body, font in sections:
        title_lines = [f"{title}:"] if title else []
        body_lines = wrap_text(dummy_draw, body, font, text_width)
        wrapped_sections.append((title_lines, body_lines, font))
        total_height += measure_multiline_height(dummy_draw, title_lines, FONT_TEXT, 4)
        total_height += measure_multiline_height(dummy_draw, body_lines, font, 6)
        total_height += 16

    header_height = max(120, total_height)
    canvas = Image.new("RGB", (width, header_height + height), "white")
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle([0, 0, width, header_height], fill=(248, 249, 252, 255))
    draw.line([0, header_height - 1, width, header_height - 1], fill=(210, 214, 220, 255), width=2)

    y = 16
    for title_lines, body_lines, font in wrapped_sections:
        if title_lines:
            draw_multiline(draw, (20, y), title_lines, FONT_TEXT, (40, 44, 52), 4)
            y += measure_multiline_height(draw, title_lines, FONT_TEXT, 4) + 4
        draw_multiline(draw, (20, y), body_lines, font, (15, 23, 42), 6)
        y += measure_multiline_height(draw, body_lines, font, 6) + 14
    return canvas, header_height


def simple_decode(text):
    raw = str(text).strip()
    match = re.search(
        r"(CLICK|LONG_PRESS|SCROLL|TYPE|PRESS_HOME|PRESS_BACK|PRESS_RECENT|COMPLETE|IMPOSSIBLE)\s*:\s*(.+)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        action = match.group(1).upper()
        info = match.group(2).strip()
        if action in {"CLICK", "LONG_PRESS"}:
            info = safe_literal_eval(info)
        return {"action": action, "info": info}

    match = re.search(r"\b(COMPLETE|IMPOSSIBLE|PRESS_HOME|PRESS_BACK|PRESS_RECENT)\b", raw, re.IGNORECASE)
    if match:
        return {"action": match.group(1).upper(), "info": ""}
    raise ValueError(f"Cannot parse action from: {raw}")


def resolve_android_src_root():
    return os.environ.get("ANDROID_CONTROL_CURATED_ROOT", "/mnt/storage2/users/ytshen_data/AndroidControl_Curated").strip()


def resolve_android_image_root():
    return os.environ.get("ANDROID_CONTROL_CURATED_IMAGE_ROOT", "/mnt/storage2/users/ytshen_data/AndroidControl_Curated/images").strip()


def resolve_android_json_path(dataset_name):
    if dataset_name not in ANDROID_BENCHMARK_MAP:
        raise ValueError(f"Unsupported AndroidControl dataset: {dataset_name}")
    return os.path.join(resolve_android_src_root(), "benchmark_resource", ANDROID_BENCHMARK_MAP[dataset_name])


def resolve_android_image_path(image_value):
    image_value = str(image_value)
    if os.path.isabs(image_value) and os.path.exists(image_value):
        return image_value
    candidates = [
        os.path.join(resolve_android_image_root(), image_value),
        os.path.join(resolve_android_src_root(), image_value),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def resolve_gui_root():
    return os.environ.get("GUI_ODYSSEY_ROOT", "/mnt/storage2/users/ytshen_data/GUIOdyssey").strip()


def resolve_gui_json_path(dataset_name):
    short_name = str(dataset_name).replace("GUIOdyssey_", "")
    return os.path.join(resolve_gui_root(), "test_anno", f"{short_name}.json")


def resolve_gui_image_path(image_value):
    image_value = str(image_value)
    if os.path.exists(image_value):
        return image_value
    candidate = os.path.join(resolve_gui_root(), "screenshots", os.path.basename(image_value))
    if os.path.exists(candidate):
        return candidate
    return image_value


def render_android_step(row):
    image_path = str(row["image_path"])
    image = Image.open(image_path)
    task_instruction = row.get("instruction", "")
    step_instruction = row.get("step_instruction", "")
    gt_action = str(row.get("gt_action", "")).strip()
    gt_bbox = row.get("gt_max_bbox", None) or row.get("gt_min_bbox", None)
    gt_point = row.get("gt_coordinate", None)

    canvas, header_height = build_canvas(
        image=image,
        task_instruction=task_instruction,
        step_instruction=step_instruction,
        gt_summary=f"action={gt_action}, bbox={gt_bbox}, point={gt_point}",
    )
    draw = ImageDraw.Draw(canvas, "RGBA")

    if isinstance(gt_bbox, str):
        gt_bbox = safe_literal_eval(gt_bbox)
    if isinstance(gt_point, str):
        gt_point = safe_literal_eval(gt_point)

    if isinstance(gt_bbox, (list, tuple)) and len(gt_bbox) == 4:
        shifted_bbox = [float(gt_bbox[0]), float(gt_bbox[1]) + header_height, float(gt_bbox[2]), float(gt_bbox[3]) + header_height]
        draw_bbox(draw, shifted_bbox, color=(48, 196, 107, 255))
    if isinstance(gt_point, (list, tuple)) and len(gt_point) >= 2:
        point = [float(gt_point[0]), float(gt_point[1]) + header_height]
        draw_point(draw, point, color=(230, 57, 70, 255))

    draw_badge(draw, f"GT Action: {gt_action}", canvas.size)
    return canvas, {
        "image_path": image_path,
        "task_instruction": task_instruction,
        "step_instruction": step_instruction,
        "gt_action": gt_action,
        "gt_bbox": gt_bbox,
        "gt_point": gt_point,
    }


def render_gui_step(row):
    image_path = str(row["image_path"])
    image = Image.open(image_path)
    task_instruction = row.get("question", "")
    raw_answer = str(row.get("answer", "")).strip()
    step_instruction = row.get("step_instruction", "") or "N/A"
    decoded = None
    try:
        decoded = simple_decode(raw_answer)
    except Exception:
        decoded = {"action": raw_answer, "info": ""}

    action = str(decoded.get("action", "")).strip()
    info = decoded.get("info", "")
    sam2_bbox = row.get("sam2_bbox", None)

    canvas, header_height = build_canvas(
        image=image,
        task_instruction=task_instruction,
        step_instruction=step_instruction,
        gt_summary=f"answer={raw_answer}",
    )
    draw = ImageDraw.Draw(canvas, "RGBA")

    scaled_bbox = scale_gui_bbox(sam2_bbox, image.size)
    if scaled_bbox is not None:
        scaled_bbox = [scaled_bbox[0], scaled_bbox[1] + header_height, scaled_bbox[2], scaled_bbox[3] + header_height]
        draw_bbox(draw, scaled_bbox, color=(48, 196, 107, 255))

    if action in {"CLICK", "LONG_PRESS"}:
        point = scale_gui_coord(info, image.size)
        if point is not None:
            point[1] += header_height
            draw_point(draw, point, color=(230, 57, 70, 255))

    draw_badge(draw, f"GT Action: {raw_answer}", canvas.size)
    return canvas, {
        "image_path": image_path,
        "task_instruction": task_instruction,
        "step_instruction": step_instruction,
        "gt_action": action,
        "gt_info": info,
        "gt_bbox": sam2_bbox,
        "raw_answer": raw_answer,
    }


def sample_groups(grouped_rows, num_tasks, seed):
    task_keys = sorted(grouped_rows.keys())
    rng = random.Random(seed)
    if num_tasks >= len(task_keys):
        return task_keys
    return sorted(rng.sample(task_keys, num_tasks))


def has_all_images(rows):
    for row in rows:
        image_path = str(row.get("image_path", row.get("image", "")))
        if not image_path or not os.path.exists(image_path):
            return False
    return True


def group_android_dataset(dataset_name):
    with open(resolve_android_json_path(dataset_name), "r", encoding="utf-8") as f:
        records = json.load(f)
    grouped = {}
    for row in records:
        row_dict = dict(row)
        row_dict["image_path"] = resolve_android_image_path(row_dict.get("image", ""))
        key = android_task_key(row_dict)
        grouped.setdefault(key, []).append(row_dict)
    for key in grouped:
        grouped[key].sort(key=lambda item: step_index_from_path(item.get("image_path", item.get("image", ""))))
    return grouped


def group_gui_dataset(dataset_name):
    with open(resolve_gui_json_path(dataset_name), "r", encoding="utf-8") as f:
        records = json.load(f)
    grouped = {}
    for row in records:
        row_dict = dict(row)
        row_dict["image_path"] = resolve_gui_image_path(row_dict.get("image_path", row_dict.get("image", "")))
        grouped.setdefault(gui_task_key(row_dict), []).append(row_dict)
    for key in grouped:
        grouped[key].sort(key=lambda item: step_index_from_path(item.get("image_path", item.get("image", ""))))
    return grouped


def dataset_family(dataset_name):
    if str(dataset_name).startswith("AndroidControl_Curated"):
        return "android"
    if str(dataset_name).startswith("GUIOdyssey_"):
        return "gui_odyssey"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def export_dataset(dataset_name, num_tasks, seed, output_dir):
    family = dataset_family(dataset_name)
    if family == "android":
        grouped = group_android_dataset(dataset_name)
        renderer = render_android_step
    else:
        grouped = group_gui_dataset(dataset_name)
        renderer = render_gui_step

    complete_grouped = {key: rows for key, rows in grouped.items() if has_all_images(rows)}
    sampled_task_keys = sample_groups(complete_grouped, num_tasks=num_tasks, seed=seed)
    dataset_out_dir = Path(output_dir) / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": dataset_name,
        "family": family,
        "num_requested": num_tasks,
        "num_available_tasks": len(grouped),
        "num_complete_tasks": len(complete_grouped),
        "num_sampled_tasks": len(sampled_task_keys),
        "seed": seed,
        "tasks": [],
    }

    for task_rank, task_key in enumerate(sampled_task_keys, start=1):
        rows = complete_grouped[task_key]
        task_dir = dataset_out_dir / f"{task_rank:03d}_{sanitize_name(task_key)}"
        task_dir.mkdir(parents=True, exist_ok=True)
        task_meta = {
            "dataset": dataset_name,
            "task_key": task_key,
            "task_rank": task_rank,
            "num_steps": len(rows),
            "steps": [],
        }
        for step_rank, row in enumerate(rows, start=1):
            canvas, step_meta = renderer(row)
            step_filename = f"step_{step_rank:03d}.png"
            canvas.save(task_dir / step_filename)
            step_meta["output_file"] = step_filename
            step_meta["step_rank"] = step_rank
            task_meta["steps"].append(step_meta)
        with open(task_dir / "task_meta.json", "w", encoding="utf-8") as f:
            json.dump(task_meta, f, ensure_ascii=False, indent=2)
        summary["tasks"].append(
            {
                "task_key": task_key,
                "task_dir": str(task_dir),
                "num_steps": len(rows),
            }
        )

    with open(dataset_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[visualize_multistep_gui_tasks] dataset={dataset_name} sampled={len(sampled_task_keys)} output={dataset_out_dir}")


def main():
    args = parse_args()
    for dataset_name in args.datasets:
        export_dataset(
            dataset_name=dataset_name,
            num_tasks=max(0, args.num_tasks),
            seed=args.seed,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
