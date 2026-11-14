#!/usr/bin/env python3
"""Render visual smoke-test boards for AndroidControl history crop policies."""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATE_PACKET_PATH = REPOSITORY_ROOT / "vlmeval/dataset/AndroidControl_Curated/state_packet.py"
_spec = importlib.util.spec_from_file_location("android_control_state_packet_preview", STATE_PACKET_PATH)
assert _spec is not None and _spec.loader is not None
_state_packet = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _state_packet
_spec.loader.exec_module(_state_packet)
build_state_packet = _state_packet.build_state_packet


POLICIES = ("action", "center", "random", "interface_change")


def _fit(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return result


def _render_board(
    source: Image.Image,
    roi: Image.Image,
    crop_xyxy: tuple[int, int, int, int],
    policy: str,
    meta: dict,
    output_path: Path,
) -> None:
    font = ImageFont.load_default()
    source_view = _fit(source, 360, 500)
    scale_x = source_view.width / source.width
    scale_y = source_view.height / source.height
    annotated = source_view.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(
        tuple(int(round(v)) for v in (crop_xyxy[0] * scale_x, crop_xyxy[1] * scale_y,
                                      crop_xyxy[2] * scale_x, crop_xyxy[3] * scale_y)),
        outline="#ff3030",
        width=4,
    )
    roi_view = _fit(roi, 360, 500)
    board_height = 610
    board = Image.new("RGB", (760, board_height), "white")
    board.paste(annotated, (20, 76))
    board.paste(roi_view, (400, 76))
    draw = ImageDraw.Draw(board)
    draw.text((20, 14), f"History crop policy: {policy}", fill="black", font=font)
    draw.text((20, 38), "Before: original historical screenshot (red = selected ROI)", fill="black", font=font)
    draw.text((400, 38), "After: local ROI passed to the packet", fill="black", font=font)
    details = (
        f"box={list(crop_xyxy)}  source={meta.get('crop_selection_source', 'action_anchor')}"
        f"  change_score={meta.get('crop_change_score', 0.0):.2f}"
    )
    draw.text((20, 580), details, fill="black", font=font)
    board.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Historical screenshot path")
    parser.add_argument("--next-image", required=True, help="Successor/current screenshot path")
    parser.add_argument("--action-json", required=True, help="Historical action packet as JSON")
    parser.add_argument("--sample-index", default="smoke")
    parser.add_argument("--history-index", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    action_packet = json.loads(args.action_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ANDROID_CONTROL_STATE_PACKET_CACHE_DIR", str(out_dir / "packet_cache"))
    os.environ["ANDROID_CONTROL_STATE_PACKET_DEBUG"] = "1"

    manifest = {"image": args.image, "next_image": args.next_image, "policies": {}}
    source = Image.open(args.image).convert("RGB")
    for policy in POLICIES:
        os.environ["ANDROID_CONTROL_STATE_PACKET_CROP_POLICY"] = policy
        packet_images, meta = build_state_packet(
            image_path=args.image,
            action_packet=action_packet,
            sample_index=args.sample_index,
            history_index=args.history_index,
            next_image_path=args.next_image,
        )
        roi_item = next(item for item in packet_images if item.kind == "action_roi")
        roi = Image.open(roi_item.path).convert("RGB")
        board_path = out_dir / f"{policy}_comparison.png"
        _render_board(source, roi, roi_item.crop_xyxy, policy, meta, board_path)
        manifest["policies"][policy] = {
            "comparison_image": str(board_path),
            "crop_xyxy": list(roi_item.crop_xyxy),
            "metadata": meta,
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
