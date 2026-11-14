import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw


MODULE_PATH = Path(__file__).parents[1] / "vlmeval/dataset/AndroidControl_Curated/state_packet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("android_control_state_packet_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_crop_policy_controls_are_geometry_matched_and_reproducible(tmp_path, monkeypatch):
    module = _load_module()
    image = Image.new("RGB", (400, 300), "black")
    action = {"gt_action": "click", "gt_bbox": [20, 30, 80, 90]}
    action_box = module._roi_box_for_action(action, image.size)

    monkeypatch.delenv("ANDROID_CONTROL_STATE_PACKET_CROP_POLICY", raising=False)
    selected, policy, _, _ = module._select_roi_box(
        img=image, action_packet=action, sample_index="sample", history_index=0, next_image_path=None
    )
    assert policy == "action"
    assert selected == action_box

    monkeypatch.setenv("ANDROID_CONTROL_STATE_PACKET_CROP_POLICY", "center")
    center, _, _, _ = module._select_roi_box(
        img=image, action_packet=action, sample_index="sample", history_index=0, next_image_path=None
    )
    assert module._box_size(center) == module._box_size(action_box)
    assert center == module._box_at_center(400, 300, *module._box_size(action_box))

    monkeypatch.setenv("ANDROID_CONTROL_STATE_PACKET_CROP_POLICY", "random")
    first, _, _, _ = module._select_roi_box(
        img=image, action_packet=action, sample_index="sample", history_index=0, next_image_path=None
    )
    second, _, _, _ = module._select_roi_box(
        img=image, action_packet=action, sample_index="sample", history_index=0, next_image_path=None
    )
    assert first == second
    assert module._box_size(first) == module._box_size(action_box)
    assert 0 <= first[0] < first[2] <= 400 and 0 <= first[1] < first[3] <= 300

    successor = image.copy()
    ImageDraw.Draw(successor).rectangle((300, 220, 390, 290), fill="white")
    successor_path = tmp_path / "successor.png"
    successor.save(successor_path)
    monkeypatch.setenv("ANDROID_CONTROL_STATE_PACKET_CROP_POLICY", "interface_change")
    changed, policy, score, source = module._select_roi_box(
        img=image,
        action_packet=action,
        sample_index="sample",
        history_index=0,
        next_image_path=str(successor_path),
    )
    assert policy == "interface_change"
    assert source == "successor_difference" and score > 0
    assert module._box_size(changed) == module._box_size(action_box)
    assert changed[2] > 300 and changed[3] > 220


def test_default_action_packet_keeps_legacy_metadata_schema(tmp_path, monkeypatch):
    module = _load_module()
    image_path = tmp_path / "history.png"
    Image.new("RGB", (400, 300), "black").save(image_path)
    monkeypatch.delenv("ANDROID_CONTROL_STATE_PACKET_CROP_POLICY", raising=False)
    monkeypatch.setenv("ANDROID_CONTROL_STATE_PACKET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ANDROID_CONTROL_STATE_PACKET_DEBUG", "0")

    images, meta = module.build_state_packet(
        image_path=str(image_path),
        action_packet={"gt_action": "click", "gt_bbox": [20, 30, 80, 90]},
        sample_index="sample",
        history_index=0,
    )
    assert [item.kind for item in images] == ["thumbnail", "action_roi"]
    assert "crop_policy" not in meta
    assert "crop_selection_source" not in meta
