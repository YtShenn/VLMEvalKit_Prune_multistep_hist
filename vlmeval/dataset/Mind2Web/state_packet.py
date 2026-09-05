import os

from ..AndroidControl_Curated.state_packet import (
    build_state_packet as _android_build_state_packet,
)


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_with_fallback(name: str, fallback_name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return os.getenv(fallback_name, default).strip()


def state_packet_enabled() -> bool:
    return _env_flag("MIND2WEB_STATE_PACKET_ENABLE", os.getenv("ANDROID_CONTROL_STATE_PACKET_ENABLE", "0"))


def state_packet_debug_enabled() -> bool:
    return _env_flag("MIND2WEB_STATE_PACKET_DEBUG", os.getenv("ANDROID_CONTROL_STATE_PACKET_DEBUG", "0"))


def _sync_android_state_packet_env_from_mind2web() -> None:
    mapping = {
        "STATE_PACKET_ENABLE": "0",
        "STATE_PACKET_DEBUG": "0",
        "STATE_PACKET_CACHE_DIR": "/tmp/mind2web_state_packet_cache",
        "STATE_PACKET_PATCH_SIZE": "16",
        "STATE_PACKET_MERGE_SIZE": "2",
        "STATE_PACKET_THUMB_LONG_EDGE": "192",
        "STATE_PACKET_ROI_LONG_EDGE": "224",
        "STATE_PACKET_ROI_SHORT_SIDE_RATIO": "0.22",
        "STATE_PACKET_ROI_MIN_SIDE_PX": "160",
    }
    for suffix, default in mapping.items():
        mind2web_name = f"MIND2WEB_{suffix}"
        android_name = f"ANDROID_CONTROL_{suffix}"
        os.environ[android_name] = _env_with_fallback(mind2web_name, android_name, default)


def build_state_packet(*, image_path: str, action_packet: dict, sample_index: str, history_index: int):
    _sync_android_state_packet_env_from_mind2web()
    return _android_build_state_packet(
        image_path=image_path,
        action_packet=action_packet,
        sample_index=sample_index,
        history_index=history_index,
    )


__all__ = [
    "build_state_packet",
    "state_packet_debug_enabled",
    "state_packet_enabled",
]
