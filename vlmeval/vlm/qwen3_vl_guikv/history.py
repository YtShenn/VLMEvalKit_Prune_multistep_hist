from __future__ import annotations

import os


def _set_default_env(name: str, value: str) -> None:
    if name not in os.environ:
        os.environ[name] = value


def configure_guikv_history_defaults(history_steps: int = 4) -> None:
    """Use the repository's existing multi-image history builders with GUI-KV defaults."""
    steps = str(max(0, int(history_steps)))
    _set_default_env("GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS", "1")
    _set_default_env("GUI_ODYSSEY_MAX_HISTORY_IMAGES", steps)
    _set_default_env("ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS", "1")
    _set_default_env("ANDROID_CONTROL_MAX_HISTORY_IMAGES", steps)
    _set_default_env("AITW_HIS_NUM", steps)
    _set_default_env("MIND2WEB_HIS_NUM", steps)
