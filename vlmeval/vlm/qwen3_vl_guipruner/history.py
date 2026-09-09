"""History defaults isolated from every existing pruning backend."""

from __future__ import annotations

import os


def configure_guipruner_history_defaults(history_steps: int = 4) -> None:
    if not 0 <= int(history_steps) <= 4:
        raise ValueError("GUIPruner-reproduction supports at most four history screenshots.")
    steps = str(int(history_steps))
    for name, value in {
        "GUI_ODYSSEY_USE_HISTORY_SCREENSHOTS": "1",
        "GUI_ODYSSEY_MAX_HISTORY_IMAGES": steps,
        "ANDROID_CONTROL_USE_HISTORY_SCREENSHOTS": "1",
        "ANDROID_CONTROL_MAX_HISTORY_IMAGES": steps,
        "AITW_HIS_NUM": steps,
        "MIND2WEB_HIS_NUM": steps,
    }.items():
        os.environ.setdefault(name, value)
