"""Action ids used by the Android in the Wild annotations."""

SCROLL_DOWN = 0
SCROLL_UP = 1
UNKNOWN = 2
TYPE = 3
DUAL_POINT = 4
PRESS_BACK = 5
PRESS_HOME = 6
PRESS_ENTER = 7
SCROLL_LEFT = 8
SCROLL_RIGHT = 9
COMPLETE = 10

SCROLL_ACTIONS = {SCROLL_DOWN, SCROLL_UP, SCROLL_LEFT, SCROLL_RIGHT}
GLOBAL_ACTIONS = {UNKNOWN, PRESS_BACK, PRESS_HOME, PRESS_ENTER, COMPLETE}

ACTION_ID_TO_NAME = {
    SCROLL_DOWN: "scroll down",
    SCROLL_UP: "scroll up",
    UNKNOWN: "unknown",
    TYPE: "input text",
    DUAL_POINT: "dual point",
    PRESS_BACK: "press back",
    PRESS_HOME: "press home",
    PRESS_ENTER: "press enter",
    SCROLL_LEFT: "scroll left",
    SCROLL_RIGHT: "scroll right",
    COMPLETE: "complete",
}

SCROLL_NAME_TO_ID = {
    "scroll down": SCROLL_DOWN,
    "swipe up": SCROLL_DOWN,
    "scroll up": SCROLL_UP,
    "swipe down": SCROLL_UP,
    "scroll left": SCROLL_LEFT,
    "swipe right": SCROLL_LEFT,
    "scroll right": SCROLL_RIGHT,
    "swipe left": SCROLL_RIGHT,
}
