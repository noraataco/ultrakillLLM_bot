import ctypes
import time
from ctypes import wintypes
from utils import lock_ultrakill_focus

# Ensure correct pointer type
# Windows SendInput boilerplate
INPUT_KEYBOARD     = 1
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP    = 0x0002

class _KI(ctypes.Structure):
    _fields_ = [
        ("wVk",         wintypes.WORD),
        ("wScan",       wintypes.WORD),
        ("dwFlags",     wintypes.DWORD),
        ("time",        wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]

class _INPUT_I(ctypes.Union):
    _fields_ = [("ki", _KI)]

class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("ii",   _INPUT_I),
    ]

# Default forward scan code
SCAN = {"MOVE_FORWARD": 0x11}


def send_scan(scancode: int, up: bool = False):
    """Press or release a key by scancode."""
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    inp = _INPUT(type=INPUT_KEYBOARD, ii=_INPUT_I(ki=_KI(0, scancode, flags, 0, 0)))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def press_scancode(scancode: int):
    """Press & hold a key by scancode."""
    send_scan(scancode, up=False)


def release_scancode(scancode: int):
    """Release a key by scancode."""
    send_scan(scancode, up=True)


def press_forward(duration: float):
    """
    Blockingly hold the FORWARD key (scancode) for the given duration.
    Ensures the game window retains focus.
    """
    # ensure focus stays on game
    lock_ultrakill_focus()
    time.sleep(0.1)

    # press forward scancode down
    press_scancode(SCAN["MOVE_FORWARD"])
    # hold for duration
    time.sleep(duration)
    # release forward
    release_scancode(SCAN["MOVE_FORWARD"])
    # small buffer
    time.sleep(0.05)
