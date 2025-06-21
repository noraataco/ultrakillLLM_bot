# ultrakill_ai.py – full move-set, scan-code SendInput
import ctypes, time, os, json, requests, threading, sys
from   ctypes import wintypes
import logging
import re, random
import dxcam, cv2, numpy as np, os, ctypes, win32gui
import atexit, signal, sys, traceback, ctypes
from ctypes import wintypes
import traceback
from input_helper import send_scan, press_scancode, release_scancode, press_forward, SCAN


ctypes.windll.kernel32.FreeConsole()


user32 = ctypes.windll.user32
KEYEVENTF_KEYUP = 0x0002

# virtual-keys we might hold (SHIFT, CTRL, ALT, CAPS, WASD, SPACE)
SAFETY_VK = [0x10, 0xA0, 0xA1,     # Shift + L/R variants
             0x11, 0xA2, 0xA3,     # Ctrl   (not used but safe)
             0x12, 0xA4, 0xA5,     # Alt    (same)
             0x14,                 # Caps Lock
             0x57, 0x41, 0x53, 0x44,  # W A S D
             0x20]                 # Space

def _vk_up(vk):
    user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

def emergency_release():
    # 1) release anything the driver knows it’s holding
    try:
        from ultrakill_ai import clear_all
        clear_all()
    except Exception:
        pass
    # 2) send an "UP" event for every VK in the safety list
    for vk in SAFETY_VK:
        _vk_up(vk)
    print("[safety] all keys released")

# run on normal exit
atexit.register(emergency_release)

# run on uncaught exception
def _excepthook(etype, value, tb):
    emergency_release()
    traceback.print_exception(etype, value, tb)
sys.excepthook = _excepthook

# run on SIGINT, SIGTERM, etc.
for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGABRT):
    signal.signal(sig, lambda *_: sys.exit(1))

# ───────── action vocabulary ─────────
ACTIONS = [
    # movement
    "MOVE_FORWARD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT",
    # camera (8-way nudges)
    "TURN_LEFT", "TURN_RIGHT",
    "LOOK_UP", "LOOK_DOWN",
    "TURN_LEFT_UP", "TURN_LEFT_DOWN",
    "TURN_RIGHT_UP", "TURN_RIGHT_DOWN",
    # weapons
    "SHOOT",        # left-click
    "ALT_FIRE",     # right-click   ← NEW
    # misc
    "JUMP", "DASH", "STOP_ALL"
]
ALLOWED      = " ".join(ACTIONS)
ALLOWED_SET  = set(ACTIONS)


sys.stdout = open("driver.log", "w", buffering=1)
sys.stderr = sys.stdout         # catch traceback too

# ──────────────── Config (tweak to taste) ────────────────
DEBUG        = True            # press F1 while running to toggle
TICK_RATE    = 0.02            # 50 Hz poll loop
DASH_MS      = 0.15            # hold Shift this long for a dash
TURN_PIXELS  = 40              # mouse dx per TURN_LEFT/RIGHT tick
TURN_HOLD_MS = 0.0             # >0 → continuous turn (0 = one flick)
OLLAMA_URL   = os.getenv("OLLAMA_URL",  "http://10.3.1.101:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b")
WARMUP_TIME  = 4.0

# Allowed action keywords
HOLD_ACTIONS = {"MOVE_FORWARD","MOVE_BACK","MOVE_LEFT","MOVE_RIGHT"}
TAP_ACTIONS  = {"JUMP"}
SPECIALS     = {"DASH","SHOOT","ALT_FIRE", "TURN_LEFT","TURN_RIGHT","STOP_ALL"}



PROMPT = f"""
# ROLE
You are V1, the player character in ULTRAKILL.

# HOW TO ANSWER
Reply with **ONE** word only, chosen from this list (all caps):
{ALLOWED}

No punctuation, no extra words, no sentences.  
Example: `MOVE_FORWARD`

If you repeat the same action 3 times in a row, choose a different one.

# CONTEXT
{{state}}
"""

camera = dxcam.create(output_idx=0)  # 0 = primary monitor

def get_ultrakill_rect():
    hwnd = win32gui.FindWindow(None, None)
    def _enum(h, p):
        if "ultrakill" in win32gui.GetWindowText(h).lower():
            p.append(h)
        return True
    wins=[]; win32gui.EnumWindows(_enum, wins)
    if not wins:
        raise RuntimeError("ULTRAKILL window not found")
    left, top, right, bot = win32gui.GetClientRect(wins[0])
    x, y = win32gui.ClientToScreen(wins[0], (0,0))
    return (x, y, x+right, y+bot)

RECT = get_ultrakill_rect()

def grab_frame():
    try:
        img = camera.grab(region=RECT)          # only the game area
        if img is None:
            raise RuntimeError("grab returned None – window probably minimized")
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
        return small[...,None].astype(np.uint8)
    except Exception as e:
        logging.error(f"Capture failed: {e}")
        return np.zeros((84,84,1), np.uint8)  # Return blank frame instead of None

# Helper to recognize the scoreboard / death screen
def is_score_screen(frame: np.ndarray) -> bool:
    gray = frame.squeeze()
    return gray.mean() < 55 and gray.std() < 20

# ── soft reset (ESC→Enter) ──────────────────────────────────
VK_ESC, VK_ENTER = 0x1B, 0x0D
def soft_reset():
    ctypes.windll.user32.keybd_event(VK_ESC,0,0,0)
    ctypes.windll.user32.keybd_event(VK_ESC,0,KEYEVENTF_KEYUP,0)
    time.sleep(0.3)
    ctypes.windll.user32.keybd_event(VK_ENTER,0,0,0)
    ctypes.windll.user32.keybd_event(VK_ENTER,0,KEYEVENTF_KEYUP,0)
    time.sleep(3)


# ─────────────── Windows & SendInput setup ───────────────
if not hasattr(wintypes, "ULONG_PTR"):
    wintypes.ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p)==8 else ctypes.c_uint32

INPUT_KEYBOARD, INPUT_MOUSE  = 1, 0
KEYEVENTF_SCANCODE, KEYEVENTF_KEYUP = 0x0008, 0x0002
MOUSEEVENTF_MOVE, MOUSEEVENTF_LDOWN, MOUSEEVENTF_LUP = 0x1, 0x2, 0x4
MOUSEEVENTF_RDOWN = 0x0008     # ← NEW
MOUSEEVENTF_RUP   = 0x0010     # ← NEW
# right-button
SCAN = {"MOVE_FORWARD":0x11,"MOVE_BACK":0x1F,"MOVE_LEFT":0x1E,"MOVE_RIGHT":0x20,"JUMP":0x39,"DASH":0x2A}

class K(ctypes.Structure):
    _fields_= [("wVk",wintypes.WORD),("wScan",wintypes.WORD),
               ("dwFlags",wintypes.DWORD),("time",wintypes.DWORD),
               ("dwExtraInfo",wintypes.ULONG_PTR)]
class M(ctypes.Structure):
    _fields_= [("dx",wintypes.LONG),("dy",wintypes.LONG),
               ("mouseData",wintypes.DWORD),("dwFlags",wintypes.DWORD),
               ("time",wintypes.DWORD),("dwExtraInfo",wintypes.ULONG_PTR)]
class U(ctypes.Union): _fields_=[("ki",K),("mi",M)]
class I(ctypes.Structure): _anonymous_=("u",); _fields_=[("type",wintypes.DWORD),("u",U)]
SendInput=ctypes.windll.user32.SendInput; user32=ctypes.windll.user32

def send_scan(scan, up=False):
    flags=KEYEVENTF_SCANCODE|(KEYEVENTF_KEYUP if up else 0)
    inp=I(type=INPUT_KEYBOARD,ki=K(0,scan,flags,0,0))
    ctypes.windll.user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
    time.sleep(0.005)  # Brief pause between inputs
def mouse_move(dx,dy):
    SendInput(1,ctypes.byref(I(INPUT_MOUSE,mi=M(dx,dy,0,MOUSEEVENTF_MOVE,0,0))),ctypes.sizeof(I))
def mouse_click():
    SendInput(1,ctypes.byref(I(INPUT_MOUSE,mi=M(0,0,0,MOUSEEVENTF_LDOWN,0,0))),ctypes.sizeof(I))
    SendInput(1,ctypes.byref(I(INPUT_MOUSE,mi=M(0,0,0,MOUSEEVENTF_LUP,0,0))),ctypes.sizeof(I))
def mouse_alt_click():
    """Tap right mouse button (alt-fire)."""
    SendInput(1, ctypes.byref(I(INPUT_MOUSE,
            mi=M(0,0,0,MOUSEEVENTF_RDOWN,0,0))), ctypes.sizeof(I))
    SendInput(1, ctypes.byref(I(INPUT_MOUSE,
            mi=M(0,0,0,MOUSEEVENTF_RUP,  0,0))), ctypes.sizeof(I))


turn_accumulator = [0, 0]  # [x, y]

def handle_turn(action):
    if action == "TURN_LEFT": 
        turn_accumulator[0] -= TURN_PIXELS
    elif action == "TURN_RIGHT":
        turn_accumulator[0] += TURN_PIXELS
    
    # Apply accumulated movement
    if abs(turn_accumulator[0]) > 1:
        mouse_move(int(turn_accumulator[0]), 0)
        turn_accumulator[0] = 0

CONFLICTING_ACTIONS = {
    "MOVE_FORWARD": "MOVE_BACK",
    "MOVE_LEFT": "MOVE_RIGHT"
}

def resolve_conflicts(action):
    conflict = CONFLICTING_ACTIONS.get(action)
    if conflict and conflict in HELD:
        release(conflict)

# ─────────────── stateful key helpers ───────────────
HELD=set()
def hold(act):     send_scan(SCAN[act]);       HELD.add(act)
def release(act):  send_scan(SCAN[act],True);  HELD.discard(act)
def tap(act,d=0.08): send_scan(SCAN[act]); time.sleep(d); send_scan(SCAN[act],True)
def clear_all():
    for act in list(HELD):
        if is_key_down(SCAN[act]):  # Only release if actually pressed
            release(act)
def is_key_down(vk_code):
    return user32.GetAsyncKeyState(vk_code) & 0x8000 != 0

# ─────────────── focus ULTRAKILL window ─────────────
_hwnd_cache = None
UNITY_CLASS = b"UnityWndClass"        # ← constant

def focus_ultrakill() -> bool:
    global _hwnd_cache
    fg = user32.GetForegroundWindow()

    # cached handle still valid?
    if _hwnd_cache and user32.IsWindow(_hwnd_cache):
        if fg != _hwnd_cache:
            user32.SetForegroundWindow(_hwnd_cache)
        return True

    # (re)find a Unity window whose title begins with "ULTRAKILL"
    def _enum(hwnd, _):
        global _hwnd_cache
        class_name = ctypes.create_string_buffer(256)
        user32.GetClassNameA(hwnd, class_name, 256)
        if class_name.value != UNITY_CLASS:
            return True            # skip non-Unity windows

        title = ctypes.create_unicode_buffer(256)
        user32.GetWindowTextW(hwnd, title, 256)

        if title.value.upper().startswith("ULTRAKILL"):
            _hwnd_cache = hwnd
            user32.SetForegroundWindow(hwnd)
            return False           # stop EnumWindows
        return True

    EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(EnumProc(_enum), 0)
    return _hwnd_cache is not None


# ─────────────── LLM helper ───────────────
ALLOWED_SET = set(ALLOWED.split())
history     = []          # last 3 actions

def ask_llm():
    state  = "HISTORY " + " ".join(history[-3:]) if history else "START"
    body   = {
        "model":  OLLAMA_MODEL,
        "stream": False,
        "prompt": PROMPT.format(state=state)
    }
    try:
        r = requests.post(OLLAMA_URL, json=body, timeout=4)
        r.raise_for_status()
        text = r.json()["response"].upper()
    except Exception as e:
        if DEBUG: print("LLM error:", e)
        text = ""

    # pick the first allowed token we can find
    m = re.search(r"\b(" + "|".join(ALLOWED_SET) + r")\b", text)
    action = m.group(1) if m else random.choice(tuple(ALLOWED_SET))

    history.append(action)
    return action

# ─────────────── hotkey thread (F1 toggle, Esc exit) ──────────
def hotkeys():
    global DEBUG; VK_F1=0x70; VK_ESC=0x1B
    while True:
        if user32.GetAsyncKeyState(VK_F1)&1: DEBUG=not DEBUG; print(f"[debug {'on' if DEBUG else 'off'}]")
        if user32.GetAsyncKeyState(VK_ESC)&1: os._exit(0)
        time.sleep(0.05)
threading.Thread(target=hotkeys,daemon=True).start()

# ─────────────── main loop ───────────────
def wait_for_ultrakill(max_wait=10) -> bool:
    """Try to focus the ULTRAKILL window for <max_wait> seconds."""
    t0 = time.time()
    while time.time() - t0 < max_wait:
        if focus_ultrakill():
            return True
        time.sleep(0.5)
    return False

def main():
    try:
        # ────────────────────────────────────────────────────────────
        # 1.  Focus the ULTRAKILL window (wait up to 10 s)
        # ────────────────────────────────────────────────────────────
        if not wait_for_ultrakill(10):
            print("[abort] ULTRAKILL window not found after 10 s")
            return

        # ────────────────────────────────────────────────────────────
        # 2.  Load PPO checkpoint (optional arg after "ppo")
        # ────────────────────────────────────────────────────────────
        ckpt = (
            sys.argv[2]                       # python ultrakill_ai.py ppo my.ckpt.zip
            if len(sys.argv) >= 3 else
            "ppo_ultrakill_curiosity.zip"     # default
        )
        if not os.path.isfile(ckpt):
            print(f"[abort] checkpoint '{ckpt}' not found")
            return

        print("ULTRAKILL AI driver running  (Esc = quit, F1 = debug toggle)")

        # ────────────────────────────────────────────────────────────
        # 3.  Main loop
        # ────────────────────────────────────────────────────────────
        auto_forward_active = True
        auto_forward_end    = time.time() + WARMUP_TIME
        in_score_screen     = False
        last_jump_time      = 0.0
        hold("MOVE_FORWARD")

        last_action        = None
        action_repeat_cnt  = 0
        turn_accumulator_x = 0
        turn_accumulator_y = 0
        frame_counter      = 0

        while True:
            frame_counter += 1

            frame = grab_frame()

            if is_score_screen(frame):
                if not in_score_screen:
                    in_score_screen = True
                    clear_all()
                    last_jump_time = 0.0
                if time.time() - last_jump_time > 1.0:
                    tap("JUMP")
                    last_jump_time = time.time()
                time.sleep(TICK_RATE)
                continue
            else:
                if in_score_screen:
                    in_score_screen = False
                    auto_forward_active = True
                    auto_forward_end = time.time() + WARMUP_TIME
                    hold("MOVE_FORWARD")

            if auto_forward_active:
                if time.time() >= auto_forward_end:
                    release("MOVE_FORWARD")
                    auto_forward_active = False
                else:
                    if "MOVE_FORWARD" not in HELD:
                        hold("MOVE_FORWARD")
                    time.sleep(TICK_RATE)
                    continue

            # re-focus every 30 ticks (~0.6 s) in case something stole it
            if frame_counter % 30 == 0:
                focus_ultrakill()

            # ── ask the LLM / PPO policy ─────────────────────────
            act = ask_llm()

            if DEBUG:
                print("LLM →", act)

            # Prevent getting stuck on one action
            if act == last_action:
                action_repeat_cnt += 1
                if action_repeat_cnt >= 3:
                    # pick a different random allowed action
                    alt_choices = list(ALLOWED_SET - {act})
                    act = random.choice(alt_choices)
                    action_repeat_cnt = 0
                    if DEBUG:
                        print("forced change →", act)
            else:
                action_repeat_cnt = 0
            last_action = act

            # ── resolve mutually exclusive holds ────────────────
            conflict = CONFLICTING_ACTIONS.get(act)
            if conflict in HELD:
                release(conflict)

            # ── dispatch action ────────────────────────────────
            if act in HOLD_ACTIONS:
                hold(act)

            elif act == "STOP_ALL":
                clear_all()

            elif act == "JUMP":
                tap("JUMP")

            elif act == "DASH":
                send_scan(SCAN["DASH"])             # key down
                time.sleep(DASH_MS)
                send_scan(SCAN["DASH"], True)       # key up

            elif act == "SHOOT":
                mouse_click()

            elif act == "ALT_FIRE":
                mouse_alt_click()

            elif act in (
                "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN",
                "TURN_LEFT_UP", "TURN_LEFT_DOWN",
                "TURN_RIGHT_UP", "TURN_RIGHT_DOWN"
            ):
                dx, dy = {
                    "TURN_LEFT"      : (-TURN_PIXELS,  0),
                    "TURN_RIGHT"     : ( TURN_PIXELS,  0),
                    "LOOK_UP"        : ( 0, -TURN_PIXELS),
                    "LOOK_DOWN"      : ( 0,  TURN_PIXELS),
                    "TURN_LEFT_UP"   : (-TURN_PIXELS, -TURN_PIXELS),
                    "TURN_LEFT_DOWN" : (-TURN_PIXELS,  TURN_PIXELS),
                    "TURN_RIGHT_UP"  : ( TURN_PIXELS, -TURN_PIXELS),
                    "TURN_RIGHT_DOWN": ( TURN_PIXELS,  TURN_PIXELS),
                }[act]

                turn_accumulator_x += dx
                turn_accumulator_y += dy

                # apply immediately if magnitude ≥ 1 pixel
                if abs(turn_accumulator_x) >= 1 or abs(turn_accumulator_y) >= 1:
                    mouse_move(int(turn_accumulator_x), int(turn_accumulator_y))
                    turn_accumulator_x = turn_accumulator_y = 0

            # ── wait for next tick ─────────────────────────────
            time.sleep(TICK_RATE)

    # ───────────────────────────────────────────────────────────────
    except Exception:
        print("[fatal] unhandled exception:\n", traceback.format_exc())

    finally:
        clear_all()
        print("[exit] all keys released – safe to type again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("Fatal error:", e)
    finally:
        clear_all()               # <–– releases any held key
        # make sure CapsLock is off
        VK_CAPS = 0x14
        ctypes.windll.user32.keybd_event(VK_CAPS, 0, 0,    0)  # down
        ctypes.windll.user32.keybd_event(VK_CAPS, 0, 0x2, 0)  # up
        print("All keys released – safe to type again.")

