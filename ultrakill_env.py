# ultrakill_env.py  –  exploration-friendly wrapper for ULTRAKILL
# --------------------------------------------------------------
import time, os
import ctypes
import mss, cv2, numpy as np, win32gui
import gymnasium as gym
from gymnasium.spaces import Box
from utils import lock_ultrakill_focus
from ultrakill_ai import send_scan, tap, SCAN         # ← restore real driver logic
from typing import Tuple, Optional
import win32gui
import ctypes.wintypes as wintypes
import numpy as np

# Mouse input constants
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("_input",)
    _fields_ = [("type", wintypes.DWORD), ("_input", _INPUT)]

def mouse_click():
    """Simulate a left mouse button click using SendInput"""
    # Mouse down
    ii_down = INPUT()
    ii_down.type = 0  # INPUT_MOUSE
    ii_down.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
    
    # Mouse up
    ii_up = INPUT()
    ii_up.type = 0  # INPUT_MOUSE
    ii_up.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)
    
    # Send both events
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_down), ctypes.sizeof(INPUT))
    time.sleep(0.05)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii_up), ctypes.sizeof(INPUT))

SCAN = {
    "MOVE_FORWARD": 0x11,
    "MOVE_BACK": 0x1F,
    "MOVE_LEFT": 0x1E,
    "MOVE_RIGHT": 0x20
}

#def send_scan(scan, up=False):
#    KEYEVENTF_SCANCODE = 0x0008
#    KEYEVENTF_KEYUP = 0x0002
#    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
#    
#    class KBDINPUT(ctypes.Structure):
#        _fields_ = [
#            ("wVk", ctypes.c_ushort),
#            ("wScan", ctypes.c_ushort),
#            ("dwFlags", ctypes.c_ulong),
#            ("time", ctypes.c_ulong),
#            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
#        ]
#    
#    class INPUT(ctypes.Structure):
#        _fields_ = [
#            ("type", ctypes.c_ulong),
#            ("ki", KBDINPUT)
#        ]
#    
#    extra = ctypes.c_ulong(0)
#    ii = INPUT()
#    ii.type = 1  # INPUT_KEYBOARD
#    ii.ki = KBDINPUT(0, scan, flags, 0, ctypes.pointer(extra))
#    ctypes.windll.user32.SendInput(1, ctypes.pointer(ii), ctypes.sizeof(ii))

# -------------------------------------------------------------------
os.makedirs("debug_frames", exist_ok=True)
user32 = ctypes.windll.user32            # low-level mouse

# ---------------------------- constants ----------------------------
PIXEL_THRESH   = 0.005      # “stuck” detector
STALE_LIMIT    = 500
STALE_PENALTY  = -0.5

TURN_PIXELS    = 100         # max pixel delta per frame
MOVE_R         = 0.01
TURN_R         = 0.05
CURI_SCALE     = 0.05
VEL_SCALE      = 0.04
DELTA_ERR_SCALE = 0.2
ON_CENTER_BONUS = 0.2

HIT_BONUS      = 1.5
TARGET_PENALTY = -0.02

# -------------------------------------------------------------------
def pitch_penalty(frame: np.ndarray) -> float:
    """Penalize staring at the sky/ceiling (simple vertical gradient)."""
    return 0.1 if (frame[-20:].mean() - frame[:20].mean()) < -15 else 0.0

def release_all_movement_keys():
    keys = ["MOVE_FORWARD", "MOVE_BACK", "MOVE_LEFT", "MOVE_RIGHT"]
    for key in keys:
        send_scan(SCAN[key], True)  # Release each key
        time.sleep(0.01)  # Short delay between keys

# ------------------------- screen capture ---------------------------
_sct = mss.mss()
def grab_frame() -> np.ndarray:
    """Grab ULTRAKILL window specifically"""
    try:
        # Get game window dimensions
        wins = []
        def _enum(h, p):
            if "ultrakill" in win32gui.GetWindowText(h).lower():
                p.append(h)
            return True
        win32gui.EnumWindows(_enum, wins)
        if not wins:
            return np.zeros((360, 640, 3), np.uint8)
        
        hwnd = wins[0]
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        x, y = win32gui.ClientToScreen(hwnd, (0,0))
        
        # Capture window region
        with mss.mss() as sct:
            monitor = {"top": y, "left": x, "width": right, "height": bot}
            img = np.array(sct.grab(monitor))
            
            # Convert BGRA to BGR (remove alpha channel)
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.shape[2] == 1:  # Grayscale edge case
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
            return cv2.resize(img, (640, 360))
    except Exception as e:
        print(f"Capture error: {e}")
        return np.zeros((360, 640, 3), np.uint8)


# ---------------- visual heuristics / helpers ----------------------
def red_center_bonus(rgb: np.ndarray) -> float:
    """Detect red hit-flash in the 40×40 cross-hair region."""
    h, w = rgb.shape[:2]
    c = rgb[h//2-20:h//2+20, w//2-20:w//2+20]
    r, g, b = c[...,2], c[...,1], c[...,0]
    return 0.4 * ((r>150)&(g<90)&(b<90)).mean()

def detect_target_offset(frame: np.ndarray) -> Optional[Tuple[float,float]]:
    """Largest red/blue/yellow blob centroid offset in (−1..1, −1..1)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV); h,w = frame.shape[:2]

    # colour masks
    lr1,ur1 = (0,120,70),  (10,255,255)
    lr2,ur2 = (170,120,70),(180,255,255)
    lb,ub   = (100,120,70),(130,255,255)
    ly,uy   = (15,100,100),(40,255,255)

    mask  = cv2.inRange(hsv, lr1, ur1) | cv2.inRange(hsv, lr2, ur2)
    mask |= cv2.inRange(hsv, lb,  ub)
    mask |= cv2.inRange(hsv, ly,  uy)

    # cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, 1)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 50: return None
    M = cv2.moments(best);  cx,cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
    return ( (cx-w/2)/(w/2), (cy-h/2)/(h/2) )

def detect_targets(frame: np.ndarray) -> float:
    """Rudimentary 'enemy pixels' score."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (0,120,70), (10,255,255)),
        cv2.inRange(hsv, (170,120,70),(180,255,255)),
        cv2.inRange(hsv, (100,120,70),(130,255,255)),
        cv2.inRange(hsv, (15,100,100),(40,255,255)),
    ]
    combined = masks[0]
    for m in masks[1:]:
        combined |= m
    return combined.mean()                 # 0‒1 proportion


# ----------------------- the Gym environment ------------------------
class UltrakillEnv(gym.Env):
    WARMUP_STEPS   = 20     # number of 0.02s ticks to hold W (20×0.02=0.4s; bump to taste)t
    FRAME_DELAY = 0.02

    def __init__(self, *, aim_only: bool = False):
        super().__init__()
        self.aim_only = aim_only
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = Box(0, 255, (360, 640, 3), np.uint8)
        # state vars:
        self.prev_frame = None
        self.prev_offset = None
        self.stale_cnt = 0
        self.t = 0
        self.episode_id = 0
        self.warmup_start_time = 0
        self.in_warmup = False
        self.warmup_cnt    = 0

    def reset(self, *, seed=None, options=None):
        print("\n=== RESET CALLED ===", flush=True)
#        # Clear keys and focus
#        release_all_movement_keys()
#        time.sleep(0.5)
#        lock_ultrakill_focus()
#        time.sleep(0.2)

#        print(f"→ BLOCKING WALK-IN: holding W for {self.WARMUP_DURATION}s …", flush=True)
#        press_forward(self.WARMUP_DURATION)
#        print("→ WALK-IN complete, handing control to agent", flush=True)

        # ── kick off a fixed-length W-hold ─────────────────
        self.warmup_cnt = self.WARMUP_STEPS
        # on the first tick of step(), we’ll press‐down W

        # grab a single fresh frame for your agent
        frame = grab_frame()
        self.prev_frame  = frame.copy()
        self.prev_offset = None
        self.stale_cnt    = 0
        self.t           = 0
        self.episode_id += 1
        print(f"=== RESET DONE (episode {self.episode_id}) ===", flush=True)
        return frame, {}

    def step(self, action: np.ndarray):
        current_time = time.time()
        warmup_elapsed = current_time - self.warmup_start_time
        # ── handle warm-up period by frames, not time ───────
        if self.warmup_cnt > 0:
            if self.warmup_cnt == self.WARMUP_STEPS:
                send_scan(SCAN["MOVE_FORWARD"])   # key-down
            self.warmup_cnt -= 1
            if self.warmup_cnt == 0:
                send_scan(SCAN["MOVE_FORWARD"], True)  # key-up
            time.sleep(self.FRAME_DELAY)
            frame = grab_frame()
            # continue issuing zero‐reward frames until done
            self.prev_frame = frame.copy()
            self.t += 1
            return frame, 0.0, False, False, {}

        # NORMAL OPERATION: AI control
        dx, dy, shoot_p = map(float, action)
        dx, dy = np.clip([dx, dy], -1, 1)
        user32.mouse_event(0x0001,
                           int(dx * TURN_PIXELS),
                           int(dy * TURN_PIXELS),
                           0, 0)
        if shoot_p > 0.5:
            mouse_click()

        # Grab frame & death check
        time.sleep(self.FRAME_DELAY)
        frame = grab_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if gray < 12 or gray > 240:
            release_all_movement_keys()
            return frame, -50.0, True, False, {}

        # Reward shaping
        diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        motion_frac = (diff > 15).mean()
        target_score = detect_targets(frame)
        hit_bonus = red_center_bonus(frame) * HIT_BONUS
        offset = detect_target_offset(frame)

        # base reward: time penalty + movement incentive + curiosity/velocity bonus
        r = -0.02 + TURN_R * (abs(dx) + abs(dy)) + (CURI_SCALE + VEL_SCALE) * (diff.mean() / 255)

        # penalize shooting into emptiness
        if shoot_p > 0.5 and target_score < 0.01:
            r -= 0.1

        # target engagement reward/penalty
        if target_score > 0.02:
            if shoot_p > 0.5:
                r += 2.0 + 2.0 * target_score + hit_bonus
            else:
                r -= 0.005
        else:
            r += -0.05 if shoot_p > 0.5 else 0.02 * motion_frac

        # on-center firing bonus
        if shoot_p > 0.5 and offset and abs(offset[0]) < 0.1 and abs(offset[1]) < 0.1:
            r += ON_CENTER_BONUS * 1.5

        # penalty for excessive vertical aim
        if offset:
            vert_err = max(0.0, abs(offset[1]) - 0.5)
            r -= vert_err * 0.2

        # keep look-down bias
        r -= pitch_penalty(frame)

        # Update & length cap
        self.prev_frame = frame.copy()
        self.t += 1
        done = self.t >= 2000
        return frame, float(r), done, False, {}

    def close(self):
        """Ensure keys are released when environment closes"""
        release_all_movement_keys()
        print("Environment closed - keys released")

# -------------------------- quick smoke-test -------------------------
if __name__ == "__main__":
    print("Testing screen capture...")
    for i in range(5):
        frame = grab_frame()
        print(f"Frame {i} shape: {frame.shape}, channels: {frame.shape[2] if len(frame.shape) > 2 else 1}")
        cv2.imwrite(f"test_frame_{i}.png", frame)
    print("Test complete! Check saved images.")
