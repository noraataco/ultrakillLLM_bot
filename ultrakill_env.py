# ultrakill_env.py  –  exploration-friendly wrapper for ULTRAKILL
# --------------------------------------------------------------
import time, os, ctypes
import mss, cv2, numpy as np, win32gui
import gymnasium as gym
from gymnasium.spaces import Box
from utils import lock_ultrakill_focus, PAUSED
import pytesseract, re # type: ignore
from ultrakill_ai import soft_reset, send_scan, SCAN, mouse_click
from typing import Tuple, Optional
import ctypes.wintypes as wintypes
from input_helper import press_forward
import ctypes
import win32gui
import win32con
import win32api



# Allow overriding the Tesseract executable location via environment
tesseract_override = os.environ.get("TESSERACT_CMD")
if tesseract_override:
    pytesseract.pytesseract.tesseract_cmd = tesseract_override

# Ensure output directory exists
os.makedirs("debug_frames", exist_ok=True)
user32 = ctypes.windll.user32

# Create a single MSS instance to avoid handle leaks on Windows
sct = mss.mss()

# Constants
TURN_PIXELS    = 100
PIXEL_THRESH   = 0.005
STALE_LIMIT    = 500
STALE_PENALTY  = -0.5
MOVE_R         = 0.01
TURN_R         = 0.05
CURI_SCALE     = 0.05
VEL_SCALE      = 0.04
DELTA_ERR_SCALE= 0.2
ON_CENTER_BONUS= 0.2
HIT_BONUS      = 1.5
TARGET_PENALTY = -0.02
DAMAGE_MOVE_WEIGHT = 0.1
DAMAGE_THRESHOLD    = 0.05

MOUSEEVENTF_MOVE     = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

STEP_DELAY = 0.02
# Reward granted each step the agent stays alive
SURVIVAL_BONUS = 0.05

# Utility functions
def pitch_penalty(frame: np.ndarray, target_present: bool) -> float:
    """Return a penalty for extreme up/down pitch."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top = gray[:20].mean()
    bottom = gray[-20:].mean()

    penalty = 0.0
    if bottom - top < -15:
        penalty = 0.4
    elif top - bottom > 15:
        penalty = 0.3

    if target_present:
        penalty *= 0.5

    return penalty

def release_all_movement_keys():
    for key in ["MOVE_FORWARD","MOVE_BACK","MOVE_LEFT","MOVE_RIGHT"]:
        send_scan(SCAN[key], True)
        time.sleep(0.01)

# Screen capture
def grab_frame() -> np.ndarray:
    t0 = time.time()
    try:
        wins = []
        win32gui.EnumWindows(
            lambda h, p: p.append(h) if "ultrakill" in win32gui.GetWindowText(h).lower() else True,
            wins
        )
        if not wins:
            img = np.zeros((360,640,3), np.uint8)
        else:
            hwnd = wins[0]
            left, top, right, bot = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (0,0))
            monitor = {"top": y, "left": x, "width": right, "height": bot}
            img = np.array(sct.grab(monitor))
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (640,360))
    except Exception:
        img = np.zeros((360,640,3), np.uint8)
    dt = time.time() - t0
    if dt > STEP_DELAY:
        print(f"[warn] grab_frame took {dt:.3f}s (> {STEP_DELAY}s)")
    return img

def eye_level_bonus(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    center_y = h // 2
    eye_zone = frame[center_y-30:center_y+30, :]
    return 0.1 * (eye_zone.mean() > 100)

def is_score_screen(frame: np.ndarray) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean, std = gray.mean(), gray.std()
    # Dark AND very uniform → scoreboard / death screen
    return (mean < 50) and (std < 15)

def red_center_bonus(rgb: np.ndarray) -> float:
    h,w = rgb.shape[:2]
    c = rgb[h//2-20:h//2+20, w//2-20:w//2+20]
    r,g,b = c[...,2], c[...,1], c[...,0]
    return 0.4 * ((r>150)&(g<90)&(b<90)).mean()

def detect_target_offset(frame: np.ndarray) -> Optional[Tuple[float,float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, lo, hi) for lo,hi in
             [((0,120,70),(10,255,255)),
              ((170,120,70),(180,255,255)),
              ((100,120,70),(130,255,255)),
              ((15,100,100),(40,255,255))]]
    mask = masks[0]
    for m in masks[1:]:
        mask |= m
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or cv2.contourArea(max(cnts, key=cv2.contourArea)) < 50:
        return None
    M = cv2.moments(max(cnts, key=cv2.contourArea))
    cx,cy = M['m10']/M['m00'], M['m01']/M['m00']
    h,w = frame.shape[:2]
    return ((cx-w/2)/(w/2), (cy-h/2)/(h/2))

def detect_targets(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, lo, hi) for lo, hi in
             [((0,120,70),(10,255,255)),
              ((170,120,70),(180,255,255)),
              ((100,120,70),(130,255,255)),
              ((15,100,100),(40,255,255))]]
    combined = masks[0]
    for m in masks[1:]:
        combined |= m
    return combined.mean()

def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    """Return a copy of ``img`` with saturation scaled by ``factor``."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def read_health(frame: np.ndarray) -> Optional[int]:
    """Return the current health as an integer if detected."""
    h, w = frame.shape[:2]
    region = frame[h-45:h-5, 5:140]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    digits = re.findall(r"\d+", text)
    try:
        return int(digits[0]) if digits else None
    except ValueError:
        return None

def detect_damage(frame: np.ndarray) -> Tuple[float, float, float]:
    """Return overall damage level and a rough direction vector."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0,50,50),(10,255,255)) |
            cv2.inRange(hsv, (170,50,50),(180,255,255)))
    edge = 20
    left   = mask[:, :edge].mean()
    right  = mask[:, -edge:].mean()
    top    = mask[:edge, :].mean()
    bottom = mask[-edge:, :].mean()

    level = max(left, right, top, bottom) / 255.0
    dx = (right - left) / 255.0
    dy = (bottom - top) / 255.0
    return level, dx, dy


def detect_dashes(frame: np.ndarray) -> int:
    """Return the number of dash charges available (0-3)."""
    h, w = frame.shape[:2]
    x0, x1 = int(w*0.03), int(w*0.18)
    y0, y1 = int(h*0.84), int(h*0.89)
    bar = frame[y0:y1, x0:x1]
    if bar.size == 0:
        return 0
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (80,50,180), (110,255,255))
    seg_w = mask.shape[1] // 3
    return sum((mask[:, i*seg_w:(i+1)*seg_w].mean() > 40) for i in range(3))

def read_health(frame: np.ndarray) -> Optional[int]:
    """Return the current health as an integer if detected."""
    h, w = frame.shape[:2]
    region = frame[h-45:h-5, 5:140]
    if region.size == 0:
        return None
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    digits = re.findall(r"\d+", text)
    return int(digits[0]) if digits else None

class UltrakillEnv(gym.Env):
    WARMUP_TIME  = 4.0
    FRAME_DELAY  = STEP_DELAY

    def __init__(self, *, aim_only: bool=False):
        super().__init__()
        self.aim_only = aim_only
        self.observation_space = Box(
            low=0, high=255, shape=(360, 640, 3), dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1,-1,0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.prev_frame   = None
        self.prev_offset  = None
        self.t            = 0
        self.episode_id   = 0
        self.resets       = 0
        self.dash_count   = 0
        self.health       = None
        self.score_screen_frames = 0  # debounce counter
        self.SCORE_FRAMES        = 5

    def reset(self, *, seed=None, options=None):
        # 1) bookkeeping
        self.episode_id += 1
        self.resets    += 1
        self.t          = 0
        release_all_movement_keys()
        time.sleep(0.5)
        lock_ultrakill_focus()
        time.sleep(2.0)  # let the death screen unload

        # 2) if we’re on the score screen, click twice to advance
        frame = grab_frame()
        mean, std = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()
        print(f"SCORECHECK: mean={mean:.1f}, std={std:.1f}")
        if is_score_screen(frame):
            for attempt in range(8):  # try up to 8 times
                lock_ultrakill_focus()     # re‐grab focus
                send_scan(SCAN["JUMP"])    # key‐down
                send_scan(SCAN["JUMP"], True)  # key‐up
                time.sleep(0.3)
                frame = grab_frame()
                mean, std = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).std()
                print(f"  attempt {attempt+1}: mean={mean:.1f}, std={std:.1f}")
                if not is_score_screen(frame):
                    break
            else:
                # fallback to a full soft reset if it never cleared
                print("SCORECHECK: never cleared, doing soft_reset()")
                soft_reset()
                time.sleep(3.0)
                frame = grab_frame()

        # 3) now we’re clean in the live map—walk forward 4s in one go
        print("WALK: pressing forward for", self.WARMUP_TIME, "seconds")
        # key-down using your ultrakill_ai.send_scan, so SCAN matches what step() uses
        send_scan(SCAN["MOVE_FORWARD"])   
        time.sleep(self.WARMUP_TIME)
        print("WALK: releasing forward")
        send_scan(SCAN["MOVE_FORWARD"], True)
        # small buffer so the game actually registers the release
        time.sleep(0.05)

        # 4) grab the first “real” observation
        frame = grab_frame()
        self.prev_frame = frame.copy()
        self.dash_count = detect_dashes(frame)
        self.health     = read_health(frame)

        return frame, {
            "dash_count": self.dash_count,
            "health":     self.health,
            "resets":     self.resets,
        }

    def step(self, action):
        dx, dy, shoot = map(float, action)
        dx, dy = np.clip([dx, dy], -1, 1)

        while PAUSED:
            release_all_movement_keys()
            time.sleep(0.05)

        # movement
        if not self.aim_only:
            if abs(dx) < 0.1 and abs(dy) < 0.1:
                release_all_movement_keys()
            else:
                if dx > 0.1: send_scan(SCAN["MOVE_FORWARD"])
                elif dx < -0.1: send_scan(SCAN["MOVE_BACK"])
                if dy > 0.1: send_scan(SCAN["MOVE_RIGHT"])
                elif dy < -0.1: send_scan(SCAN["MOVE_LEFT"])

        # camera
        user32.mouse_event(MOUSEEVENTF_MOVE,
            int(dx * TURN_PIXELS),
            int(dy * TURN_PIXELS), 0, 0)

        # shoot
        if shoot > 0.5:
            mouse_click()

        time.sleep(self.FRAME_DELAY)
        frame = grab_frame()
        self.dash_count = detect_dashes(frame)
        self.health     = read_health(frame)

        # score/death detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if is_score_screen(frame) or gray < 12 or gray > 240:
            release_all_movement_keys()
            return frame, -50.0, True, False, {
                "dash_count": self.dash_count,
                "health":     self.health,
                "resets":     self.resets,
            }

        # reward shaping...
        diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        motion_frac = (diff > 15).mean()
        hit_bonus   = red_center_bonus(frame) * HIT_BONUS
        offset      = detect_target_offset(frame)
        damage_level, dmg_dx, dmg_dy = detect_damage(frame)

        r = -0.02 + (CURI_SCALE + VEL_SCALE) * (diff.mean() / 255)
        target_score   = detect_targets(frame)
        target_present = target_score > 0.02

        if target_score > 0.01:
            r += TURN_R * (abs(dx) + abs(dy))
        if shoot > 0.5 and target_score < 0.02:
            r -= 0.5

        if target_score > 0.02:
            if shoot > 0.5:
                r += 2.0 + 2.0 * target_score + hit_bonus
            else:
                r -= 0.005
        else:
            r += 0.02 * motion_frac

        if shoot > 0.5 and offset and abs(offset[0]) < 0.1 and abs(offset[1]) < 0.1:
            r += ON_CENTER_BONUS * 1.5

        if not target_present:
            r += 0.05 * abs(dx)
            r -= 0.05 * abs(dy)

        if offset is not None:
            up_err = max(0.0, -offset[1])
            down_b = max(0.0, offset[1])
            r -= up_err * 0.5
            r += down_b * 0.2
            if not target_present:
                r -= 0.05 * abs(offset[1])
                if abs(offset[1]) < 0.1:
                    r += 0.01

        if not target_present:
            r += eye_level_bonus(frame)

        if damage_level > DAMAGE_THRESHOLD:
            move_dot = dx * dmg_dx + dy * dmg_dy
            r -= damage_level * 2.0
            r -= DAMAGE_MOVE_WEIGHT * move_dot

        r -= pitch_penalty(frame, target_present)
        r += SURVIVAL_BONUS

        self.prev_frame = frame.copy()
        self.t += 1
        done   = self.t >= 2000
        return frame, float(r), done, False, {
            "dash_count": self.dash_count,
            "health":     self.health,
            "resets":     self.resets,
        }

    def close(self):
        release_all_movement_keys()
        try: sct.close()
        except: pass
        print("Environment closed - keys released")
