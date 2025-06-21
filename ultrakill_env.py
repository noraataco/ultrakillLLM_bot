# ultrakill_env.py  –  exploration-friendly wrapper for ULTRAKILL
# --------------------------------------------------------------
import time, os, ctypes
import mss, cv2, numpy as np, win32gui
import gymnasium as gym
from gymnasium.spaces import Box
from utils import lock_ultrakill_focus
from ultrakill_ai import soft_reset, send_scan, SCAN, mouse_click
from typing import Tuple, Optional
import ctypes.wintypes as wintypes

# Ensure output directory exists
os.makedirs("debug_frames", exist_ok=True)
user32 = ctypes.windll.user32

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

MOUSEEVENTF_MOVE     = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP   = 0x0004

STEP_DELAY = 0.02

# Utility functions
# Replace the existing pitch_penalty function with:
def pitch_penalty(frame: np.ndarray, target_present: bool) -> float:
    """
    Penalize looking at ground/sky, but allow when enemies are present
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    top = gray[:20].mean()
    bottom = gray[-20:].mean()
    
    # Only apply penalty when no enemies are visible
    if not target_present:
        if bottom - top < -15:  # Looking down at ground
            return 0.3
        elif top - bottom > 15:  # Looking up at sky
            return 0.2
    return 0.0

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
            lambda h,p: p.append(h) if "ultrakill" in win32gui.GetWindowText(h).lower() else True,
            wins
        )
        if not wins:
            img = np.zeros((360,640,3), np.uint8)
        else:
            hwnd = wins[0]
            left,top,right,bot = win32gui.GetClientRect(hwnd)
            x,y = win32gui.ClientToScreen(hwnd,(0,0))
            monitor = {"top":y, "left":x, "width":right, "height":bot}
            img = np.array(mss.mss().grab(monitor))
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
    """Reward for keeping crosshair near eye-level (center of screen)"""
    h, w = frame.shape[:2]
    center_y = h // 2
    # Check a horizontal strip around eye-level (20% of screen height)
    eye_zone = frame[center_y-30:center_y+30, :]
    return 0.1 * (eye_zone.mean() > 100)  # Reward if not looking at dark ground

# Scoreboard detection helper
def is_score_screen(frame: np.ndarray) -> bool:
    """Return True if the frame looks like the ULTRAKILL scoreboard."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < 40 and gray.std() < 15

# Vision helpers
def red_center_bonus(rgb: np.ndarray) -> float:
    h,w = rgb.shape[:2]
    c = rgb[h//2-20:h//2+20, w//2-20:w//2+20]
    r,g,b = c[...,2], c[...,1], c[...,0]
    return 0.4 * ((r>150)&(g<90)&(b<90)).mean()

def detect_target_offset(frame: np.ndarray) -> Optional[Tuple[float,float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (0,120,70), (10,255,255)),
        cv2.inRange(hsv, (170,120,70), (180,255,255)),
        cv2.inRange(hsv, (100,120,70), (130,255,255)),
        cv2.inRange(hsv, (15,100,100), (40,255,255)),
    ]
    mask = masks[0]
    for m in masks[1:]:
        mask |= m
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < 50:
        return None
    M = cv2.moments(best)
    cx,cy = M['m10']/M['m00'], M['m01']/M['m00']
    h,w = frame.shape[:2]
    return ((cx-w/2)/(w/2), (cy-h/2)/(h/2))

def detect_targets(frame: np.ndarray) -> float:
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
    return combined.mean()


class UltrakillEnv(gym.Env):
    """
    Wrapper that walks forward on reset, then hands off to the agent for aiming/shooting.
    """
    WARMUP_TIME  = 4.0    # seconds
    FRAME_DELAY  = STEP_DELAY

    def __init__(self, *, aim_only: bool=False):
        super().__init__()
        self.aim_only = aim_only

        # **here**: explicitly pass the shape tuple
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(360, 640, 3),
            dtype=np.uint8,
        )

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1,  1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.prev_frame   = None
        self.prev_offset  = None
        self.t            = 0
        self.episode_id   = 0
        self.in_warmup    = False

    def reset(self, *, seed=None, options=None):
        self.episode_id += 1
        self.t = 0
        release_all_movement_keys()
        time.sleep(0.5)
        lock_ultrakill_focus()
        time.sleep(0.2)
        
        # Wait briefly for the scoreboard to appear after death
        time.sleep(2.0)

        attempts = 0
        while True:
            frame = grab_frame()
            if not is_score_screen(frame):
                break
                
            # For first episode: use soft reset (Esc→Enter)
            if self.episode_id == 1:
                soft_reset()
            # For subsequent episodes: press JUMP with proper timing
            else:
                # Press JUMP with increasing delays between attempts
                send_scan(SCAN["JUMP"])
                send_scan(SCAN["JUMP"], True)
                time.sleep(1.0 + attempts * 0.5)  # 1s, 1.5s, 2s...
            
            attempts += 1
            if attempts > 5:  # Safety break
                soft_reset()  # Fallback to full reset
                time.sleep(3.0)
                break
        
        # Make extra sure no keys are stuck before we walk in
        release_all_movement_keys()
        # Start walking in
        send_scan(SCAN["MOVE_FORWARD"])
        self._spawn_time = time.time()
        self.in_warmup = True
        
        # Get initial observation
        frame = grab_frame()
        self.prev_frame = frame.copy()
        return frame, {}

    def step(self, action):
        elapsed = time.time() - self._spawn_time
        if self.in_warmup:
            if elapsed < self.WARMUP_TIME:
                time.sleep(self.FRAME_DELAY)
                frame = grab_frame()
                self.prev_frame = frame.copy()
                self.t += 1
                return frame, 0.0, False, False, {}
            else:
                send_scan(SCAN["MOVE_FORWARD"], True)
                time.sleep(0.05)

                self.in_warmup = False

        # from here on, normal unpack/action/reward logic…
        dx_move, dy_move, shoot_p = map(float, action)
        dx_move, dy_move          = np.clip([dx_move, dy_move], -1, 1)

        # 2) Full-body locomotion if not aim-only
        if not self.aim_only:
            # forward/back
            if   dx_move >  0.1:
                send_scan(SCAN["MOVE_FORWARD"])
            elif dx_move < -0.1:
                send_scan(SCAN["MOVE_BACK"], True)
            else:
                send_scan(SCAN["MOVE_FORWARD"], True)
                send_scan(SCAN["MOVE_BACK"],    True)

            # strafe
            if   dy_move >  0.1:
                send_scan(SCAN["MOVE_RIGHT"])
            elif dy_move < -0.1:
                send_scan(SCAN["MOVE_LEFT"], True)
            else:
                send_scan(SCAN["MOVE_RIGHT"], True)
                send_scan(SCAN["MOVE_LEFT"],  True)

        # 3) Always turn camera
        user32.mouse_event(
            MOUSEEVENTF_MOVE,
            int(dx_move * TURN_PIXELS),
            int(dy_move * TURN_PIXELS),
            0, 0
        )

        # 4) Shooting
        if shoot_p > 0.5:
            mouse_click()

        # 6) Normal frame + reward
        time.sleep(self.FRAME_DELAY)
        frame = grab_frame()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_score_screen(frame) or gray_img.mean() < 12 or gray_img.mean() > 240:
            release_all_movement_keys()
            return frame, -50.0, True, False, {}
        gray = gray_img.mean()

        target_score = detect_targets(frame)
        target_present = target_score > 0.02

        # ——— Reward shaping ———
        diff         = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        motion_frac  = (diff > 15).mean()
        target_score = detect_targets(frame)
        hit_bonus    = red_center_bonus(frame) * HIT_BONUS
        offset       = detect_target_offset(frame)
        # base reward: time penalty + curiosity/velocity bonus
        r = -0.02 + (CURI_SCALE + VEL_SCALE) * (diff.mean() / 255)

        # only reward turning when there’s actually something to turn toward
        if target_score > 0.01:
            r += TURN_R * (abs(dx_move) + abs(dy_move))

        # penalize blind firing
        if shoot_p > 0.5 and target_score < 0.02:
            r -= 0.5

        # target engagement
        if target_score > 0.02:
            if shoot_p > 0.5:
                r += 2.0 + 2.0 * target_score + hit_bonus
            else:
                r -= 0.005
        else:
            r += 0.02 * motion_frac

        # on-center bonus
        if shoot_p > 0.5 and offset and abs(offset[0]) < 0.1 and abs(offset[1]) < 0.1:
            r += ON_CENTER_BONUS * 1.5

        # Vertical aim adjustments
        if offset is not None:
            up_error   = max(0.0, -offset[1])
            down_bonus = max(0.0, offset[1])
            r -= up_error * 0.5            # harsh penalty for looking up
            r += down_bonus * 0.2          # mild bonus for looking down

            if not target_present:
                # Penalize deviation from center but reward slight steadiness
                r -= 0.05 * abs(offset[1])
                if abs(offset[1]) < 0.1:
                    r += 0.01

        if not target_present:
            r += eye_level_bonus(frame)

        # keep your old sky/ceiling penalty too (you can scale it up):
        r -= pitch_penalty(frame, target_present)    # make that penalty twice as harsh

        # 7) Finalize
        self.prev_frame = frame.copy()
        self.t += 1
        done = self.t >= 2000
        return frame, float(r), done, False, {}

    def close(self):
        release_all_movement_keys()
        print("Environment closed - keys released")

