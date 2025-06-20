# ultrakill_env.py  –  exploration-friendly wrapper for ULTRAKILL
# --------------------------------------------------------------
import time, os, ctypes
import mss, cv2, numpy as np, win32gui
import gymnasium as gym
from gymnasium.spaces import Box
from utils import lock_ultrakill_focus
from ultrakill_ai import send_scan, SCAN, mouse_click
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

# Utility functions
def pitch_penalty(frame: np.ndarray) -> float:
    return 0.1 if (frame[-20:].mean() - frame[:20].mean()) < -15 else 0.0

def release_all_movement_keys():
    for key in ["MOVE_FORWARD","MOVE_BACK","MOVE_LEFT","MOVE_RIGHT"]:
        send_scan(SCAN[key], True)
        time.sleep(0.01)

# Screen capture
def grab_frame() -> np.ndarray:
    try:
        wins = []
        win32gui.EnumWindows(lambda h,p: p.append(h) if "ultrakill" in win32gui.GetWindowText(h).lower() else True, wins)
        if not wins:
            return np.zeros((360,640,3), np.uint8)
        hwnd = wins[0]
        left,top,right,bot = win32gui.GetClientRect(hwnd)
        x,y = win32gui.ClientToScreen(hwnd,(0,0))
        monitor = {"top":y, "left":x, "width":right, "height":bot}
        img = np.array(mss.mss().grab(monitor))
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.resize(img, (640,360))
    except Exception:
        return np.zeros((360,640,3), np.uint8)

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
    WARMUP_TIME = 4.0
    FRAME_DELAY = 0.02

    def __init__(self, *, aim_only: bool=False):
        super().__init__()
        self.aim_only = aim_only
        self.action_space = gym.spaces.Box(
            low=np.array([-1,-1,0], dtype=np.float32),
            high=np.array([1,1,1], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = Box(0,255,(360,640,3), dtype=np.uint8)
        self.prev_frame = None
        self.prev_offset = None
        self.t = 0
        self.episode_id = 0
        self.in_warmup = False

    def reset(self, *, seed=None, options=None):
        # skip warmup on the very first reset
        first = (self.episode_id == 0)
        print(f"\n=== RESET CALLED (first={first}) ===", flush=True)
        release_all_movement_keys()
        time.sleep(0.5)
        lock_ultrakill_focus()
        time.sleep(0.2)

        if not first:
            # only do walk-in after initial load
            print(f"→ BLOCKING WALK-IN: holding FORWARD for {self.WARMUP_TIME}s …", flush=True)
            send_scan(SCAN["MOVE_FORWARD"])  # key-down
            self._warmup_start = time.time()
            self.in_warmup = True
        else:
            self.in_warmup = False

        frame = grab_frame()
        self.prev_frame = frame.copy()
        self.prev_offset = None
        self.t = 0
        self.episode_id += 1
        print(f"=== RESET DONE (episode {self.episode_id}) ===", flush=True)
        return frame, {}

    def step(self, action: np.ndarray):
        if self.in_warmup:
            if time.time() - self._warmup_start >= self.WARMUP_TIME:
                send_scan(SCAN["MOVE_FORWARD"], True)
                self.in_warmup = False
            time.sleep(self.FRAME_DELAY)
            frame = grab_frame()
            self.prev_frame = frame.copy()
            self.t += 1
            return frame, 0.0, False, False, {}

        dx,dy,shoot_p = map(float, action)
        dx,dy = np.clip([dx,dy], -1,1)
        user32.mouse_event(0x0001, int(dx*TURN_PIXELS), int(dy*TURN_PIXELS), 0,0)
        if shoot_p > 0.5:
            mouse_click()
        time.sleep(self.FRAME_DELAY)
        frame = grab_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        if gray < 12 or gray > 240:
            release_all_movement_keys()
            return frame, -50.0, True, False, {}

        diff = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        motion_frac = (diff > 15).mean()
        target_score = detect_targets(frame)
        hit_bonus = red_center_bonus(frame) * HIT_BONUS
        offset = detect_target_offset(frame)

        r = -0.02 + (CURI_SCALE+VEL_SCALE)*(diff.mean()/255)
        if target_score>0.01:
            r += TURN_R*(abs(dx)+abs(dy))
        if shoot_p>0.5 and target_score<0.02:
            r -= 0.5
        if target_score>0.02:
            r += (2.0+2.0*target_score+hit_bonus) if shoot_p>0.5 else -0.005
        else:
            r += 0.02*motion_frac
        if shoot_p>0.5 and offset and abs(offset[0])<0.1 and abs(offset[1])<0.1:
            r += ON_CENTER_BONUS*1.5
        if offset:
            vert_err = max(0.0, abs(offset[1]) - 0.5)
            r -= vert_err*0.2
        r -= pitch_penalty(frame)

        self.prev_frame = frame.copy()
        self.t += 1
        done = self.t >= 2000
        return frame, float(r), done, False, {}

    def close(self):
        release_all_movement_keys()
        print("Environment closed - keys released")

# Smoke test
if __name__ == "__main__":
    for i in range(5):
        frame = grab_frame()
        print(frame.shape)
        cv2.imwrite(f"test_frame_{i}.png", frame)
