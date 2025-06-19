# ultrakill_env.py  –  exploration-friendly wrapper for ULTRAKILL
# --------------------------------------------------------------
import ctypes, time, logging, os
from typing import Tuple, Optional

import numpy as np
import cv2
import mss
import gymnasium as gym
from gymnasium.spaces import Box
from ultrakill_ai import tap, send_scan, SCAN, mouse_click

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


# ------------------------- screen capture ---------------------------
_sct = mss.mss()
def grab_frame() -> np.ndarray:
    """Grab full virtual screen and resize to 360×640."""
    try:
        img = np.array(_sct.grab(_sct.monitors[0]))[:, :, :3]
        return cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
    except Exception:
        logging.warning("grab_frame failed")
        return np.zeros((360,640,3), np.uint8)


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
    """
    Continuous controller:
        action[0] ∈ [-1,1] → mouse Δx  (–1 = hard-left,  +1 = hard-right)
        action[1] ∈ [-1,1] → mouse Δy  (–1 = up,         +1 = down)
        action[2] ∈ [0,1]  → shoot probability (> 0.5 fires this frame)
    """
    metadata = {"render_modes": []}
    WARMUP_STEPS = 20

    def __init__(self, *, aim_only: bool = False):
        super().__init__()
        self.aim_only = aim_only
        # continuous 3-D action: dx_norm, dy_norm (−1..1), shoot_prob (0..1)
        self.action_space      = gym.spaces.Box(
            low  = np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high = np.array([ 1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = Box(
            low=0, high=255, shape=(360, 640, 3), dtype=np.uint8
        )

        self.prev_frame  = np.zeros((360,640,3),np.uint8)
        self.prev_offset = None
        self.stale_cnt   = 0
        self.t           = 0
        self.warmup_cnt  = 0   # set in reset()

    # ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        self.episode_id = int(time.time())
        os.makedirs(f"debug_frames/ep_{self.episode_id}", exist_ok=True)

        send_scan(SCAN["MOVE_FORWARD"], True)   # release W
        tap("JUMP",0.05)                        # quick respawn tap
        time.sleep(1.0)

        self.prev_frame  = grab_frame()
        self.prev_offset = None
        self.stale_cnt   = 0
        self.t           = 0
        self.warmup_cnt  = self.WARMUP_STEPS

        return self.prev_frame.copy(), {}
    
    def _dump_debug(self, frame: np.ndarray):
        """Save this frame to disk with overlays."""
        debug = frame.copy()
        h, w = frame.shape[:2]
        size = 40
        y0, y1 = (h - size)//2, (h + size)//2
        x0, x1 = (w - size)//2, (w + size)//2
        cv2.rectangle(debug, (x0,y0), (x1,y1), (0,255,0), 2)
        os.makedirs(f"debug_frames/ep_{self.episode_id}", exist_ok=True)
        cv2.imwrite(f"debug_frames/ep_{self.episode_id}/frame_{self.t:05d}.png", debug)


    # ---------------------------------------------------------------
    def step(self, action: np.ndarray):
        done = truncated = False

        # ---------- warm-up hold forward ----------
        if self.warmup_cnt > 0:
            if self.warmup_cnt == self.WARMUP_STEPS:
                send_scan(SCAN["MOVE_FORWARD"])          # key-down
            self.warmup_cnt -= 1
            if self.warmup_cnt == 0:
                send_scan(SCAN["MOVE_FORWARD"], True)    # key-up
            time.sleep(0.02)
            frame = grab_frame()
            #self._dump_debug(frame)
            # ----- DEBUG DUMP -----
            debug_frame = frame.copy()
            if hasattr(self, "episode_id") and self.t < 500:
                # draw your box / crosshair if you like:
                h, w = frame.shape[:2]
                size = 40
                y0, y1 = (h - size)//2, (h + size)//2
                x0, x1 = (w - size)//2, (w + size)//2
                cv2.rectangle(debug_frame, (x0,y0), (x1,y1), (0,255,0), 2)
                # write the file
                #os.makedirs(f"debug_frames/ep_{self.episode_id}", exist_ok=True)
                #cv2.imwrite(
                #    f"debug_frames/ep_{self.episode_id}/frame_{self.t:05d}.png",
                #    debug_frame
                #)
            self.prev_frame = frame
            self.t += 1
            if frame.mean() < 5: done = True
            return frame, 0.0, done, truncated, {}

        # ---------- unpack continuous action ----------
        dx_n, dy_n, shoot_p = map(float, action)
        dx_n, dy_n = np.clip([dx_n, dy_n], -1, 1)
        shoot_p    = np.clip(shoot_p, 0, 1)

        # mouse motion & shooting
        user32.mouse_event(0x0001, int(dx_n*TURN_PIXELS), int(dy_n*TURN_PIXELS), 0, 0)
        if shoot_p > 0.5:
            mouse_click()

        time.sleep(0.02)
        frame = grab_frame()
        #self._dump_debug(frame)

        # ---------- death / score screen ----------
        if frame.mean() < 5 or (frame[69:].mean(axis=(1,2)) < 25).mean() > 0.6:
            return frame, -50.0, True, truncated, {}

        # ---------- vision & motion metrics ----------
        diff   = np.abs(frame.astype(np.int16) - self.prev_frame.astype(np.int16))
        motion_frac = (diff > 15).mean()

        target_score = detect_targets(frame)
        hit_bonus    = red_center_bonus(frame) * HIT_BONUS
        offset       = detect_target_offset(frame)

        # ---------- reward shaping ----------
        r  = -0.02
        r += TURN_R * (abs(dx_n)+abs(dy_n))
        r += (CURI_SCALE+VEL_SCALE) * (diff.mean()/255)

        # aim delta-error
        if self.prev_offset is not None and offset is not None:
            prev_err = abs(self.prev_offset[0]) + abs(self.prev_offset[1])
            curr_err = abs(offset[0]) + abs(offset[1])
            r += DELTA_ERR_SCALE * (prev_err - curr_err)
        self.prev_offset = offset

        fired = shoot_p > 0.5
        if target_score > 0.02:
            if fired:
                # much bigger shooting bonus
                r += 2.0 + 2.0 * target_score + hit_bonus
            else:
                r -= 0.005
        else:
            if fired:
                r -= 0.05
            else:
                r += 0.02 * motion_frac


        if fired and offset and abs(offset[0])<0.1 and abs(offset[1])<0.1:
            r += ON_CENTER_BONUS

        r -= pitch_penalty(frame)

        self.stale_cnt = self.stale_cnt+1 if (diff.mean()/255)<PIXEL_THRESH else 0
        if self.stale_cnt >= STALE_LIMIT:
            r += STALE_PENALTY
            done = True

        # quick debug for first 100 steps
        if self.t < 100:
            print(f"t={self.t:03d} score={target_score:.2f} offset={offset} r={r:.3f}")

        # ---------- book-keeping ----------
        self.prev_frame = frame
        self.t += 1
        if self.t >= 2000:
            done = True

        return frame, float(r), done, truncated, {}


# -------------------------- quick smoke-test -------------------------
if __name__ == "__main__":
    env = UltrakillEnv()
    obs,_ = env.reset()
    print("reset OK, initial mean", obs.mean())
    for _ in range(5):
        print(env.step(env.action_space.sample()))
