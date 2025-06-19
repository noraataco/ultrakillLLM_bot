import sys
import ctypes
import win32gui
import win32con
import time
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from ultrakill_env import UltrakillEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from datetime import datetime

log = open("ppo_train.log", "w", buffering=1)
sys.stdout = log
sys.stderr = log

# Lock focus to ULTRAKILL permanently
def lock_ultrakill_focus():
    hwnd = None
    for _ in range(10):  # Try for 5 seconds
        hwnds = []
        win32gui.EnumWindows(lambda h, p: p.append(h) if "ultrakill" in win32gui.GetWindowText(h).lower() else True, hwnds)
        if hwnds:
            hwnd = hwnds[0]
            break
        time.sleep(0.5)
    
    if not hwnd:
        raise RuntimeError("ULTRAKILL window not found")
    
    # Set as topmost and force focus
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    win32gui.SetForegroundWindow(hwnd)
    ctypes.windll.user32.BlockInput(True)  # Block keyboard/mouse input to other windows
    return hwnd

# Lock focus before training
try:
    print("Locked focus to ULTRAKILL window")
    
    print("Creating env...")

    def make_env():
        env = UltrakillEnv(aim_only=True)
        env = RecordEpisodeStatistics(env)
        return env
    # 1) Build a DummyVecEnv that constructs your aim_only env
    vec = DummyVecEnv([make_env])

    # 3) Since UltrakillEnv produces (H,W,C), we need to transpose to (C,H,W):
    vec = VecTransposeImage(vec)

    # 4) Now stack 4 of those channels to get shape (4*C, H, W):
    vec = VecFrameStack(vec, n_stack=4)

    # 5) Pass that to PPO:
    model = PPO("CnnPolicy", vec, n_steps=512, batch_size=256, verbose=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = CheckpointCallback(
        save_freq=200_000,
        save_path="./checkpoints/",
        name_prefix=f"ultrakill_{timestamp}"
    )

    print("Starting training...")
    model.learn(total_timesteps=200_000, callback=checkpoint_callback)
    print("Training complete. Saving...")
    model.save("ppo_ultrakill_curiosity")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Fatal error: {e}")

