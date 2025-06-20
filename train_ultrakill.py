# train_ultrakill.py (corrected)

import os
import sys
import time
import msvcrt
from datetime import datetime
from gymnasium import ObservationWrapper
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from ppo_episodes import PPOEpisodes
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import BaseCallback
from utils import lock_ultrakill_focus, start_esc_watcher, start_pause_watcher, PAUSED
from ultrakill_env import UltrakillEnv, release_all_movement_keys
import threading

start_esc_watcher()
start_pause_watcher(on_pause=release_all_movement_keys)

# pick GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────── immediate-Esc callback ────────────
class EscCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.stop_flag = False

    def _on_step(self) -> bool:
        # if Esc hit, set flag and tell SB3 to stop learn()
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
            print("ESC pressed → stopping training immediately")
            self.stop_flag = True
            return False
        return True

class ChannelValidator(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.expected_channels = 3
        
    def observation(self, observation):
        if observation.shape[-1] != self.expected_channels:
            print(f"Invalid channel count: {observation.shape} - forcing to 3 channels")
            if observation.shape[-1] == 4:
                return observation[..., :3]  # Drop alpha channel
            elif observation.shape[-1] == 1:  # Grayscale
                return np.repeat(observation, 3, axis=-1)
            else:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)
        return observation

class Downsample(gym.ObservationWrapper):
    def __init__(self, env, ratio=2):
        super().__init__(env)
        self.ratio = ratio
        orig_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(orig_space.shape[0]//ratio, orig_space.shape[1]//ratio, orig_space.shape[2]),
            dtype=orig_space.dtype
        )
        
    def observation(self, observation):
        return observation[::self.ratio, ::self.ratio, :]

def make_env():
    env = UltrakillEnv(aim_only=False)
    env = ChannelValidator(env)
    # now downsample by 2 → halves both height and width
    env = Downsample(env, ratio=2)
    return Monitor(env)

def _esc_watcher():
    while True:
        if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
            print("ESC pressed → exiting process immediately")
            os._exit(0)
        time.sleep(0.01)

# In train_ultrakill.py

def main():
    # First focus the window BEFORE redirecting logs
    print("Locking focus to ULTRAKILL window")
    lock_ultrakill_focus()
    threading.Thread(target=_esc_watcher, daemon=True).start()
    # Now redirect logs
    log = open("ppo_train.log", "w", buffering=1, encoding="utf-8")
    sys.stdout = log
    sys.stderr = log

    print("Building vectorized environments")
    vec = DummyVecEnv([make_env])
    
    # Test observation shape
    test_obs = vec.reset()
    print("Test observation shape:", test_obs.shape)
    if test_obs.shape[-1] != 3:
        print(f"Error: Expected 3 channels, got {test_obs.shape[-1]}")
        return
    # Proper frame stacking for color images
    vec = VecTransposeImage(vec)  # (H,W,C) -> (C,H,W)
    print("Observation space before stacking:", vec.observation_space)
    vec = VecFrameStack(vec, n_stack=4, channels_order='first')  # Stack along channel dim

    # Observation space adjustments
    print("Observation space shape:", vec.observation_space.shape)
    vec.observation_space.sample = lambda: np.zeros(
        vec.observation_space.shape, dtype=vec.observation_space.dtype
    )
    # enforce image bounds
    vec.observation_space.low = 0
    vec.observation_space.high = 255

    print("Instantiating PPO agent")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = PPOEpisodes(
        "CnnPolicy",
        vec,
        # collect experience from 250 lives before each update
        n_steps=256,
        n_episodes=250,
        # use smaller minibatches
        batch_size=64,
        verbose=1,
        device=device,
        tensorboard_log="./tb_logs",
        policy_kwargs={
            "normalize_images": False
        }
    )

    # checkpoint every 200k global steps
    checkpoint_freq = 20_000 // vec.num_envs
    cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path="./checkpoints/",
        name_prefix=f"ppo_ultrakill_{timestamp}"
    )

    print("Starting training (2000000 timesteps, checkpoint every 200000 steps)")
    print("Press ESC to stop training and save model")
    print("Initial warmup period...")
    time.sleep(2.0)  # Extra time for game to stabilize
    
    # Windows-specific key detection
    def is_esc_pressed():
        return msvcrt.kbhit() and msvcrt.getch() == b'\x1b'
    
    # single-shot training for 250 full episodes before saving
    EPISODE_STEPS = 2000  # max steps per episode in UltrakillEnv
    target_timesteps = 250 * EPISODE_STEPS
    print(f"Training for {target_timesteps} timesteps (~250 episodes)...")
    # Wrap learn() in try/except/finally so we always save & clean up
    try:
        # We still want our EscCallback here:
        esc_cb = EscCallback()
        model.learn(
            total_timesteps=target_timesteps,
            callback=[cb, esc_cb],
            reset_num_timesteps=False
        )
    except Exception as e:
        print(f"Training error: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        # Clean up resources
        print("Saving final model...", flush=True)
        # detach env & buffers so we don't hit pickle issues
        model.env = None
        model._last_obs = None
        model.rollout_buffer = None
        model.save(
            f"ppo_ultrakill_final_{timestamp}",
            exclude=["env", "policy.optimizer"]  # Skip problematic objects
        )

        print("Closing environment and releasing keys...", flush=True)
        vec.close()
        release_all_movement_keys()

        print("Training stopped safely", flush=True)

if __name__ == "__main__":
    main()
