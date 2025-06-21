# explore_wrapper.py

import gymnasium as gym
import numpy as np


class EpsilonGreedy(gym.Wrapper):
    """Simple epsilon-greedy exploration wrapper for discrete actions."""

    def __init__(self, env: gym.Env, eps: float = 0.2) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            "EpsilonGreedy requires a discrete action space"
        )
        super().__init__(env)
        self.eps = eps

    def step(self, action):
        if np.random.rand() < self.eps:
            # 40% of random picks are a trigger pull when indices are available
            if np.random.rand() < 0.4 and hasattr(self.env.action_space, "index"):
                shoot = self.env.action_space.index("SHOOT")
                alt   = self.env.action_space.index("ALT_FIRE")
                action = np.random.choice([shoot, alt])
            else:
                action = self.env.action_space.sample()
        return self.env.step(action)

