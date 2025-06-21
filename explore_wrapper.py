# explore_wrapper.py

import gymnasium as gym
import numpy as np


class EpsilonGreedy(gym.Wrapper):
    """Epsilon-greedy exploration wrapper.

    With probability ``eps`` choose a random action instead of the policy's.
    Works for any ``Discrete`` action space (the ULTRAKILL environment has
    10 actions).
    """

    def __init__(self, env: gym.Env, eps: float = 0.2):
        super().__init__(env)
        self.eps = eps
        assert isinstance(env.action_space, gym.spaces.Discrete)

    def step(self, action):
        if np.random.rand() < self.eps:
            # 40 % of random picks are a trigger pull
            if np.random.rand() < 0.4:
                action = np.random.choice([
                    self.env.action_space.index("SHOOT"),
                    self.env.action_space.index("ALT_FIRE"),
                ])
            else:
                action = self.action_space.sample()
        return self.env.step(action)


