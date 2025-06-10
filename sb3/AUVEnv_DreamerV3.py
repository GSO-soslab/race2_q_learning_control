# AUVEnv_DreamerV3.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AUVEnvDreamerV3(gym.Env):
    """
    A simplified Gym wrapper for the AUV environment, providing only the necessary
    metadata (observation and action spaces) for DreamerV3's offline training.
    The step() and reset() methods are placeholders and will not be called
    by the offline training loop.
    """
    def __init__(self, **kwargs):
        super().__init__()
        
        # Define action space. This MUST match the action vector in your .npz files.
        # Based on the processing script, this is 6D.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Define observation space. This MUST match the observation vector in your .npz files.
        # Your env has shape (17,). Let's be explicit here.
        # 3 (pos/vel err) + 2*3 (ori err sin/cos) + 3 (vel) + 3 (ang_rate) + 2 (accel) = 17
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        print("AUVEnvDreamerV3 initialized for metadata.")
        print(f"Action Space: {self.action_space}")
        print(f"Observation Space: {self.observation_space}")

    def step(self, action):
        # This will not be called during offline training
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        # This will not be called during offline training
        return self.observation_space.sample(), {}

    def close(self):
        pass

# Factory function that DreamerV3 will look for
def auv_control(**kwargs):
    return AUVEnvDreamerV3(**kwargs)