import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo


class RoboboIREnv(gym.Env):
    """Simple env: action = two continuous times (forward_seconds, spin_seconds).
    The larger value determines the executed primitive. Obs = 8 IRs.
    Rewards are negative; collision gives a large negative penalty.
    """

    def __init__(self, rob: IRobobo = None, base_reward: float = -0.1, collision_penalty: float = -50.0, max_action_time: float = 1.0):
        self.rob = rob
        self.base_reward = float(base_reward)
        self.collision_penalty = float(collision_penalty)
        self.max_action_time = float(max_action_time)

        self.action_space = spaces.Box(low=0.0, high=self.max_action_time, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(8,), dtype=np.float32)

        self.collision_threshold = 80.0
        self._forward_speed = 20
        self._turn_speed = 20

    def reset(self, *, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        obs = np.zeros(8, dtype=np.float32)
        return obs, {}

    def step(self, action):
        f = float(action[0])
        s = float(action[1])
        # execute the primitive with larger requested time
        if f >= s:
            ms = int(1000 * min(f, self.max_action_time))
            if self.rob:
                self.rob.move_blocking(self._forward_speed, self._forward_speed, ms)
        else:
            ms = int(1000 * min(s, self.max_action_time))
            if self.rob:
                self.rob.move_blocking(-self._turn_speed, self._turn_speed, ms)

        irs = self.rob.read_irs()

        obs = np.array([v or 0.0 for v in irs], dtype=np.float32)
        reward = float(self.base_reward)
        terminated = False
        if obs.max() > self.collision_threshold:
            reward += float(self.collision_penalty)
            terminated = True

        return obs, reward, terminated, False, {}

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()
