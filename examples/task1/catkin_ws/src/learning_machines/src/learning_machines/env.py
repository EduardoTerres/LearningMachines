import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo
from typing import List, Optional

# Minimal RL helpers (extracted and simplified)
ACTIONS = [
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "BACKWARD",
]

NUM_ACTIONS = len(ACTIONS)

MAX_SPEED = 10
TURN_SPEED = 5
ACTION_TO_SPEEDS = {
    "FORWARD": (MAX_SPEED, MAX_SPEED, 800),
    "TURN_LEFT": (-TURN_SPEED, TURN_SPEED, 300),
    "TURN_RIGHT": (TURN_SPEED, -TURN_SPEED, 300),
    "FORWARD_LEFT": (MAX_SPEED / 2, MAX_SPEED, 800),
    "FORWARD_RIGHT": (MAX_SPEED, MAX_SPEED / 2, 800),
    "BACKWARD": (-MAX_SPEED, -MAX_SPEED, 800),
}

def normalize_irs(
    ir_values: List[Optional[float]],
    sensor_min_values: List[float],
    sensor_max_values: List[float],
) -> List[float]:
    """Normalize IR sensor values to [0,1] range."""
    normalized = []
    for i, val in enumerate(ir_values):
        if val is None:
            normalized.append(0.0)
        else:
            norm_val = (val - sensor_min_values[i]) / (sensor_max_values[i] - sensor_min_values[i])
            normalized.append(min(max(norm_val, 0.0), 1.0))
    return normalized


def extract_state(
        rob: IRobobo,
        sensor_min_values: Optional[List[float]] = None,
        sensor_max_values: Optional[List[float]] = None,
    ) -> np.ndarray:
    """Read IRs from `rob` and normalize into 8-element float32 array in [0,1]."""
    ir_values = rob.read_irs()

    state = []
    for i, val in enumerate(ir_values):
        if val is None:
            state.append(0.0)
        else:
            state.append(normalize_irs([val], [0.0], [sensor_max_values[i]])[0])
    return np.array(state, dtype=np.float32)


def execute_action(rob: IRobobo, action_idx: int) -> None:
    """Execute discrete action by index on the robot (blocking)."""
    action = ACTIONS[action_idx]
    left_speed, right_speed, duration_ms = ACTION_TO_SPEEDS[action]
    rob.move_blocking(left_speed, right_speed, duration_ms)


def detect_collision(state: np.ndarray, threshold: float = 0.8) -> bool:
    """Detect collision given normalized state (uses 0-1 scale)."""
    return float(np.max(state)) > threshold


def compute_reward(state: np.ndarray, action_idx: int, collision: bool, distance_traveled: float = 0.0) -> float:
    """Simple reward: small positive for FORWARD, penalty on collision."""
    reward = 0.0
    if ACTIONS[action_idx] == "FORWARD":
        reward += 0.1

    # reward += distance_traveled * 0.01

    if collision:
        reward -= np.max(state) * 10.0
    return float(reward)


class RoboboIREnv(gym.Env):
    """Gymnasium environment wrapping a Robobo IR obstacle-avoidance task.

    Action: Discrete 6 (primitive moves). Observation: 8 normalized IRs.
    """

    metadata = {"render.modes": []}
    sensor_max_values = [80] * 8
    sensor_min_values = [0.0] * 8

    def __init__(self, rob: IRobobo = None):
        self.rob = rob
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self._collision_threshold = 0.8

    def reset(self, *, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        obs = extract_state(self.rob, self.sensor_min_values, self.sensor_max_values)
        return obs, {}

    def step(self, action):
        execute_action(self.rob, int(action))
        # small wait for sensors to update
        # self.rob.sleep(0.1)
        next_state = extract_state(self.rob, self.sensor_min_values, self.sensor_max_values)

        collision = detect_collision(next_state, threshold=self._collision_threshold)
        if collision:
            print("Collision detected!")
            
        reward = compute_reward(next_state, int(action), collision)
        terminated = bool(collision)
        truncated = False
        return next_state, reward, terminated, truncated, {}

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()

def test_env(mode: str = "--simulation", actions: list = None) -> None:
    """Run a predefined action sequence and print rewards to terminal.

    Usage: import and call `test_actions()` or run this file as a script.
    """
    from robobo_interface import SimulationRobobo, HardwareRobobo

    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    else:
        rob = SimulationRobobo()

    env = RoboboIREnv(rob=rob)
    obs, _ = env.reset()

    if actions is None:
        actions = [0, 3, 4, 1, 2, 5]

    print("Running test action sequence:")
    for i, a in enumerate(actions):
        next_obs, reward, done, truncated, info = env.step(int(a))
        print(f"Step {i+1}: action={ACTIONS[int(a)]}  reward={reward:.3f}")
        if done:
            print("Terminated (collision detected)")
            break

    env.close()
