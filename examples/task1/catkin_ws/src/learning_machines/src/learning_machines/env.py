import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo
from typing import List, Optional

# Speeds
MAX_SPEED_SIM = 10
TURN_SPEED_SIM = 5

MAX_SPEED_HW = 40
TURN_SPEED_HW = 20

# Sensor values
# These values were not working properly
# HARDWARE_SENSOR_MAX_VALUES = [
#     23000.0,  # BL - Back Left
#     4500.0,   # BR - Back Right
#     200.0,    # FL - Front Left
#     90.0,     # FR - Front Right
#     175.0,    # FC - Front Center
#     1400.0,   # FRR - Front Right Right
#     15000.0,  # BC - Back Center
#     150.0,    # FLL - Front Left Left
# ]
HARDWARE_SENSOR_MAX_VALUES = [
    300.0,  # BL - Back Left
    300.0,   # BR - Back Right
    9999999999.0,    # FL - Front Left
    9999999999.0,     # FR - Front Right
    300.0,    # FC - Front Center
    300.0,   # FRR - Front Right Right
    300.0,  # BC - Back Center
    300.0,    # FLL - Front Left Left
]

# Simulation sensor min and max values
SIMULATION_SENSOR_MAX_VALUES = [
    300.0,   # BL - Back Left
    300.0,    # BR - Back Right
    9999999999.0,   # FL - Front Left
    9999999999.0,   # FR - Front Right
    300.0,   # FC - Front Center
    300.0,   # FRR - Front Right Right
    300.0,   # BC - Back Center
    300.0,  # FLL - Front Left Left
]

SENSOR_MIN_VALUES = {
    "simulation": [0.0] * 8,
    "hardware": [0.0] * 8,
}
SENSOR_MAX_VALUES = {
    "simulation": SIMULATION_SENSOR_MAX_VALUES,
    "hardware": HARDWARE_SENSOR_MAX_VALUES,
}



class RoboboIREnv(gym.Env):
    """Gymnasium environment wrapping a Robobo IR obstacle-avoidance task.

    Action: Discrete 6 (primitive moves). Observation: 8 normalized IRs.
    """

    metadata = {"render.modes": []}

    def __init__(self, rob: IRobobo = None):
        self.rob = rob
        self.instance = "simulation" if isinstance(rob, SimulationRobobo) else "hardware"
        self._collision_threshold = 0.5
        # Episode length control
        self._max_steps = 10
        self._step_count = 0

        self.sensor_min_values = SENSOR_MIN_VALUES[self.instance]
        self.sensor_max_values = SENSOR_MAX_VALUES[self.instance]

        self.actions = [
            "FORWARD",
            "TURN_LEFT",
            "TURN_RIGHT",
            "FORWARD_LEFT",
            "FORWARD_RIGHT",
            "BACKWARD",
        ]

        self.num_actions = len(self.actions)

        max_speed = MAX_SPEED_SIM if isinstance(self.rob, SimulationRobobo) else MAX_SPEED_HW
        turn_speed = TURN_SPEED_SIM if isinstance(self.rob, SimulationRobobo) else TURN_SPEED_HW

        self.actions_to_speed = {
            "FORWARD": (max_speed, max_speed, 800),
            "TURN_LEFT": (-turn_speed, turn_speed, 300),
            "TURN_RIGHT": (turn_speed, -turn_speed, 300),
            "FORWARD_LEFT": (int(max_speed // 2), max_speed, 800),
            "FORWARD_RIGHT": (max_speed, int(max_speed // 2), 800),
            "BACKWARD": (-max_speed, -max_speed, 800),
        }

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        # reset step counter
        self._step_count = 0
        obs = self.extract_state()
        return obs, {}

    def step(self, action):
        self.execute_action(int(action))
        # small wait for sensors to update
        # self.rob.sleep(0.1)
        next_state = self.extract_state()

        collision = self.detect_collision(next_state)
        if collision:
            print("Collision detected!")
        
        reward_function = (
            self.compute_reward_simulation
            if self.instance == "simulation"
            else self.compute_reward_hardware
        )
        reward = reward_function(next_state, int(action), collision)
        # Do NOT terminate on collision; only truncate (time-limit) after max steps.
        self._step_count += 1
        terminated = False
        truncated = bool(self._step_count >= self._max_steps)
        return next_state, reward, terminated, truncated, {}

    def close(self):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.stop_simulation()

    def normalize_irs(
        self,
        ir_values: List[Optional[float]],
        sensor_min_values: Optional[List[float]] = None,
        sensor_max_values: Optional[List[float]] = None,
    ) -> List[float]:
        """Normalize IR sensor values to [0,1] range."""
        if sensor_min_values is None:
            sensor_min_values = self.sensor_min_values
        if sensor_max_values is None:
            sensor_max_values = self.sensor_max_values
        normalized = []
        for i, val in enumerate(ir_values):
            if val is None:
                normalized.append(0.0)
            else:
                norm_val = (val - sensor_min_values[i]) / (sensor_max_values[i] - sensor_min_values[i])
                normalized.append(min(max(norm_val, 0.0), 1.0))
        return normalized

    def extract_state(
        self,
    ) -> np.ndarray:
        """Read IRs from `rob` and normalize into 8-element float32 array in [0,1]."""
        ir_values = self.rob.read_irs()
        # print(f"IR: {ir_values}")

        state = []
        for i, val in enumerate(ir_values):
            if val is None:
                state.append(0.0)
            else:
                state.append(self.normalize_irs([val], [0.0], [self.sensor_max_values[i]])[0])

        # print(f"IR norm: {state}")
        return np.array(state, dtype=np.float32)

    def execute_action(self, action_idx: int) -> None:
        """Execute discrete action by index on the robot (blocking)."""
        action = self.actions[int(action_idx)]
        left_speed, right_speed, duration_ms = self.actions_to_speed[action]

        # Different move functions for sim vs hardware 
        moving_function = self.rob.move_blocking # if self.instance == "simulation" else self.rob.move
        moving_function(left_speed, right_speed, duration_ms)

    def detect_collision(self, state: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Detect collision given normalized state (uses 0-1 scale)."""
        if threshold is None:
            threshold = self._collision_threshold
        return float(np.max(state)) > threshold

    # def compute_reward(self, state: np.ndarray, action_idx: int, collision: bool) -> float:
    #     """Simple reward: small positive for FORWARD, penalty on collision."""
    #     return np.sum(state ** 2) * ( -10.0 if collision else 0.1)
    
    def compute_reward_simulation(self, state: np.ndarray, action_idx: int, collision: bool, distance_traveled: float = 0.0) -> float:
        """Simple reward: small positive for FORWARD, penalty on collision."""
        reward = 0.0
        if self.actions[int(action_idx)] == "FORWARD":
            reward += 0.5
        elif self.actions[int(action_idx)] == "BACKWARD":
            reward -= 0.5

        if collision:
            reward -= np.max(state)
        reward -= np.sum(state ** 4)
        return float(reward)

    def compute_reward_hardware(self, state: np.ndarray, action_idx: int, collision: bool, distance_traveled: float = 0.0) -> float:
        """Simple reward: small positive for FORWARD, penalty on collision."""
        reward = 0.0
        if self.actions[int(action_idx)] == "FORWARD":
            reward += 0.1
        elif self.actions[int(action_idx)] == "BACKWARD":
            reward -= 0.1
        if collision:
            reward -= np.max(state)
        reward -= np.sum(state ** 4)
        return float(reward)

    @staticmethod
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
            print(f"Step {i+1}: action={env.actions[int(a)]}  reward={reward:.3f}")
            if done:
                print("Terminated (collision detected)")
                break

        env.close()
