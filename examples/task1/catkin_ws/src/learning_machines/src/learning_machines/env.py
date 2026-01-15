import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robobo_interface import IRobobo, SimulationRobobo
from typing import List, Optional

# Speeds
MAX_SPEED_SIM = 100
TURN_SPEED_SIM = 5

MAX_SPEED_HW = 10
TURN_SPEED_HW = 5

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
    1000.0,  # BL - Back Left
    1000.0,   # BR - Back Right
    9999999999.0,    # FL - Front Left
    9999999999.0,     # FR - Front Right
    1000.0,    # FC - Front Center
    1000.0,   # FRR - Front Right Right
    1000.0,  # BC - Back Center
    1000.0,    # FLL - Front Left Left
]

# Simulation sensor min and max values
SIMULATION_SENSOR_MAX_VALUES = [
    100.0,   # BL - Back Left
    100.0,    # BR - Back Right
    1000000000.0,   # FL - Front Left
    1000000000.0,   # FR - Front Right
    100.0,   # FC - Front Center
    100.0,   # FRR - Front Right Right
    100.0,   # BC - Back Center
    100.0,  # FLL - Front Left Left
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
        """
        Improved reward function using sensor readings and actions.
        
        Reward components:
        1. Collision penalty (strong, fixed)
        2. Proactive obstacle avoidance (based on front sensors)
        3. Action-based rewards (forward progress vs backward)
        4. Safe distance maintenance (reward for keeping distance)
        5. Sensor-based action guidance (turn away from obstacles)
        """
        reward = 0.0
        action = self.actions[int(action_idx)]
        
        # 1. STRONG collision penalty (most important!)
        if collision:
            reward -= 10.0
        
        # 2. Extract sensor groups for analysis
        front_sensors = np.array([state[2], state[3], state[4], state[5], state[7]])  # FL, FR, FC, FRR, FLL
        left_sensors = np.array([state[2], state[7]])  # FL, FLL
        right_sensors = np.array([state[3], state[5]])  # FR, FRR
        center_front = state[4]  # FC
        max_front = np.max(front_sensors)
        max_left = np.max(left_sensors) if len(left_sensors) > 0 else 0.0
        max_right = np.max(right_sensors) if len(right_sensors) > 0 else 0.0
        
        # 3. Proactive obstacle avoidance rewards
        # If obstacle detected in front (threshold: 0.4)
        if max_front > 0.4:
            # Reward turning away from obstacle
            if action == "TURN_LEFT" and max_right > max_left:
                # Turn left when obstacle is more on the right
                reward += 0.8
            elif action == "TURN_RIGHT" and max_left > max_right:
                # Turn right when obstacle is more on the left
                reward += 0.8
            elif action == "TURN_LEFT" or action == "TURN_RIGHT":
                # General turning reward when obstacle ahead
                reward += 0.5
            elif action == "FORWARD":
                # Penalty for going forward into obstacle
                reward -= 0.5
        
        # 4. Center obstacle detection (most critical)
        if center_front > 0.5:
            # Strong obstacle directly ahead
            if action == "FORWARD":
                reward -= 1.0  # Strong penalty
            elif action == "TURN_LEFT" or action == "TURN_RIGHT":
                reward += 1.0  # Strong reward for turning
            elif action == "BACKWARD":
                reward += 0.3  # Backward is okay to avoid collision
        
        # 5. Action-based rewards (with sensor awareness)
        if action == "FORWARD":
            # Reward forward movement, but reduce if obstacles are close
            obstacle_penalty = max_front * 0.8
            if max_front < 0.3:  # Clear path ahead
                reward += 0.5 - obstacle_penalty  # Base reward minus obstacle penalty
            else:
                reward += 0.1 - obstacle_penalty  # Small reward if obstacles present
        
        elif action == "FORWARD_LEFT" or action == "FORWARD_RIGHT":
            # Reward forward-turning actions (good for navigation)
            obstacle_penalty = max_front * 0.5
            if max_front < 0.4:  # Some clearance
                reward += 0.4 - obstacle_penalty
            else:
                reward += 0.1 - obstacle_penalty
        
        elif action == "TURN_LEFT" or action == "TURN_RIGHT":
            # Pure turning: reward if avoiding obstacle, small penalty if path is clear
            if max_front > 0.3:
                reward += 0.3  # Good - avoiding obstacle
            else:
                reward += 0.05  # Small reward - path is clear, turning is okay
        
        elif action == "BACKWARD":
            # Small penalty for backward (unless avoiding collision)
            if max_front > 0.6:
                reward += 0.2  # Backward is okay to avoid collision
            else:
                reward -= 0.1  # Penalty if path is clear
        
        # 6. Safe distance maintenance
        # Reward for maintaining safe distance from obstacles
        if max_front < 0.2:  # Very safe distance
            reward += 0.2  # Bonus for safety
        elif max_front < 0.3:  # Safe distance
            reward += 0.1  # Small bonus
        
        # Penalty for getting too close (even without collision)
        if max_front > 0.6 and not collision:
            reward -= 0.5  # Getting dangerously close
        
        # 7. Side obstacle awareness
        # If obstacle on one side, reward moving away from it
        if max_left > 0.5 and max_right < 0.3:
            # Obstacle on left, clear on right
            if action == "TURN_RIGHT" or action == "FORWARD_RIGHT":
                reward += 0.4  # Good - moving away from obstacle
        
        elif max_right > 0.5 and max_left < 0.3:
            # Obstacle on right, clear on left
            if action == "TURN_LEFT" or action == "FORWARD_LEFT":
                reward += 0.4  # Good - moving away from obstacle
        
        return float(reward)

    def compute_reward_hardware(self, state: np.ndarray, action_idx: int, collision: bool, distance_traveled: float = 0.0) -> float:
        """
        Same improved reward function for hardware (scaled appropriately).
        Hardware rewards are typically smaller to avoid large value issues.
        """
        reward = 0.0
        action = self.actions[int(action_idx)]
        
        # 1. Collision penalty (scaled for hardware)
        if collision:
            reward -= 2.0
        
        # 2. Extract sensor groups
        front_sensors = np.array([state[2], state[3], state[4], state[5], state[7]])
        left_sensors = np.array([state[2], state[7]])
        right_sensors = np.array([state[3], state[5]])
        center_front = state[4]
        max_front = np.max(front_sensors)
        max_left = np.max(left_sensors) if len(left_sensors) > 0 else 0.0
        max_right = np.max(right_sensors) if len(right_sensors) > 0 else 0.0
        
        # 3. Proactive obstacle avoidance (scaled down for hardware)
        if max_front > 0.4:
            if action == "TURN_LEFT" and max_right > max_left:
                reward += 0.08
            elif action == "TURN_RIGHT" and max_left > max_right:
                reward += 0.08
            elif action == "TURN_LEFT" or action == "TURN_RIGHT":
                reward += 0.05
            elif action == "FORWARD":
                reward -= 0.05
        
        # 4. Center obstacle
        if center_front > 0.5:
            if action == "FORWARD":
                reward -= 0.1
            elif action == "TURN_LEFT" or action == "TURN_RIGHT":
                reward += 0.1
            elif action == "BACKWARD":
                reward += 0.03
        
        # 5. Action-based rewards (scaled)
        if action == "FORWARD":
            obstacle_penalty = max_front * 0.08
            if max_front < 0.3:
                reward += 0.05 - obstacle_penalty
            else:
                reward += 0.01 - obstacle_penalty
        
        elif action == "FORWARD_LEFT" or action == "FORWARD_RIGHT":
            obstacle_penalty = max_front * 0.05
            if max_front < 0.4:
                reward += 0.04 - obstacle_penalty
            else:
                reward += 0.01 - obstacle_penalty
        
        elif action == "TURN_LEFT" or action == "TURN_RIGHT":
            if max_front > 0.3:
                reward += 0.03
            else:
                reward += 0.005
        
        elif action == "BACKWARD":
            if max_front > 0.6:
                reward += 0.02
            else:
                reward -= 0.01
        
        # 6. Safe distance
        if max_front < 0.2:
            reward += 0.02
        elif max_front < 0.3:
            reward += 0.01
        
        if max_front > 0.6 and not collision:
            reward -= 0.05
        
        # 7. Side obstacle awareness
        if max_left > 0.5 and max_right < 0.3:
            if action == "TURN_RIGHT" or action == "FORWARD_RIGHT":
                reward += 0.04
        
        elif max_right > 0.5 and max_left < 0.3:
            if action == "TURN_LEFT" or action == "FORWARD_LEFT":
                reward += 0.04
        
        return float(reward)

   
        """Improved reward with strong collision penalty."""
        reward = 0.0
        
        # 1. STRONG, FIXED collision penalty (same as simulation!)
        if collision:
            reward -= 2.0  # Strong, consistent penalty
        
        # Action-based rewards
        if self.actions[int(action_idx)] == "FORWARD":
            reward += 0.01
        elif self.actions[int(action_idx)] == "BACKWARD":
            reward -= 0.005
        
        # Remove the always-on penalty (line 217) - it's confusing
        # reward -= np.sum(state ** 4) * 2.0  # REMOVED

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
