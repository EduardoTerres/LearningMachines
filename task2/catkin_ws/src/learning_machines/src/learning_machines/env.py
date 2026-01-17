import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from robobo_interface import IRobobo, SimulationRobobo
from typing import List, Optional, Tuple
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Speeds
MAX_SPEED_SIM = 10
TURN_SPEED_SIM = 5

MAX_SPEED_HW = 40
TURN_SPEED_HW = 20

# Sensor values
# These values were not working properly
HARDWARE_SENSOR_MAX_VALUES = [
    300.0,  # BL - Back Left
    300.0,   # BR - Back Right
    300.0,    # FL - Front Left
    300.0,     # FR - Front Right
    300.0,    # FC - Front Center
    300.0,   # FRR - Front Right Right
    300.0,  # BC - Back Center
    300.0,    # FLL - Front Left Left
]

# Simulation sensor min and max values
SIMULATION_SENSOR_MAX_VALUES = [
    300.0,   # BL - Back Left
    300.0,    # BR - Back Right
    300.0,   # FL - Front Left
    300.0,   # FR - Front Right
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

        # Continuous action space: [left_wheel_speed, right_wheel_speed]
        # Normalized to [-1, 1], will be scaled to actual speeds
        self.max_speed = MAX_SPEED_SIM if isinstance(self.rob, SimulationRobobo) else MAX_SPEED_HW
        self.action_duration = 800  # milliseconds
        
        # Action space: continuous wheel speeds
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 8 IR sensors + 1 green detection + 1 ViT food score
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # Food collection tracking
        self.food_collected_this_episode = 0
        
        # Vision transformer for food detection (lazy loaded)
        self.food_detector = None

    def reset(self, *, seed=None, options=None):
        if isinstance(self.rob, SimulationRobobo):
            self.rob.play_simulation()
        # reset step counter and food counter
        self._step_count = 0
        self.food_collected_this_episode = 0
        obs = self.extract_state()
        return obs, {}

    def step(self, action):
        # Check food before action
        food_before = self.get_food_count()
        
        self.execute_action(action)
        next_state = self.extract_state()
        
        # Check if food was collected
        food_after = self.get_food_count()
        food_collected = food_before - food_after
        if food_collected > 0:
            self.food_collected_this_episode += food_collected
            print(f"Food collected! Total this episode: {self.food_collected_this_episode}")

        collision = self.detect_collision(next_state)
        if collision:
            print("Collision detected!")
        
        reward_function = (
            self.compute_reward_simulation
            if self.instance == "simulation"
            else self.compute_reward_hardware
        )
        reward = reward_function(next_state, action, collision, food_collected)
        
        # Do NOT terminate on collision; only truncate (time-limit) after max steps.
        self._step_count += 1
        terminated = False
        truncated = bool(self._step_count >= self._max_steps)
        return next_state, reward, terminated, truncated, {"food_collected": food_collected}

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

    def detect_green_in_image(self, image: np.ndarray) -> float:
        """Detect green color in image and return normalized value [0,1].
        Higher value means more green (food) detected."""
        if image is None or image.size == 0:
            return 0.0
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for green color in HSV
        # Green hue is around 60 degrees (in OpenCV: 0-179 scale)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        
        # Create mask for green pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Calculate percentage of green pixels
        green_ratio = np.sum(mask > 0) / mask.size
        
        return float(np.clip(green_ratio * 5.0, 0.0, 1.0))  # Scale up and clip

    def detect_food_with_vit(self, image: np.ndarray) -> float:
        """Detect food in image using Vision Transformer (ViT) model.
        
        Uses a pre-trained ViT model to classify whether the image contains
        edible/food items. Returns normalized probability [0,1].
        
        Note: This requires transformers library. Falls back to 0.0 if unavailable.
        """
        if not HAS_TRANSFORMERS or image is None or image.size == 0:
            return 0.0
        
        try:
            # Lazy load the food detector on first call
            if self.food_detector is None:
                # Using zero-shot classification with "food" vs "not food"
                # This is lightweight and doesn't require fine-tuning
                self.food_detector = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32"
                )
            
            # Prepare image for CLIP (convert BGR to RGB)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Candidate labels for classification
            candidate_labels = ["green block", "red block", "obstacle", "wall"]
            
            # Run zero-shot classification
            results = self.food_detector(image_rgb, candidate_labels)
            
            # Extract probability for food-related labels
            food_score = 0.0
            for result in results:
                if result["label"] in ["green block", "red block"]:
                    food_score = max(food_score, result["score"])
            
            return float(np.clip(food_score, 0.0, 1.0))
        except Exception as e:
            # Silently fail and return 0 if model loading or inference fails
            return 0.0

    def extract_state(self) -> np.ndarray:
        """Read IRs and camera, return 10-element state: 8 IRs + green detection + ViT food score."""
        ir_values = self.rob.read_irs()

        state = []
        for i, val in enumerate(ir_values):
            if val is None:
                state.append(0.0)
            else:
                state.append(self.normalize_irs([val], [0.0], [self.sensor_max_values[i]])[0])
        
        # Add green detection from camera
        try:
            image = self.rob.read_image_front()
            green_value = self.detect_green_in_image(image)
            # Add ViT-based food detection
            vit_food_score = self.detect_food_with_vit(image)
        except:
            green_value = 0.0
            vit_food_score = 0.0
        
        state.append(green_value)
        state.append(vit_food_score)
        return np.array(state, dtype=np.float32)

    def execute_action(self, action: np.ndarray) -> None:
        """Execute continuous action (normalized wheel speeds) on the robot.
        action: [left_speed, right_speed] in range [-1, 1]
        """
        # Scale normalized actions to actual speed range
        left_speed = int(action[0] * self.max_speed)
        right_speed = int(action[1] * self.max_speed)
        
        # Execute movement
        self.rob.move_blocking(left_speed, right_speed, self.action_duration)

    def detect_collision(self, state: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Detect collision given normalized state (uses 0-1 scale)."""
        if threshold is None:
            threshold = self._collision_threshold
        # Only check IR sensors (first 8 values), not green detection
        return float(np.max(state[:8])) > threshold
    
    def get_food_count(self) -> int:
        """Get number of food items collected (only works in simulation)."""
        if isinstance(self.rob, SimulationRobobo):
            try:
                return self.rob.get_nr_food_collected()
            except:
                return 0
        return 0

    def compute_reward_simulation(self, state: np.ndarray, action: np.ndarray, 
                                   collision: bool, food_collected: int = 0) -> float:
        """Reward for food collection task.
        - Large positive reward for collecting food
        - Penalty for collision
        - Small reward for forward movement
        - Bonus for approaching green (food)
        """
        reward = 0.0
        
        # Huge reward for collecting food
        if food_collected > 0:
            reward += 10.0 * food_collected
        
        # Penalty for collision
        if collision:
            reward -= 5.0
        
        # Small reward for moving forward (encourage exploration)
        forward_speed = (action[0] + action[1]) / 2.0
        reward += 0.1 * forward_speed
        
        # Bonus for detecting green (approaching food)
        green_value = state[8]  # Last element is green detection
        reward += 0.5 * green_value
        
        # Small penalty for proximity to obstacles (encourage clearance)
        obstacle_penalty = np.mean(state[:8] ** 2)
        reward -= 0.1 * obstacle_penalty
        
        return float(reward)

    def compute_reward_hardware(self, state: np.ndarray, action: np.ndarray, 
                                 collision: bool, food_collected: int = 0) -> float:
        """Hardware reward (food collection not available, focus on green detection)."""
        reward = 0.0
        
        # Penalty for collision
        if collision:
            reward -= 3.0
        
        # Reward for forward movement
        forward_speed = (action[0] + action[1]) / 2.0
        reward += 0.2 * forward_speed
        
        # Reward for detecting green
        green_value = state[8]
        reward += 1.0 * green_value
        
        # Penalty for proximity to obstacles
        obstacle_penalty = np.mean(state[:8] ** 2)
        reward -= 0.2 * obstacle_penalty
        
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
