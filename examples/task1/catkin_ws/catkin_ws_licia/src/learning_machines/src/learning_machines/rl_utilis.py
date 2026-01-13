

import numpy as np
from robobo_interface import IRobobo
from typing import List, Optional
from collections import deque
import random
from typing import Tuple
import os
from datetime import datetime
from robobo_interface import SimulationRobobo

 # state representation (and normalization)
def extract_state(rob: IRobobo, sensor_max_values: Optional[List[float]] = None) -> np.ndarray:

    ir_values = rob.read_irs()
    
    # Default max values need to do observations to find the max values
    if sensor_max_values is None:
        sensor_max_values = [
            2000.0, 
            2000.0,  
            2000.0,  
            2000.0,  
            50.0,   
            2000.0,  
            2000.0, 
            2000.0,  
        ]
    
    state = []
    for i, val in enumerate(ir_values):
        if val is None:
            state.append(0.0)  
        else:
            normalized = min(val / sensor_max_values[i], 1.0) # Normalize for its own max
            state.append(normalized)
    
    return np.array(state, dtype=np.float32)



def find_sensor_max_values(rob: IRobobo, num_samples: int = 1000, 
                          output_file: Optional[str] = None) -> List[float]:
    
    max_values = [0.0] * 8
    sensor_names = ["BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL"]
    
    if output_file is None:
        results_dir = "/root/results/sensor_values_hardware"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(results_dir, f"sensor_max_values_{timestamp}.txt")

    

    with open(output_file, 'w') as f:

        header = " ".join(f"{name:>10}" for name in sensor_names)
        f.write(header + "\n")

        for i in range(num_samples):
            ir_values = rob.read_irs()
            
            for j, val in enumerate(ir_values):
                if val is not None:
                    max_values[j] = max(max_values[j], val)
            
            row_values = []
            for val in ir_values:
                if val is None:
                    row_values.append("None")
                else:
                    row_values.append(f"{val:10.1f}")
            f.write(" ".join(row_values) + "\n")
        
            
            rob.sleep(0.1)
    return max_values

# Action discrete 
ACTIONS = [
    "FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "BACKWARD",
]

NUM_ACTIONS = len(ACTIONS)  # 6
ACTION_TO_SPEEDS = {
    "FORWARD": (10, 10, 800),           
    "TURN_LEFT": (-10, 10, 100),       
    "TURN_RIGHT": (10, -10, 100),       
    "FORWARD_LEFT": (5, 10, 800),       
    "FORWARD_RIGHT": (10, 5, 800),      
    "BACKWARD": (-10, -10, 800),        
}

# For hardware, turns need different duration, we can think of a more elegant solution
ACTION_TO_SPEEDS_HARDWARE = {
    "FORWARD": (10, 10, 800),
    "TURN_LEFT": (-10, 10, 300),        
    "TURN_RIGHT": (10, -10, 300),       
    "FORWARD_LEFT": (5, 10, 800),
    "FORWARD_RIGHT": (10, 5, 800),
    "BACKWARD": (-10, -10, 800),
}


def execute_action(rob: IRobobo, action: str) -> None:
    is_simulation = isinstance(rob, SimulationRobobo)
    
    if is_simulation:
        left_speed, right_speed, duration_ms = ACTION_TO_SPEEDS[action]
    else:
        left_speed, right_speed, duration_ms = ACTION_TO_SPEEDS_HARDWARE[action]
    
    rob.move_blocking(left_speed, right_speed, duration_ms)


# reward function
# 0.8 is 80% of the max range of the sensors that we normalized before
def detect_collision(state: np.ndarray, threshold: float = 0.8) -> bool: 
    return np.max(state) > threshold


def compute_reward(state: np.ndarray, action: str, collision_detected: bool, 
                   distance_traveled: float = 0.0) -> float:
    reward = 0.0
    
    # if moving forward good, maybe to add other actions!
    if action == "FORWARD":
        reward += 0.1
    
    # more distance more good
    reward += distance_traveled * 0.01

    # if collision, bad
    if collision_detected:
        reward -= 10.0
    
    # im commenting this out for now, we can add it back later, scared that the 
    # sensor will have weird behavior depending on lights etc
    # min_sensor = np.min(state)  # Closest obstacle
    # if min_sensor < 0.2:  # Too close! (20% of max range)
    #     reward -= 0.5
    
    return reward

# replay buffer--> read and copyed from github, explain better why we need it

class ReplayBuffer:

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        return len(self.buffer)