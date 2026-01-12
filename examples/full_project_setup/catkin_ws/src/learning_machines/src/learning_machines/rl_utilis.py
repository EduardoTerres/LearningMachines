

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