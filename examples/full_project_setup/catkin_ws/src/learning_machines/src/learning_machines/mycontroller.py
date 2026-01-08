from robobo_interface import IRobobo, SimulationRobobo

sensors = {
    "front_left": [],
    "front_right": [],
    "front_center": [],
    "front_right_right": [],
    "front_left_left": [],
    "back_left": [],
    "back_right": [],
    "back_center": [],
}

def read_and_log_irs(rob: IRobobo) -> None:
    """
    Read the IR sensors and log the values.
    """
    ir_values = rob.read_irs()

    front_left = ir_values[2] or 0.0  
    front_right = ir_values[3] or 0.0
    front_center = ir_values[4] or 0.0
    front_right_right = ir_values[5] or 0.0
    front_left_left = ir_values[7] or 0.0
    
    back_left = ir_values[0] or 0.0
    back_right = ir_values[1] or 0.0
    back_center = ir_values[6] or 0.0

    # Log sensor values
    sensors["front_left"].append(front_left)
    sensors["front_right"].append(front_right)
    sensors["front_center"].append(front_center)
    sensors["front_right_right"].append(front_right_right)
    sensors["front_left_left"].append(front_left_left)
    sensors["back_left"].append(back_left)
    sensors["back_right"].append(back_right)
    sensors["back_center"].append(back_center)

    return (
        front_left,
        front_right,
        front_center,
        front_right_right,
        front_left_left,
        back_left,
        back_right,
        back_center,
    )

def find_object_and_turnR(rob: IRobobo) -> None:
    """
    Example 1: Robot goes straight until it senses an object getting near,
    then turns right without touching it.
    Uses front AND back sensors for complete awareness.
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    DETECTION_THRESHOLD = 80.0
    FORWARD_SPEED = 10
    TURN_SPEED = 5

    # Phase 1: forward
    while True:
        (
            front_left,
            front_right,
            front_center,
            front_right_right,
            front_left_left,
            back_left,
            back_right,
            back_center,
        ) = read_and_log_irs(rob)
        
        
        front_sensors = [front_left, front_right, front_center, front_right_right, front_left_left]
        front_max = max(front_sensors)
        
        back_sensors = [back_left, back_right, back_center]
        back_max = max(back_sensors)
        
        # Stop if object is detected
        if front_max > DETECTION_THRESHOLD:
            break

        rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 800)
        rob.talk("Moving forward")

    # Phase 2: turn right in 20 steps
    for _ in range(20):
        read_and_log_irs(rob)
        rob.move_blocking(-TURN_SPEED, TURN_SPEED, 5)
        rob.talk("Turning right")

    # Phase 3: Move a bit
    for _ in range(10):
        read_and_log_irs(rob)
        rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 800)
        rob.talk("Moving forward (after turning right)")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def find_object_and_back(rob: IRobobo) -> None:
    """
    Example 2: Robot goes straight until it touches the wall,
    then goes backward.
    Uses front sensors for wall detection.
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    TOUCH_THRESHOLD = 50.0
    FORWARD_SPEED = 40
    BACKWARD_SPEED = 40
    BACKWARD_DURATION = 1500
    
    while True:
        ir_values = rob.read_irs()
        
        front_left = ir_values[2] if ir_values[2] is not None else 0.0
        front_right = ir_values[3] if ir_values[3] is not None else 0.0
        front_center = ir_values[4] if ir_values[4] is not None else 0.0
        
        back_left = ir_values[0] if ir_values[0] is not None else 0.0
        back_right = ir_values[1] if ir_values[1] is not None else 0.0
        back_center = ir_values[6] if ir_values[6] is not None else 0.0
        
        front_max = max(front_left, front_right, front_center)
        back_max = max(back_left, back_right, back_center)
        
        print(f"Front - L:{front_left:.1f} C:{front_center:.1f} R:{front_right:.1f} | Max:{front_max:.1f}")
        print(f"Back  - L:{back_left:.1f} C:{back_center:.1f} R:{back_right:.1f}")
        
        if front_max > TOUCH_THRESHOLD:
            rob.move_blocking(-BACKWARD_SPEED, -BACKWARD_SPEED, BACKWARD_DURATION)
            if back_max > TOUCH_THRESHOLD:
                rob.move_blocking(-BACKWARD_SPEED*0.5, -BACKWARD_SPEED*0.5, BACKWARD_DURATION) #slower if there is something behind
        else:
            rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 200)
        
        rob.sleep(0.1)