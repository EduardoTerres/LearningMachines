from robobo_interface import IRobobo, SimulationRobobo

def read_and_log_irs(rob: IRobobo, sensors: dict) -> None:
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
    ), sensors


def find_object_and_turnR(rob: IRobobo) -> dict:
    """
    Example 1: Robot goes straight until it senses an object getting near,
    then turns right without touching it.
    Uses front AND back sensors for complete awareness.
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    DETECTION_THRESHOLD = 80.0
    FORWARD_SPEED = 10
    TURN_SPEED = 10

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

    # Phase 1: forward
    while True:
        (
            (
                front_left,
                front_right,
                front_center,
                front_right_right,
                front_left_left,
                back_left,
                back_right,
                back_center,
            ),
            sensors
        ) = read_and_log_irs(rob, sensors)
        
        
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
        _, sensors = read_and_log_irs(rob, sensors)
        rob.move_blocking(-TURN_SPEED, TURN_SPEED, 100)
        rob.talk("Turning right")

    # Phase 3: Move a bit
    for _ in range(20):
        _, sensors = read_and_log_irs(rob, sensors)
        rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 800)
        rob.talk("Moving forward (after turning right)")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

    return sensors
    