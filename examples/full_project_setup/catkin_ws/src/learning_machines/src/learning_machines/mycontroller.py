from robobo_interface import IRobobo, SimulationRobobo


def find_object_and_turnR(rob: IRobobo) -> None:
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    DETECTION_THRESHOLD = 20.0
    FORWARD_SPEED = 30
    TURN_SPEED = 40
    TURN_DURATION = 800
    
    while True:
        ir_values = rob.read_irs()
        
        front_center = ir_values[4] if ir_values[4] is not None else 0.0
        
        print(f"Front Center IR: {front_center:.1f}")
        
        if front_center > DETECTION_THRESHOLD:
            print(f"Object detected in front center ({front_center:.1f}) - Turning right")
            rob.move_blocking(TURN_SPEED, -TURN_SPEED, TURN_DURATION)
            print("Turn completed")
        else:
            rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 200)
        
        rob.sleep(0.1)


def find_object_and_back(rob: IRobobo) -> None:
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    TOUCH_THRESHOLD = 50.0
    FORWARD_SPEED = 40
    BACKWARD_SPEED = 40
    BACKWARD_DURATION = 1500
    
    while True:
        ir_values = rob.read_irs()
        
        front_center = ir_values[4] if ir_values[4] is not None else 0.0
        
        print(f"Front Center IR: {front_center:.1f}")
        
        if front_center > TOUCH_THRESHOLD:
            print("Wall touched! Moving backward...")
            rob.move_blocking(-BACKWARD_SPEED, -BACKWARD_SPEED, BACKWARD_DURATION)
            print("Backward movement completed")
        else:
            rob.move_blocking(FORWARD_SPEED, FORWARD_SPEED, 200)
        
        rob.sleep(0.1)