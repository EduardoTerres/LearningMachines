#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import find_object_and_turnR, plot_all_sensors


if __name__ == "__main__":
    # You can do better argument parsing than this!
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware` or `--simulation` to specify."""
        )
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    NUM_RUNS = 5
    sensors_all_runs = []
    phase_durations_all_runs = []
    for _ in range(NUM_RUNS):
        sensors, phase_durations = find_object_and_turnR(rob)
        sensors_all_runs.append(sensors)
        phase_durations_all_runs.append(phase_durations)

    plot_all_sensors(sensors_all_runs, phase_durations_all_runs)


