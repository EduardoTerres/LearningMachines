import matplotlib.pyplot as plt
import numpy as np

from typing import List

def plot_unit_sensor(sensor_unit: List[List[float]], filename_base: str):
    """
    data: dict[str, list[float]] or list[list[float]]
    filename_base: str, e.g., 'output/timeseries'
    """
    arr = np.array(sensor_unit)
    mean_series = np.mean(arr, axis=0)
    plt.figure()
    plt.plot(mean_series)
    plt.title("Mean of Timeseries")
    plt.xlabel("Timestep")
    plt.ylabel("Mean Value")
    plt.tight_layout()
    plt.savefig(f"{filename_base}.png")
    plt.savefig(f"{filename_base}.pdf")
    plt.close()

def plot_all_sensors(sensors: dict):
    for sensor_unit_name, sensor_unit in sensors.items():
        plot_unit_sensor(sensor_unit, sensor_unit_name)