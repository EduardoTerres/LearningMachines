import matplotlib.pyplot as plt
import numpy as np

def plot_unit_sensor(sensor_unit_mean: np.array, filename_base: str, phase_durations: dict):
    """
    data: dict[str, list[float]] or list[list[float]]
    filename_base: str, e.g., 'output/timeseries'
    """
    dir = "results/figures"
    plt.figure()
    plt.plot(sensor_unit_mean)

    # Add vertical lines at phase transitions
    keys = list(phase_durations.keys())
    phase1_end = phase_durations[keys[0]]
    phase2_end = phase1_end + phase_durations[keys[1]]
    plt.axvline(x=phase1_end, color='red', linestyle='--', label=f'End {keys[0]}')
    plt.axvline(x=phase2_end, color='green', linestyle='--', label=f'End {keys[1]}')

    # Plot series
    plt.title(f"{filename_base}")
    plt.xlabel("Timestep (non-uniform grid)")
    plt.ylabel("Mean value accros runs")
    plt.tight_layout()
    plt.savefig(f"../{dir}/{filename_base}.png")
    plt.savefig(f"../{dir}/{filename_base}.pdf")
    plt.close()

def plot_all_sensors(sensors_all_runs, phase_durations_all_runs):  # list[dict[str, list[float]]]  I get errors in this type
    """
    For each sensor name, compute the mean across all runs at each timestep
    and plot using plot_unit_sensor.
    """
    if not sensors_all_runs:
        return

    # Take the mean of all phase durations (keywise)
    phase_duration_keys = phase_durations_all_runs[0].keys()
    phase_durations_mean = {
        key: np.mean([d[key] for d in phase_durations_all_runs])
        for key in phase_duration_keys
    }

    
    sensor_names = sensors_all_runs[0].keys()
    for sensor_name in sensor_names:
        # Gather the time series for this sensor over all runs
        sensor_series_all_runs = [run[sensor_name] for run in sensors_all_runs]
        # Take mean across all runs (axis=0: per timestep)
        mean_across_runs = np.mean(sensor_series_all_runs, axis=0)
        print(f"Plotting sensor, {sensor_name}...")
        plot_unit_sensor(mean_across_runs, sensor_name, phase_durations_mean)
        