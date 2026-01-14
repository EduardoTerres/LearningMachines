#!/usr/bin/env python3
from pathlib import Path
import json
import sys

# Running from examples/task1
MODEL_DIR = Path("../full_project_setup/results/model")

EXPECTED_EPISODES = 100 #to adjust
MIN_MEAN_REWARD = -1e9  #to adjust once rewards are known

def fail(msg):
    print("validation failed", msg)
    sys.exit(1)

def newest(paths):
    return max(paths, key=lambda p: p.stat().st_mtime)

def main():
    if not MODEL_DIR.exists():
        fail(f"Missing model dir: {MODEL_DIR.resolve()}")

    models = list(MODEL_DIR.glob("*_model_*.h5"))
    stats_files = list(MODEL_DIR.glob("stats_*.json"))

    if not models:
        fail("No model file found (*_model_*.h5)")
    if not stats_files:
        fail("No stats file found (stats_*.json)")

    model = newest(models)
    stats_path = newest(stats_files)

    if model.stat().st_size == 0:
        fail(f"Model file is empty: {model.name}")
    if stats_path.stat().st_size == 0:
        fail(f"Stats file is empty: {stats_path.name}")

    try:
        stats = json.loads(stats_path.read_text())
    except Exception as e:
        fail(f"Could not parse {stats_path.name}: {e}")

    if not isinstance(stats, list) or len(stats) == 0:
        fail(f"{stats_path.name} is not a non-empty list")

    n = len(stats)
    mean_reward = sum(s.get("reward", 0.0) for s in stats) / n

    if n < EXPECTED_EPISODES:
        fail(f"Too few episodes in stats: {n} < {EXPECTED_EPISODES}")
    if mean_reward < MIN_MEAN_REWARD:
        fail(f"Mean reward too low: {mean_reward:.3f} < {MIN_MEAN_REWARD}")

    print("validation succeeded")
    print(f"Model : {model.name} ({model.stat().st_size} bytes)")
    print(f"Stats : {stats_path.name} (episodes={n}, mean_reward={mean_reward:.3f})")

if __name__ == "__main__":
    main()
