#!/usr/bin/env python3
"""Run a trained policy for evaluation and log rollouts.

Usage:
  python deploy.py --simulation
  python deploy.py --hardware
"""
import sys
import json
import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from robobo_interface import HardwareRobobo, SimulationRobobo
from learning_machines import RoboboIREnv, SACAgent, DQNAgent

# Global configuration
AGENT_TYPE = "dqn"
MODEL_PATH = "/root/results/dqn_15-01-2026_14-02-54/dqn_model_15-01-2026_14-02-54.h5"
NUM_STEPS = 1000
RESULTS_DIR = "/root/results/dqn_hardware"


def build_env(mode: str) -> Tuple[RoboboIREnv, str]:
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
        hw_type = "hardware"
    else:
        rob = SimulationRobobo(identifier=1)
        hw_type = "simulation"
    env = RoboboIREnv(rob=rob)
    return env, hw_type


def load_agent(state_dim: int, action_dim: int):
    if AGENT_TYPE == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    else:
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, epsilon_start=0.0, epsilon_end=0.0)
    agent.load_model(MODEL_PATH)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def run_rollout(env: RoboboIREnv, agent, steps: int):
    obs, _ = env.reset()
    states, rewards, actions = [], [], []
    for _ in range(steps):
        states.append(obs.tolist())
        action = agent.select_action(obs, training=False)
        actions.append(int(action))
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(states, dtype=np.float32), np.array(rewards, dtype=np.float32), np.array(actions, dtype=np.int32)


def save_raw(states: np.ndarray, rewards: np.ndarray, actions: np.ndarray, basepath: str) -> None:
    np.savez(f"{basepath}.npz", states=states, rewards=rewards, actions=actions)
    with open(f"{basepath}.json", "w") as f:
        json.dump({"states": states.tolist(), "rewards": rewards.tolist(), "actions": actions.tolist()}, f)


def plot_rewards(rewards: np.ndarray, basepath: str) -> None:
    plt.figure()
    plt.plot(rewards)
    plt.title("Reward per step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(f"{basepath}_rewards.png")
    plt.savefig(f"{basepath}_rewards.pdf")
    plt.close()


def plot_states(states: np.ndarray, basepath: str) -> None:
    plt.figure()
    for i in range(states.shape[1]):
        plt.plot(states[:, i], label=f"sensor_{i}")
    plt.title("Normalized sensor readings")
    plt.xlabel("Step")
    plt.ylabel("Sensor value")
    plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{basepath}_states.png")
    plt.savefig(f"{basepath}_states.pdf")
    plt.close()


def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]
    if mode == "--hardware":
        hw_type = "hardware"
    elif mode == "--simulation":
        hw_type = "simulation"
    else:
        raise ValueError("Invalid mode")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    env, hw_type = build_env(mode)
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])
    action_dim = env.action_space.n

    agent = load_agent(state_dim, action_dim)

    states, rewards, actions = run_rollout(env, agent, NUM_STEPS)
    env.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    basepath = os.path.join(RESULTS_DIR, f"deploy_{AGENT_TYPE}_{hw_type}_{ts}")

    save_raw(states, rewards, actions, basepath)
    plot_rewards(rewards, basepath)
    plot_states(states, basepath)
    print(f"Saved rollout data and plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
