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
AGENT = "sac"
timestamp = "15-01-2026_17-41-54"
MODEL_PATH = f"/root/results/{AGENT}_simulation_{timestamp}/{AGENT}_model_final.h5"
NUM_STEPS = 400
RESULTS_DIR = f"/root/results/{AGENT}_hardware"


def get_action_probabilities(agent, state, agent_type):
    """Get action probabilities from agent's policy."""
    state_reshaped = state.reshape(1, -1).astype(np.float32)
    
    if agent_type == 'dqn':
        q_vals = agent.q_network.predict(state_reshaped, verbose=0)[0]
        # Convert Q-values to probabilities using softmax
        exp_q = np.exp(q_vals - np.max(q_vals))
        probs = exp_q / np.sum(exp_q)
        return probs, q_vals
    elif agent_type == 'sac':
        # Get logits from actor network
        logits = agent.actor.predict(state_reshaped, verbose=0)[0]
        # Convert logits to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs, logits
    else:
        raise ValueError("Invalid agent type")


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
    if AGENT == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    elif AGENT == "sac":
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, epsilon_start=0.0, epsilon_end=0.0)
    else:
        raise ValueError("Invalid agent type")

    agent.load_model(MODEL_PATH)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    return agent


def run_rollout(env: RoboboIREnv, agent, steps: int, log_file):
    obs, _ = env.reset()
    states, rewards, actions = [], [], []
    
    for step in range(steps):
        states.append(obs.tolist())
        
        # Get action probabilities BEFORE selecting action
        action_probs, q_vals = get_action_probabilities(agent, obs, AGENT)
        
        # Select action
        action = agent.select_action(obs, training=False)
        actions.append(int(action))
        
        # Format state as readable string
        state_str = "[" + ", ".join([f"{s:.3f}" for s in obs]) + "]"
        
        # Format action distribution as readable string
        action_dist_str = " | ".join([f"{env.actions[i]}: {action_probs[i]:.4f}" for i in range(len(env.actions))])
        
        # Print to screen
        print(f"\n--- Step {step+1}/{steps} ---")
        print(f"State: {state_str}")
        print(f"Action Distribution: {action_dist_str}")
        print(f"Selected Action: {env.actions[action]} (probability: {action_probs[action]:.4f})")
        
        # Write to log file
        log_file.write(f"Step {step+1} | ")
        log_file.write(f"State: {state_str} | ")
        log_file.write(f"Action Distribution: {action_dist_str} | ")
        log_file.write(f"Selected: {env.actions[action]} (prob: {action_probs[action]:.4f})\n")
        log_file.flush()
        
        # Execute action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        
        print(f"Reward: {reward:.4f}")
        log_file.write(f"  -> Reward: {reward:.4f}\n")
        
        obs = next_obs
        if terminated or truncated:
            print("Episode terminated, resetting...")
            log_file.write("  -> Episode terminated, resetting...\n")
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

    # Create log file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    basepath = os.path.join(RESULTS_DIR, f"deploy_{AGENT}_{hw_type}_{ts}")
    log_file_path = f"{basepath}_log.txt"
    
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Deployment Log - Step | State | Action Distribution | Probability of Selected Action | Reward\n")
        log_file.write("=" * 100 + "\n")
        
        states, rewards, actions = run_rollout(env, agent, NUM_STEPS, log_file)
    
    env.close()

    save_raw(states, rewards, actions, basepath)
    plot_rewards(rewards, basepath)
    plot_states(states, basepath)
    
    print(f"\nSaved rollout data and plots to {RESULTS_DIR}")
    print(f"Saved log to {log_file_path}")


if __name__ == "__main__":
    main()
