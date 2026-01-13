#!/usr/bin/env python3
"""Minimal trainer using the gym `RoboboIREnv` and `DQNAgent`.

Usage: `python train.py --simulation` or `--hardware`.
"""
import sys
import os
import json
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.env import RoboboIREnv
from learning_machines.agent import DQNAgent


def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    elif mode == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError("Invalid mode")

    env = RoboboIREnv(rob=rob)

    agent = DQNAgent(state_dim=8, num_actions=6, epsilon_start=1.0,
                     epsilon_end=0.01, epsilon_decay=0.995, batch_size=32,
                     target_update_frequency=100)

    num_episodes = 100
    max_steps = 500
    stats = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for t in range(max_steps):
            a = agent.select_action(obs, training=True)
            next_obs, reward, done, _, _ = env.step(a)
            agent.replay_buffer.add(obs, a, reward, next_obs, done)
            agent.train_step()
            obs = next_obs
            total_reward += reward
            if done:
                break
        agent.decay_epsilon()
        stats.append({"episode": ep, "reward": total_reward, "steps": t + 1, "epsilon": agent.epsilon})
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}  reward={total_reward:.2f} eps={agent.epsilon:.3f}")

    # Save model and stats
    results_dir = "/root/results/model"
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(results_dir, f"dqn_model_{ts}.h5")
    agent.save_model(model_path)
    with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved model to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
from spinup import td3
from examples.task1.catkin_ws.src.learning_machines.env import RoboboIREnv


def main():
    env_fn = lambda: RoboboIREnv()
    td3(env_fn, epochs=50)


if __name__ == '__main__':
    main()
