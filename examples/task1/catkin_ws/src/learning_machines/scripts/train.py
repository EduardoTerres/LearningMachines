#!/usr/bin/env python3
"""Minimal trainer using the gym `RoboboIREnv` and `DQNAgent`.

Usage: `python train.py --simulation` or `--hardware`.
"""
import sys
import os
import json
import numpy as np
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo

from learning_machines import RoboboIREnv, DQNAgent, SACAgent

# Agent selection: change default here or pass `--agent sac` on CLI. Options: 'dqn', 'sac'
AGENT = "sac"

# Dummy run
# from learning_machines import test_env
# RoboboIREnv.test_env(mode="--simulation")
# exit(0)

def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    elif mode == "--simulation":
        rob = SimulationRobobo(identifier=1)
    else:
        raise ValueError("Invalid mode")

    env = RoboboIREnv(rob=rob)

    # allow CLI override: --agent sac or --agent dqn
    agent_type = AGENT
    if '--agent' in sys.argv:
        i = sys.argv.index('--agent')
        if i + 1 < len(sys.argv):
            agent_type = sys.argv[i + 1]
        else:
            raise ValueError("Provide agent type after --agent")

    # create a small sample to infer dims
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])

    if agent_type == 'dqn':
        agent = DQNAgent(state_dim=state_dim, action_dim=env.action_space.n)
    elif agent_type == 'sac':
        agent = SACAgent(state_dim=state_dim, action_dim=env.action_space.n)
    else:
        raise ValueError("Invalid agent type")

    num_episodes = 100
    max_steps = 10
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
            rob.sleep(0.5)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
        stats.append({"episode": ep, "reward": total_reward, "steps": t + 1, "epsilon": getattr(agent, 'epsilon', None)})
        # if (ep + 1) % 10 == 0:
        print(f"Episode {ep+1}/{num_episodes}  reward={total_reward:.2f} eps={agent.epsilon:.3f}")

    # Save model and stats
    results_dir = "/root/results/model"
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(results_dir, f"{agent_type}_model_{ts}.h5")
    agent.save_model(model_path)
    with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved model to {model_path}")

    env.close()


if __name__ == "__main__":
    main()
