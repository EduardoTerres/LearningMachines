#!/usr/bin/env python3
"""
Trainer/Evaluator using the gym `RoboboIREnv` and agents `DQNAgent` / `SACAgent`.

Usage:
  Train: python train.py --simulation --agent sac
  Eval : python train.py --simulation --eval --agent sac --checkpoint /path/to/model.h5
"""
import sys
import os
import json
import numpy as np
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import RoboboIREnv, DQNAgent, SACAgent

# Default agent: can be overridden with --agent sac|dqn
AGENT = "sac"


def get_arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        raise ValueError(f"Provide value after {flag}")
    return default


def has_flag(flag):
    return flag in sys.argv


def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]

    # switch from train to eval if needed
    run_mode = "eval" if (has_flag("--eval") or has_flag("--validate")) else "train"

    # override defaults from CLI
    agent_type = get_arg("--agent", AGENT)
    num_episodes = int(get_arg("--episodes", 100 if run_mode == "train" else 50))
    max_steps = int(get_arg("--max_steps", 10))
    seed = get_arg("--seed", None)
    checkpoint = get_arg("--checkpoint", None)

    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)

    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    elif mode == "--simulation":
        rob = SimulationRobobo(identifier=1)
    else:
        raise ValueError("Invalid mode")

    env = RoboboIREnv(rob=rob)

    # infer dims
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])

    if agent_type == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=env.action_space.n)
    elif agent_type == "sac":
        agent = SACAgent(state_dim=state_dim, action_dim=env.action_space.n)
    else:
        raise ValueError("Invalid agent type (use 'dqn' or 'sac')")

    # load checkpoint for eval (and optionally for resume training)
    if checkpoint is not None:
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        if not hasattr(agent, "load_model"):
            raise AttributeError("Agent does not implement load_model()")
        agent.load_model(checkpoint)
        print(f"Loaded checkpoint: {checkpoint}")

    if run_mode == "eval" and checkpoint is None:
        raise ValueError("Eval requires --checkpoint /path/to/model")

    # outputs 
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "/root/results/model" if run_mode == "train" else "/root/results/validation"
    os.makedirs(out_dir, exist_ok=True)

    stats = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        for t in range(max_steps):
            a = agent.select_action(obs, training=(run_mode == "train"))
            next_obs, reward, done, _, _info = env.step(a)

            if run_mode == "train":
                agent.replay_buffer.add(obs, a, reward, next_obs, done)
                agent.train_step()

            obs = next_obs
            total_reward += reward
            rob.sleep(0.2)

            if done:
                break

        # only decay epsilon in training
        if run_mode == "train" and hasattr(agent, "decay_epsilon"):
            agent.decay_epsilon()

        # keeping the same schema as train.py
        stats.append({
            "episode": ep,
            "reward": float(total_reward),
            "steps": int(t + 1),
            "epsilon": getattr(agent, "epsilon", None),
        })

        if run_mode == "train":
            print(f"[TRAIN] Ep {ep+1}/{num_episodes} reward={total_reward:.2f} eps={getattr(agent,'epsilon',0):.3f}")
        else:
            print(f"[EVAL ] Ep {ep+1}/{num_episodes} reward={total_reward:.2f}")

    # Save stats 
    stats_path = os.path.join(out_dir, f"stats_{ts}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Save model once only
    model_path = os.path.join(out_dir, f"{agent_type}_model_{ts}.h5")
    agent.save_model(model_path)
    if run_mode == "train":
        print(f"Saved model to {model_path}")
    else:
        print(f"Saved eval model copy to {model_path}")

    env.close()


if __name__ == "__main__":
    main()

