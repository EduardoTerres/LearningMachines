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

from learning_machines import RoboboIREnv, DQNAgent, SACAgent, plot_training_statistics

# Agent selection: change default here or pass `--agent sac` on CLI. Options: 'dqn', 'sac'
AGENT = "sac"

INIT_MODEL_PATH = None
INIT_MODEL_PATH = "/root/results/sac_15-01-2026_15-56-29/sac_model_final.h5"

# Dummy run
# from learning_machines import test_env
# RoboboIREnv.test_env(mode="--simulation")
# exit(0)

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

    # Load initial model if provided
    if INIT_MODEL_PATH:
        agent.load_model(INIT_MODEL_PATH)
        print(f"Initialized training from model: {INIT_MODEL_PATH}")

    num_episodes = 200
    max_steps = 10
    stats = []

    # Make logging directory
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    results_dir = f"/root/results/{AGENT}_{ts}"
    os.makedirs(results_dir, exist_ok=True)

    # Save properties file
    properties = {
        **agent.get_properties(),
        "num_episodes": int(num_episodes),
        "num_steps": int(max_steps)
    }
    with open(os.path.join(results_dir, "properties.json"), "w") as f:
        json.dump(properties, f, indent=2)

    log_file_path = os.path.join(results_dir, f"training_log_{ts}.txt")
    log_file = open(log_file_path, 'w')
    log_file.write(f"Training Log - Episode | Step | State | Action Distribution | Probability of Selected Action\n")
    log_file.write("=" * 100 + "\n")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        episode_losses = []
        episode_q_values = []
        episode_collisions = 0
        
        for t in range(max_steps):
            # Get action probabilities BEFORE selecting action
            action_probs, q_vals = get_action_probabilities(agent, obs, agent_type)
            
            # Track Q-values
            if q_vals is not None:
                episode_q_values.append(np.mean(q_vals))
            
            # Select action (existing code - unchanged)
            a = agent.select_action(obs, training=True)
            
            # Format state as readable string
            state_str = "[" + ", ".join([f"{s:.3f}" for s in obs]) + "]"
            
            # Format action distribution as readable string
            action_dist_str = " | ".join([f"{env.actions[i]}: {action_probs[i]:.4f}" for i in range(len(env.actions))])
            
            # Print to screen
            print(f"\n--- Episode {ep+1} | Step {t+1} ---")
            print(f"State: {state_str}")
            print(f"Action Distribution: {action_dist_str}")
            print(f"Selected Action: {env.actions[a]} (probability: {action_probs[a]:.4f})")
            
            # Write to log file
            log_file.write(f"Episode {ep+1} | Step {t+1} | ")
            log_file.write(f"State: {state_str} | ")
            log_file.write(f"Action Distribution: {action_dist_str} | ")
            log_file.write(f"Selected: {env.actions[a]} (prob: {action_probs[a]:.4f})\n")
            log_file.flush()
            
            # Rest of existing code - UNCHANGED
            next_obs, reward, done, _, _ = env.step(a)
            
            # Track collisions
            if env.detect_collision(next_obs):
                episode_collisions += 1
            
            agent.replay_buffer.add(obs, a, reward, next_obs, done)
            loss = agent.train_step()
            
            # Track loss
            if loss is not None:
                episode_losses.append(loss)
            
            obs = next_obs
            total_reward += reward
            rob.sleep(0.2)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
        if hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
        
        # Store comprehensive statistics - CONVERT NUMPY TYPES TO PYTHON TYPES FOR JSON
        episode_stat = {
            "episode": int(ep),
            "steps": int(t + 1),
            "reward": float(total_reward),
            "epsilon": float(getattr(agent, 'epsilon', 0)) if getattr(agent, 'epsilon', None) is not None else None,
            "collisions": int(episode_collisions),
            "mean_q_value": float(np.mean(episode_q_values)) if episode_q_values else None,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else None,
        }
        stats.append(episode_stat)

        # Save intermediate model
        if ep % 20 == 0:
            intermediate_model_path = os.path.join(results_dir, f"{agent_type}_model_ep{ep+1}.h5")
            agent.save_model(intermediate_model_path)
            with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
                json.dump(stats, f, indent=2)

        print(f"\nEpisode {ep+1}/{num_episodes}  reward={total_reward:.2f} eps={getattr(agent, 'epsilon', 'N/A')} collisions={episode_collisions}")

    # Close log file
    log_file.close()

    # Save model and stats
    model_path = os.path.join(results_dir, f"{agent_type}_model_final.h5")
    agent.save_model(model_path)
    with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations using the separate function
    plot_training_statistics(stats, results_dir, ts, agent_type)
    
    print(f"\nSaved model to {model_path}")
    print(f"Saved log to {log_file_path}")
    print(f"Saved stats to {os.path.join(results_dir, f'stats_{ts}.json')}")

    env.close()


if __name__ == "__main__":
    main()