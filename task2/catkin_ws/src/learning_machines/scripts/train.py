#!/usr/bin/env python3
"""
Usage: `python train.py --simulation` or `--hardware`.
"""
import sys
import os
import json
import numpy as np
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo

from learning_machines import RoboboIREnv, SACAgent, plot_training_statistics

AGENT = "sac"

INIT_MODEL_PATH = None
# INIT_MODEL_PATH = "/root/results/sac_15-01-2026_16-19-31/sac_model_final.h5"
INSTANCE = None

def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=True)
        INSTANCE = "hardware"
    elif mode == "--simulation":
        rob = SimulationRobobo()
        INSTANCE = "simulation"
    else:
        raise ValueError("Invalid mode")

    env = RoboboIREnv(rob=rob)

    # create a small sample to infer dims
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])

    agent = SACAgent(state_dim=state_dim, action_dim=2)

    # Load initial model if provided
    if INIT_MODEL_PATH:
        agent.load_model(INIT_MODEL_PATH)
        print(f"Initialized training from model: {INIT_MODEL_PATH}")

    num_episodes = 150
    max_steps = 10
    stats = []

    # Make logging directory
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    results_dir = f"/root/results/{AGENT}_{INSTANCE}_{ts}"
    os.makedirs(results_dir, exist_ok=True)

    # Save properties file
    properties = {
        **agent.get_properties(),
        "num_episodes": int(num_episodes),
        "num_steps": int(max_steps),
        "init_model_path": INIT_MODEL_PATH,
        "instance": INSTANCE,
    }
    with open(os.path.join(results_dir, "properties.json"), "w") as f:
        json.dump(properties, f, indent=2)

    log_file_path = os.path.join(results_dir, f"training_log_{ts}.txt")
    log_file = open(log_file_path, 'w')
    log_file.write(f"Training Log - Episode | Step | State | Green Detection | ViT Food Score | Action\n")
    log_file.write("=" * 130 + "\n")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        episode_losses = []
        episode_collisions = 0
        episode_food_collected = 0
        
        for t in range(max_steps):
            # Select continuous action
            a = agent.select_action(obs, training=True)
            
            # Format state as readable string
            state_str = "[" + ", ".join([f"{s:.3f}" for s in obs[:8]]) + "]"
            green_value = obs[8] if len(obs) > 8 else 0.0
            vit_food_score = obs[9] if len(obs) > 9 else 0.0
            
            # Format action as readable string
            action_str = f"[L:{a[0]:.3f}, R:{a[1]:.3f}]"
            
            # Print to screen
            print(f"\n--- Episode {ep+1} | Step {t+1} ---")
            print(f"State (IRs): {state_str}")
            print(f"Green Detection: {green_value:.4f}")
            print(f"ViT Food Score: {vit_food_score:.4f}")
            print(f"Action (wheel speeds): {action_str}")
            
            # Write to log file
            log_file.write(f"Episode {ep+1} | Step {t+1} | ")
            log_file.write(f"State: {state_str} | ")
            log_file.write(f"Green: {green_value:.4f} | ViT: {vit_food_score:.4f} | ")
            log_file.write(f"Action: {action_str}\n")
            log_file.flush()
            
            # Execute action
            next_obs, reward, done, truncated, info = env.step(a)
            
            # Track food collection
            if 'food_collected' in info and info['food_collected'] > 0:
                episode_food_collected += info['food_collected']
                print(f">>> FOOD COLLECTED! Total: {episode_food_collected}")
            
            # Track collisions
            if env.detect_collision(next_obs):
                episode_collisions += 1
            
            agent.replay_buffer.add(obs, a, reward, next_obs, done or truncated)
            loss = agent.train_step()
            
            # Track loss
            if loss is not None:
                episode_losses.append(loss)
            
            obs = next_obs
            total_reward += reward
            rob.sleep(0.2)
            if done or truncated:
                print(f"Episode finished after {t+1} timesteps")
                break
        
        # Store comprehensive statistics
        episode_stat = {
            "episode": int(ep),
            "steps": int(t + 1),
            "reward": float(total_reward),
            "collisions": int(episode_collisions),
            "food_collected": int(episode_food_collected),
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else None,
        }
        stats.append(episode_stat)

        # Save intermediate model
        if ep % 20 == 0:
            intermediate_model_path = os.path.join(results_dir, f"{AGENT}_model_ep{ep+1}.h5")
            agent.save_model(intermediate_model_path)
            with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
                json.dump(stats, f, indent=2)

        print(f"\nEpisode {ep+1}/{num_episodes}  reward={total_reward:.2f} collisions={episode_collisions} food={episode_food_collected}")

    # Close log file
    log_file.close()

    # Save model and stats
    model_path = os.path.join(results_dir, f"{AGENT}_model_final.h5")
    agent.save_model(model_path)
    with open(os.path.join(results_dir, f"stats_{ts}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Create visualizations using the separate function
    plot_training_statistics(stats, results_dir, ts, AGENT)
    
    print(f"\nSaved model to {model_path}")
    print(f"Saved log to {log_file_path}")
    print(f"Saved stats to {os.path.join(results_dir, f'stats_{ts}.json')}")

    env.close()


if __name__ == "__main__":
    main()