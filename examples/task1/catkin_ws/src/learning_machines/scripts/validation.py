#!/usr/bin/env python3
"""Validate/evaluate trained RL model by testing it on the environment.

This tests the trained policy WITHOUT exploration to measure actual performance.

Usage:
  python validation.py --simulation
  python validation.py --hardware --model /path/to/model.h5
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from robobo_interface import HardwareRobobo, SimulationRobobo
from learning_machines import RoboboIREnv, DQNAgent, SACAgent

# Agent selection: change default here or pass `--agent sac` on CLI. Options: 'dqn', 'sac'
AGENT = "dqn"


def load_agent(agent_type: str, state_dim: int, action_dim: int, model_path: str):
    """Load trained agent and disable exploration."""
    if agent_type == "dqn":
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    else:
        agent = SACAgent(state_dim=state_dim, action_dim=action_dim, 
                        epsilon_start=0.0, epsilon_end=0.0)
    
    agent.load_model(model_path)
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0  # Disable exploration for evaluation
    
    return agent


def validate_model(env: RoboboIREnv, agent, num_episodes: int = 50, max_steps: int = 10):
    """Run validation episodes and collect metrics."""
    episode_rewards = []
    episode_collisions = []
    episode_lengths = []
    
    print(f"\nRunning {num_episodes} validation episodes...")
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        collisions = 0
        
        for t in range(max_steps):
            action = agent.select_action(obs, training=False)  # Greedy policy, no exploration
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            if env.detect_collision(next_obs):
                collisions += 1
            
            episode_reward += reward
            obs = next_obs
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_collisions.append(collisions)
        episode_lengths.append(t + 1)
        
        if (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{num_episodes} episodes...")
    
    # Calculate metrics
    metrics = {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_collisions": float(np.mean(episode_collisions)),
        "std_collisions": float(np.std(episode_collisions)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean([1 if c == 0 else 0 for c in episode_collisions])),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_collisions": [int(c) for c in episode_collisions],
        "episode_lengths": [int(l) for l in episode_lengths],
    }
    
    return metrics


def plot_validation_results(metrics: dict, model_name: str, agent_type: str, 
                            hw_type: str, output_path: Path):
    """Generate plots for validation results."""
    episode_rewards = metrics['episode_rewards']
    episode_collisions = metrics['episode_collisions']
    episode_lengths = metrics['episode_lengths']
    episodes = list(range(1, len(episode_rewards) + 1))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Validation Results - {agent_type.upper()} ({hw_type})', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards Over Time
    ax1 = axes[0, 0]
    ax1.plot(episodes, episode_rewards, alpha=0.6, color='blue', marker='o', markersize=4, linewidth=1.5)
    ax1.axhline(y=metrics['mean_reward'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {metrics['mean_reward']:.3f}")
    ax1.fill_between(episodes, 
                     metrics['mean_reward'] - metrics['std_reward'],
                     metrics['mean_reward'] + metrics['std_reward'],
                     alpha=0.2, color='red', label=f"±1 std")
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Distribution (Histogram)
    ax2 = axes[0, 1]
    ax2.hist(episode_rewards, bins=min(20, len(episode_rewards)//2), color='skyblue', 
             edgecolor='black', alpha=0.7)
    ax2.axvline(x=metrics['mean_reward'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {metrics['mean_reward']:.3f}")
    ax2.set_xlabel('Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Collisions Per Episode
    ax3 = axes[0, 2]
    ax3.plot(episodes, episode_collisions, alpha=0.6, color='red', marker='o', markersize=4, linewidth=1.5)
    ax3.axhline(y=metrics['mean_collisions'], color='orange', linestyle='--', linewidth=2,
                label=f"Mean: {metrics['mean_collisions']:.2f}")
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Collisions', fontsize=11)
    ax3.set_title('Collisions Per Episode', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Episode Lengths
    ax4 = axes[1, 0]
    ax4.plot(episodes, episode_lengths, alpha=0.6, color='green', marker='o', markersize=4, linewidth=1.5)
    ax4.axhline(y=metrics['mean_length'], color='darkgreen', linestyle='--', linewidth=2,
                label=f"Mean: {metrics['mean_length']:.1f}")
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Steps', fontsize=11)
    ax4.set_title('Episode Lengths', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Success Rate Visualization
    ax5 = axes[1, 1]
    success_episodes = sum(1 for c in episode_collisions if c == 0)
    failure_episodes = len(episode_collisions) - success_episodes
    colors = ['#2ecc71', '#e74c3c']
    ax5.pie([success_episodes, failure_episodes], 
            labels=[f'Success ({success_episodes})', f'Failure ({failure_episodes})'],
            autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 11})
    ax5.set_title(f'Success Rate: {metrics["success_rate"]:.1%}', fontsize=12, fontweight='bold')
    
    # 6. Summary Statistics Text
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_text = f"""
Model: {Path(model_name).name}

REWARD STATISTICS:
  Mean: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}
  Range: [{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]

COLLISION STATISTICS:
  Mean: {metrics['mean_collisions']:.2f} ± {metrics['std_collisions']:.2f}
  Success Rate: {metrics['success_rate']:.1%}

EPISODE LENGTH:
  Mean: {metrics['mean_length']:.1f} steps

TOTAL EPISODES: {metrics['num_episodes']}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nValidation plot saved to: {output_path}")
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
        hw_type = "hardware"
    elif mode == "--simulation":
        rob = SimulationRobobo(identifier=1)
        hw_type = "simulation"
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

    # Parse optional arguments
    model_path = None
    if '--model' in sys.argv:
        i = sys.argv.index('--model')
        if i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
        else:
            raise ValueError("Provide model path after --model")

    num_episodes = 50
    if '--episodes' in sys.argv:
        i = sys.argv.index('--episodes')
        if i + 1 < len(sys.argv):
            num_episodes = int(sys.argv[i + 1])
        else:
            raise ValueError("Provide number of episodes after --episodes")

    max_steps = 10
    if '--max-steps' in sys.argv:
        i = sys.argv.index('--max-steps')
        if i + 1 < len(sys.argv):
            max_steps = int(sys.argv[i + 1])
        else:
            raise ValueError("Provide max steps after --max-steps")

    save_results = '--save-results' in sys.argv
    no_plot = '--no-plot' in sys.argv

    # create a small sample to infer dims
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])
    action_dim = env.action_space.n

    # Find model if not specified
    if model_path is None:
        # Check multiple possible model directories
        model_dirs = [
            Path("/root/results/model"),
            Path("/root/results/dqn_visibility"),
        ]
        
        models = []
        for model_dir in model_dirs:
            if model_dir.exists():
                found = list(model_dir.glob(f"*{agent_type}_model_*.h5"))
                models.extend(found)
        
        if not models:
            print(f"ERROR: No {agent_type} model files found in any model directory")
            sys.exit(1)
        
        # Get newest model
        model_path = str(max(models, key=lambda p: p.stat().st_mtime))
        print(f"Using newest model: {Path(model_path).name}")
    
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)

    # Load agent
    print(f"\nLoading model: {Path(model_path).name}")
    agent = load_agent(agent_type, state_dim, action_dim, model_path)

    # Run validation
    metrics = validate_model(env, agent, num_episodes, max_steps)
    env.close()

    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Model: {Path(model_path).name}")
    print(f"Agent: {agent_type.upper()}")
    print(f"Mode: {hw_type}")
    print(f"Episodes: {metrics['num_episodes']}")
    print()
    print(f"Reward Statistics:")
    print(f"  Mean: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
    print(f"  Range: [{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]")
    print()
    print(f"Collision Statistics:")
    print(f"  Mean: {metrics['mean_collisions']:.2f} ± {metrics['std_collisions']:.2f}")
    print(f"  Success Rate (0 collisions): {metrics['success_rate']:.1%}")
    print()
    print(f"Episode Length:")
    print(f"  Mean: {metrics['mean_length']:.1f} steps")
    print("=" * 60)

    # Generate plots (unless disabled)
    if not no_plot:
        # Determine output directory
        model_path_obj = Path(model_path)
        if model_path_obj.parent.name == "model":
            results_dir = Path("/root/results/validation")
        else:
            results_dir = model_path_obj.parent
        
        results_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = results_dir / f"validation_{agent_type}_{hw_type}_{ts}.png"
        
        plot_validation_results(metrics, model_path, agent_type, hw_type, plot_path)

    # Save results if requested
    if save_results:
        results_dir = Path("/root/results/validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"validation_{agent_type}_{hw_type}_{ts}.json"
        
        results = {
            "model_path": model_path,
            "agent_type": agent_type,
            "mode": hw_type,
            "timestamp": ts,
            "metrics": metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")

    sys.exit(0)


if __name__ == "__main__":
    main()