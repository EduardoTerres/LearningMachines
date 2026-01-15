#!/usr/bin/env python3
"""Standalone script to plot training statistics from JSON file.

Usage: python plot_stats.py <path_to_stats.json>
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_from_json(json_path):
    """Plot training statistics from a JSON stats file."""
    
    try:
        # Load stats
        with open(json_path, 'r') as f:
            content = f.read()
            # Try to fix incomplete JSON if needed
            if not content.strip().endswith(']'):
                # Find last complete episode
                lines = content.strip().split('\n')
                complete_lines = []
                for line in lines:
                    if line.strip() and not line.strip().endswith(':'):
                        complete_lines.append(line)
                content = '\n'.join(complete_lines)
                if not content.endswith(']'):
                    content = content.rstrip(',') + '\n]'
            
            stats = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print("Trying to fix incomplete JSON...")
        # Try to manually fix
        with open(json_path, 'r') as f:
            content = f.read()
        # Remove incomplete last entry
        if '"mean_q_value":' in content and not content.strip().endswith('}'):
            # Find last complete entry
            last_complete = content.rfind('}')
            if last_complete > 0:
                content = content[:last_complete+1] + '\n]'
                stats = json.loads(content)
            else:
                print("Cannot parse JSON file. File may be corrupted.")
                return
        else:
            print("Cannot parse JSON file.")
            return
    
    if not stats:
        print("No data found in stats file!")
        return
    
    print(f"Loaded {len(stats)} episodes")
    
    # Extract data (handle missing fields)
    episodes = [s.get('episode', i) for i, s in enumerate(stats)]
    rewards = [s.get('reward', 0) for s in stats]
    steps = [s.get('steps', 10) for s in stats]
    epsilons = [s.get('epsilon', 0) for s in stats]
    collisions = [s.get('collisions', 0) for s in stats]
    
    # Handle optional fields
    mean_q_values = []
    q_episodes = []
    for i, s in enumerate(stats):
        if s.get('mean_q_value') is not None:
            try:
                mean_q_values.append(float(s['mean_q_value']))
                q_episodes.append(i)
            except (TypeError, ValueError):
                pass
    
    losses = []
    loss_episodes = []
    for i, s in enumerate(stats):
        if s.get('mean_loss') is not None:
            try:
                losses.append(float(s['mean_loss']))
                loss_episodes.append(i)
            except (TypeError, ValueError):
                pass
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Statistics - DQN', fontsize=16)
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward', marker='o', markersize=3)
    window = min(20, max(1, len(rewards) // 4))
    if window > 1 and len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Length
    ax2 = axes[0, 1]
    ax2.plot(episodes, steps, alpha=0.5, color='green', marker='o', markersize=3)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon
    ax3 = axes[0, 2]
    ax3.plot(episodes, epsilons, color='orange', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss
    ax4 = axes[1, 0]
    if losses and len(losses) > 0:
        ax4.plot(loss_episodes, losses, alpha=0.5, color='purple', marker='o', markersize=3)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.set_yscale('log')
    else:
        ax4.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Training Loss (No Data)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Q-Values
    ax5 = axes[1, 1]
    if mean_q_values and len(mean_q_values) > 0:
        ax5.plot(q_episodes, mean_q_values, alpha=0.5, color='cyan', marker='o', markersize=3)
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Mean Q-Value')
        ax5.set_title('Mean Q-Value Over Episodes')
    else:
        ax5.text(0.5, 0.5, 'No Q-value data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Mean Q-Value (No Data)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Collision Rate
    ax6 = axes[1, 2]
    collision_rate = [c / s if s > 0 else 0 for c, s in zip(collisions, steps)]
    ax6.plot(episodes, collision_rate, alpha=0.5, color='red', marker='o', markersize=3)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Collision Rate')
    ax6.set_title('Collision Rate (collisions/steps)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot next to JSON file
    json_path_obj = Path(json_path)
    plot_path = json_path_obj.parent / f"plot_{json_path_obj.stem}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Episodes: {len(stats)}")
    print(f"Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    if len(rewards) > 0:
        print(f"Best Reward: {np.max(rewards):.4f} (Episode {np.argmax(rewards) + 1})")
        print(f"Worst Reward: {np.min(rewards):.4f} (Episode {np.argmin(rewards) + 1})")
    print(f"Mean Episode Length: {np.mean(steps):.2f} steps")
    print(f"Total Collisions: {sum(collisions)}")
    if len(collision_rate) > 0:
        print(f"Mean Collision Rate: {np.mean(collision_rate):.4f}")
    if len(epsilons) > 0:
        print(f"Final Epsilon: {epsilons[-1]:.4f}")
    if mean_q_values:
        print(f"Final Mean Q-Value: {mean_q_values[-1]:.4f}")
    print("="*60)
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_stats.py <path_to_stats.json>")
        print("Example: python plot_stats.py ../../results/dqn_visibility/stats_20260115_123943.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    plot_from_json(json_path)