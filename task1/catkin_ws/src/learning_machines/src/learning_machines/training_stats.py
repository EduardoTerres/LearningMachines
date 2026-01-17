import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_statistics(stats, results_dir, timestamp, agent_type):
    """Create comprehensive training visualization plots.
    
    Args:
        stats: List of dictionaries with episode statistics
        results_dir: Directory to save plots
        timestamp: Timestamp string for filename
        agent_type: Type of agent ('dqn' or 'sac')
    """
    
    episodes = [s['episode'] for s in stats]
    rewards = [s['reward'] for s in stats]
    steps = [s['steps'] for s in stats]
    epsilons = [s.get('epsilon', 0) for s in stats]
    losses = [s.get('mean_loss', 0) for s in stats if s.get('mean_loss') is not None]
    mean_q_values = [s.get('mean_q_value', 0) for s in stats if s.get('mean_q_value') is not None]
    collisions = [s.get('collisions', 0) for s in stats]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Statistics - {agent_type.upper()}', fontsize=16)
    
    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    # Moving average
    window = min(20, len(rewards) // 4)
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Episode Length (Steps)
    ax2 = axes[0, 1]
    ax2.plot(episodes, steps, alpha=0.5, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax3 = axes[0, 2]
    ax3.plot(episodes, epsilons, color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss (if available)
    ax4 = axes[1, 0]
    if losses:
        loss_episodes = [s['episode'] for s in stats if s.get('mean_loss') is not None]
        ax4.plot(loss_episodes, losses, alpha=0.5, color='purple')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Training Loss')
        ax4.set_yscale('log')  # Log scale for loss
    else:
        ax4.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Training Loss (No Data)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Mean Q-Values
    ax5 = axes[1, 1]
    if mean_q_values:
        q_episodes = [s['episode'] for s in stats if s.get('mean_q_value') is not None]
        ax5.plot(q_episodes, mean_q_values, alpha=0.5, color='cyan')
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
    ax6.plot(episodes, collision_rate, alpha=0.5, color='red')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Collision Rate')
    ax6.set_title('Collision Rate (collisions/steps)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    plot_path = os.path.join(results_dir, f"training_stats_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training plots to {plot_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Episodes: {len(stats)}")
    print(f"Mean Reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"Best Reward: {np.max(rewards):.4f} (Episode {np.argmax(rewards) + 1})")
    print(f"Worst Reward: {np.min(rewards):.4f} (Episode {np.argmin(rewards) + 1})")
    print(f"Mean Episode Length: {np.mean(steps):.2f} steps")
    print(f"Total Collisions: {sum(collisions)}")
    print(f"Mean Collision Rate: {np.mean(collision_rate):.4f}")
    print(f"Final Epsilon: {epsilons[-1]:.4f}")
    if mean_q_values:
        print(f"Final Mean Q-Value: {mean_q_values[-1]:.4f}")
    print("="*60)