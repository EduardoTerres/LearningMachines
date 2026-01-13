#!/usr/bin/env python3

import sys
import os
import json
from datetime import datetime

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines.agent import DQNAgent
from learning_machines.rl_utilis import (
    extract_state, execute_action, compute_reward, detect_collision, ACTIONS
)


def train_episode(agent: DQNAgent, rob, max_steps: int = 500, episode_num: int = 0) -> dict:
    """Train DQN agent for one episode."""

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        rob.play_simulation()
    
    state = extract_state(rob)
    total_reward = 0.0
    steps = 0
    collision_count = 0
    
    while steps < max_steps:
        # Select action
        action_idx = agent.select_action(state, training=True)
        action = ACTIONS[action_idx]
        
        # Execute action
        execute_action(rob, action)
        rob.sleep(0.1)
        
        # Get next state
        next_state = extract_state(rob)
        
        # Check collision
        collision = detect_collision(next_state)
        if collision:
            collision_count += 1
        
        # Compute reward
        reward = compute_reward(state, action, collision, distance_traveled=0.0)
        
        # Check done
        done = collision or steps >= max_steps - 1
        
        # Store experience
        agent.replay_buffer.add(state, action_idx, reward, next_state, done)
        
        # Train agent
        agent.train_step()
        
        # Update state
        state = next_state
        total_reward += reward
        steps += 1
        
        if collision:
            break
    
    # Decay epsilon
    agent.decay_epsilon()
    
    return {
        'episode': episode_num,
        'total_reward': total_reward,
        'steps': steps,
        'collisions': collision_count,
        'epsilon': agent.epsilon
    }

    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    elif mode == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=8,
        num_actions=6,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_frequency=100
    )
    
    # Training parameters
    num_episodes = 100
    max_steps_per_episode = 500
    
    print("=" * 70)
    print("DQN Reinforcement Learning Training - Obstacle Avoidance")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Episodes: {num_episodes}")
    print(f"Actions: 6 discrete actions")
    print("=" * 70)
    
    training_stats = []
    
    # Training loop
    for episode in range(num_episodes):
        stats = train_episode(agent, rob, max_steps_per_episode, episode)
        training_stats.append(stats)
        
        if (episode + 1) % 10 == 0:
            recent_stats = training_stats[-10:]
            avg_reward = sum(s['total_reward'] for s in recent_stats) / 10
            avg_steps = sum(s['steps'] for s in recent_stats) / 10
            avg_collisions = sum(s['collisions'] for s in recent_stats) / 10
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Avg Collisions: {avg_collisions:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print("-" * 70)
    
    # Save model
    results_dir = "/root/results/model"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(results_dir, f"dqn_model_{timestamp}.h5")
    agent.save_model(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save statistics
    stats_path = os.path.join(results_dir, f"dqn_stats_{timestamp}.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"✓ Statistics saved to: {stats_path}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    print("\nTraining Complete!")


def validate_agent(agent: DQNAgent, rob, num_episodes: int = 7, max_steps: int = 500) -> dict:
    # Store original epsilon to restore later
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during validation (always use best action)
    
    validation_stats = []
    
    for episode in range(num_episodes):
        # Reset simulation
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
            rob.play_simulation()
        
        state = extract_state(rob)
        total_reward = 0.0
        steps = 0
        collision = False
        
        while steps < max_steps:
            # Select action WITHOUT exploration (always best action)
            action_idx = agent.select_action(state, training=False)
            action = ACTIONS[action_idx]
            
            # Execute action
            execute_action(rob, action)
            rob.sleep(0.1)
            
            # Get next state
            next_state = extract_state(rob)
            
            # Check collision
            collision = detect_collision(next_state)
            
            # Compute reward
            reward = compute_reward(state, action, collision, distance_traveled=0.0)
            
            # Update
            state = next_state
            total_reward += reward
            steps += 1
            
            if collision:
                break
        
        validation_stats.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'collision': bool(collision)
        })
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    # Compute aggregate metrics
    avg_reward = sum(s['total_reward'] for s in validation_stats) / len(validation_stats)
    avg_steps = sum(s['steps'] for s in validation_stats) / len(validation_stats)
    collision_rate = sum(1 for s in validation_stats if s['collision']) / len(validation_stats)
    
    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'collision_rate': collision_rate,
        'episodes': validation_stats
    }



def main():
    if len(sys.argv) < 2:
        raise ValueError("Pass --hardware or --simulation")
    
    mode = sys.argv[1]
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
    elif mode == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=8,
        num_actions=6,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_frequency=100
    )
    
    # Training parameters
    num_episodes = 2
    max_steps_per_episode = 500
    validation_frequency = 10  # Validate every N episodes
    
    print("=" * 70)
    print("DQN Reinforcement Learning Training - Obstacle Avoidance")
    print("=" * 70)
    print(f"Mode: {mode}")
    print(f"Episodes: {num_episodes}")
    print(f"Validation: Every {validation_frequency} episodes")
    print(f"Actions: 6 discrete actions")
    print("=" * 70)
    
    training_stats = []
    validation_results = []
    best_validation_reward = float('-inf')
    
    # Training loop
    for episode in range(num_episodes):
        # Train episode
        stats = train_episode(agent, rob, max_steps_per_episode, episode)
        training_stats.append(stats)
        
        # Print training progress
        if (episode + 1) % 10 == 0:
            recent_stats = training_stats[-10:]
            avg_reward = sum(s['total_reward'] for s in recent_stats) / 10
            avg_steps = sum(s['steps'] for s in recent_stats) / 10
            avg_collisions = sum(s['collisions'] for s in recent_stats) / 10
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Steps: {avg_steps:.1f}")
            print(f"  Avg Collisions: {avg_collisions:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print("-" * 70)
        
        # Validation
        if (episode + 1) % validation_frequency == 0:
            print("\n" + "=" * 70)
            print(f"Running validation after episode {episode + 1}...")
            print("=" * 70)
            
            val_results = validate_agent(agent, rob, num_episodes=7, max_steps=max_steps_per_episode)
            validation_results.append({
                'episode': episode + 1,
                'validation': val_results
            })
            
            print(f"Validation Results:")
            print(f"  Avg Reward: {val_results['avg_reward']:.2f}")
            print(f"  Avg Steps: {val_results['avg_steps']:.1f}")
            print(f"  Collision Rate: {val_results['collision_rate']:.2%}")
            print("=" * 70)
            
            # Save best model
            if val_results['avg_reward'] > best_validation_reward:
                best_validation_reward = val_results['avg_reward']
                results_dir = "/root/results/model"
                os.makedirs(results_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                best_model_path = os.path.join(results_dir, f"dqn_best_model_ep{episode+1}_{timestamp}.h5")
                agent.save_model(best_model_path)
                print(f"✓ New best model saved! (Reward: {best_validation_reward:.2f})")
            
            # Save validation results
            results_dir = "/root/results/model"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            val_stats_path = os.path.join(results_dir, f"validation_ep{episode+1}_{timestamp}.json")
            with open(val_stats_path, 'w') as f:
                json.dump(val_results, f, indent=2)
            print(f"✓ Validation results saved")
    
    # Save final model
    results_dir = "/root/results/model"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(results_dir, f"dqn_model_final_{timestamp}.h5")
    agent.save_model(model_path)
    print(f"\n✓ Final model saved to: {model_path}")
    
    # Save training statistics
    stats_path = os.path.join(results_dir, f"dqn_stats_{timestamp}.json")
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2)
    print(f"✓ Training statistics saved to: {stats_path}")
    
    # Save validation results
    val_path = os.path.join(results_dir, f"validation_all_{timestamp}.json")
    with open(val_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    print(f"✓ Validation results saved to: {val_path}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    print("\nTraining Complete!")
    print(f"Best validation reward: {best_validation_reward:.2f}")
    
if __name__ == "__main__":
    main()