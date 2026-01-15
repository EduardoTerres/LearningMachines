#!/usr/bin/env python3
"""Validate/evaluate trained RL model by testing it on the environment.

This tests the trained policy WITHOUT exploration to measure actual performance.

Usage:
  python validation.py --simulation --model /path/to/model.h5 --agent dqn
  python validation.py --hardware --model /path/to/model.h5 --agent sac
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

from robobo_interface import HardwareRobobo, SimulationRobobo
from learning_machines import RoboboIREnv, DQNAgent, SACAgent


def build_env(mode: str) -> Tuple[RoboboIREnv, str]:
    """Build environment based on mode."""
    if mode == "--hardware":
        rob = HardwareRobobo(camera=False)
        hw_type = "hardware"
    else:
        rob = SimulationRobobo(identifier=1)
        hw_type = "simulation"
    env = RoboboIREnv(rob=rob)
    return env, hw_type


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
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate/evaluate trained RL model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate latest model in simulation
  python validation.py --simulation --agent dqn
  
  # Validate specific model
  python validation.py --simulation --agent sac --model /root/results/model/sac_model_20260115_094032.h5
  
  # Validate with more episodes
  python validation.py --simulation --agent dqn --episodes 100
        """
    )
    
    parser.add_argument("mode", choices=["--simulation", "--hardware"], 
                       help="Environment mode")
    parser.add_argument("--agent", choices=["sac", "dqn"], default="sac",
                       help="Agent type (default: sac)")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file. If not specified, uses newest model in /root/results/model/")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of validation episodes (default: 50)")
    parser.add_argument("--max-steps", type=int, default=10,
                       help="Maximum steps per episode (default: 10)")
    parser.add_argument("--save-results", action="store_true",
                       help="Save validation results to JSON file")
    
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model is None:
        model_dir = Path("/root/results/model")
        if not model_dir.exists():
            print(f"ERROR: Model directory not found: {model_dir}")
            sys.exit(1)
        
        models = list(model_dir.glob(f"*{args.agent}_model_*.h5"))
        if not models:
            print(f"ERROR: No {args.agent} model files found in {model_dir}")
            sys.exit(1)
        
        # Get newest model
        args.model = str(max(models, key=lambda p: p.stat().st_mtime))
        print(f"Using newest model: {Path(args.model).name}")
    
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    
    # Setup environment
    env, hw_type = build_env(args.mode)
    sample_obs, _ = env.reset()
    state_dim = int(np.array(sample_obs).reshape(-1).shape[0])
    action_dim = env.action_space.n
    
    # Load agent
    print(f"\nLoading model: {Path(args.model).name}")
    agent = load_agent(args.agent, state_dim, action_dim, args.model)
    
    # Run validation
    metrics = validate_model(env, agent, args.episodes, args.max_steps)
    env.close()
    
    # Print results
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Model: {Path(args.model).name}")
    print(f"Agent: {args.agent.upper()}")
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
    
    # Save results if requested
    if args.save_results:
        from datetime import datetime
        results_dir = Path("/root/results/validation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"validation_{args.agent}_{hw_type}_{ts}.json"
        
        results = {
            "model_path": args.model,
            "agent_type": args.agent,
            "mode": hw_type,
            "timestamp": ts,
            "metrics": metrics
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
    
    # Exit with appropriate code
    # You can add thresholds here if needed
    # For now, always succeeds
    sys.exit(0)


if __name__ == "__main__":
    main()