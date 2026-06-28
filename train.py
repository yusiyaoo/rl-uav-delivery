"""
Training script for UAV City Delivery DQN agent.

Usage:
    python train.py
"""

import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from env.config import *
from env.uav_env import UAVDeliveryEnv
from agent.dqn_agent import DoubleDQNAgent
from utils.visualize import plot_training_curves, plot_3d_environment, plot_wind_field


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(agent: DoubleDQNAgent, env: UAVDeliveryEnv, n_episodes: int = EVAL_EPISODES):
    """
    Evaluate agent (greedy, no exploration).
    Returns: (success_rate, avg_steps, avg_min_distance)
    """
    successes = 0
    total_steps = 0
    total_min_dist = 0.0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_min_dist = float('inf')
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, info = env.step(action)
            # Track min obstacle distance
            for ob in env.obstacles:
                d = ob.distance_to(env.pos)
                if d < episode_min_dist:
                    episode_min_dist = d
            steps += 1

        total_steps += steps
        total_min_dist += episode_min_dist if episode_min_dist < float('inf') else 0.0
        if info.get('arrived', False):
            successes += 1

    success_rate = successes / n_episodes
    avg_steps = total_steps / n_episodes
    avg_min_dist = total_min_dist / n_episodes
    return success_rate, avg_steps, avg_min_dist


def main():
    print("=" * 60)
    print("UAV City Delivery — Double DQN Training")
    print("=" * 60)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Training episodes: {N_EPISODES}, Batch size: {BATCH_SIZE}")
    print(f"Buffer capacity: {BUFFER_CAPACITY}, Learning rate: {LR}")
    print("=" * 60)

    set_seed(SEED)

    # Create results directory
    Path("results").mkdir(parents=True, exist_ok=True)

    # Initialize
    env = UAVDeliveryEnv(seed=SEED)
    agent = DoubleDQNAgent()

    # Logging
    log = defaultdict(list)
    best_success_rate = 0.0

    # Generate wind field visualization once
    plot_wind_field()

    # Progress bar
    pbar = tqdm(range(1, N_EPISODES + 1), desc="Training", unit="ep")

    for episode in pbar:
        obs = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.learn()

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if episode_steps >= MAX_STEPS_PER_EPISODE:
                break

        # Logging
        log['episode_rewards'].append(episode_reward)
        log['losses'].append(loss)
        log['epsilons'].append(agent.epsilon)

        # Decay exploration and anneal PER beta
        agent.decay_epsilon()
        agent.anneal_beta()

        # Evaluation
        if episode % EVAL_FREQ == 0:
            sr, avg_steps, avg_min_dist = evaluate(agent, env)
            log['eval_success_rates'].append(sr)
            log['eval_episodes'].append(episode)
            log['eval_avg_steps'].append(avg_steps)
            log['eval_avg_min_dist'].append(avg_min_dist)

            # Save best model
            if sr >= best_success_rate:
                best_success_rate = sr
                agent.save(SAVE_PATH)
                # Save a successful trajectory
                _save_best_trajectory(agent, env)

            pbar.set_postfix({
                'reward': f'{episode_reward:.1f}',
                'ε': f'{agent.epsilon:.3f}',
                'SR': f'{sr:.2f}',
                'loss': f'{loss:.4f}',
                'buffer': len(agent.buffer),
            })

    pbar.close()

    # Save final model
    agent.save("results/model_final.pth")

    # Save training log
    np.savez(LOG_PATH, **{k: np.array(v, dtype=object) for k, v in log.items()})

    # --- Generate training curves ---
    print("\nGenerating training curves...")
    plot_training_curves(dict(log))

    # --- Final evaluation and trajectory ---
    print("Running final evaluation...")
    _save_best_trajectory(agent, env, suffix="final")

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best evaluation success rate: {best_success_rate:.2%}")
    print(f"Results saved to results/")
    print("=" * 60)


def _save_best_trajectory(agent: DoubleDQNAgent, env: UAVDeliveryEnv, suffix: str = "best"):
    """Run a greedy trajectory and save the 3D plot."""
    obs = env.reset()
    done = False
    steps = 0
    while not done and steps < MAX_STEPS_PER_EPISODE:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, done, info = env.step(action)
        steps += 1
    trajectory = env.get_trajectory()
    arrived = info.get('arrived', False)
    status = "Arrived" if arrived else "Failed"
    title = f"UAV Delivery Path ({status})"
    plot_3d_environment(env, trajectory,
                        save_path=f"results/path_3d_{suffix}.png",
                        title=title)
    print(f"  Trajectory: {len(trajectory)} steps, arrived={arrived}")


if __name__ == '__main__':
    main()
