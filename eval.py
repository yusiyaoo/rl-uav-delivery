"""
Evaluation script — generates all figures.

Usage:
    python eval.py
"""

import numpy as np
from pathlib import Path

from env.config import *
from env.uav_env import UAVDeliveryEnv
from agent.dqn_agent import DoubleDQNAgent
from utils.visualize import (
    plot_3d_environment,
    plot_training_curves,
    plot_cost_analysis,
    plot_comparison_bar,
    plot_wind_field,
)


def main():
    print("=" * 60)
    print("UAV Delivery — Evaluation & Report Figure Generation")
    print("=" * 60)

    Path("results").mkdir(parents=True, exist_ok=True)

    # Load trained agent
    agent = DoubleDQNAgent()
    model_path = SAVE_PATH if Path(SAVE_PATH).exists() else "results/model_final.pth"

    if not Path(model_path).exists():
        print(f"Model not found at {model_path}. Run train.py first.")
        print("Generating figures from log data only...")
        _generate_from_log()
        return

    agent.load(model_path)
    agent.epsilon = 0.0  # greedy for evaluation
    print(f"Loaded model from {model_path}")

    # --- Generate wind field ---
    print("\n1. Generating wind field visualization...")
    plot_wind_field()

    # --- Run multiple evaluation episodes ---
    print("\n2. Running evaluation episodes...")
    env = UAVDeliveryEnv(seed=999)

    for i in range(5):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < MAX_STEPS_PER_EPISODE:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, info = env.step(action)
            steps += 1

        trajectory = env.get_trajectory()
        arrived = info.get('arrived', False)
        print(f"  Episode {i+1}: {len(trajectory)} steps, arrived={arrived}")

        if arrived and i == 0:
            # Save the best trajectory
            title = "UAV Optimal Delivery Path (Trained DQN)"
            plot_3d_environment(env, trajectory,
                                save_path="results/path_3d_optimal.png",
                                title=title)

            # Cost analysis for this trajectory
            plot_cost_analysis(trajectory, env,
                               save_path="results/cost_analysis.png")

    # --- Training curves from log ---
    print("\n3. Generating training curves from log...")
    _generate_from_log()

    # --- Ablation comparison (placeholder values from log) ---
    print("\n4. Generating comparison chart...")
    log_path = Path(LOG_PATH)
    if log_path.exists():
        data = np.load(log_path, allow_pickle=True)
        log = dict(data)
        if 'eval_success_rates' in log and len(log['eval_success_rates']) > 0:
            final_sr = log['eval_success_rates'][-1]
            final_steps = log.get('eval_avg_steps', [0])[-1] if 'eval_avg_steps' in log else 0
            final_dist = log.get('eval_avg_min_dist', [0])[-1] if 'eval_avg_min_dist' in log else 0
        else:
            final_sr, final_steps, final_dist = 0, 0, 0
    else:
        final_sr, final_steps, final_dist = 0, 0, 0

    metrics = {
        'DQN\n(Dueling+PER)': {
            'Success Rate': final_sr,
            'Avg Steps': final_steps,
            'Min Safety Dist (m)': final_dist,
        },
        # Placeholder comparison — can be filled after running ablations
        # 'Vanilla DQN': {
        #     'Success Rate': 0.55,
        #     'Avg Steps': 320,
        #     'Min Safety Dist (m)': 12.0,
        # },
    }
    if final_sr > 0:
        plot_comparison_bar(metrics, save_path="results/comparison.png")

    print("\n" + "=" * 60)
    print("All figures generated in results/")
    print("Files:")
    for f in sorted(Path("results").glob("*.png")):
        print(f"  {f.name}")
    print("=" * 60)


def _generate_from_log():
    """Generate training curves from saved log."""
    log_path = Path(LOG_PATH)
    if not log_path.exists():
        print("  No training log found. Skipping.")
        return

    data = np.load(log_path, allow_pickle=True)
    log = dict(data)

    # Convert object arrays to proper types
    for k, v in log.items():
        if isinstance(v, np.ndarray) and v.dtype == object:
            log[k] = v.astype(np.float32)
    # Handle eval_episodes specially (may contain varying-length arrays)
    if 'eval_episodes' in log:
        flat_eps = []
        for item in log['eval_episodes']:
            if isinstance(item, (np.ndarray, list)):
                flat_eps.extend(item)
            else:
                flat_eps.append(item)
        log['eval_episodes'] = np.array(flat_eps, dtype=np.int32)

    if 'eval_success_rates' in log:
        flat_sr = []
        for item in log['eval_success_rates']:
            if isinstance(item, (np.ndarray, list)):
                flat_sr.extend(item)
            else:
                flat_sr.append(item)
        log['eval_success_rates'] = np.array(flat_sr, dtype=np.float32)

    plot_training_curves(log)


if __name__ == '__main__':
    main()
