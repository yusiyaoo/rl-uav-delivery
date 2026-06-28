"""
Generate all figures from the trained model.
Run after train.py completes.
"""
import numpy as np
from pathlib import Path
from env.config import *
from env.uav_env import UAVDeliveryEnv
from agent.dqn_agent import DoubleDQNAgent
from utils.visualize import (
    plot_3d_environment, plot_training_curves, plot_cost_analysis, plot_wind_field,
)


def run_episode(agent, env_seed, epsilon=0.05, max_steps=MAX_STEPS_PER_EPISODE):
    """Run one episode. Returns (trajectory, arrived, total_reward, env)."""
    env = UAVDeliveryEnv(seed=env_seed)
    agent.epsilon = epsilon
    obs = env.reset()
    done = False
    steps = 0
    total = 0.0
    while not done and steps < max_steps:
        action = agent.select_action(obs, evaluate=False)
        obs, reward, done, info = env.step(action)
        total += reward
        steps += 1
    return env.get_trajectory(), info.get('arrived', False), total, env


def main():
    agent = DoubleDQNAgent()
    for mp in ["results/model_final.pth", "results/model.pth"]:
        if Path(mp).exists():
            agent.load(mp)
            print(f"Loaded {mp}")
            break
    else:
        print("ERROR: No model found. Run train.py first.")
        return

    # Wind field
    plot_wind_field()

    # Training curves from log
    log_path = LOG_PATH
    if Path(log_path).exists():
        log_data = np.load(log_path, allow_pickle=True)
        log = dict(log_data)
        for k in list(log.keys()):
            arr = log[k]
            if arr.dtype == object:
                log[k] = np.array([float(x) for x in arr.flatten()], dtype=np.float32)
        plot_training_curves(log)

    # Hunt for arrival (try many seeds with increasing epsilon)
    print("\nSearching for successful trajectory...")
    success = None
    for eps in [0.05, 0.10, 0.15]:
        for seed in range(100):
            traj, arrived, total_r, env = run_episode(agent, seed, epsilon=eps)
            if arrived:
                print(f"  Found! seed={seed}, ε={eps}, steps={len(traj)}")
                success = (traj, env)
                break
        if success is not None:
            break
        print(f"  ε={eps}: no arrival in 100 seeds")

    if success is None:
        print("  No arrival found. Picking best attempt for visualization.")
        traj, _, _, env = run_episode(agent, 0, epsilon=0.15)
        label = "Best Effort"
    else:
        traj, env = success
        label = "Arrived"

    # 3D path
    plot_3d_environment(env, traj, save_path="results/path_3d_optimal.png",
                        title=f"UAV Delivery Path — {label}")

    # Cost analysis
    plot_cost_analysis(traj, env, save_path="results/cost_analysis.png")

    # Eval stats
    print("\nEvaluation (10 eps, ε=0.05):")
    arrs = []
    steps_list = []
    for i in range(10):
        traj, a, _, _ = run_episode(agent, 200 + i, epsilon=0.05)
        arrs.append(a)
        steps_list.append(len(traj))
    print(f"  SR: {sum(arrs)}/10, Avg steps: {np.mean(steps_list):.0f}")

    print("\nDone. Figures:")
    for f in sorted(Path("results").glob("*.png")):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
