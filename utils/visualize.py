"""
Visualization utilities for UAV path planning results.

Generates publication-quality figures using matplotlib.
No screenshots — all figures are programmatically rendered.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from pathlib import Path
from env.config import *


# Matplotlib global style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})


def plot_3d_environment(env, trajectory: np.ndarray, save_path: str = "results/path_3d.png",
                        title: str = "UAV Delivery Path in 3D City Environment"):
    """
    Plot the 3D city environment with buildings, no-fly zones, and UAV trajectory.

    Args:
        env: UAVDeliveryEnv instance (after episode, with obstacles generated)
        trajectory: (N, 3) array of UAV positions
        save_path: file path to save the figure
        title: plot title
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot buildings
    for b in env.buildings:
        x_verts = [b.cx - b.w/2, b.cx + b.w/2, b.cx + b.w/2, b.cx - b.w/2]
        y_verts = [b.cy - b.d/2, b.cy - b.d/2, b.cy + b.d/2, b.cy + b.d/2]
        z_bottom = 0
        z_top = b.h

        # Draw building as translucent box
        verts = []
        # Bottom face
        verts.append([(x_verts[0], y_verts[0], z_bottom),
                      (x_verts[1], y_verts[1], z_bottom),
                      (x_verts[2], y_verts[2], z_bottom),
                      (x_verts[3], y_verts[3], z_bottom)])
        # Top face
        verts.append([(x_verts[0], y_verts[0], z_top),
                      (x_verts[1], y_verts[1], z_top),
                      (x_verts[2], y_verts[2], z_top),
                      (x_verts[3], y_verts[3], z_top)])
        # Side faces
        for i in range(4):
            j = (i + 1) % 4
            verts.append([(x_verts[i], y_verts[i], z_bottom),
                          (x_verts[j], y_verts[j], z_bottom),
                          (x_verts[j], y_verts[j], z_top),
                          (x_verts[i], y_verts[i], z_top)])

        building_collection = Poly3DCollection(verts, alpha=0.3, facecolor='gray',
                                                edgecolor='darkgray', linewidth=0.5)
        ax.add_collection3d(building_collection)

    # Plot no-fly zones as red translucent cylinders
    for nf in env.nofly_zones:
        u = np.linspace(0, 2 * np.pi, 40)
        z_bottom = nf.z_min
        z_top = nf.z_max
        # Draw as wireframe circles at top and bottom
        x_circle = nf.cx + nf.radius * np.cos(u)
        y_circle = nf.cy + nf.radius * np.sin(u)
        ax.plot(x_circle, y_circle, z_bottom, color='red', linewidth=1.0, alpha=0.7)
        ax.plot(x_circle, y_circle, z_top, color='red', linewidth=1.0, alpha=0.7)
        # Vertical lines
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            ax.plot([nf.cx + nf.radius * np.cos(angle)] * 2,
                    [nf.cy + nf.radius * np.sin(angle)] * 2,
                    [z_bottom, z_top], color='red', linewidth=0.8, alpha=0.5)

    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
            color='blue', linewidth=2.0, label='UAV Path', zorder=10)

    # Mark start and goal
    ax.scatter(*START_POS, color='green', s=120, marker='o',
               label='Start (Delivery Station)', zorder=11)
    ax.scatter(*GOAL_POS, color='orange', s=120, marker='*',
               label='Goal (Delivery Point)', zorder=11)

    # Labels and formatting
    ax.set_xlabel('X (m) — East')
    ax.set_ylabel('Y (m) — North')
    ax.set_zlabel('Z (m) — Altitude')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_zlim(0, SPACE_Z)
    ax.view_init(elev=30, azim=-60)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualize] Saved 3D path plot to {save_path}")


def plot_training_curves(log_data: dict, save_path: str = "results/training_curves.png"):
    """
    Plot training metrics: reward, loss, success rate, epsilon over episodes.

    Args:
        log_data: dict with keys 'episode_rewards', 'losses', 'eval_success_rates',
                  'eval_episodes', 'epsilons'
        save_path: file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = np.arange(1, len(log_data['episode_rewards']) + 1)

    # (a) Episode Reward
    ax = axes[0, 0]
    ax.plot(episodes, log_data['episode_rewards'], color='steelblue', linewidth=0.6, alpha=0.6)
    # Smoothed reward
    window = max(1, len(episodes) // 100)
    if len(episodes) >= window:
        smoothed = np.convolve(log_data['episode_rewards'],
                               np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], smoothed, color='darkblue', linewidth=2.0,
                label=f'Moving Avg (window={window})')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('(a) Episode Reward')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) TD Loss
    ax = axes[0, 1]
    loss_episodes = np.arange(1, len(log_data['losses']) + 1)
    ax.plot(loss_episodes, log_data['losses'], color='coral', linewidth=0.5, alpha=0.5)
    if len(loss_episodes) >= window:
        smoothed_loss = np.convolve(log_data['losses'],
                                    np.ones(window)/window, mode='valid')
        ax.plot(loss_episodes[window-1:], smoothed_loss, color='darkred', linewidth=2.0,
                label=f'Moving Avg (window={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('TD Loss')
    ax.set_title('(b) Training Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (c) Evaluation Success Rate
    ax = axes[1, 0]
    if len(log_data.get('eval_success_rates', [])) > 0:
        eval_eps = log_data.get('eval_episodes',
                                np.arange(1, len(log_data['eval_success_rates'])+1) * EVAL_FREQ)
        ax.plot(eval_eps, np.array(log_data['eval_success_rates'])*100,
                color='seagreen', marker='o', markersize=3, linewidth=1.5)
        ax.set_ylim(-5, 105)
        ax.axhline(y=80, color='gray', linestyle='--', linewidth=0.8, label='80% threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('(c) Evaluation Success Rate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Epsilon Decay
    ax = axes[1, 1]
    ax.plot(episodes, log_data['epsilons'], color='purple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('ε')
    ax.set_title('(d) Exploration Rate ε Decay')
    ax.grid(True, alpha=0.3)

    fig.suptitle('DQN Training Progress', fontsize=14, fontweight='bold')
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualize] Saved training curves to {save_path}")


def plot_cost_analysis(trajectory: np.ndarray, env,
                       save_path: str = "results/cost_analysis.png"):
    """
    Plot energy, safety, and altitude analysis along the trajectory.

    Args:
        trajectory: (N, 3) UAV positions
        env: environment instance with obstacles
        save_path: file path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = np.arange(len(trajectory))

    # (a) Distance to Goal over time
    ax = axes[0, 0]
    dist_to_goal = np.linalg.norm(trajectory - GOAL_POS, axis=1)
    ax.plot(steps, dist_to_goal, color='steelblue', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Goal (m)')
    ax.set_title('(a) Distance to Goal vs. Time')
    ax.grid(True, alpha=0.3)
    ax.fill_between(steps, 0, dist_to_goal, color='steelblue', alpha=0.1)

    # (b) Safety: Minimum distance to obstacles
    ax = axes[0, 1]
    min_dists = []
    for pos in trajectory:
        min_d = float('inf')
        for obs in env.obstacles:
            d = obs.distance_to(pos)
            if d < min_d:
                min_d = d
        min_dists.append(min_d)
    min_dists = np.array(min_dists)
    ax.plot(steps, min_dists, color='coral', linewidth=1.5)
    ax.axhline(y=SAFETY_MARGIN, color='red', linestyle='--', linewidth=1.0,
               label=f'Safety Margin ({SAFETY_MARGIN}m)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Min Obstacle Distance (m)')
    ax.set_title('(b) Safety Distance vs. Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.fill_between(steps, 0, min_dists, color='coral', alpha=0.1)

    # (c) Altitude Profile
    ax = axes[1, 0]
    ax.plot(steps, trajectory[:, 2], color='seagreen', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('(c) Altitude Profile')
    ax.grid(True, alpha=0.3)
    # Mark goal altitude
    ax.axhline(y=GOAL_POS[2], color='orange', linestyle='--', linewidth=1.0, label='Goal Altitude')
    ax.legend(fontsize=9)

    # (d) Step-wise displacement (speed proxy)
    ax = axes[1, 1]
    step_dists = np.linalg.norm(np.diff(trajectory, axis=0, prepend=trajectory[:1]), axis=1)
    ax.plot(steps, step_dists, color='purple', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Step Displacement (m)')
    ax.set_title('(d) Step-wise Movement Magnitude')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Trajectory Cost Analysis', fontsize=14, fontweight='bold')
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualize] Saved cost analysis to {save_path}")


def plot_comparison_bar(metrics: dict, save_path: str = "results/comparison.png"):
    """
    Bar chart comparing multiple metrics (e.g., DQN vs baselines or different configs).

    Args:
        metrics: dict of {label: {metric_name: value, ...}, ...}
        save_path: file path
    """
    if not metrics:
        return

    labels = list(metrics.keys())
    metric_names = list(metrics[labels[0]].keys())
    n_metrics = len(metric_names)
    n_groups = len(labels)

    x = np.arange(n_groups)
    width = 0.7 / n_metrics
    colors = ['steelblue', 'coral', 'seagreen', 'purple', 'orange']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mname in enumerate(metric_names):
        values = [metrics[l][mname] for l in labels]
        offset = (i - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, values, width, label=mname, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualize] Saved comparison bar to {save_path}")


def plot_wind_field(save_path: str = "results/wind_field.png"):
    """Visualize the wind vector field in 2D."""
    resolution = 20
    x = np.linspace(0, SPACE_X, resolution)
    y = np.linspace(0, SPACE_Y, resolution)
    X, Y = np.meshgrid(x, y)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(resolution):
        for j in range(resolution):
            pos = np.array([X[i, j], Y[i, j], 50.0])
            wind = WIND_PREVAILING + np.array([
                WIND_VARIANCE * np.sin(2 * np.pi * Y[i, j] / SPACE_Y + np.pi / 4),
                WIND_VARIANCE * np.cos(2 * np.pi * X[i, j] / SPACE_X),
            ])
            U[i, j] = wind[0]
            V[i, j] = wind[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(X, Y, U, V, color='steelblue', alpha=0.7, scale=50, width=0.003)
    ax.scatter(START_POS[0], START_POS[1], color='green', s=100, marker='o',
               label='Start', zorder=5)
    ax.scatter(GOAL_POS[0], GOAL_POS[1], color='orange', s=100, marker='*',
               label='Goal', zorder=5)
    ax.set_xlabel('X (m) — East')
    ax.set_ylabel('Y (m) — North')
    ax.set_title('Wind Field (Horizontal Plane)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, SPACE_X)
    ax.set_ylim(0, SPACE_Y)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Visualize] Saved wind field to {save_path}")
