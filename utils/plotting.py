"""Simple plotting helpers for saved training metrics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

EpisodeMetrics = Sequence[Mapping[str, float | int]]


def moving_average(values: Sequence[float], window_size: int = 50) -> list[float]:
    """Compute a simple trailing moving average."""

    if not values:
        return []

    averages: list[float] = []
    for index in range(len(values)):
        start_index = max(0, index - window_size + 1)
        window = values[start_index : index + 1]
        averages.append(sum(window) / len(window))
    return averages


def plot_training_curves(
    metrics: EpisodeMetrics,
    *,
    output_dir: str | Path,
    reward_filename: str = "reward_curve.png",
    success_filename: str = "success_rate.png",
    window_size: int = 50,
) -> dict[str, Path]:
    """Save a reward curve and success-rate trend for training metrics."""

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training plots.")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    episodes = [int(row["episode"]) for row in metrics]
    rewards = [float(row["total_reward"]) for row in metrics]
    successes = [float(row["success"]) for row in metrics]

    reward_trend = moving_average(rewards, window_size=window_size)
    success_trend = moving_average(successes, window_size=window_size)

    reward_path = output_path / reward_filename
    success_path = output_path / success_filename

    reward_figure, reward_axis = plt.subplots(figsize=(9, 4.5))
    reward_axis.plot(episodes, rewards, label="episode reward", alpha=0.4, linewidth=1.0)
    reward_axis.plot(
        episodes,
        reward_trend,
        label=f"{window_size}-episode average",
        linewidth=2.0,
    )
    reward_axis.set_title("Q-Learning Reward Curve")
    reward_axis.set_xlabel("Episode")
    reward_axis.set_ylabel("Total Reward")
    reward_axis.legend()
    reward_axis.grid(alpha=0.25)
    reward_figure.tight_layout()
    reward_figure.savefig(reward_path, dpi=150)
    plt.close(reward_figure)

    success_figure, success_axis = plt.subplots(figsize=(9, 4.5))
    success_axis.plot(
        episodes,
        success_trend,
        label=f"{window_size}-episode success rate",
        linewidth=2.0,
        color="tab:green",
    )
    success_axis.set_title("Q-Learning Success Trend")
    success_axis.set_xlabel("Episode")
    success_axis.set_ylabel("Success Rate")
    success_axis.set_ylim(0.0, 1.05)
    success_axis.legend()
    success_axis.grid(alpha=0.25)
    success_figure.tight_layout()
    success_figure.savefig(success_path, dpi=150)
    plt.close(success_figure)

    return {
        "reward_curve": reward_path,
        "success_rate": success_path,
    }
