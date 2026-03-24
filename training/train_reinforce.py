"""Training utilities for the PyTorch REINFORCE baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from algorithms.reinforce import (
    EpisodeTrajectory,
    ReinforceConfig,
    build_reinforce_agent,
)
from config import DEFAULT_LAYOUT_NAME, DEFAULT_MAX_STEPS, DEFAULT_SEED, RESULTS_DIR, SAVED_MODELS_DIR
from env.maze_env import MazeEnv
from utils.logging_utils import (
    finish_wandb,
    log_metrics,
    maybe_init_wandb,
    save_episode_metrics_csv,
    save_json,
)
from utils.plotting import plot_training_curves
from utils.seed import set_global_seeds


def run_reinforce_training(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 1000,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    hidden_size: int = 64,
    normalize_returns: bool = True,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = False,
    model_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a REINFORCE policy and save its outputs."""

    set_global_seeds(seed)

    training_config = ReinforceConfig(
        learning_rate=learning_rate,
        gamma=gamma,
        episodes=episodes,
        max_steps=max_steps,
        hidden_size=hidden_size,
        normalize_returns=normalize_returns,
    )

    model_dir = SAVED_MODELS_DIR / "reinforce"
    output_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR / "reinforce"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = Path(model_path) if model_path is not None else model_dir / "policy.pt"
    metrics_path = output_dir / "training_metrics.csv"
    summary_path = output_dir / "training_summary.json"

    env = MazeEnv(layout_name=layout_name, max_steps=max_steps, render_mode=None)
    observation_size = int(np.prod(env.observation_space.shape))
    action_size = int(env.action_space.n)
    observation_scale = np.asarray(env.observation_space.high, dtype=np.float32)

    agent = build_reinforce_agent(
        observation_size=observation_size,
        action_size=action_size,
        config=training_config,
        observation_scale=observation_scale,
    )

    wandb_run = maybe_init_wandb(
        enabled=use_wandb,
        project="rl-algorithm-zoo",
        run_name="reinforce",
        config={
            "algorithm": "reinforce",
            "layout_name": layout_name,
            "seed": seed,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "hidden_size": hidden_size,
            "normalize_returns": normalize_returns,
            "episodes": episodes,
            "max_steps": max_steps,
        },
    )

    print(
        f"Training REINFORCE for {episodes} episodes on "
        f"layout='{layout_name}' with seed={seed}."
    )

    episode_metrics: list[dict[str, float | int]] = []
    log_interval = max(1, episodes // 10)
    pending_trajectories: list[EpisodeTrajectory] = []
    latest_loss = 0.0

    for episode in range(1, episodes + 1):
        observation, _ = env.reset(seed=seed + episode - 1)
        trajectory = EpisodeTrajectory()

        total_reward = 0.0
        success = False
        steps_taken = 0

        for step in range(1, max_steps + 1):
            action, log_prob, entropy = agent.select_action(observation, deterministic=False)
            next_observation, reward, terminated, truncated, _ = env.step(action)

            trajectory.add_step(
                state=observation,
                action=action,
                reward=reward,
                log_prob=log_prob,
                entropy=entropy,
            )

            total_reward += reward
            steps_taken = step
            observation = next_observation

            if terminated or truncated:
                success = terminated
                break

        pending_trajectories.append(trajectory)
        if len(pending_trajectories) >= training_config.batch_episodes or episode == episodes:
            latest_loss = agent.update_batch(pending_trajectories)
            pending_trajectories.clear()

        metrics_row = {
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "success": int(success),
            "steps": steps_taken,
            "policy_loss": round(latest_loss, 6),
        }
        episode_metrics.append(metrics_row)

        recent_metrics = episode_metrics[-50:]
        average_recent_reward = sum(float(item["total_reward"]) for item in recent_metrics) / len(
            recent_metrics
        )
        recent_success_rate = sum(int(item["success"]) for item in recent_metrics) / len(
            recent_metrics
        )

        log_metrics(
            {
                "episode_reward": float(metrics_row["total_reward"]),
                "episode_success": int(metrics_row["success"]),
                "episode_steps": int(metrics_row["steps"]),
                "policy_loss": float(metrics_row["policy_loss"]),
                "recent_reward_mean": average_recent_reward,
                "recent_success_rate": recent_success_rate,
            },
            step=episode,
            run=wandb_run,
        )

        if episode == 1 or episode % log_interval == 0 or episode == episodes:
            print(
                f"episode={episode:04d} "
                f"reward={float(metrics_row['total_reward']):>6.2f} "
                f"success={int(metrics_row['success'])} "
                f"steps={int(metrics_row['steps']):02d} "
                f"loss={float(metrics_row['policy_loss']):>8.3f} "
                f"recent_success={recent_success_rate:.2f}"
            )

    env.close()

    agent.save(
        final_model_path,
        metadata={
            "algorithm": "reinforce",
            "layout_name": layout_name,
            "seed": seed,
            "episodes": episodes,
            "max_steps": max_steps,
            "observation_size": observation_size,
            "action_size": action_size,
        },
    )
    save_episode_metrics_csv(episode_metrics, metrics_path)

    plot_paths = plot_training_curves(
        episode_metrics,
        output_dir=output_dir,
        reward_filename="reward_curve.png",
        success_filename="success_rate.png",
        algorithm_name="REINFORCE",
    )

    last_window = episode_metrics[-50:] if episode_metrics else []
    summary = {
        "algorithm": "reinforce",
        "layout_name": layout_name,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "hidden_size": hidden_size,
        "normalize_returns": normalize_returns,
        "batch_episodes": training_config.batch_episodes,
        "entropy_coefficient": training_config.entropy_coefficient,
        "average_reward": round(
            sum(float(item["total_reward"]) for item in episode_metrics) / max(len(episode_metrics), 1),
            4,
        ),
        "success_rate": round(
            sum(int(item["success"]) for item in episode_metrics) / max(len(episode_metrics), 1),
            4,
        ),
        "recent_reward_mean": round(
            sum(float(item["total_reward"]) for item in last_window) / max(len(last_window), 1),
            4,
        ),
        "recent_success_rate": round(
            sum(int(item["success"]) for item in last_window) / max(len(last_window), 1),
            4,
        ),
        "model_path": str(final_model_path),
        "metrics_path": str(metrics_path),
        "plot_paths": {name: str(path) for name, path in plot_paths.items()},
    }
    save_json(summary, summary_path)
    finish_wandb(wandb_run)

    print(f"Saved REINFORCE model to {final_model_path}")
    print(f"Saved training metrics to {metrics_path}")
    print(f"Saved training summary to {summary_path}")

    return summary
