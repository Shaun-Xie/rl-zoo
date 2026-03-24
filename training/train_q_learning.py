"""Training utilities for the tabular Q-learning baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from algorithms.q_learning import (
    QLearningConfig,
    build_q_learning_agent,
    observation_to_state,
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


def run_q_learning_training(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 800,
    max_steps: int = DEFAULT_MAX_STEPS,
    alpha: float = 0.2,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = False,
    model_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train a tabular Q-learning agent and save its outputs."""

    set_global_seeds(seed)

    training_config = QLearningConfig(
        alpha=alpha,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        episodes=episodes,
        max_steps=max_steps,
    )

    model_dir = SAVED_MODELS_DIR / "q_learning"
    output_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR / "q_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = Path(model_path) if model_path is not None else model_dir / "q_table.pkl"
    metrics_path = output_dir / "training_metrics.csv"
    summary_path = output_dir / "training_summary.json"

    env = MazeEnv(layout_name=layout_name, max_steps=max_steps, render_mode=None)
    agent = build_q_learning_agent(
        action_size=env.action_space.n,
        config=training_config,
        seed=seed,
    )

    wandb_run = maybe_init_wandb(
        enabled=use_wandb,
        project="rl-algorithm-zoo",
        run_name="q-learning",
        config={
            "algorithm": "q_learning",
            "layout_name": layout_name,
            "seed": seed,
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "episodes": episodes,
            "max_steps": max_steps,
        },
    )

    print(
        f"Training Q-learning for {episodes} episodes on "
        f"layout='{layout_name}' with seed={seed}."
    )

    episode_metrics: list[dict[str, float | int]] = []
    log_interval = max(1, episodes // 10)

    for episode in range(1, episodes + 1):
        observation, _ = env.reset(seed=seed + episode - 1)
        state = observation_to_state(observation)

        epsilon_used = agent.epsilon
        total_reward = 0.0
        success = False
        steps_taken = 0

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, explore=True)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state(next_observation)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            total_reward += reward
            steps_taken = step
            state = next_state

            if done:
                success = terminated
                break

        agent.decay_epsilon_value()

        metrics_row = {
            "episode": episode,
            "total_reward": round(total_reward, 4),
            "success": int(success),
            "steps": steps_taken,
            "epsilon": round(epsilon_used, 6),
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
                "epsilon": float(metrics_row["epsilon"]),
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
                f"epsilon={float(metrics_row['epsilon']):.3f} "
                f"recent_success={recent_success_rate:.2f}"
            )

    env.close()

    agent.save(
        final_model_path,
        metadata={
            "algorithm": "q_learning",
            "layout_name": layout_name,
            "seed": seed,
            "episodes": episodes,
            "max_steps": max_steps,
            "state_encoding": "integer tuple from compact observation vector",
        },
    )
    save_episode_metrics_csv(episode_metrics, metrics_path)

    plot_paths = plot_training_curves(
        episode_metrics,
        output_dir=output_dir,
        reward_filename="reward_curve.png",
        success_filename="success_rate.png",
        algorithm_name="Q-Learning",
    )

    last_window = episode_metrics[-50:] if episode_metrics else []
    summary = {
        "algorithm": "q_learning",
        "layout_name": layout_name,
        "seed": seed,
        "episodes": episodes,
        "max_steps": max_steps,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": round(agent.epsilon, 6),
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

    print(f"Saved Q-table to {final_model_path}")
    print(f"Saved training metrics to {metrics_path}")
    print(f"Saved training summary to {summary_path}")

    return summary
