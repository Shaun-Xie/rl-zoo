"""Training utilities for the Stable-Baselines3 PPO baseline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from algorithms.sb3_ppo import build_ppo_model, make_ppo_env
from config import (
    DEFAULT_LAYOUT_NAME,
    DEFAULT_MAX_STEPS,
    DEFAULT_PPO_TIMESTEPS,
    DEFAULT_SEED,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
)
from utils.logging_utils import (
    finish_wandb,
    log_metrics,
    maybe_init_wandb,
    save_episode_metrics_csv,
    save_json,
)
from utils.plotting import plot_training_curves
from utils.seed import set_global_seeds


def run_ppo_training(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    total_timesteps: int = DEFAULT_PPO_TIMESTEPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    learning_rate: float = 0.0003,
    gamma: float = 0.99,
    n_steps: int = 128,
    batch_size: int = 128,
    ent_coef: float = 0.01,
    num_envs: int = 4,
    seed: int = DEFAULT_SEED,
    use_wandb: bool = False,
    model_path: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train an SB3 PPO agent and save its artifacts."""

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 is required for PPO training. "
            "Install the dependencies in requirements.txt."
        ) from exc

    set_global_seeds(seed)

    model_dir = SAVED_MODELS_DIR / "ppo"
    output_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR / "ppo"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = Path(model_path) if model_path is not None else model_dir / "model.zip"
    metrics_path = output_dir / "training_metrics.csv"
    summary_path = output_dir / "training_summary.json"

    env = DummyVecEnv(
        [
            make_ppo_env(layout_name=layout_name, max_steps=max_steps, seed=seed + index)
            for index in range(num_envs)
        ]
    )
    model = build_ppo_model(
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        seed=seed,
    )

    wandb_run = maybe_init_wandb(
        enabled=use_wandb,
        project="rl-algorithm-zoo",
        run_name="ppo",
        config={
            "algorithm": "ppo",
            "layout_name": layout_name,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "ent_coef": ent_coef,
            "num_envs": num_envs,
        },
    )

    print(
        f"Training PPO for {total_timesteps} timesteps on "
        f"layout='{layout_name}' with seed={seed}."
    )

    class EpisodeMetricsCallback(BaseCallback):
        """Collect lightweight per-episode stats from the monitored env."""

        def __init__(self, run: Any | None = None) -> None:
            super().__init__()
            self.run = run
            self.rows: list[dict[str, float | int]] = []

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            infos = self.locals.get("infos")

            if dones is None or infos is None:
                return True

            for done, info in zip(dones, infos):
                if not done:
                    continue

                episode_info = info.get("episode")
                if episode_info is None:
                    continue

                success = int(
                    tuple(info.get("position", ())) == tuple(info.get("goal_position", (None, None)))
                )
                row = {
                    "episode": len(self.rows) + 1,
                    "timesteps": int(self.num_timesteps),
                    "total_reward": round(float(episode_info["r"]), 4),
                    "success": success,
                    "steps": int(episode_info["l"]),
                }
                self.rows.append(row)

                recent_rows = self.rows[-50:]
                recent_reward_mean = sum(
                    float(item["total_reward"]) for item in recent_rows
                ) / len(recent_rows)
                recent_success_rate = sum(
                    int(item["success"]) for item in recent_rows
                ) / len(recent_rows)

                log_metrics(
                    {
                        "episode_reward": float(row["total_reward"]),
                        "episode_success": int(row["success"]),
                        "episode_steps": int(row["steps"]),
                        "total_timesteps": int(row["timesteps"]),
                        "recent_reward_mean": recent_reward_mean,
                        "recent_success_rate": recent_success_rate,
                    },
                    step=int(row["episode"]),
                    run=self.run,
                )

                if row["episode"] == 1 or row["episode"] % 25 == 0:
                    print(
                        f"episode={int(row['episode']):04d} "
                        f"reward={float(row['total_reward']):>6.2f} "
                        f"success={int(row['success'])} "
                        f"steps={int(row['steps']):02d} "
                        f"timesteps={int(row['timesteps'])} "
                        f"recent_success={recent_success_rate:.2f}"
                    )

            return True

    callback = EpisodeMetricsCallback(run=wandb_run)
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    model.save(str(final_model_path))
    env.close()

    save_episode_metrics_csv(callback.rows, metrics_path)
    plot_paths: dict[str, str] = {}
    if callback.rows:
        saved_plots = plot_training_curves(
            callback.rows,
            output_dir=output_dir,
            reward_filename="reward_curve.png",
            success_filename="success_rate.png",
            algorithm_name="PPO",
        )
        plot_paths = {name: str(path) for name, path in saved_plots.items()}

    last_window = callback.rows[-50:] if callback.rows else []
    summary = {
        "algorithm": "ppo",
        "layout_name": layout_name,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "ent_coef": ent_coef,
        "num_envs": num_envs,
        "episodes_completed": len(callback.rows),
        "average_reward": round(
            sum(float(item["total_reward"]) for item in callback.rows) / max(len(callback.rows), 1),
            4,
        ),
        "success_rate": round(
            sum(int(item["success"]) for item in callback.rows) / max(len(callback.rows), 1),
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
        "plot_paths": plot_paths,
    }
    save_json(summary, summary_path)
    finish_wandb(wandb_run)

    print(f"Saved PPO model to {final_model_path}")
    print(f"Saved training metrics to {metrics_path}")
    print(f"Saved training summary to {summary_path}")

    return summary
