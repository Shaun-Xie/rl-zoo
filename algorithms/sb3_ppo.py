"""Stable-Baselines3 PPO helpers for the maze environment.

PPO is the planned bridge algorithm for later MicroRTS experiments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from config import DEFAULT_LAYOUT_NAME, DEFAULT_MAX_STEPS, DEFAULT_SEED
from env.maze_env import MazeEnv
from utils.seed import seed_action_space

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def make_ppo_env(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
) -> Callable[[], Any]:
    """Build a monitored maze factory for PPO."""

    def _make_env() -> Any:
        try:
            from stable_baselines3.common.monitor import Monitor
        except ImportError as exc:
            raise ImportError(
                "Stable-Baselines3 is required for PPO support. "
                "Install the dependencies in requirements.txt."
            ) from exc

        env = MazeEnv(layout_name=layout_name, max_steps=max_steps, render_mode=None)
        env.reset(seed=seed)
        seed_action_space(env.action_space, seed)
        return Monitor(env)

    return _make_env


def build_ppo_model(
    *,
    env: Any,
    learning_rate: float = 0.0003,
    gamma: float = 0.99,
    n_steps: int = 128,
    batch_size: int = 128,
    ent_coef: float = 0.01,
    seed: int = DEFAULT_SEED,
    verbose: int = 0,
) -> Any:
    """Create a small SB3 PPO model for vector maze observations."""

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 is required for PPO support. "
            "Install the dependencies in requirements.txt."
        ) from exc

    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        seed=seed,
        verbose=verbose,
    )


def load_ppo_model(model_path: str | Path) -> Any:
    """Load a saved SB3 PPO model from disk."""

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 is required for PPO support. "
            "Install the dependencies in requirements.txt."
        ) from exc

    return PPO.load(str(model_path))
