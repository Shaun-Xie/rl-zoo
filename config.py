from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_LAYOUT_NAME = "classic_8x8"
DEFAULT_SEED = 7
DEFAULT_MAX_STEPS = 80
DEFAULT_RANDOM_ROLLOUT_STEPS = 20
DEFAULT_A2C_TIMESTEPS = 100_000
DEFAULT_PPO_TIMESTEPS = 20_000

ACTION_LABELS: dict[int, str] = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}


@dataclass(frozen=True)
class EnvConfig:
    """Shared environment settings for the maze experiments."""

    layout_name: str = DEFAULT_LAYOUT_NAME
    max_steps: int = DEFAULT_MAX_STEPS
    render_mode: str | None = None
    include_wall_indicators: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Optional experiment logging settings."""

    use_wandb: bool = False
    wandb_project: str = "rl-algorithm-zoo"
    wandb_run_name: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Top-level runtime settings used by quick scripts."""

    seed: int = DEFAULT_SEED
    episodes: int = 1
    random_rollout_steps: int = DEFAULT_RANDOM_ROLLOUT_STEPS
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def ensure_project_dirs() -> None:
    """Create output folders used later by training and evaluation scripts."""

    for directory in (SAVED_MODELS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
