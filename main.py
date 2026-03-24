from __future__ import annotations

import argparse

from config import (
    ACTION_LABELS,
    DEFAULT_LAYOUT_NAME,
    DEFAULT_RANDOM_ROLLOUT_STEPS,
    DEFAULT_SEED,
    EnvConfig,
    RunConfig,
    ensure_project_dirs,
)
from env.maze_env import MazeEnv
from env.maze_layouts import list_layout_names
from utils.seed import seed_action_space, set_global_seeds


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the environment sanity check."""

    parser = argparse.ArgumentParser(description="Run a random rollout in the maze.")
    parser.add_argument(
        "--layout",
        type=str,
        default=DEFAULT_LAYOUT_NAME,
        choices=list_layout_names(),
        help="Maze layout to load.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of random-rollout episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_RANDOM_ROLLOUT_STEPS,
        help="Episode truncation limit for the sanity check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base random seed used for Python, NumPy, Torch, and the action space.",
    )
    parser.add_argument(
        "--render",
        type=str,
        default="none",
        choices=("none", "human", "ansi", "rgb_array"),
        help="Render mode used during the rollout.",
    )
    return parser.parse_args()


def run_random_rollout(config: RunConfig) -> None:
    """Create the environment and run a short random-policy sanity test."""

    set_global_seeds(config.seed)

    render_mode = config.env.render_mode
    env = MazeEnv(
        layout_name=config.env.layout_name,
        max_steps=config.env.max_steps,
        render_mode=render_mode,
        include_wall_indicators=config.env.include_wall_indicators,
    )
    seed_action_space(env.action_space, config.seed)

    print(
        f"Running {config.episodes} random rollout(s) on "
        f"layout='{config.env.layout_name}' with max_steps={config.env.max_steps}."
    )

    for episode_index in range(config.episodes):
        observation, info = env.reset(seed=config.seed + episode_index)
        print(
            f"\nEpisode {episode_index + 1}: "
            f"start_position={info['position']} goal={info['goal_position']}"
        )
        print(f"Initial observation: {observation.tolist()}")

        total_reward = 0.0
        terminated = False
        truncated = False

        if render_mode == "ansi":
            print(env.render())
        elif render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                print(f"Initial rgb_array frame shape: {frame.shape}")

        for step_index in range(config.random_rollout_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(
                f"step={step_index + 1:02d} "
                f"action={ACTION_LABELS[int(action)]:<5} "
                f"reward={reward:>5.1f} "
                f"position={info['position']} "
                f"invalid={info['invalid_move']} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

            if render_mode == "ansi":
                print(env.render())
            elif render_mode == "rgb_array" and step_index == 0:
                frame = env.render()
                if frame is not None:
                    print(f"rgb_array frame shape: {frame.shape}")

            if terminated or truncated:
                break

        print(
            f"Episode {episode_index + 1} finished with total_reward={total_reward:.2f}, "
            f"terminated={terminated}, truncated={truncated}."
        )

    env.close()


def main() -> None:
    """CLI entry point for the scaffold sanity test."""

    args = parse_args()
    ensure_project_dirs()

    run_config = RunConfig(
        seed=args.seed,
        episodes=args.episodes,
        random_rollout_steps=args.max_steps,
        env=EnvConfig(
            layout_name=args.layout,
            max_steps=args.max_steps,
            render_mode=None if args.render == "none" else args.render,
        ),
    )
    run_random_rollout(run_config)


if __name__ == "__main__":
    main()
