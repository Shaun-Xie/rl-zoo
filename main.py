from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    ACTION_LABELS,
    DEFAULT_A2C_TIMESTEPS,
    DEFAULT_LAYOUT_NAME,
    DEFAULT_MAX_STEPS,
    DEFAULT_PPO_TIMESTEPS,
    DEFAULT_RANDOM_ROLLOUT_STEPS,
    DEFAULT_SEED,
    EnvConfig,
    RunConfig,
    SAVED_MODELS_DIR,
    ensure_project_dirs,
)
from env.maze_env import MazeEnv
from env.maze_layouts import list_layout_names
from evaluation.compare_results import compare_runs
from evaluation.record_demo import play_all_live_demos, play_live_demo, record_all_demos
from evaluation.evaluate import (
    evaluate_a2c_model,
    evaluate_ppo_model,
    evaluate_q_learning_model,
    evaluate_reinforce_model,
)
from training.train_a2c import run_a2c_training
from training.train_ppo import run_ppo_training
from training.train_q_learning import run_q_learning_training
from training.train_reinforce import run_reinforce_training
from utils.seed import seed_action_space, set_global_seeds


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sanity checks, training, and evaluation."""

    parser = argparse.ArgumentParser(
        description="Run maze sanity checks, train or evaluate RL baselines, and play live maze demos."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sanity",
        choices=(
            "sanity",
            "train-q",
            "eval-q",
            "train-reinforce",
            "eval-reinforce",
            "train-a2c",
            "eval-a2c",
            "train-ppo",
            "eval-ppo",
            "compare-results",
            "record-demos",
            "live-demo",
        ),
        help="Program mode.",
    )
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
        default=None,
        help="Episode count. Defaults depend on the selected mode.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Episode step limit. Defaults depend on the selected mode.",
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
        help="Render mode used during sanity checks or evaluation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Q-learning learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Q-learning discount factor.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for Q-learning.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Minimum epsilon for Q-learning.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Per-episode epsilon decay for Q-learning.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for REINFORCE, A2C, or PPO. Uses a mode-specific default when omitted.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size for the REINFORCE policy network.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path used to save or load a model. If omitted, the mode-specific default is used.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps for A2C or PPO. Uses a mode-specific default when omitted.",
    )
    parser.add_argument(
        "--eval-policy",
        type=str,
        default="greedy",
        choices=("greedy", "sample"),
        help="Action-selection mode used for REINFORCE evaluation.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ppo",
        choices=("q_learning", "reinforce", "a2c", "ppo", "all"),
        help="Algorithm used for live-demo mode. Use 'all' to play every saved policy in sequence.",
    )
    parser.add_argument(
        "--playback",
        type=str,
        default="gui",
        choices=("gui", "ansi"),
        help="Playback mode used for live-demo. GUI falls back to ANSI if no window backend is available.",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=250,
        help="Per-frame playback delay in milliseconds for live-demo playback.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable optional Weights & Biases logging during training.",
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
    """CLI entry point for environment checks and Q-learning experiments."""

    args = parse_args()
    ensure_project_dirs()
    render_mode = None if args.render == "none" else args.render

    if args.mode == "sanity":
        sanity_episodes = args.episodes if args.episodes is not None else 1
        sanity_steps = args.max_steps if args.max_steps is not None else DEFAULT_RANDOM_ROLLOUT_STEPS

        run_config = RunConfig(
            seed=args.seed,
            episodes=sanity_episodes,
            random_rollout_steps=sanity_steps,
            env=EnvConfig(
                layout_name=args.layout,
                max_steps=sanity_steps,
                render_mode=render_mode,
            ),
        )
        run_random_rollout(run_config)
        return

    if args.mode == "compare-results":
        summary = compare_runs()
        print(f"Saved comparison table to {summary['csv_path']}")
        print(f"Saved comparison report to {summary['report_path']}")
        return

    if args.mode == "record-demos":
        saved_paths = record_all_demos(
            layout_name=args.layout,
            max_steps=args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS,
            seed=args.seed,
        )
        for algorithm, path in saved_paths.items():
            print(f"{algorithm}: {path}")
        return

    if args.mode == "live-demo":
        demo_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        if args.algorithm == "all":
            summaries = play_all_live_demos(
                layout_name=args.layout,
                max_steps=demo_steps,
                seed=args.seed,
                playback=args.playback,
                delay_ms=args.delay_ms,
            )
            for summary in summaries:
                print(
                    f"{summary['algorithm']}: success={summary['success']} "
                    f"reward={summary['total_reward']:.2f} "
                    f"steps={summary['steps']} "
                    f"model_path={summary['model_path']}"
                )
            return

        summary = play_live_demo(
            algorithm=args.algorithm,
            model_path=args.model_path,
            layout_name=args.layout,
            max_steps=demo_steps,
            seed=args.seed,
            playback=args.playback,
            delay_ms=args.delay_ms,
        )
        print(
            f"Live demo finished. success={summary['success']} "
            f"reward={summary['total_reward']:.2f} "
            f"steps={summary['steps']} "
            f"model_path={summary['model_path']}"
        )
        return

    if args.mode == "train-q":
        training_episodes = args.episodes if args.episodes is not None else 800
        training_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        q_model_path = Path(args.model_path) if args.model_path is not None else SAVED_MODELS_DIR / "q_learning" / "q_table.pkl"

        summary = run_q_learning_training(
            layout_name=args.layout,
            episodes=training_episodes,
            max_steps=training_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
            use_wandb=args.use_wandb,
            model_path=q_model_path,
        )
        print(
            f"Training complete. success_rate={summary['success_rate']:.2f} "
            f"recent_success={summary['recent_success_rate']:.2f}"
        )
        return

    if args.mode == "train-reinforce":
        training_episodes = args.episodes if args.episodes is not None else 1000
        training_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.001
        reinforce_model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else SAVED_MODELS_DIR / "reinforce" / "policy.pt"
        )

        summary = run_reinforce_training(
            layout_name=args.layout,
            episodes=training_episodes,
            max_steps=training_steps,
            learning_rate=learning_rate,
            gamma=args.gamma,
            hidden_size=args.hidden_size,
            seed=args.seed,
            use_wandb=args.use_wandb,
            model_path=reinforce_model_path,
        )
        print(
            f"Training complete. success_rate={summary['success_rate']:.2f} "
            f"recent_success={summary['recent_success_rate']:.2f}"
        )
        return

    if args.mode == "train-a2c":
        training_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.0003
        total_timesteps = args.timesteps
        if total_timesteps is None:
            total_timesteps = (
                args.episodes * training_steps
                if args.episodes is not None
                else DEFAULT_A2C_TIMESTEPS
            )

        a2c_model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else SAVED_MODELS_DIR / "a2c" / "model.zip"
        )

        summary = run_a2c_training(
            layout_name=args.layout,
            total_timesteps=total_timesteps,
            max_steps=training_steps,
            learning_rate=learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            use_wandb=args.use_wandb,
            model_path=a2c_model_path,
        )
        print(
            f"Training complete. success_rate={summary['success_rate']:.2f} "
            f"recent_success={summary['recent_success_rate']:.2f}"
        )
        return

    if args.mode == "train-ppo":
        training_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        learning_rate = args.learning_rate if args.learning_rate is not None else 0.0003
        total_timesteps = args.timesteps
        if total_timesteps is None:
            total_timesteps = (
                args.episodes * training_steps
                if args.episodes is not None
                else DEFAULT_PPO_TIMESTEPS
            )

        ppo_model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else SAVED_MODELS_DIR / "ppo" / "model.zip"
        )

        summary = run_ppo_training(
            layout_name=args.layout,
            total_timesteps=total_timesteps,
            max_steps=training_steps,
            learning_rate=learning_rate,
            gamma=args.gamma,
            seed=args.seed,
            use_wandb=args.use_wandb,
            model_path=ppo_model_path,
        )
        print(
            f"Training complete. success_rate={summary['success_rate']:.2f} "
            f"recent_success={summary['recent_success_rate']:.2f}"
        )
        return

    if args.mode == "eval-q":
        evaluation_episodes = args.episodes if args.episodes is not None else 100
        evaluation_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        q_model_path = Path(args.model_path) if args.model_path is not None else SAVED_MODELS_DIR / "q_learning" / "q_table.pkl"

        summary = evaluate_q_learning_model(
            model_path=q_model_path,
            layout_name=args.layout,
            episodes=evaluation_episodes,
            max_steps=evaluation_steps,
            seed=args.seed,
            render_mode=render_mode,
        )
        print(
            f"Evaluation complete. avg_reward={summary['average_reward']:.2f} "
            f"success_rate={summary['success_rate']:.2f} "
            f"avg_steps={summary['average_steps']:.2f}"
        )
        return

    if args.mode == "eval-a2c":
        evaluation_episodes = args.episodes if args.episodes is not None else 100
        evaluation_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        a2c_model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else SAVED_MODELS_DIR / "a2c" / "model.zip"
        )

        summary = evaluate_a2c_model(
            model_path=a2c_model_path,
            layout_name=args.layout,
            episodes=evaluation_episodes,
            max_steps=evaluation_steps,
            seed=args.seed,
            render_mode=render_mode,
        )
        print(
            f"Evaluation complete. avg_reward={summary['average_reward']:.2f} "
            f"success_rate={summary['success_rate']:.2f} "
            f"avg_steps={summary['average_steps']:.2f}"
        )
        return

    if args.mode == "eval-ppo":
        evaluation_episodes = args.episodes if args.episodes is not None else 100
        evaluation_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
        ppo_model_path = (
            Path(args.model_path)
            if args.model_path is not None
            else SAVED_MODELS_DIR / "ppo" / "model.zip"
        )

        summary = evaluate_ppo_model(
            model_path=ppo_model_path,
            layout_name=args.layout,
            episodes=evaluation_episodes,
            max_steps=evaluation_steps,
            seed=args.seed,
            render_mode=render_mode,
        )
        print(
            f"Evaluation complete. avg_reward={summary['average_reward']:.2f} "
            f"success_rate={summary['success_rate']:.2f} "
            f"avg_steps={summary['average_steps']:.2f}"
        )
        return

    evaluation_episodes = args.episodes if args.episodes is not None else 100
    evaluation_steps = args.max_steps if args.max_steps is not None else DEFAULT_MAX_STEPS
    reinforce_model_path = (
        Path(args.model_path)
        if args.model_path is not None
        else SAVED_MODELS_DIR / "reinforce" / "policy.pt"
    )
    summary = evaluate_reinforce_model(
        model_path=reinforce_model_path,
        layout_name=args.layout,
        episodes=evaluation_episodes,
        max_steps=evaluation_steps,
        seed=args.seed,
        render_mode=render_mode,
        policy_mode=args.eval_policy,
    )
    print(
        f"Evaluation complete. avg_reward={summary['average_reward']:.2f} "
        f"success_rate={summary['success_rate']:.2f} "
        f"avg_steps={summary['average_steps']:.2f}"
    )


if __name__ == "__main__":
    main()
