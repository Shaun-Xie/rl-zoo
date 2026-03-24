"""Evaluation helpers for trained maze agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from algorithms.q_learning import load_q_learning_model, observation_to_state
from algorithms.reinforce import load_reinforce_agent
from config import DEFAULT_LAYOUT_NAME, DEFAULT_MAX_STEPS, DEFAULT_SEED
from env.maze_env import MazeEnv
from utils.seed import set_global_seeds

PolicyFn = Callable[[np.ndarray], int]


def evaluate_policy(
    *,
    policy_fn: PolicyFn,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 100,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    render_mode: str | None = None,
) -> dict[str, Any]:
    """Evaluate a policy callable on the maze and return aggregate metrics."""

    set_global_seeds(seed)
    env = MazeEnv(layout_name=layout_name, max_steps=max_steps, render_mode=render_mode)

    rewards: list[float] = []
    successes: list[int] = []
    steps_per_episode: list[int] = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)

        if render_mode == "ansi":
            print(env.render())

        total_reward = 0.0
        steps_taken = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(policy_fn(observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps_taken += 1

            if render_mode == "ansi":
                print(env.render())

        rewards.append(total_reward)
        successes.append(int(terminated))
        steps_per_episode.append(steps_taken)

    env.close()

    return {
        "episodes": episodes,
        "average_reward": round(float(np.mean(rewards)) if rewards else 0.0, 4),
        "success_rate": round(float(np.mean(successes)) if successes else 0.0, 4),
        "average_steps": round(float(np.mean(steps_per_episode)) if steps_per_episode else 0.0, 4),
        "max_steps": max_steps,
        "layout_name": layout_name,
        "seed": seed,
    }


def evaluate_q_learning_model(
    *,
    model_path: str | Path,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 100,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    render_mode: str | None = None,
) -> dict[str, Any]:
    """Load a saved Q-table and run greedy evaluation episodes."""

    model_payload = load_q_learning_model(model_path)
    q_table = model_payload["q_table"]
    action_size = int(model_payload["action_size"])

    def greedy_policy(observation: np.ndarray) -> int:
        state = observation_to_state(observation)
        q_values = q_table.get(state)
        if q_values is None:
            return 0

        q_values_array = np.asarray(q_values, dtype=np.float32)
        best_actions = np.flatnonzero(q_values_array == np.max(q_values_array))
        return int(best_actions[0]) if len(best_actions) else 0

    summary = evaluate_policy(
        policy_fn=greedy_policy,
        layout_name=layout_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        render_mode=render_mode,
    )
    summary["algorithm"] = "q_learning"
    summary["model_path"] = str(Path(model_path))
    summary["action_size"] = action_size

    return summary


def evaluate_reinforce_model(
    *,
    model_path: str | Path,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 100,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    render_mode: str | None = None,
    policy_mode: str = "greedy",
) -> dict[str, Any]:
    """Load a saved REINFORCE policy and evaluate it on the maze."""

    if policy_mode not in {"greedy", "sample"}:
        raise ValueError("policy_mode must be 'greedy' or 'sample'.")

    agent = load_reinforce_agent(model_path)

    def reinforce_policy(observation: np.ndarray) -> int:
        action, _, _ = agent.select_action(
            observation,
            deterministic=(policy_mode == "greedy"),
        )
        return action

    summary = evaluate_policy(
        policy_fn=reinforce_policy,
        layout_name=layout_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        render_mode=render_mode,
    )
    summary["algorithm"] = "reinforce"
    summary["model_path"] = str(Path(model_path))
    summary["policy_mode"] = policy_mode

    return summary


def evaluate_a2c_model(
    *,
    model_path: str | Path,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    episodes: int = 100,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    render_mode: str | None = None,
) -> dict[str, Any]:
    """Load a saved A2C policy and evaluate it greedily on the maze."""

    from algorithms.sb3_a2c import load_a2c_model

    model = load_a2c_model(model_path)

    def a2c_policy(observation: np.ndarray) -> int:
        action, _ = model.predict(observation, deterministic=True)
        return int(action)

    summary = evaluate_policy(
        policy_fn=a2c_policy,
        layout_name=layout_name,
        episodes=episodes,
        max_steps=max_steps,
        seed=seed,
        render_mode=render_mode,
    )
    summary["algorithm"] = "a2c"
    summary["model_path"] = str(Path(model_path))

    return summary


def evaluate_agent(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility wrapper for the generic policy evaluation helper."""

    return evaluate_policy(*args, **kwargs)
