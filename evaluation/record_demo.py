"""Demo recording helpers for qualitative policy rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from algorithms.q_learning import load_q_learning_model, observation_to_state
from algorithms.reinforce import load_reinforce_agent
from config import (
    DEFAULT_LAYOUT_NAME,
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    PROJECT_ROOT,
    SAVED_MODELS_DIR,
)
from env.maze_env import MazeEnv
from utils.seed import set_global_seeds

PolicyFn = Callable[[np.ndarray], int]

DEFAULT_GIF_DIR = PROJECT_ROOT / "assets" / "gifs"
DEFAULT_MODEL_PATHS = {
    "q_learning": SAVED_MODELS_DIR / "q_learning" / "q_table.pkl",
    "reinforce": SAVED_MODELS_DIR / "reinforce" / "policy.pt",
    "a2c": SAVED_MODELS_DIR / "a2c" / "model.zip",
    "ppo": SAVED_MODELS_DIR / "ppo" / "model.zip",
}
DEFAULT_GIF_NAMES = {
    "q_learning": "q_learning_demo.gif",
    "reinforce": "reinforce_demo.gif",
    "a2c": "a2c_demo.gif",
    "ppo": "ppo_demo.gif",
}


def record_all_demos(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path = DEFAULT_GIF_DIR,
) -> dict[str, Path]:
    """Generate one short GIF demo for each saved algorithm."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for algorithm_key, model_path in DEFAULT_MODEL_PATHS.items():
        gif_path = output_path / DEFAULT_GIF_NAMES[algorithm_key]
        result = record_demo(
            algorithm=algorithm_key,
            model_path=model_path,
            output_path=gif_path,
            layout_name=layout_name,
            max_steps=max_steps,
            seed=seed,
        )
        saved_paths[algorithm_key] = result["gif_path"]
        print(
            f"Saved {algorithm_key} demo to {result['gif_path']} "
            f"(success={result['success']}, reward={result['total_reward']:.2f}, steps={result['steps']})"
        )

    return saved_paths


def record_demo(
    *,
    algorithm: str,
    model_path: str | Path,
    output_path: str | Path,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    max_attempts: int = 8,
) -> dict[str, Any]:
    """Record a short rollout GIF for a trained policy."""

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Saved model not found: {model_file}")

    policy_fn = build_policy_fn(algorithm, model_file)

    best_rollout: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        rollout = capture_rollout(
            policy_fn=policy_fn,
            layout_name=layout_name,
            max_steps=max_steps,
            seed=seed + attempt,
        )
        if best_rollout is None or rollout["total_reward"] > best_rollout["total_reward"]:
            best_rollout = rollout
        if rollout["success"]:
            best_rollout = rollout
            break

    if best_rollout is None:
        raise RuntimeError(f"Unable to capture a rollout for algorithm '{algorithm}'.")

    save_gif(best_rollout["frames"], Path(output_path))
    return {
        "algorithm": algorithm,
        "gif_path": Path(output_path),
        "success": best_rollout["success"],
        "total_reward": best_rollout["total_reward"],
        "steps": best_rollout["steps"],
    }


def build_policy_fn(algorithm: str, model_path: Path) -> PolicyFn:
    """Load a saved model and return a deterministic policy callable."""

    if algorithm == "q_learning":
        payload = load_q_learning_model(model_path)
        q_table = payload["q_table"]

        def q_policy(observation: np.ndarray) -> int:
            state = observation_to_state(observation)
            q_values = q_table.get(state)
            if q_values is None:
                return 0

            q_values_array = np.asarray(q_values, dtype=np.float32)
            best_actions = np.flatnonzero(q_values_array == np.max(q_values_array))
            return int(best_actions[0]) if len(best_actions) else 0

        return q_policy

    if algorithm == "reinforce":
        agent = load_reinforce_agent(model_path)

        def reinforce_policy(observation: np.ndarray) -> int:
            action, _, _ = agent.select_action(observation, deterministic=True)
            return action

        return reinforce_policy

    if algorithm == "a2c":
        from algorithms.sb3_a2c import load_a2c_model

        model = load_a2c_model(model_path)

        def a2c_policy(observation: np.ndarray) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        return a2c_policy

    if algorithm == "ppo":
        from algorithms.sb3_ppo import load_ppo_model

        model = load_ppo_model(model_path)

        def ppo_policy(observation: np.ndarray) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        return ppo_policy

    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def capture_rollout(
    *,
    policy_fn: PolicyFn,
    layout_name: str,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Capture frames from one deterministic rollout."""

    set_global_seeds(seed)
    env = MazeEnv(layout_name=layout_name, max_steps=max_steps, render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)

    frames: list[np.ndarray] = []
    initial_frame = env.render()
    if isinstance(initial_frame, np.ndarray):
        frames.append(initial_frame.copy())

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = int(policy_fn(observation))
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        frame = env.render()
        if isinstance(frame, np.ndarray):
            frames.append(frame.copy())

    env.close()

    if frames:
        frames.extend([frames[-1].copy() for _ in range(6)])

    return {
        "frames": frames,
        "success": bool(terminated),
        "total_reward": float(total_reward),
        "steps": steps,
    }


def save_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int = 250) -> Path:
    """Save a short GIF with Pillow."""

    if not frames:
        raise ValueError("Cannot save a GIF without frames.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return output_path
