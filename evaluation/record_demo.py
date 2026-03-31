"""Demo recording helpers for qualitative policy rollouts."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

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
from env.renderer import render_text_grid
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
DISPLAY_NAMES = {
    "q_learning": "Q-Learning",
    "reinforce": "REINFORCE",
    "a2c": "A2C",
    "ppo": "PPO",
}
ALGORITHM_ALIASES = {
    "q": "q_learning",
    "q-learning": "q_learning",
    "q_learning": "q_learning",
    "reinforce": "reinforce",
    "a2c": "a2c",
    "ppo": "ppo",
}
LIVE_DEMO_ALGORITHMS = ("q_learning", "reinforce", "a2c", "ppo")


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

    normalized_algorithm = normalize_algorithm_name(algorithm)
    model_file = resolve_model_path(normalized_algorithm, model_path=model_path)
    policy_fn = build_policy_fn(normalized_algorithm, model_file)
    best_rollout = select_best_rollout(
        policy_fn=policy_fn,
        layout_name=layout_name,
        max_steps=max_steps,
        seed=seed,
        max_attempts=max_attempts,
    )

    save_gif(best_rollout["frames"], Path(output_path))
    return {
        "algorithm": normalized_algorithm,
        "gif_path": Path(output_path),
        "success": best_rollout["success"],
        "total_reward": best_rollout["total_reward"],
        "steps": best_rollout["steps"],
        "seed": best_rollout["seed"],
    }


def normalize_algorithm_name(algorithm: str) -> str:
    """Normalize a user-facing algorithm label to the canonical key."""

    normalized = ALGORITHM_ALIASES.get(str(algorithm).strip().lower())
    if normalized is None:
        supported = ", ".join(sorted(DEFAULT_MODEL_PATHS))
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Expected one of: {supported}."
        )
    return normalized


def get_default_model_path(algorithm: str) -> Path:
    """Return the default saved-model path for a supported algorithm."""

    normalized_algorithm = normalize_algorithm_name(algorithm)
    return DEFAULT_MODEL_PATHS[normalized_algorithm]


def resolve_model_path(algorithm: str, model_path: str | Path | None = None) -> Path:
    """Resolve an explicit or default model path and validate that it exists."""

    normalized_algorithm = normalize_algorithm_name(algorithm)
    resolved_path = (
        Path(model_path)
        if model_path is not None
        else get_default_model_path(normalized_algorithm)
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Saved model not found: {resolved_path}")
    return resolved_path


def build_policy_fn(algorithm: str, model_path: Path) -> PolicyFn:
    """Load a saved model and return a deterministic policy callable."""

    normalized_algorithm = normalize_algorithm_name(algorithm)

    if normalized_algorithm == "q_learning":
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

    if normalized_algorithm == "reinforce":
        agent = load_reinforce_agent(model_path)

        def reinforce_policy(observation: np.ndarray) -> int:
            action, _, _ = agent.select_action(observation, deterministic=True)
            return action

        return reinforce_policy

    if normalized_algorithm == "a2c":
        from algorithms.sb3_a2c import load_a2c_model

        model = load_a2c_model(model_path)

        def a2c_policy(observation: np.ndarray) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        return a2c_policy

    if normalized_algorithm == "ppo":
        from algorithms.sb3_ppo import load_ppo_model

        model = load_ppo_model(model_path)

        def ppo_policy(observation: np.ndarray) -> int:
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        return ppo_policy

    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def select_best_rollout(
    *,
    policy_fn: PolicyFn,
    layout_name: str,
    max_steps: int,
    seed: int,
    max_attempts: int = 8,
) -> dict[str, Any]:
    """Pick a successful rollout when possible to make demos reliable in class."""

    best_rollout: dict[str, Any] | None = None
    for attempt in range(max_attempts):
        attempt_seed = seed + attempt
        rollout = capture_rollout(
            policy_fn=policy_fn,
            layout_name=layout_name,
            max_steps=max_steps,
            seed=attempt_seed,
        )
        if best_rollout is None or rollout["total_reward"] > best_rollout["total_reward"]:
            best_rollout = rollout
        if rollout["success"]:
            best_rollout = rollout
            break

    if best_rollout is None:
        raise RuntimeError("Unable to capture a rollout.")

    return best_rollout


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
    ansi_frames: list[str] = []
    reward_history: list[float] = [0.0]
    step_counts: list[int] = [0]
    initial_frame = env.render()
    if isinstance(initial_frame, np.ndarray):
        frames.append(initial_frame.copy())
    ansi_frames.append(render_text_grid(env.grid, env.agent_position))

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
        ansi_frames.append(render_text_grid(env.grid, env.agent_position))
        reward_history.append(float(total_reward))
        step_counts.append(steps)

    env.close()

    if frames:
        frames.extend([frames[-1].copy() for _ in range(6)])
    if ansi_frames:
        ansi_frames.extend([ansi_frames[-1] for _ in range(6)])
    if reward_history:
        reward_history.extend([reward_history[-1] for _ in range(6)])
    if step_counts:
        step_counts.extend([step_counts[-1] for _ in range(6)])

    return {
        "frames": frames,
        "ansi_frames": ansi_frames,
        "reward_history": reward_history,
        "step_counts": step_counts,
        "success": bool(terminated),
        "total_reward": float(total_reward),
        "steps": steps,
        "seed": seed,
    }


def play_live_demo(
    *,
    algorithm: str,
    model_path: str | Path | None = None,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    playback: str = "gui",
    delay_ms: int = 250,
    max_attempts: int = 8,
) -> dict[str, Any]:
    """Play back one deterministic maze rollout in a GUI window or the terminal."""

    normalized_algorithm = normalize_algorithm_name(algorithm)
    model_file = resolve_model_path(normalized_algorithm, model_path=model_path)
    policy_fn = build_policy_fn(normalized_algorithm, model_file)
    rollout = select_best_rollout(
        policy_fn=policy_fn,
        layout_name=layout_name,
        max_steps=max_steps,
        seed=seed,
        max_attempts=max_attempts,
    )
    playback_mode = playback.strip().lower()
    if playback_mode not in {"gui", "ansi"}:
        raise ValueError("playback must be either 'gui' or 'ansi'.")

    print(
        f"Starting {DISPLAY_NAMES[normalized_algorithm]} live demo "
        f"with seed={rollout['seed']} using playback='{playback_mode}'."
    )
    if playback_mode == "gui":
        used_gui = playback_rollout_gui(
            algorithm=normalized_algorithm,
            rollout=rollout,
            delay_ms=delay_ms,
        )
        if not used_gui:
            print("Falling back to ANSI terminal playback.")
            playback_mode = "ansi"

    if playback_mode == "ansi":
        playback_rollout_ansi(
            algorithm=normalized_algorithm,
            rollout=rollout,
            delay_ms=delay_ms,
        )

    return {
        "algorithm": normalized_algorithm,
        "model_path": str(model_file),
        "playback": playback_mode,
        "success": rollout["success"],
        "total_reward": rollout["total_reward"],
        "steps": rollout["steps"],
        "seed": rollout["seed"],
    }


def play_all_live_demos(
    *,
    layout_name: str = DEFAULT_LAYOUT_NAME,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = DEFAULT_SEED,
    playback: str = "gui",
    delay_ms: int = 250,
    max_attempts: int = 8,
) -> list[dict[str, Any]]:
    """Play one live demo for each supported algorithm in sequence."""

    summaries: list[dict[str, Any]] = []
    for algorithm in LIVE_DEMO_ALGORITHMS:
        summary = play_live_demo(
            algorithm=algorithm,
            layout_name=layout_name,
            max_steps=max_steps,
            seed=seed,
            playback=playback,
            delay_ms=delay_ms,
            max_attempts=max_attempts,
        )
        summaries.append(summary)
    return summaries


def format_playback_status(
    *,
    algorithm: str,
    step_index: int,
    total_reward: float,
    is_final_frame: bool,
    success: bool,
) -> str:
    """Create a compact status line shared by GUI and ANSI playback."""

    result_label = "RUNNING"
    if is_final_frame:
        result_label = "SUCCESS" if success else "FAILURE"

    return (
        f"Algorithm: {DISPLAY_NAMES[algorithm]} | "
        f"Step: {step_index:02d} | "
        f"Reward: {total_reward:6.2f} | "
        f"Status: {result_label}"
    )


def playback_rollout_ansi(
    *,
    algorithm: str,
    rollout: dict[str, Any],
    delay_ms: int,
) -> None:
    """Animate a rollout in the terminal using ANSI screen clearing."""

    delay_seconds = max(delay_ms, 0) / 1000.0
    frame_count = len(rollout["ansi_frames"])
    final_frame_index = max(rollout["steps"], 0)

    for frame_index in range(frame_count):
        is_final_frame = frame_index >= final_frame_index
        step_index = rollout["step_counts"][frame_index]
        total_reward = rollout["reward_history"][frame_index]

        print("\033[2J\033[H", end="")
        print(
            format_playback_status(
                algorithm=algorithm,
                step_index=step_index,
                total_reward=total_reward,
                is_final_frame=is_final_frame,
                success=rollout["success"],
            )
        )
        print(rollout["ansi_frames"][frame_index])
        sys.stdout.flush()
        if delay_seconds > 0:
            time.sleep(delay_seconds)


def playback_rollout_gui(
    *,
    algorithm: str,
    rollout: dict[str, Any],
    delay_ms: int,
) -> bool:
    """Animate a rollout in a matplotlib window when a GUI backend is available."""

    if not is_gui_playback_available():
        return False

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not rollout["frames"]:
        return False

    delay_seconds = max(delay_ms, 0) / 1000.0
    final_frame_index = max(rollout["steps"], 0)
    figure, axis = plt.subplots(figsize=(6.4, 7.2))
    figure.patch.set_facecolor("#f6f6f3")
    axis.set_facecolor("#f6f6f3")
    figure.subplots_adjust(top=0.88, bottom=0.12)
    image = axis.imshow(rollout["frames"][0])
    axis.axis("off")
    figure.suptitle(
        f"{DISPLAY_NAMES[algorithm]}",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.text(
        0.5,
        0.92,
        "Live Maze Demo",
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )
    status_text = axis.text(
        0.5,
        -0.08,
        "",
        transform=axis.transAxes,
        ha="center",
        va="top",
        family="monospace",
        fontsize=10,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#dddddd",
        },
    )

    try:
        plt.show(block=False)
        for frame_index, frame in enumerate(rollout["frames"]):
            if not plt.fignum_exists(figure.number):
                break

            is_final_frame = frame_index >= final_frame_index
            step_index = rollout["step_counts"][frame_index]
            total_reward = rollout["reward_history"][frame_index]

            image.set_data(frame)
            status_text.set_text(
                format_playback_status(
                    algorithm=algorithm,
                    step_index=step_index,
                    total_reward=total_reward,
                    is_final_frame=is_final_frame,
                    success=rollout["success"],
                )
            )
            figure.canvas.draw_idle()
            figure.canvas.flush_events()
            plt.pause(delay_seconds if delay_seconds > 0 else 0.001)
    except Exception:
        plt.close(figure)
        return False

    plt.close(figure)
    return True


def is_gui_playback_available() -> bool:
    """Return whether the current machine likely supports a live matplotlib window."""

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    try:
        import matplotlib
    except Exception:
        return False

    backend = str(matplotlib.get_backend()).lower()
    non_interactive_backends = {
        "agg",
        "cairo",
        "pdf",
        "pgf",
        "ps",
        "svg",
        "template",
        "module://matplotlib_inline.backend_inline",
    }
    return backend not in non_interactive_backends


def save_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int = 250) -> Path:
    """Save a short GIF with Pillow."""

    if not frames:
        raise ValueError("Cannot save a GIF without frames.")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required to save GIF demos. Install the dependencies in requirements.txt."
        ) from exc

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
