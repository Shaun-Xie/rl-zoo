"""Tabular Q-learning components for the maze environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
from typing import Any, Sequence

import numpy as np

StateKey = tuple[int, ...]
QTable = dict[StateKey, np.ndarray]


@dataclass(frozen=True)
class QLearningConfig:
    """Core hyperparameters for a tabular Q-learning run."""

    alpha: float = 0.2
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    episodes: int = 800
    max_steps: int = 80


def observation_to_state(observation: Sequence[float] | np.ndarray) -> StateKey:
    """Convert the compact maze observation into a stable discrete state key."""

    values = np.asarray(observation, dtype=np.float32).flatten()
    return tuple(int(round(value)) for value in values.tolist())


class QLearningAgent:
    """Dictionary-backed tabular Q-learning agent."""

    def __init__(
        self,
        *,
        action_size: int,
        config: QLearningConfig | None = None,
        seed: int | None = None,
        q_table: QTable | None = None,
    ) -> None:
        self.action_size = action_size
        self.config = config or QLearningConfig()
        self.alpha = self.config.alpha
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.q_table: QTable = {}
        if q_table is not None:
            for state_key, q_values in q_table.items():
                self.q_table[tuple(state_key)] = np.asarray(q_values, dtype=np.float32).copy()

    def get_q_values(self, state: StateKey) -> np.ndarray:
        """Return the action values for a state, creating them on first use."""

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size, dtype=np.float32)
        return self.q_table[state]

    def select_greedy_action(
        self,
        state: StateKey,
        *,
        random_tie_break: bool = False,
    ) -> int:
        """Select the highest-value action for a state."""

        q_values = self.get_q_values(state)
        best_actions = np.flatnonzero(q_values == np.max(q_values))

        if random_tie_break and len(best_actions) > 1:
            return int(self.rng.choice(best_actions))
        return int(best_actions[0])

    def select_action(self, state: StateKey, *, explore: bool = True) -> int:
        """Select an action using epsilon-greedy exploration."""

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return self.select_greedy_action(state, random_tie_break=explore)

    def update(
        self,
        state: StateKey,
        action: int,
        reward: float,
        next_state: StateKey,
        done: bool,
    ) -> None:
        """Apply the standard one-step Q-learning update."""

        current_q_values = self.get_q_values(state)
        next_q_values = self.get_q_values(next_state)

        td_target = reward
        if not done:
            td_target += self.gamma * float(np.max(next_q_values))

        td_error = td_target - float(current_q_values[action])
        current_q_values[action] += self.alpha * td_error

    def decay_epsilon_value(self) -> float:
        """Decay epsilon after an episode and return the updated value."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save the learned Q-table and a small metadata payload."""

        return save_q_learning_model(
            path=path,
            q_table=self.q_table,
            action_size=self.action_size,
            config=self.config,
            epsilon=self.epsilon,
            metadata=metadata,
        )


def build_q_learning_agent(
    *,
    action_size: int,
    config: QLearningConfig | None = None,
    seed: int | None = None,
    q_table: QTable | None = None,
) -> QLearningAgent:
    """Construct a tabular Q-learning agent for the maze."""

    return QLearningAgent(
        action_size=action_size,
        config=config,
        seed=seed,
        q_table=q_table,
    )


def save_q_learning_model(
    *,
    path: str | Path,
    q_table: QTable,
    action_size: int,
    config: QLearningConfig,
    epsilon: float,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist the Q-table in a simple pickle payload."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "q_table": q_table,
        "action_size": action_size,
        "config": asdict(config),
        "epsilon": float(epsilon),
        "metadata": dict(metadata or {}),
    }

    with output_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    return output_path


def load_q_learning_model(path: str | Path) -> dict[str, Any]:
    """Load a saved Q-learning payload from disk."""

    input_path = Path(path)
    with input_path.open("rb") as file_handle:
        payload = pickle.load(file_handle)

    if "q_table" not in payload or "action_size" not in payload:
        raise ValueError(f"File '{input_path}' is not a valid Q-learning model payload.")

    return payload
