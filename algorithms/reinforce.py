"""REINFORCE policy-gradient components built with PyTorch."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class ReinforceConfig:
    """Core hyperparameters for a REINFORCE run."""

    learning_rate: float = 0.001
    gamma: float = 0.99
    episodes: int = 1000
    max_steps: int = 80
    hidden_size: int = 64
    normalize_returns: bool = True
    batch_episodes: int = 10
    entropy_coefficient: float = 0.01


@dataclass
class EpisodeTrajectory:
    """Simple container for one episode of policy-gradient data."""

    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)

    def add_step(
        self,
        *,
        state: Sequence[float] | np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
    ) -> None:
        """Store a single transition from the current episode."""

        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)


class PolicyNetwork(nn.Module):
    """Small MLP policy for compact maze observations."""

    def __init__(self, observation_size: int, hidden_size: int, action_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Return action logits for the given observations."""

        return self.network(observations)


class ReinforceAgent:
    """Policy-gradient agent with episodic REINFORCE updates."""

    def __init__(
        self,
        *,
        observation_size: int,
        action_size: int,
        config: ReinforceConfig | None = None,
        device: str | torch.device = "cpu",
        observation_scale: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.config = config or ReinforceConfig()
        self.device = torch.device(device)

        if observation_scale is None:
            observation_scale = np.ones(observation_size, dtype=np.float32)
        self.observation_scale = torch.as_tensor(
            np.maximum(np.asarray(observation_scale, dtype=np.float32), 1.0),
            dtype=torch.float32,
            device=self.device,
        )

        self.policy = PolicyNetwork(
            observation_size=observation_size,
            hidden_size=self.config.hidden_size,
            action_size=action_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
        )

    def select_action(
        self,
        observation: Sequence[float] | np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Choose an action from the current policy."""

        observation_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        observation_tensor = (observation_tensor / self.observation_scale).unsqueeze(0)

        logits = self.policy(observation_tensor)
        distribution = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = distribution.sample()

        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()
        return int(action.item()), log_prob.squeeze(0), entropy.squeeze(0)

    def compute_discounted_returns(
        self,
        rewards: Sequence[float],
        *,
        normalize: bool | None = None,
    ) -> torch.Tensor:
        """Compute discounted returns for one episode."""

        if normalize is None:
            normalize = self.config.normalize_returns

        discounted_returns: list[float] = []
        running_return = 0.0

        for reward in reversed(rewards):
            running_return = float(reward) + self.config.gamma * running_return
            discounted_returns.append(running_return)

        discounted_returns.reverse()
        returns_tensor = torch.tensor(
            discounted_returns,
            dtype=torch.float32,
            device=self.device,
        )

        if normalize and len(discounted_returns) > 1:
            mean = returns_tensor.mean()
            std = returns_tensor.std(unbiased=False)
            if float(std.item()) > 1e-8:
                returns_tensor = (returns_tensor - mean) / (std + 1e-8)

        return returns_tensor

    def compute_episode_loss(
        self,
        trajectory: EpisodeTrajectory,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute policy and entropy terms for one episode."""
        log_prob_tensor = torch.stack(trajectory.log_probs)
        entropy_tensor = torch.stack(trajectory.entropies)

        policy_loss = -(log_prob_tensor * returns).mean()
        mean_entropy = entropy_tensor.mean()
        return policy_loss, mean_entropy

    def update_batch(self, trajectories: Sequence[EpisodeTrajectory]) -> float:
        """Apply one REINFORCE update from a small batch of episodes."""

        if not trajectories:
            return 0.0

        policy_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        returns_per_trajectory: list[torch.Tensor] = []
        valid_trajectories: list[EpisodeTrajectory] = []

        for trajectory in trajectories:
            if not trajectory.log_probs:
                continue
            valid_trajectories.append(trajectory)
            returns_per_trajectory.append(
                self.compute_discounted_returns(trajectory.rewards, normalize=False)
            )

        if not returns_per_trajectory:
            return 0.0

        if self.config.normalize_returns:
            stacked_returns = torch.cat(returns_per_trajectory)
            mean = stacked_returns.mean()
            std = stacked_returns.std(unbiased=False)
            if float(std.item()) > 1e-8:
                returns_per_trajectory = [
                    (returns - mean) / (std + 1e-8) for returns in returns_per_trajectory
                ]

        for trajectory, returns in zip(valid_trajectories, returns_per_trajectory):
            policy_loss, mean_entropy = self.compute_episode_loss(trajectory, returns)
            policy_losses.append(policy_loss)
            entropies.append(mean_entropy)

        if not policy_losses:
            return 0.0

        mean_policy_loss = torch.stack(policy_losses).mean()
        mean_entropy = torch.stack(entropies).mean()
        loss = mean_policy_loss - self.config.entropy_coefficient * mean_entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_episode(self, trajectory: EpisodeTrajectory) -> float:
        """Compatibility wrapper for a single-episode update."""

        return self.update_batch([trajectory])

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        """Save the policy network and run metadata to disk."""

        return save_reinforce_model(
            path=path,
            policy=self.policy,
            observation_size=self.observation_size,
            action_size=self.action_size,
            config=self.config,
            observation_scale=self.observation_scale.detach().cpu().numpy(),
            metadata=metadata,
        )


def build_reinforce_agent(
    *,
    observation_size: int,
    action_size: int,
    config: ReinforceConfig | None = None,
    device: str | torch.device = "cpu",
    observation_scale: Sequence[float] | np.ndarray | None = None,
) -> ReinforceAgent:
    """Construct a REINFORCE agent for the maze."""

    return ReinforceAgent(
        observation_size=observation_size,
        action_size=action_size,
        config=config,
        device=device,
        observation_scale=observation_scale,
    )


def save_reinforce_model(
    *,
    path: str | Path,
    policy: PolicyNetwork,
    observation_size: int,
    action_size: int,
    config: ReinforceConfig,
    observation_scale: Sequence[float] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist a trained REINFORCE policy."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if observation_scale is None:
        observation_scale = np.ones(observation_size, dtype=np.float32)

    payload = {
        "model_state_dict": policy.state_dict(),
        "observation_size": observation_size,
        "action_size": action_size,
        "config": asdict(config),
        "observation_scale": np.asarray(observation_scale, dtype=np.float32),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, output_path)
    return output_path


def load_reinforce_model(path: str | Path) -> dict[str, Any]:
    """Load a saved REINFORCE checkpoint payload."""

    input_path = Path(path)
    payload = torch.load(input_path, map_location="cpu", weights_only=False)

    required_keys = {"model_state_dict", "observation_size", "action_size"}
    if not required_keys.issubset(payload):
        raise ValueError(f"File '{input_path}' is not a valid REINFORCE checkpoint.")

    return payload


def load_reinforce_agent(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ReinforceAgent:
    """Rebuild a REINFORCE agent from a saved checkpoint."""

    payload = load_reinforce_model(path)
    config = ReinforceConfig(**payload.get("config", {}))
    agent = build_reinforce_agent(
        observation_size=int(payload["observation_size"]),
        action_size=int(payload["action_size"]),
        config=config,
        device=device,
        observation_scale=payload.get("observation_scale"),
    )
    agent.policy.load_state_dict(payload["model_state_dict"])
    agent.policy.eval()
    return agent
