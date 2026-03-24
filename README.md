# RL Algorithm Zoo

A fresh Game AI course project that compares four reinforcement learning approaches on a custom maze/gridworld environment:

- Q-Learning
- REINFORCE
- A2C
- PPO

The current scaffold focuses on the environment, configuration, logging helpers, and project organization. PPO is included as the planned candidate algorithm for later MicroRTS-oriented work.

## Environment

- Fixed 8x8 maze with walls, a start cell, and a goal cell
- Compact numeric observation vector:
  - agent row / col
  - goal row / col
  - four binary wall indicators
- Discrete actions: up, down, left, right
- Rewards:
  - `+10.0` for reaching the goal
  - `-0.1` for a valid non-goal step
  - `-1.0` for an invalid move into a wall or boundary
- Episode truncation after a configurable max step count
- Text rendering plus a simple `rgb_array` renderer

## Setup

```bash
cd rl_algorithm_zoo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want to prepare the full training stack later, install the planned extras after the base setup:

```bash
pip install torch stable-baselines3 matplotlib wandb
```

## Sanity Check

Run a short random-policy rollout:

```bash
cd rl_algorithm_zoo
python main.py --render human
```

A no-render terminal check also works:

```bash
cd rl_algorithm_zoo
python main.py --render none --episodes 1 --max-steps 20
```

## Notes

- `wandb` hooks are optional. The project still runs if W&B is not configured or not installed.
- The algorithm modules and training scripts are intentionally left as clean TODO stubs for the next implementation step.
- The base `requirements.txt` is intentionally lightweight for this first scaffold so the maze environment can be tested immediately.
