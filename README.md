# RL Algorithm Zoo

A Game AI course project that compares four reinforcement learning approaches on a custom maze/gridworld environment:

- Q-Learning
- REINFORCE
- A2C
- PPO

The project now includes a working tabular Q-Learning pipeline for the maze. PPO remains the planned candidate algorithm for later MicroRTS-oriented work.

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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you want optional W&B logging later, install it separately:

```bash
pip install wandb
```

## Commands

Run a short environment sanity check:

```bash
python3 main.py --mode sanity --render ansi
```

Train Q-Learning:

```bash
python3 main.py --mode train-q --episodes 800 --max-steps 80
```

Evaluate a saved Q-Learning model:

```bash
python3 main.py --mode eval-q --model-path saved_models/q_learning/q_table.pkl --episodes 100
```

## Notes

- Q-learning outputs are saved under `saved_models/q_learning/` and `results/q_learning/`.
- The tabular state key is an integer tuple built from the maze observation vector.
- `wandb` is optional. Training still runs normally without it.
