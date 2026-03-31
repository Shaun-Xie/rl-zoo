# RL Algorithm Zoo

An original Game AI course project that compares four reinforcement learning algorithms on a custom 8x8 maze environment with compact vector observations. The project includes tabular, policy-gradient, and Stable-Baselines3 baselines, plus saved comparison outputs and short policy demos.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 main.py --mode train-q
python3 main.py --mode train-reinforce
python3 main.py --mode train-a2c
python3 main.py --mode train-ppo
python3 main.py --mode live-demo --algorithm ppo
python3 main.py --mode live-demo --algorithm ppo --playback ansi
python3 main.py --mode compare-results
python3 main.py --mode record-demos
```

## Live Demo

Recommended in-class command:

```bash
python3 main.py --mode live-demo --algorithm ppo --playback gui --delay-ms 250
```

Play every algorithm with the same GUI in sequence:

```bash
python3 main.py --mode live-demo --algorithm all --playback gui --delay-ms 250
```

Terminal-only fallback:

```bash
python3 main.py --mode live-demo --algorithm ppo --playback ansi --delay-ms 250
```

Notes:

- `live-demo` loads the saved model for `q_learning`, `reinforce`, `a2c`, or `ppo` automatically unless you override `--model-path`.
- `--algorithm all` plays Q-Learning, REINFORCE, A2C, and PPO one after another using the same live demo window style.
- GUI playback uses a lightweight `matplotlib` window over the existing `rgb_array` renderer.
- If GUI playback is unavailable, `--playback gui` falls back to ANSI terminal playback automatically.
- Example alternate algorithms:

```bash
python3 main.py --mode live-demo --algorithm q_learning
python3 main.py --mode live-demo --algorithm reinforce
python3 main.py --mode live-demo --algorithm a2c
```

## GIF Demos

Regenerate the saved GIF demos:

```bash
python3 main.py --mode record-demos
```

## Algorithms

- Q-Learning
- REINFORCE
- A2C
- PPO

## Results

Saved summaries, plots, and comparison files are written under `results/`. The comparison step creates a final table, report, and presentation-friendly plots in `results/comparison/`.

## Q-Learning

![Q-Learning demo](assets/gifs/q_learning_demo.gif)

## REINFORCE

![REINFORCE demo](assets/gifs/reinforce_demo.gif)

## A2C

![A2C demo](assets/gifs/a2c_demo.gif)

## PPO

![PPO demo](assets/gifs/ppo_demo.gif)
