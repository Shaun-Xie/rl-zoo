"""Comparison helpers for cross-algorithm experiment outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from config import RESULTS_DIR

ALGORITHM_ORDER = ["q_learning", "reinforce", "a2c", "ppo"]
ALGORITHM_LABELS = {
    "q_learning": "Q-Learning",
    "reinforce": "REINFORCE",
    "a2c": "A2C",
    "ppo": "PPO",
}


def compare_runs(
    *,
    results_dir: str | Path = RESULTS_DIR,
) -> dict[str, Any]:
    """Build a lightweight comparison report from saved experiment outputs."""

    base_dir = Path(results_dir)
    comparison_dir = base_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = build_comparison_rows(base_dir)
    if not comparison_rows:
        raise FileNotFoundError(
            "No saved training summaries were found under results/. "
            "Run the algorithm training commands before comparing results."
        )

    comparison_csv_path = comparison_dir / "comparison_table.csv"
    report_path = comparison_dir / "comparison_report.md"

    save_comparison_csv(comparison_rows, comparison_csv_path)
    plot_paths = save_comparison_plots(comparison_rows, comparison_dir)
    report_text = build_markdown_report(comparison_rows, plot_paths)
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "comparison_rows": comparison_rows,
        "csv_path": comparison_csv_path,
        "report_path": report_path,
        "plot_paths": plot_paths,
    }


def build_comparison_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Read saved summaries and metrics into a single comparison table."""

    rows: list[dict[str, Any]] = []

    for algorithm_key in ALGORITHM_ORDER:
        summary_path = results_dir / algorithm_key / "training_summary.json"
        metrics_path = results_dir / algorithm_key / "training_metrics.csv"

        if not summary_path.exists() or not metrics_path.exists():
            continue

        summary = load_json(summary_path)
        metrics_rows = load_metrics_csv(metrics_path)

        average_steps = mean_value(metrics_rows, "steps")
        recent_average_steps = mean_value(metrics_rows[-50:], "steps")

        budget_type, budget_value = get_budget_field(summary)

        rows.append(
            {
                "algorithm": ALGORITHM_LABELS.get(algorithm_key, algorithm_key),
                "algorithm_key": algorithm_key,
                "average_reward": round(float(summary.get("average_reward", 0.0)), 4),
                "success_rate": round(float(summary.get("success_rate", 0.0)), 4),
                "recent_success_rate": round(float(summary.get("recent_success_rate", 0.0)), 4),
                "recent_reward_mean": round(float(summary.get("recent_reward_mean", 0.0)), 4),
                "average_steps": round(average_steps, 4) if average_steps is not None else "",
                "recent_average_steps": round(recent_average_steps, 4)
                if recent_average_steps is not None
                else "",
                "budget_type": budget_type,
                "budget_value": budget_value,
                "training_budget": f"{budget_value} {budget_type}" if budget_type else "",
                "model_path": str(summary.get("model_path", "")),
            }
        )

    return rows


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON summary file."""

    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_metrics_csv(path: Path) -> list[dict[str, str]]:
    """Load a training metrics CSV file."""

    with path.open("r", newline="", encoding="utf-8") as file_handle:
        return list(csv.DictReader(file_handle))


def mean_value(rows: list[dict[str, str]], key: str) -> float | None:
    """Compute a simple mean for a numeric CSV column."""

    values: list[float] = []
    for row in rows:
        raw_value = row.get(key)
        if raw_value in (None, ""):
            continue
        values.append(float(raw_value))

    if not values:
        return None

    return sum(values) / len(values)


def get_budget_field(summary: dict[str, Any]) -> tuple[str, int | float | str]:
    """Return the main training budget field used by a run."""

    if "episodes" in summary:
        return "episodes", summary["episodes"]
    if "total_timesteps" in summary:
        return "timesteps", summary["total_timesteps"]
    return "", ""


def save_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Save the merged comparison table as CSV."""

    fieldnames = [
        "algorithm",
        "average_reward",
        "success_rate",
        "recent_success_rate",
        "average_steps",
        "recent_average_steps",
        "budget_type",
        "budget_value",
        "training_budget",
        "model_path",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path


def save_comparison_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    """Save simple presentation-friendly cross-algorithm plots."""

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping comparison plots.")
        return {}

    labels = [str(row["algorithm"]) for row in rows]
    x_positions = list(range(len(rows)))
    bar_width = 0.36

    success_path = output_dir / "success_rate_comparison.png"
    reward_path = output_dir / "reward_comparison.png"

    overall_success = [float(row["success_rate"]) for row in rows]
    recent_success = [float(row["recent_success_rate"]) for row in rows]

    success_figure, success_axis = plt.subplots(figsize=(9, 4.8))
    success_axis.bar(
        [position - bar_width / 2 for position in x_positions],
        overall_success,
        width=bar_width,
        label="overall success",
        color="tab:blue",
    )
    success_axis.bar(
        [position + bar_width / 2 for position in x_positions],
        recent_success,
        width=bar_width,
        label="recent success",
        color="tab:green",
    )
    success_axis.set_title("Cross-Algorithm Success Rate")
    success_axis.set_ylabel("Success Rate")
    success_axis.set_xticks(x_positions)
    success_axis.set_xticklabels(labels)
    success_axis.set_ylim(0.0, 1.05)
    success_axis.grid(axis="y", alpha=0.25)
    success_axis.legend()
    success_figure.tight_layout()
    success_figure.savefig(success_path, dpi=150)
    plt.close(success_figure)

    overall_reward = [float(row["average_reward"]) for row in rows]
    recent_reward = [float(row["recent_reward_mean"]) for row in rows]

    reward_figure, reward_axis = plt.subplots(figsize=(9, 4.8))
    reward_axis.bar(
        [position - bar_width / 2 for position in x_positions],
        overall_reward,
        width=bar_width,
        label="overall reward",
        color="tab:orange",
    )
    reward_axis.bar(
        [position + bar_width / 2 for position in x_positions],
        recent_reward,
        width=bar_width,
        label="recent reward",
        color="tab:red",
    )
    reward_axis.set_title("Cross-Algorithm Reward")
    reward_axis.set_ylabel("Reward")
    reward_axis.set_xticks(x_positions)
    reward_axis.set_xticklabels(labels)
    reward_axis.grid(axis="y", alpha=0.25)
    reward_axis.legend()
    reward_figure.tight_layout()
    reward_figure.savefig(reward_path, dpi=150)
    plt.close(reward_figure)

    return {
        "success_rate_comparison": success_path,
        "reward_comparison": reward_path,
    }


def build_markdown_report(
    rows: list[dict[str, Any]],
    plot_paths: dict[str, Path],
) -> str:
    """Create a concise markdown comparison report."""

    lines = [
        "# Comparison Report",
        "",
        "This report summarizes the saved training outputs in `results/<algorithm>/`.",
        "",
        "## Comparison Table",
        "",
        "| Algorithm | Avg Reward | Success Rate | Recent Success | Avg Steps | Budget | Model |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['algorithm']} | "
            f"{float(row['average_reward']):.4f} | "
            f"{float(row['success_rate']):.4f} | "
            f"{float(row['recent_success_rate']):.4f} | "
            f"{float(row['average_steps']):.2f} | "
            f"{row['training_budget']} | "
            f"`{row['model_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Short Takeaways",
            "",
            "- Q-Learning and PPO produced the strongest saved maze runs by overall success rate and recent performance.",
            "- REINFORCE improved over training but remained less stable than the other methods in this fixed maze.",
            "- A2C eventually became strong, but it needed a larger timestep budget than PPO in these runs.",
            "",
            "## PPO For Later MicroRTS Work",
            "",
            "PPO is the strongest candidate for later MicroRTS-oriented experiments because it matched top maze performance while still using a neural policy and a stable on-policy training setup. Compared with tabular Q-Learning, PPO is a much better fit for larger state spaces, and compared with REINFORCE and A2C here, it reached strong results with a cleaner training budget.",
            "",
            "## W&B Note",
            "",
            "Weights & Biases is optional in this repo. Final official runs can be logged there and used as supporting material when writing the course report.",
        ]
    )

    if plot_paths:
        lines.extend(
            [
                "",
                "## Saved Comparison Plots",
                "",
            ]
        )
        for plot_name, plot_path in plot_paths.items():
            lines.append(f"- `{plot_name}`: `{plot_path}`")

    lines.append("")
    return "\n".join(lines)
