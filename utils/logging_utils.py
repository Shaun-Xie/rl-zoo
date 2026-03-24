from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def wandb_is_available() -> bool:
    """Return True when the wandb package can be imported."""

    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def maybe_init_wandb(
    *,
    enabled: bool,
    project: str,
    config: Mapping[str, Any] | None = None,
    run_name: str | None = None,
) -> Any | None:
    """Start an optional W&B run and fall back quietly when unavailable."""

    if not enabled:
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing without W&B logging.")
        return None

    wandb_mode = os.getenv("WANDB_MODE", "").strip().lower()
    has_api_key = bool(os.getenv("WANDB_API_KEY"))

    if wandb_mode not in {"offline", "disabled"} and not has_api_key:
        print("W&B logging requested, but no WANDB_API_KEY was found. Skipping W&B.")
        return None

    try:
        return wandb.init(
            project=project,
            name=run_name,
            config=dict(config or {}),
            reinit=True,
        )
    except Exception as exc:
        print(f"Unable to start W&B logging: {exc}")
        return None


def log_metrics(
    metrics: Mapping[str, float | int],
    *,
    step: int | None = None,
    run: Any | None = None,
) -> None:
    """Log a flat metric dictionary to an active W&B run if one exists."""

    if run is None:
        return

    payload = dict(metrics)
    try:
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=step)
    except Exception as exc:
        print(f"Unable to log metrics to W&B: {exc}")


def finish_wandb(run: Any | None) -> None:
    """Close a W&B run if logging was enabled successfully."""

    if run is None:
        return

    try:
        run.finish()
    except Exception as exc:
        print(f"Unable to finish W&B run cleanly: {exc}")


def save_episode_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> Path:
    """Save per-episode metrics to a CSV file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Save a small JSON artifact to disk."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(_make_json_safe(data), file_handle, indent=2)
        file_handle.write("\n")

    return output_path


def _make_json_safe(value: Any) -> Any:
    """Convert common Python and NumPy values into JSON-friendly types."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
