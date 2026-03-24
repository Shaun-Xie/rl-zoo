from __future__ import annotations

import os
from typing import Any, Mapping


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
