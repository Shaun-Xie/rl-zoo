from __future__ import annotations

import random
from typing import Any

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch if it is installed."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_action_space(action_space: Any, seed: int) -> None:
    """Seed an environment action space if it exposes Gymnasium's seed method."""

    if hasattr(action_space, "seed"):
        action_space.seed(seed)
