"""Seed handling.

One entry point sets every generator the pipeline uses, and the seed is written
into every checkpoint so a resumed run restores the same stream.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed the standard, numpy and torch generators and return the seed."""
    if seed < 0:
        raise ValueError("the seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def generator(seed: int) -> torch.Generator:
    handle = torch.Generator()
    handle.manual_seed(seed)
    return handle
