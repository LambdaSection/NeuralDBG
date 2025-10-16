"""Deterministic seeding utilities.

This module provides a single entry point `set_seed` that attempts to set seeds
for Python's `random`, NumPy, PyTorch (if installed), and TensorFlow (if installed).
It returns a dict indicating which frameworks were successfully seeded.
"""
from __future__ import annotations

from typing import Dict

# Standard libs
import os  # for setting PYTHONHASHSEED
import random  # for seeding Python's RNG

# Third-party (optional)
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy is a core dep but be defensive
    np = None  # type: ignore

try:  # pragma: no cover - tested when torch is available
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

try:  # pragma: no cover - tested when tensorflow is available
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None  # type: ignore


def set_seed(seed: int, deterministic: bool = True) -> Dict[str, bool]:
    """Set seeds across common frameworks for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use across RNGs.
    deterministic : bool, default True
        If True, requests deterministic behavior when supported (e.g., PyTorch).

    Returns
    -------
    Dict[str, bool]
        A mapping of framework name -> whether seeding was applied successfully.
    """
    applied: Dict[str, bool] = {
        "python": False,
        "numpy": False,
        "torch": False,
        "tensorflow": False,
    }

    # Python built-ins
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)  # affects hash() determinism (future processes)
        random.seed(seed)  # Python's RNG
        applied["python"] = True
    except Exception:
        pass

    # NumPy
    if np is not None:
        try:
            np.random.seed(seed)
            applied["numpy"] = True
        except Exception:
            pass

    # PyTorch
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Request deterministic ops when possible
            try:
                torch.use_deterministic_algorithms(deterministic)
            except Exception:
                # Older PyTorch; fall back to cudnn flags when available
                try:
                    import torch.backends.cudnn as cudnn  # type: ignore
                    cudnn.deterministic = deterministic
                    cudnn.benchmark = False
                except Exception:
                    pass
            applied["torch"] = True
        except Exception:
            pass

    # TensorFlow
    if tf is not None:
        try:
            tf.random.set_seed(seed)
            applied["tensorflow"] = True
        except Exception:
            pass

    return applied

