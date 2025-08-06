# 🔶 加载 .mat 文件并重采样
"""Utility functions for MATLAB I/O.

This module currently provides a helper for saving data to ``.mat`` files.
In the project the function is mainly used to store the learned MAML
control filter coefficients or any other intermediate results that need to
be inspected from MATLAB.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from scipy.io import loadmat, savemat
from scipy.signal import resample_poly


def save_mat(filepath: str, data: Dict[str, Any]) -> None:
    """Save variables to a ``.mat`` file.

    Parameters
    ----------
    filepath:
        Destination path of the ``.mat`` file.  Parent directories are
        created automatically.
    data:
        A mapping of variable names to the arrays that should be saved.  The
        values are typically ``numpy`` arrays but any object supported by
        :func:`scipy.io.savemat` can be used.

    Notes
    -----
    This helper wraps :func:`scipy.io.savemat` and ensures the directory
    structure exists.  It keeps the rest of the code base concise and makes
    saving learned filters straightforward.
    """

    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    savemat(filepath, data)


def load_and_resample_mat(filepath: str, key: str, fs_target: int) -> Any:
    """Load an array from a ``.mat`` file and resample to ``fs_target``."""

    data = loadmat(filepath)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {filepath}")
    arr = np.squeeze(data[key])
    fs_orig = data.get("fs", fs_target)
    return resample_poly(arr, fs_target, int(fs_orig))


__all__ = ["save_mat", "load_and_resample_mat"]


