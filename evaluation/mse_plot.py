"""Utilities for evaluating and visualising ANC performance."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def compute_mse(x: np.ndarray, win_len: int = 4096, base_db: float = 94.0) -> np.ndarray:
    """Compute sliding-window MSE in dB.

    Parameters
    ----------
    x:
        Error signal array.
    win_len:
        Window length for the moving average.
    base_db:
        Base level added to the decibel output to mimic SPL conventions.
    """

    x = np.asarray(x, dtype=float).flatten()
    win_len = min(win_len, x.size)
    window = np.ones(win_len) / win_len
    mse = np.convolve(x ** 2, window, mode="valid")
    return 10 * np.log10(mse + 1e-12) + base_db


def plot_mse(mse_curve: np.ndarray, title: str, save_path: Optional[str] = None) -> None:
    """Plot MSE curve and optionally save to disk."""

    plt.figure()
    plt.plot(mse_curve)
    plt.xlabel("Samples")
    plt.ylabel("MSE (dB)")
    plt.title(title)
    plt.grid(True)

    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        plt.savefig(save_path)

    plt.close()


__all__ = ["compute_mse", "plot_mse"]

