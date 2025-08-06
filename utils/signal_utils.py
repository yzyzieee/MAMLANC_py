# ◼️ 常用信号函数（滤波、normalize、resample）

"""Lightweight signal processing helpers used across the project."""

from __future__ import annotations

import numpy as np
from scipy import signal


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalise the signal to unit peak."""

    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak


def bandpass_filter(x: np.ndarray, fs: int, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter to ``x``."""

    b, a = signal.butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype="band")
    return signal.lfilter(b, a, x)


__all__ = ["normalize", "bandpass_filter"]

